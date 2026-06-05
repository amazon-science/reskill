"""
RESKILL Trainer: Reconciled Skill-Policy Co-Evolution.

Evolves a modular skill library through an RL-integrated pipeline:
  Sample Groups → Contrastive → Aggregate → Diagnose → Recommend → Author → Verify → A/B Test

  - Contrastive: 30 parallel per-task-group success/failure comparisons
  - Aggregate: Deterministic indexing of insights (no LLM)
  - Diagnose: Assertion CRUD + semantic grouping of insights
  - Recommend: Strategic skill operations (ADD/MODIFY/DELETE)
  - Author: Write skill content with trigger patterns
  - Verify: Trigger rate gate ensures skills fire in >=50% of episodes
  - A/B Test: Bundle-level Thompson Sampling (new vs old version)

Skills are stored as SKILL.md files and injected into the agent's prompt
at specific moments during task execution based on trigger conditions.

Trainer chain: RayPPOTrainer -> ReSkillTrainer
"""

import json
import os
import traceback

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from reskill.skill_creator.analysis.experience_reservoir import ExperienceReservoir
from reskill.skill_serving.skill_registry import SkillRegistry
from reskill.skill_serving.skill_ab_tracker import VersionABTracker
from reskill.skill_serving.trigger_matcher import TriggerMatcher
from reskill.skill_serving.skill_loader import SkillLoader
from reskill.skill_creator.diagnosis.assertion_engine import AssertionEngine


class ReSkillTrainer(RayPPOTrainer):
    """RESKILL Trainer with versioned skill evolution and bundle-level A/B testing."""

    def __init__(self, *args, **kwargs):
        self.device_name = kwargs.pop('device_name', 'cuda')
        self.traj_collector = kwargs.pop('traj_collector', None)
        self.envs = kwargs.pop('envs', None)
        self.val_envs = kwargs.pop('val_envs', None)
        super().__init__(*args, **kwargs)
        self.ref_in_actor = self.config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        self._trigger_cfg = self.config.env.get('trigger_skills', {})
        self._save_dir = self.config.trainer.get('default_local_dir', './outputs')

        self.reservoir = ExperienceReservoir(
            max_size=self._trigger_cfg.get('reservoir_max_size', 200),
            dedup_by=self._trigger_cfg.get('reservoir_dedup_by', 'task'),
        )

        self.skill_registry = SkillRegistry(
            max_active_skills=self._trigger_cfg.get('max_active_skills', 8),
        )

        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        resume_mode = self.config.trainer.get('resume_mode', 'disable')
        _has_model_ckpt = False
        if resume_mode == 'resume_path':
            _has_model_ckpt = True
        elif resume_mode != 'disable':
            ckpt_dir = self.config.trainer.get('default_local_dir', './outputs')
            _has_model_ckpt = find_latest_ckpt_path(ckpt_dir) is not None
        _should_load_skill_state = _has_model_ckpt

        registry_path = os.path.join(self._save_dir, 'skill_registry.json')
        skill_library_dir = os.path.join(self._save_dir, 'skill_library')
        _has_skill_files = os.path.exists(registry_path) or os.path.isdir(skill_library_dir)

        if _should_load_skill_state or _has_skill_files:
            self.skill_registry.load(registry_path,
                                     skill_library_dir=skill_library_dir)
            print(f"[RESKILL] Resumed registry: "
                  f"{len(self.skill_registry.get_active_skills())} active, "
                  f"{len(self.skill_registry.get_testing_skills())} testing")
        else:
            print(f"[RESKILL] Starting with empty skill registry")

        self.version_ab_tracker = VersionABTracker(
            test_steps=self._trigger_cfg.get('ab_test_steps', 5),
            min_episodes=self._trigger_cfg.get('min_version_episodes', 50),
            prob_clamp_min=self._trigger_cfg.get('prob_clamp_min', 0.15),
            prob_clamp_max=self._trigger_cfg.get('prob_clamp_max', 0.85),
            uniform_sampling=self._trigger_cfg.get('uniform_sampling', False),
            weighting_strategy=self._trigger_cfg.get('weighting_strategy', 'none'),
        )

        if _should_load_skill_state:
            vab_path = os.path.join(self._save_dir, 'version_ab_tracker.json')
            if os.path.exists(vab_path):
                self.version_ab_tracker.load(vab_path)

        self.assertion_engine = AssertionEngine()
        if _should_load_skill_state:
            assertions_path = os.path.join(self._save_dir, 'assertions_latest.json')
            self.assertion_engine.load(assertions_path)

        self.skill_loader = SkillLoader(
            registry=self.skill_registry,
        )

        self._version_history = []

        if _should_load_skill_state:
            version_history_path = os.path.join(self._save_dir, 'version_history.json')
            if os.path.exists(version_history_path):
                try:
                    with open(version_history_path) as f:
                        self._version_history = json.load(f)
                except Exception:
                    pass

        print(f"[RESKILL] Initialized: "
              f"{len(self.skill_registry.get_active_skills())} active skills, "
              f"{len(self.assertion_engine.assertions)} assertions")

    # ------------------------------------------------------------------
    # Inject components into env managers
    # ------------------------------------------------------------------

    def _inject_skill_components(self):
        """Inject prompt memory + version A/B tracker into env managers."""
        for env_mgr in [self.envs, self.val_envs]:
            if hasattr(env_mgr, 'set_skill_components'):
                env_mgr.set_skill_components(
                    skill_loader=self.skill_loader,
                    version_ab_tracker=self.version_ab_tracker,
                )
            else:
                print(f"[RESKILL] WARNING: {type(env_mgr).__name__} "
                      f"does not support set_skill_components()")

    # ------------------------------------------------------------------
    # Lazy-init pipeline stages
    # ------------------------------------------------------------------

    def _get_skill_stages(self):
        """Lazy-init pipeline stages (all share one LLMClient)."""
        if not hasattr(self, '_skill_stages'):
            from reskill.utils.llm_client import create_llm_client
            from reskill.skill_creator.analysis.episode_analyzer import EpisodeAnalyzer
            from reskill.skill_creator.diagnosis.batch_diagnoser import BatchDiagnoser
            from reskill.skill_creator.authoring.skill_recommender import SkillRecommender
            from reskill.skill_creator.authoring.skill_author import SkillAuthor

            llm_cfg = self._trigger_cfg.get('skill_llm', {})
            client = create_llm_client(llm_cfg)

            prompt_dir = self._trigger_cfg.get('skill_prompt_dir', None)

            def _prompt(name):
                if prompt_dir:
                    return os.path.join(prompt_dir, name)
                return None

            self._skill_stages = {
                'episode_analyzer': EpisodeAnalyzer(
                    client, prompt_path=_prompt('episode_analyzer.md')),
                'batch_diagnoser': BatchDiagnoser(
                    client, prompt_path=_prompt('batch_diagnoser.md')),
                'recommender': SkillRecommender(
                    client, prompt_path=_prompt('skill_recommender.md'),
                    n_version_history=self._trigger_cfg.get('n_version_history', 5)),
                'author': SkillAuthor(
                    client, prompt_path=_prompt('skill_author.md'),
                    force_general_trigger=self._trigger_cfg.get('force_general_trigger', False)),
            }
            print(f"[RESKILL] Initialized pipeline stages "
                  f"(backend={llm_cfg.get('backend', 'bedrock')}, "
                  f"model={llm_cfg.get('model', 'default')})")
        return self._skill_stages

    def _compute_per_skill_diagnostics(self):
        """Compute per-skill diagnostics (trigger rate, age)."""
        freq_stats = {}
        if hasattr(self.envs, 'get_trigger_frequency_stats'):
            freq_stats = self.envs.get_trigger_frequency_stats()

        per_skill_diagnostics = {}
        for skill in self.skill_registry.get_active_skills():
            diag = {'age_steps': self.global_steps - skill.created_at_step}
            diag['trigger_rate'] = freq_stats.get(skill.skill_id, 0.0)
            per_skill_diagnostics[skill.skill_id] = diag

        trigger_freq = dict(freq_stats) if freq_stats else {}
        if trigger_freq:
            id_to_name = self.skill_registry.id_to_name_map()
            trigger_freq = {
                id_to_name.get(sid, sid): freq
                for sid, freq in trigger_freq.items()
            }

        return per_skill_diagnostics, trigger_freq

    # ------------------------------------------------------------------
    # Hook B: Version-level A/B recording from rollout episodes
    # ------------------------------------------------------------------

    def _hook_b_after_rollout(self, gen_batch_output, metrics):
        """After rollout: collect failures, record A/B outcomes, trigger evolution."""
        train_episodes = []
        self._build_episode_trajectories(gen_batch_output, train_episodes)

        traj_uids = gen_batch_output.non_tensor_batch['traj_uid']
        unique_uids = list(dict.fromkeys(str(u) for u in traj_uids))

        version_assignments = {}
        if self.version_ab_tracker.is_testing and hasattr(self.envs, 'get_version_assignments'):
            version_assignments = self.envs.get_version_assignments()

        uid_to_condition = {}
        for slot_idx, uid in enumerate(unique_uids):
            if version_assignments:
                is_new = version_assignments.get(slot_idx, True)
                uid_to_condition[uid] = 'new' if is_new else 'old'
            else:
                uid_to_condition[uid] = 'old'

        for ep in train_episodes:
            ep['ab_condition'] = uid_to_condition.get(
                str(ep.get('traj_uid', '')), 'old')

        current_pv = str(self.skill_loader.version)
        self.reservoir.add_episodes(train_episodes, prompt_version=current_pv)

        if self.version_ab_tracker.is_testing:
            ep_success = gen_batch_output.non_tensor_batch.get(
                'episode_success_rate', np.zeros(len(traj_uids)))

            uid_to_success = {}
            for idx in range(len(traj_uids)):
                uid = str(traj_uids[idx])
                if uid not in uid_to_success:
                    uid_to_success[uid] = float(ep_success[idx])

            for slot_idx, uid in enumerate(unique_uids):
                success = uid_to_success.get(uid, 0.0)
                is_new = version_assignments.get(slot_idx, True)
                self.version_ab_tracker.record_episode_outcome(
                    is_new_version=is_new, success=success)

            if hasattr(self.envs, 'get_episode_trigger_data'):
                trigger_data = self.envs.get_episode_trigger_data()
                for slot_idx, (triggered_ids, _) in trigger_data.items():
                    for skill_id in triggered_ids:
                        skill = self.skill_registry.get_skill(skill_id)
                        if skill:
                            self.version_ab_tracker.record_skill_trigger(skill.name)

            self.version_ab_tracker.record_step()

            if self.version_ab_tracker.should_decide():
                self._handle_ab_test_decision()

            metrics.update(self.version_ab_tracker.get_test_metrics())

        if not self.version_ab_tracker.is_testing:
            try:
                self._evolve_skills()
            except Exception as e:
                print(f"[RESKILL] WARNING: Evolution trigger failed: {e}")
                traceback.print_exc()

        metrics['skills/num_active'] = len(self.skill_registry.get_active_skills())
        metrics['skills/is_testing'] = 1.0 if self.version_ab_tracker.is_testing else 0.0
        metrics['skills/reservoir_size'] = len(self.reservoir)
        metrics['skills/reservoir_failures'] = self.reservoir.num_failures()
        metrics['skills/reservoir_successes'] = self.reservoir.num_successes()
        metrics['skills/reservoir_unique_tasks'] = self.reservoir.num_unique_tasks()
        metrics['skills/evolution_cycle'] = len(self._version_history)
        metrics['assertions/count'] = len(self.assertion_engine.assertions)

        if self.assertion_engine.assertions:
            pass_rates = self.assertion_engine.get_pass_rates()
            valid_rates = [r for r in pass_rates.values() if r is not None]
            if valid_rates:
                metrics['grader/mean_pass_rate'] = sum(valid_rates) / len(valid_rates)

    def _validate(self):
        import torch

        from verl import DataProto

        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_episode_success = []
        sample_episode_f1 = {'file_f1': [], 'module_f1': [], 'entity_f1': []}
        sample_episodes = []

        if hasattr(self.val_envs, 'reward_manager') and self.val_envs.reward_manager is not None:
            self.val_envs.reward_manager.last_decomposed_results.clear()

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            sample_inputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids])

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"):
                if key in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
            del test_batch
            test_batch = test_output_gen_batch

            output_ids = test_output_gen_batch.batch["responses"]
            sample_outputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids])

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            sample_episode_success.extend(test_batch.non_tensor_batch['episode_success_rate'].tolist())

            for f1_key in ('episode_file_f1', 'episode_module_f1', 'episode_entity_f1'):
                short_key = f1_key.replace('episode_', '')
                if f1_key in test_batch.non_tensor_batch:
                    sample_episode_f1[short_key].extend(test_batch.non_tensor_batch[f1_key].tolist())

            self._build_episode_trajectories(
                test_output_gen_batch,
                sample_episodes,
                data_sources=test_batch.non_tensor_batch.get('data_source'),
            )

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])

            for k in test_batch.non_tensor_batch.keys():
                if ('success_rate' in k or k in ('file_f1', 'module_f1', 'entity_f1')) and not k.startswith('episode_'):
                    success_rate_dict.setdefault(k, []).append(test_batch.non_tensor_batch[k][0])
                    if 'success_rate' in k:
                        for i in range(1, len(test_batch.non_tensor_batch[k])):
                            assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i]

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source_reward.setdefault(data_sources[i], []).append(reward_tensor[i].item())

        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        data_source_tool_calling = {}
        for i in range(unique_tool_callings.shape[0]):
            data_source_tool_calling.setdefault(unique_data_sources[i], []).append(unique_tool_callings[i].item())

        episode_success_arr = np.array(sample_episode_success)
        unique_episode_success = episode_success_arr[unique_idx]
        data_source_episode_sr = {}
        for i in range(len(unique_episode_success)):
            data_source_episode_sr.setdefault(unique_data_sources[i], []).append(unique_episode_success[i])

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val_score/{data_source}'] = np.mean(rewards)
        for k, v in success_rate.items():
            metric_dict['val_score/overall' if k == 'success_rate' else f'val_stats/{k}'] = v
        for data_source, sr_values in data_source_episode_sr.items():
            metric_dict[f'val_score/{data_source}_sr'] = np.mean(sr_values)

        for f1_key in ('file_f1', 'module_f1', 'entity_f1'):
            if sample_episode_f1.get(f1_key):
                f1_arr = np.array(sample_episode_f1[f1_key])
                unique_f1 = f1_arr[unique_idx] if len(f1_arr) == len(data_sources) else f1_arr
                metric_dict[f'val_stats/{f1_key}'] = float(np.mean(unique_f1))
                for i in range(len(unique_f1)):
                    ds = unique_data_sources[i] if i < len(unique_data_sources) else 'unknown'
                    metric_dict.setdefault(f'val_stats/{ds}/{f1_key}', []).append(float(unique_f1[i]))
                for key in list(metric_dict.keys()):
                    if key.endswith(f'/{f1_key}') and isinstance(metric_dict[key], list):
                        metric_dict[key] = np.mean(metric_dict[key])

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val_stats/{data_source}/tool_call_count'] = np.mean(tool_calls)

        if sample_episodes:
            ep_lengths = [len(ep['trajectory']) for ep in sample_episodes]
            ep_scores = [ep.get('score', 0.0) for ep in sample_episodes]
            metric_dict['val_stats/episode_length/mean'] = np.mean(ep_lengths)
            metric_dict['val_stats/episode_length/max'] = np.max(ep_lengths)
            metric_dict['val_stats/episode_length/min'] = np.min(ep_lengths)
            metric_dict['val_stats/episode_task_score/mean'] = np.mean(ep_scores)
            metric_dict['val_stats/num_episodes'] = len(sample_episodes)

            prompt_char_lens = []
            response_char_lens = []
            for ep in sample_episodes:
                for step in ep['trajectory']:
                    prompt_char_lens.append(len(step['observation']))
                    response_char_lens.append(len(step['action']))
            if prompt_char_lens:
                metric_dict['val_stats/prompt_chars/mean'] = np.mean(prompt_char_lens)
                metric_dict['val_stats/prompt_chars/max'] = float(np.max(prompt_char_lens))
                metric_dict['val_stats/response_chars/mean'] = np.mean(response_char_lens)
                metric_dict['val_stats/response_chars/max'] = float(np.max(response_char_lens))

        if getattr(self.config.trainer, 'val_only', False) and sample_episodes:
            dump_dir = self.config.trainer.default_local_dir
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, 'episode_results.jsonl')
            with open(dump_path, 'w') as f:
                for ep in sample_episodes:
                    f.write(json.dumps({
                        'traj_uid': ep.get('traj_uid', ''),
                        'goal_idx': ep.get('goal_idx', -1),
                        'task_type': ep.get('task_type', ''),
                        'data_source': ep.get('data_source', ''),
                        'success': ep.get('success', 0.0),
                        'score': ep.get('score', 0.0),
                        'num_steps': len(ep.get('trajectory', [])),
                        'task_snippet': ep.get('task', '')[:300],
                        'trajectory': ep.get('trajectory', []),
                    }) + '\n')
            print(f"Dumped {len(sample_episodes)} episode results to {dump_path}")

        return metric_dict

    def _build_episode_trajectories(self, gen_batch_output, episode_list, data_sources=None):
        traj_uids = gen_batch_output.non_tensor_batch['traj_uid']
        responses = gen_batch_output.batch['responses']
        input_ids = gen_batch_output.batch['input_ids']
        active_masks = gen_batch_output.non_tensor_batch.get('active_masks', np.ones(len(traj_uids), dtype=bool))
        ep_success = gen_batch_output.non_tensor_batch.get('episode_success_rate', np.zeros(len(traj_uids)))
        ep_rewards = gen_batch_output.non_tensor_batch.get('episode_rewards', np.zeros(len(traj_uids)))
        ep_goal_idxs = gen_batch_output.non_tensor_batch.get('goal_idx', np.full(len(traj_uids), -1, dtype=np.int64))

        from collections import OrderedDict
        episodes = OrderedDict()
        for idx in range(len(traj_uids)):
            uid = str(traj_uids[idx])
            if uid not in episodes:
                episodes[uid] = {
                    'steps': [],
                    'success': float(ep_success[idx]),
                    'score': float(ep_rewards[idx]),
                    'goal_idx': int(ep_goal_idxs[idx]),
                    'data_source': str(data_sources[idx]) if data_sources is not None else '',
                }
            if active_masks[idx]:
                episodes[uid]['steps'].append({
                    'action': self.tokenizer.decode(responses[idx], skip_special_tokens=True),
                    'observation': self.tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                })

        for uid, ep in episodes.items():
            task = ep['steps'][0]['observation'] if ep['steps'] else ''
            episode_list.append({
                'traj_uid': uid,
                'task': task,
                'trajectory': ep['steps'],
                'task_type': self._detect_task_type_from_input(task),
                'success': ep['success'],
                'score': ep['score'],
                'goal_idx': ep['goal_idx'],
                'data_source': ep.get('data_source', ''),
            })

    def _collect_failed_trajectories(self, episodes: list) -> list:
        failed = []
        for ep in episodes:
            if ep.get('success', 1.0) < 1.0:
                failed.append({
                    'traj_uid': ep.get('traj_uid'),
                    'task': ep['task'],
                    'trajectory': ep['trajectory'],
                    'task_type': ep['task_type'],
                })
        return failed[:10]

    def _detect_task_type_from_input(self, inp: str) -> str:
        import re

        task_match = re.search(r'Your task is to:\s*(.+?)(?:\n|$)', inp)
        task_desc = task_match.group(1).lower().strip() if task_match else ''
        if 'clean' in task_desc:
            return 'pick_clean_then_place_in_recep'
        if 'hot' in task_desc or 'heat' in task_desc:
            return 'pick_heat_then_place_in_recep'
        if 'cool' in task_desc:
            return 'pick_cool_then_place_in_recep'
        if task_desc.startswith('look at') or task_desc.startswith('examine'):
            return 'look_at_obj_in_light'
        if 'two' in task_desc:
            return 'pick_two_obj_and_place'
        if task_desc:
            return 'pick_and_place_simple'
        return 'unknown'

    def _handle_ab_test_decision(self):
        """Accept or reject the tested version bundle."""
        accepted, stats = self.version_ab_tracker.decide()
        action = "ACCEPTED" if accepted else "REJECTED"

        if not accepted:
            snapshot = stats.get('snapshot', {})
            if snapshot:
                self.skill_registry.revert_version(snapshot)
                print(f"[RESKILL] Version reverted "
                      f"({len(snapshot.get('operations', []))} operations)")

        for env_mgr in [self.envs, self.val_envs]:
            if hasattr(env_mgr, 'set_version_old_registry'):
                env_mgr.set_version_old_registry(None)

        self.skill_loader.bump_version()

        print(f"[RESKILL] Version '{stats.get('version_id', '?')}' {action}: "
              f"new_mean={stats.get('new_mean', 0):.3f} "
              f"({stats.get('new_count', 0)} eps) vs "
              f"old_mean={stats.get('old_mean', 0):.3f} "
              f"({stats.get('old_count', 0)} eps)")

        os.makedirs(self._save_dir, exist_ok=True)
        skill_library_dir = os.path.join(self._save_dir, 'skill_library')
        self.skill_registry.save(os.path.join(self._save_dir, 'skill_registry.json'),
                                 skill_library_dir=skill_library_dir)
        self.version_ab_tracker.save(os.path.join(self._save_dir, 'version_ab_tracker.json'))

        self._version_history.append({
            'step': self.global_steps,
            'accepted': accepted,
            'new_mean': stats.get('new_mean', 0),
            'old_mean': stats.get('old_mean', 0),
            'operations': stats.get('operations', []),
            'per_skill_trigger_counts': stats.get('per_skill_trigger_counts', {}),
        })
        self._save_history()

    # ------------------------------------------------------------------
    # Evolution: RESKILL Pipeline
    # ------------------------------------------------------------------

    def _evolve_skills(self):
        """RESKILL skill evolution pipeline.

        Pipeline: Sample Groups → Contrastive → Aggregate → Diagnose →
                  Recommend → Author → Verify → A/B Test
        """
        trigger_cfg = self._trigger_cfg
        min_size = trigger_cfg.get('min_reservoir_size', 200)
        if len(self.reservoir) < min_size:
            return

        save_dir = self._save_dir
        os.makedirs(save_dir, exist_ok=True)
        reservoir_episodes = self.reservoir.buffer

        stages = self._get_skill_stages()

        # === Stage 0: Sample task groups ===
        n_groups = trigger_cfg.get('n_groups', 30)
        task_groups = self.reservoir.sample_task_groups(
            n_groups=n_groups,
            min_success_ratio=trigger_cfg.get('min_success_ratio', 0.3),
        )
        if not task_groups:
            return
        total_eps = sum(len(g.episodes) for g in task_groups)
        print(f"[RESKILL] Sampled {len(task_groups)} groups ({total_eps} episodes)")

        # === Stage 1: Per-Episode Analysis (parallel) ===
        insights = stages['episode_analyzer'].analyze_groups(
            task_groups=task_groups,
            skill_registry=self.skill_registry,
            max_workers=trigger_cfg.get('max_workers', 16),
            save_dir=save_dir,
            step=self.global_steps,
        )
        if not insights:
            print("[RESKILL] No insights produced, skipping")
            return
        n_high = sum(1 for ins in insights if ins.get('confidence') == 'high')
        print(f"[RESKILL] {len(insights)} insights ({n_high} high confidence)")

        # === Bridge: Summarize for batch diagnosis (deterministic) ===
        from reskill.skill_creator.analysis.episode_analyzer import EpisodeAnalyzer
        aggregation = EpisodeAnalyzer.summarize(insights)

        # === Stage 2: Batch Diagnosis ===
        diagnosis_output = None
        try:
            self.assertion_engine.grade_batch(reservoir_episodes)
            diagnosis_output = stages['batch_diagnoser'].diagnose(
                assertions_summary=self.assertion_engine.get_assertions_summary(),
                insight_summaries_text=aggregation['diagnoser_summary'],
                current_skills_summary=self.skill_registry.to_summary_text(),
                current_assertions=self.assertion_engine.assertions,
                save_dir=save_dir,
                step=self.global_steps,
            )
            if diagnosis_output:
                crud_ops = diagnosis_output.get('assertion_operations', [])
                if crud_ops:
                    self.assertion_engine.apply_crud(
                        crud_ops, current_step=self.global_steps)
                    self.assertion_engine.save(save_dir, step=self.global_steps)
                self.assertion_engine.reset_pass_rates()
        except Exception as e:
            print(f"[RESKILL] WARNING: BatchDiagnoser failed: {e}")
            traceback.print_exc()

        # === Stage 3: Skill Recommender ===
        per_skill_diagnostics, trigger_freq = self._compute_per_skill_diagnostics()
        skill_library_summary = self.skill_registry.to_diagnostic_summary(
            per_skill_diagnostics=per_skill_diagnostics)

        action_vocabulary = self._get_action_vocabulary(self.config.env)

        recommendations = stages['recommender'].recommend(
            diagnosis=diagnosis_output or {},
            skill_library_summary=skill_library_summary,
            version_history=self._version_history,
            insights=insights,
            save_dir=save_dir,
            step=self.global_steps,
            action_vocabulary=action_vocabulary,
        )
        if not recommendations or not recommendations.get('operations'):
            print("[RESKILL] Recommender returned no operations, skipping")
            return

        # === Stage 4 + Verify: Author → Verify (retry loop) ===
        min_trigger_rate = trigger_cfg.get('min_trigger_rate', 0.5)
        max_retries = trigger_cfg.get('max_proposal_retries', 3)

        version_proposal = None
        retry_feedback = None
        for attempt in range(max_retries):
            version_proposal = stages['author'].author(
                recommendations=recommendations,
                insights=insights,
                skill_registry=self.skill_registry,
                skill_library_summary=skill_library_summary,
                save_dir=save_dir,
                step=self.global_steps,
                retry_feedback=retry_feedback,
                action_vocabulary=action_vocabulary,
            )
            if not version_proposal:
                retry_feedback = None
                continue

            if self._phase_verify(version_proposal, reservoir_episodes,
                                  min_trigger_rate):
                break
            retry_feedback = self._collect_verify_feedback(
                version_proposal, reservoir_episodes, min_trigger_rate)
            version_proposal = None

        if not version_proposal:
            print("[RESKILL] No valid proposal after retries, skipping")
            return

        # === Phase 5: A/B TEST ===
        self._phase_ab_test(version_proposal)

    def _collect_verify_feedback(self, version_proposal, reservoir_episodes,
                                 min_trigger_rate):
        """Collect feedback on verify failures for author retry."""
        feedback = []
        for op in version_proposal.get('operations', []):
            pattern = None
            name = '?'
            if op['type'] == 'add':
                skill_def = op.get('skill', {})
                if skill_def.get('trigger_type') == 'action_pattern':
                    pattern = skill_def.get('trigger_pattern')
                name = skill_def.get('name', '?')
            elif op['type'] == 'modify':
                pattern = op.get('changes', {}).get('trigger_pattern')
                name = op.get('target_skill_name', '?')

            if pattern:
                rate = TriggerMatcher.estimate_trigger_rate(
                    trigger_type='action_pattern',
                    trigger_pattern=pattern,
                    trajectories=reservoir_episodes,
                )
                if rate < min_trigger_rate:
                    feedback.append({
                        "op_type": op['type'],
                        "name": name,
                        "reason": "trigger_rate_too_low",
                        "trigger_pattern": pattern,
                        "measured_rate": rate,
                        "threshold": min_trigger_rate,
                    })
        return feedback

    def _phase_verify(self, version_proposal, reservoir_episodes, min_trigger_rate):
        """Trigger rate gate for ADD/MODIFY operations."""


        for op in version_proposal.get('operations', []):
            if op['type'] == 'add':
                skill_def = op.get('skill', {})
                trigger_rate = TriggerMatcher.estimate_trigger_rate(
                    trigger_type=skill_def.get('trigger_type', 'general'),
                    trigger_pattern=skill_def.get('trigger_pattern'),
                    trajectories=reservoir_episodes,
                )
                if trigger_rate < min_trigger_rate:
                    print(f"[RESKILL] ADD '{skill_def.get('name')}' trigger rate "
                          f"too low: {trigger_rate:.2f} < {min_trigger_rate}")
                    return False

            elif op['type'] == 'modify':
                new_pattern = op.get('changes', {}).get('trigger_pattern')
                if new_pattern:
                    trigger_rate = TriggerMatcher.estimate_trigger_rate(
                        trigger_type='action_pattern',
                        trigger_pattern=new_pattern,
                        trajectories=reservoir_episodes,
                    )
                    if trigger_rate < min_trigger_rate:
                        target = op.get('target_skill_name', '?')
                        print(f"[RESKILL] MODIFY '{target}' new trigger_pattern "
                              f"rate too low: {trigger_rate:.2f} < {min_trigger_rate}")
                        return False

        max_chars = self._trigger_cfg.get('max_skill_content_chars', 500)
        if max_chars is not None:
            for op in version_proposal.get('operations', []):
                content = None
                if op['type'] == 'add':
                    content = op.get('skill', {}).get('content')
                elif op['type'] == 'modify' and 'content' in op.get('changes', {}):
                    content = op['changes']['content']

                if content and isinstance(content, dict):
                    chars = sum(len(str(v)) for v in content.values()
                                if isinstance(v, str))
                    if isinstance(content.get('examples'), list):
                        chars += sum(len(str(e)) for e in content['examples'])
                    if chars > max_chars:
                        return False

        return True

    def _phase_ab_test(self, version_proposal):
        """Snapshot, apply version, start Thompson sampling."""
        operations = version_proposal.get('operations', [])

        old_registry = self.skill_registry.snapshot()
        snapshot = self.skill_registry.apply_version(operations, step=self.global_steps)

        version_id = f"v{len(self._version_history) + 1}_step{self.global_steps}"
        self.version_ab_tracker.start_version_test(
            version_id=version_id,
            operations=operations,
            snapshot=snapshot,
            step=self.global_steps,
        )

        for env_mgr in [self.envs, self.val_envs]:
            if hasattr(env_mgr, 'set_version_old_registry'):
                env_mgr.set_version_old_registry(old_registry)

        skill_library_dir = os.path.join(self._save_dir, 'skill_library')
        self.skill_registry.save(os.path.join(self._save_dir, 'skill_registry.json'),
                                 skill_library_dir=skill_library_dir)
        self.version_ab_tracker.save(os.path.join(self._save_dir, 'version_ab_tracker.json'))

        op_summary = ', '.join(
            f"{op['type'].upper()}: {op.get('target_skill_name') or op.get('skill', {}).get('name', '?')}"
            for op in operations
        )
        print(f"[RESKILL] Version '{version_id}' proposed: [{op_summary}]")
        print(f"[RESKILL] Starting version A/B test")

    # ------------------------------------------------------------------
    # Override fit() to inject components and add Hook B
    # ------------------------------------------------------------------

    def fit(self):
        """Training loop with versioned skill hooks.

        Differences from base RayPPOTrainer.fit():
        - Injects skill components before first rollout
        - Hook B: A/B recording + evolution trigger (decoupled from validation)
        """
        from pprint import pprint
        from tqdm import tqdm

        import torch
        import ray
        from omegaconf import OmegaConf

        from verl import DataProto
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_throughout_metrics,
            compute_timing_metrics,
        )
        from verl.trainer.ppo.reward import compute_reward, compute_reward_async
        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_response_mask
        from verl.utils.debug import marked_timer
        from verl.utils.metric import reduce_metrics
        from reskill.verl_integration.algorithms import (
            adjust_batch,
            apply_invalid_action_penalty,
            compute_trajectory_grpo_advantage,
        )

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        self._inject_skill_components()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                            desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # === Rollout ===
                    with marked_timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    # === HOOK B: Version-level A/B recording ===
                    try:
                        self._hook_b_after_rollout(gen_batch_output, metrics)
                    except Exception as e:
                        print(f"[RESKILL] WARNING: Hook B failed: {e}")
                        import traceback
                        traceback.print_exc()

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()

                    skip_policy_update = self.config.trainer.get("skip_policy_update", False)

                    with marked_timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if skip_policy_update:
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["token_level_rewards"] = reward_tensor
                        response_shape = batch.batch["responses"].shape
                        batch.batch["advantages"] = torch.zeros(response_shape, device=batch.batch["responses"].device)
                        batch.batch["returns"] = torch.zeros(response_shape, device=batch.batch["responses"].device)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    else:
                        with marked_timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks,
                                                    loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                metrics.update({
                                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                })

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw):
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                                batch, invalid_metrics = apply_invalid_action_penalty(
                                    batch,
                                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.get(
                                        'invalid_action_penalty_coef', 0.1),
                                )
                                metrics.update(invalid_metrics)

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True)

                            batch = compute_trajectory_grpo_advantage(
                                batch,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )

                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # Validation
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_history(self):
        """Save version evolution history."""
        try:
            vh_path = os.path.join(self._save_dir, 'version_history.json')
            with open(vh_path, 'w') as f:
                json.dump(self._version_history, f, indent=2, default=str)
        except Exception as e:
            print(f"[RESKILL] WARNING: Failed to save version history: {e}")

    @staticmethod
    def _get_action_vocabulary(env_config) -> str:
        """Return the action vocabulary for the configured environment."""
        from reskill.environments.registry import get_action_vocabulary

        try:
            return get_action_vocabulary(env_config)
        except ValueError:
            return ""
