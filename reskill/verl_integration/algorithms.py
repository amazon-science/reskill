from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def apply_invalid_action_penalty(
    data,
    invalid_action_penalty_coef: float,
) -> tuple[DataProto, dict[str, float]]:
    reward_tensor = data.batch["token_level_scores"]
    step_rewards = data.batch.get("step_rewards")

    for i in range(len(data)):
        item = data[i]
        prompt_ids = item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
        action_valid = np.asarray(item.non_tensor_batch["is_action_valid"]).astype(np.float32)
        action_invalid = torch.tensor(
            1 - action_valid,
            dtype=torch.float32,
            device=prompt_ids.device,
        ).squeeze(0)
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalid
        if step_rewards is not None:
            step_rewards[i] -= invalid_action_penalty_coef * action_invalid

    valid_action_ratio = np.mean(data.non_tensor_batch["is_action_valid"].astype(np.float32)).item()
    return data, {"episode/valid_action_ratio": valid_action_ratio}


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add(pair)

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                values = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(values)
                id2std[idx] = torch.std(values)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = _masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.bool()
    selected = values[mask]
    if selected.numel() == 0:
        return values
    mean = selected.mean()
    var = selected.var(unbiased=False)
    return (values - mean) * torch.rsqrt(var + eps)


def compute_trajectory_grpo_advantage(
    data,
    norm_adv_by_std_in_grpo: bool = True,
) -> DataProto:
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        traj_index=data.non_tensor_batch["traj_uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def adjust_batch(config, data, mode="copy"):
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    rollout_micro_bsz = (
        config.actor_rollout_ref.rollout.get("log_prob_micro_batch_size_per_gpu")
        or config.actor_rollout_ref.rollout.get("log_prob_micro_batch_size")
        or 1
    )
    size_divisor_rollout = rollout_micro_bsz * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        ref_micro_bsz = (
            config.actor_rollout_ref.ref.get("log_prob_micro_batch_size_per_gpu")
            or config.actor_rollout_ref.ref.get("log_prob_micro_batch_size")
            or rollout_micro_bsz
        )
        size_divisor_ref = ref_micro_bsz * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        actor_micro_bsz = (
            config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu")
            or config.actor_rollout_ref.actor.get("ppo_micro_batch_size")
            or config.actor_rollout_ref.actor.ppo_mini_batch_size
        )
        size_divisor_actor = actor_micro_bsz * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data

    if mode == "delete":
        from verl import DataProto

        remove_indices = np.random.choice(bs, remainder, replace=False)
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[np.sort(remove_indices)] = False
        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        return DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)

    if mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)
        return type(data).concat([data, dup_proto])

    raise ValueError(f"Unsupported mode: {mode}")
