"""Skill registry: manages modular skill modules and their lifecycle.

Skills are stored as SKILL.md artifacts (YAML frontmatter + markdown body)
and indexed by the registry. The registry supports dual serialization:
  - JSON (skill_registry.json) for backward compatibility
  - SKILL.md files (skill_library/) for human-readable, portable artifacts
"""

import glob as glob_mod
import json
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from .trigger_matcher import TriggerMatcher

logger = logging.getLogger(__name__)


@dataclass
class SkillModule:
    """A single modular skill with trigger conditions."""
    skill_id: str
    name: str
    content: str
    trigger_type: str               # "general" | "beginning" | "action_pattern"
    trigger_pattern: Optional[str]  # Regex string (for action_pattern type)
    status: str                     # "active" | "testing" | "rejected"
    created_at_step: int = 0
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SkillModule':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# SKILL.md serialization
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str, skill_id: str = '') -> str:
    """Convert a skill name to a safe, unique filename.

    Appends the skill_id suffix to prevent collisions when different names
    (e.g. 'foo-bar' vs 'foo_bar') map to the same kebab-case string.
    """
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9\-]', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    s = s or 'unnamed-skill'
    # Append skill_id hash to guarantee uniqueness
    if skill_id:
        s = f"{s}_{skill_id}"
    return s


def _escape_yaml_value(val: str) -> str:
    """Escape a string for safe inclusion as a quoted YAML value."""
    return val.replace('\\', '\\\\').replace('"', '\\"')


def skill_to_md(skill: SkillModule) -> str:
    """Serialize a SkillModule to SKILL.md format.

    Format:
      ---
      name: <name>
      description: "<first sentence of when_to_use>"
      trigger_type: <type>
      trigger_pattern: "<pattern>"   # only if present
      version: <int>
      status: <status>
      created_at_step: <int>
      skill_id: <id>
      ---
      ## When to Use
      <content>

      ## Action
      <content>

      ## Examples
      - DO: ...
      - DON'T: ...
    """
    content = skill.content
    if isinstance(content, dict):
        when_to_use = content.get('when_to_use', '')
        action = content.get('action', '')
        examples = content.get('examples', [])
    else:
        when_to_use = ''
        action = str(content)
        examples = []

    # Build description from first sentence of when_to_use
    desc = when_to_use.split('.')[0].strip() if when_to_use else skill.name
    if desc and not desc.endswith('.'):
        desc += '.'

    # Frontmatter (quote-escape values that may contain special chars)
    lines = ['---']
    lines.append(f'name: {skill.name}')
    lines.append(f'description: "{_escape_yaml_value(desc)}"')
    lines.append(f'trigger_type: {skill.trigger_type}')
    if skill.trigger_pattern:
        lines.append(f'trigger_pattern: "{_escape_yaml_value(skill.trigger_pattern)}"')
    lines.append(f'version: {skill.version}')
    lines.append(f'status: {skill.status}')
    lines.append(f'created_at_step: {skill.created_at_step}')
    lines.append(f'skill_id: {skill.skill_id}')
    if skill.metadata:
        lines.append(f'metadata: {json.dumps(skill.metadata, default=str)}')
    lines.append('---')
    lines.append('')

    # Body
    lines.append('## When to Use')
    lines.append(when_to_use if when_to_use else '(no guidance)')
    lines.append('')
    lines.append('## Action')
    lines.append(action if action else '(no action)')
    lines.append('')

    if examples:
        lines.append('## Examples')
        for ex in examples:
            for ex_line in str(ex).strip().split('\n'):
                lines.append(f'- {ex_line}')
        lines.append('')

    return '\n'.join(lines)


def skill_from_md(md_text: str) -> SkillModule:
    """Parse a SKILL.md file back into a SkillModule.

    Parses YAML frontmatter (between --- delimiters) and extracts
    ## When to Use, ## Action, ## Examples sections from the markdown body.
    """
    # Split frontmatter
    parts = md_text.split('---', 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md must have --- delimited frontmatter")

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    # Parse frontmatter (simple key: value, no nested YAML)
    fm = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes and unescape
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            val = val[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        fm[key] = val

    # Parse integer fields
    version = int(fm.get('version', '1'))
    created_at_step = int(fm.get('created_at_step', '0'))

    # Parse metadata (JSON string)
    metadata = {}
    if 'metadata' in fm:
        try:
            metadata = json.loads(fm['metadata'])
        except (json.JSONDecodeError, TypeError):
            pass

    # Parse body sections
    sections = {}
    current_section = None
    current_lines = []
    for line in body.split('\n'):
        if line.startswith('## '):
            if current_section is not None:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_section is not None:
        sections[current_section] = '\n'.join(current_lines).strip()

    when_to_use = sections.get('When to Use', '')
    action_text = sections.get('Action', '')
    examples_text = sections.get('Examples', '')

    # Parse examples: lines starting with "- "
    examples = []
    if examples_text:
        for line in examples_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                examples.append(line[2:])

    # Build content dict
    content = {
        'when_to_use': when_to_use if when_to_use != '(no guidance)' else '',
        'action': action_text if action_text != '(no action)' else '',
        'examples': examples,
    }

    return SkillModule(
        skill_id=fm.get('skill_id', ''),
        name=fm.get('name', 'unnamed'),
        content=content,
        trigger_type=fm.get('trigger_type', 'general'),
        trigger_pattern=fm.get('trigger_pattern') or None,
        status=fm.get('status', 'active'),
        created_at_step=created_at_step,
        version=version,
        metadata=metadata,
    )


class SkillRegistry:
    """Central registry managing all skill modules and their lifecycle."""

    VALID_TRIGGER_TYPES = {"general", "beginning", "action_pattern"}
    VALID_STATUSES = {"active", "testing", "rejected"}

    def __init__(self, max_active_skills: int = 8):
        self._skills: Dict[str, SkillModule] = {}
        self._next_id: int = 0
        self._history: List[Dict] = []
        self.MAX_ACTIVE_SKILLS = max_active_skills

    def _gen_id(self, name: str) -> str:
        """Generate a unique skill ID."""
        hash_suffix = hashlib.md5(f"{name}_{self._next_id}".encode()).hexdigest()[:6]
        skill_id = f"skill_{hash_suffix}"
        self._next_id += 1
        return skill_id

    def add_skill(self, name: str, content: str, trigger_type: str,
                  trigger_pattern: Optional[str] = None,
                  status: str = "testing", step: int = 0,
                  metadata: Optional[Dict] = None) -> str:
        """Register a new skill. Returns skill_id."""
        assert trigger_type in self.VALID_TRIGGER_TYPES, \
            f"Invalid trigger_type: {trigger_type}"
        assert status in self.VALID_STATUSES, f"Invalid status: {status}"
        if trigger_type == "action_pattern" and not trigger_pattern:
            logger.warning("[SkillRegistry] action_pattern trigger without pattern, "
                           "defaulting to general")
            trigger_type = "general"

        # One testing skill at a time
        if status == "testing" and self.get_testing_skills():
            existing = self.get_testing_skills()[0]
            logger.warning(f"[SkillRegistry] Already testing {existing.skill_id}, "
                           f"cannot add another testing skill")
            return None

        skill_id = self._gen_id(name)
        skill = SkillModule(
            skill_id=skill_id,
            name=name,
            content=content,
            trigger_type=trigger_type,
            trigger_pattern=trigger_pattern,
            status=status,
            created_at_step=step,
            version=1,
            metadata=metadata or {},
        )
        self._skills[skill_id] = skill
        self._history.append({
            "action": "add", "skill_id": skill_id, "name": name,
            "trigger_type": trigger_type, "status": status, "step": step,
        })
        logger.info(f"[SkillRegistry] Added skill '{name}' ({skill_id}), "
                     f"trigger={trigger_type}, status={status}")
        return skill_id

    def get_skill(self, skill_id: str) -> Optional[SkillModule]:
        return self._skills.get(skill_id)

    def get_active_skills(self) -> List[SkillModule]:
        return [s for s in self._skills.values() if s.status == "active"]

    def get_testing_skills(self) -> List[SkillModule]:
        return [s for s in self._skills.values() if s.status == "testing"]

    def get_all_skills(self) -> List[SkillModule]:
        return list(self._skills.values())

    def id_to_name_map(self) -> Dict[str, str]:
        """Return mapping of skill_id -> skill name for all skills."""
        return {s.skill_id: s.name for s in self._skills.values()}

    def update_skill_status(self, skill_id: str, new_status: str):
        """Move skill to active/rejected/testing."""
        assert new_status in self.VALID_STATUSES
        skill = self._skills.get(skill_id)
        if not skill:
            logger.warning(f"[SkillRegistry] Skill {skill_id} not found")
            return
        old_status = skill.status
        skill.status = new_status
        self._history.append({
            "action": "status_change", "skill_id": skill_id,
            "old_status": old_status, "new_status": new_status,
        })
        logger.info(f"[SkillRegistry] {skill_id} status: {old_status} -> {new_status}")

    def edit_skill(self, skill_id: str, **updates):
        """Modify skill content/trigger. Increments version."""
        skill = self._skills.get(skill_id)
        if not skill:
            logger.warning(f"[SkillRegistry] Skill {skill_id} not found for edit")
            return
        for key, val in updates.items():
            if hasattr(skill, key) and key not in ('skill_id', 'created_at_step'):
                setattr(skill, key, val)
        skill.version += 1
        self._history.append({
            "action": "edit", "skill_id": skill_id,
            "updates": {k: str(v)[:100] for k, v in updates.items()},
        })

    def remove_skill(self, skill_id: str):
        """Remove a skill from the registry."""
        if skill_id in self._skills:
            self._history.append({
                "action": "remove", "skill_id": skill_id,
                "name": self._skills[skill_id].name,
            })
            del self._skills[skill_id]
            logger.info(f"[SkillRegistry] Removed skill {skill_id}")

    def get_triggered_skills(self, step_num: int, last_action: Optional[str],
                             testing_decisions: Optional[Dict[str, bool]] = None
                             ) -> List[SkillModule]:
        """Return skills that should be included in prompt at this step.

        Args:
            step_num: Current step number (0-indexed).
            last_action: The last action taken (None if step 0).
            testing_decisions: For skills in 'testing' status, maps
                skill_id -> True (include) or False (exclude).

        Returns:
            List of SkillModule objects to include.
        """
        testing_decisions = testing_decisions or {}
        result = []
        for skill in self._skills.values():
            if skill.status == "rejected":
                continue
            if not TriggerMatcher.should_trigger(skill, step_num, last_action):
                continue
            if skill.status == "testing":
                if not testing_decisions.get(skill.skill_id, False):
                    continue
            result.append(skill)
        return result

    def get_skill_by_name(self, name: str) -> Optional[SkillModule]:
        """Lookup a skill by its human-readable name."""
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        return None

    def apply_version(self, operations: List[Dict], step: int = 0) -> Dict:
        """Apply a bundle of operations atomically. Returns snapshot for revert.

        Operations are dicts with 'type' key ('add', 'modify', 'delete') and
        operation-specific fields.

        Returns:
            Snapshot dict with enough info to revert all operations.
        """
        snapshot = {'operations': [], 'step': step}

        for op in operations:
            op_type = op.get('type')

            if op_type == 'add':
                skill_def = op.get('skill', {})
                # Check cap
                if len(self.get_active_skills()) >= self.MAX_ACTIVE_SKILLS:
                    logger.warning(
                        f"[SkillRegistry] Cannot add skill '{skill_def.get('name')}': "
                        f"at capacity ({self.MAX_ACTIVE_SKILLS})")
                    continue

                content = skill_def.get('content', '')
                skill_id = self.add_skill(
                    name=skill_def.get('name', 'unnamed'),
                    content=content,
                    trigger_type=skill_def.get('trigger_type', 'general'),
                    trigger_pattern=skill_def.get('trigger_pattern'),
                    status='active',  # Go live immediately for version testing
                    step=step,
                )
                if skill_id:
                    snapshot['operations'].append({
                        'type': 'add', 'skill_id': skill_id,
                    })

            elif op_type == 'modify':
                target_name = op.get('target_skill_name', '')
                skill = self.get_skill_by_name(target_name)
                if not skill:
                    logger.warning(
                        f"[SkillRegistry] Cannot modify '{target_name}': not found")
                    continue

                # Save old state for revert
                old_state = {
                    'content': skill.content,
                    'trigger_type': skill.trigger_type,
                    'trigger_pattern': skill.trigger_pattern,
                }

                changes = op.get('changes', {})
                update_kwargs = {}
                if 'content' in changes:
                    # Merge content changes into existing content
                    if isinstance(skill.content, dict) and isinstance(changes['content'], dict):
                        merged = dict(skill.content)
                        merged.update(changes['content'])
                        update_kwargs['content'] = merged
                    else:
                        update_kwargs['content'] = changes['content']
                if 'trigger_pattern' in changes:
                    update_kwargs['trigger_pattern'] = changes['trigger_pattern']

                if update_kwargs:
                    self.edit_skill(skill.skill_id, **update_kwargs)
                    snapshot['operations'].append({
                        'type': 'modify', 'skill_id': skill.skill_id,
                        'old_state': old_state,
                    })

            elif op_type == 'delete':
                target_name = op.get('target_skill_name', '')
                skill = self.get_skill_by_name(target_name)
                if not skill:
                    logger.warning(
                        f"[SkillRegistry] Cannot delete '{target_name}': not found")
                    continue

                # Save full state for revert
                snapshot['operations'].append({
                    'type': 'delete', 'skill_id': skill.skill_id,
                    'skill_data': skill.to_dict(),
                })
                self.remove_skill(skill.skill_id)

        logger.info(f"[SkillRegistry] Applied version with {len(snapshot['operations'])} "
                     f"operations at step {step}")
        return snapshot

    def revert_version(self, snapshot: Dict):
        """Revert all operations from apply_version using the snapshot.

        Reverses operations in reverse order to maintain consistency.
        """
        for op in reversed(snapshot.get('operations', [])):
            op_type = op['type']

            if op_type == 'add':
                # Undo add -> remove
                self.remove_skill(op['skill_id'])

            elif op_type == 'modify':
                # Undo modify -> restore old state
                self.edit_skill(op['skill_id'], **op['old_state'])

            elif op_type == 'delete':
                # Undo delete -> re-add with original data
                data = op['skill_data']
                skill = SkillModule.from_dict(data)
                self._skills[skill.skill_id] = skill
                self._history.append({
                    'action': 'restore', 'skill_id': skill.skill_id,
                    'name': skill.name,
                })

        logger.info(f"[SkillRegistry] Reverted version "
                     f"({len(snapshot.get('operations', []))} operations)")

    def snapshot(self) -> 'SkillRegistry':
        """Create an independent deep copy of this registry.

        Used to preserve the pre-version state for A/B testing: 'old version'
        rollouts use the snapshot while 'new version' rollouts use the live
        registry (after apply_version).
        """
        import copy
        clone = SkillRegistry()
        clone._next_id = self._next_id
        clone._skills = {
            sid: copy.deepcopy(skill)
            for sid, skill in self._skills.items()
        }
        # History is not needed in the snapshot (it's for diagnostics only)
        clone._history = []
        return clone

    def to_diagnostic_summary(
        self, per_skill_diagnostics: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """Enhanced summary including per-skill diagnostics for the skill planner.

        Args:
            per_skill_diagnostics: Optional dict mapping skill_id to
                {'trigger_rate': float, 'episode_count': int, ...}
        """
        if not self._skills:
            return "No skills registered."

        per_skill_diagnostics = per_skill_diagnostics or {}
        lines = []
        for s in self._skills.values():
            if s.status == 'rejected':
                continue
            trigger_info = s.trigger_type
            if s.trigger_pattern:
                trigger_info += f" (pattern: {s.trigger_pattern})"

            # Content preview
            content = s.content
            if isinstance(content, dict):
                when_to_use = content.get("when_to_use", "")
                preview = when_to_use[:120]
                if len(when_to_use) > 120:
                    preview += "..."
            else:
                preview = str(content)[:120]
                if len(str(content)) > 120:
                    preview += "..."

            # Diagnostics
            diag = per_skill_diagnostics.get(s.skill_id, {})
            diag_parts = []
            if 'trigger_rate' in diag:
                diag_parts.append(f"trigger_rate={diag['trigger_rate']:.0%}")
            age = diag.get('age_steps')
            if age is not None:
                diag_parts.append(f"age={age} steps")
            if s.version > 1:
                diag_parts.append(f"v{s.version}")
            diag_str = f" [{', '.join(diag_parts)}]" if diag_parts else ""

            lines.append(
                f"- [{s.status}] {s.name} ({s.skill_id}): "
                f"trigger={trigger_info}{diag_str}\n  {preview}"
            )
        return "\n".join(lines)

    def save(self, path: str, skill_library_dir: Optional[str] = None):
        """Save registry to JSON and optionally as SKILL.md files.

        Args:
            path: Path for the JSON file (backward-compat format).
            skill_library_dir: If provided, also write each skill as a
                SKILL.md file in this directory.
        """
        data = {
            "next_id": self._next_id,
            "skills": [s.to_dict() for s in self._skills.values()],
            "history": self._history,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"[SkillRegistry] Saved {len(self._skills)} skills to {path}")

        if skill_library_dir:
            self.save_skill_library(skill_library_dir)

    def load(self, path: str, skill_library_dir: Optional[str] = None):
        """Load registry from SKILL.md library or JSON.

        Prefers skill_library/ if it exists, falls back to JSON.

        Args:
            path: Path to the JSON file (backward-compat).
            skill_library_dir: If provided and the directory exists,
                load from SKILL.md files instead of JSON.
        """
        if skill_library_dir and os.path.isdir(skill_library_dir):
            self.load_skill_library(skill_library_dir)
        elif os.path.exists(path):
            logger.warning(
                f"[SkillRegistry] Loading from legacy JSON ({path}). "
                f"No skill_library/ found — this checkpoint predates SKILL.md support. "
                f"Skills will be saved as SKILL.md on next save.")
            self._load_json(path)
        else:
            logger.warning(f"[SkillRegistry] No data found at {path} "
                           f"or {skill_library_dir}")

    def _load_json(self, path: str):
        """Load registry from JSON (internal helper)."""
        with open(path) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 0)
        self._skills = {}
        for d in data.get("skills", []):
            skill = SkillModule.from_dict(d)
            self._skills[skill.skill_id] = skill
        self._history = data.get("history", [])
        logger.info(f"[SkillRegistry] Loaded {len(self._skills)} skills from {path}")

    # ------------------------------------------------------------------
    # SKILL.md library I/O
    # ------------------------------------------------------------------

    def save_skill_library(self, library_dir: str):
        """Write each skill as a SKILL.md file + an _index.json.

        Removes stale .md files for skills that no longer exist.
        """
        os.makedirs(library_dir, exist_ok=True)

        # Write each skill
        current_filenames = set()
        for skill in self._skills.values():
            fname = _sanitize_filename(skill.name, skill.skill_id) + '.md'
            current_filenames.add(fname)
            filepath = os.path.join(library_dir, fname)
            with open(filepath, 'w') as f:
                f.write(skill_to_md(skill))

        # Remove stale .md files
        for existing in glob_mod.glob(os.path.join(library_dir, '*.md')):
            if os.path.basename(existing) not in current_filenames:
                os.remove(existing)
                logger.info(f"[SkillRegistry] Removed stale skill file: {existing}")

        # Write index
        index = {
            'next_id': self._next_id,
            'history': self._history,
            'skills': [
                {'skill_id': s.skill_id, 'name': s.name,
                 'status': s.status,
                 'file': _sanitize_filename(s.name, s.skill_id) + '.md'}
                for s in self._skills.values()
            ],
        }
        with open(os.path.join(library_dir, '_index.json'), 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"[SkillRegistry] Saved {len(self._skills)} skills "
                     f"to {library_dir}/")

    def load_skill_library(self, library_dir: str):
        """Load skills from SKILL.md files + _index.json."""
        index_path = os.path.join(library_dir, '_index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self._next_id = index.get('next_id', 0)
            self._history = index.get('history', [])
        else:
            self._next_id = 0
            self._history = []

        self._skills = {}
        for md_path in sorted(glob_mod.glob(os.path.join(library_dir, '*.md'))):
            try:
                with open(md_path) as f:
                    skill = skill_from_md(f.read())
                if skill.skill_id:
                    self._skills[skill.skill_id] = skill
                else:
                    logger.warning(f"[SkillRegistry] Skipping {md_path}: "
                                   f"no skill_id in frontmatter")
            except Exception as e:
                logger.warning(f"[SkillRegistry] Failed to parse {md_path}: {e}")

        logger.info(f"[SkillRegistry] Loaded {len(self._skills)} skills "
                     f"from {library_dir}/")

    def to_summary_text(self) -> str:
        """Human-readable summary for LLM optimizer context."""
        if not self._skills:
            return "No skills registered."
        lines = []
        for s in self._skills.values():
            trigger_info = s.trigger_type
            if s.trigger_pattern:
                trigger_info += f" (pattern: {s.trigger_pattern})"
            content = s.content
            if isinstance(content, dict):
                when_to_use = content.get("when_to_use", "")
                preview = when_to_use[:120]
                if len(when_to_use) > 120:
                    preview += "..."
            else:
                preview = str(content)[:120]
                if len(str(content)) > 120:
                    preview += "..."
            lines.append(
                f"- [{s.status}] {s.name} ({s.skill_id}): "
                f"trigger={trigger_info}\n  {preview}"
            )
        return "\n".join(lines)

    def __len__(self):
        return len(self._skills)

    def __repr__(self):
        active = len(self.get_active_skills())
        testing = len(self.get_testing_skills())
        return f"SkillRegistry(total={len(self)}, active={active}, testing={testing})"
