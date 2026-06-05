from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMODULE_VERL = _REPO_ROOT / "verl"
if (_SUBMODULE_VERL / "verl").is_dir() and str(_SUBMODULE_VERL) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_VERL))
