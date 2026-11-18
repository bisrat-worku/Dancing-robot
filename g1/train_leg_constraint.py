"""mjlab's train CLI with the leg-constraint variant registered (g1/leg_constraint.py).

  python3 g1/train_leg_constraint.py [mjlab train args...]     # the task id is added here
Normally called by  ./train_g1_tracking.sh outputs/<run>/ --leg-constraint ...  Re-launches itself in mjlab's venv.
"""

import os
import sys
from pathlib import Path

MJLAB_PY = Path(os.environ.get("MJLAB_DIR", "/home/bwork/mjlab")) / ".venv" / "bin" / "python"
try:
  import mjlab  # noqa: F401
except ImportError:
  os.execv(str(MJLAB_PY), [str(MJLAB_PY), os.path.abspath(__file__), *sys.argv[1:]])

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leg_constraint  # noqa: E402

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  sys.argv = [sys.argv[0], leg_constraint.TASK_ID, *sys.argv[1:]]
  main()
