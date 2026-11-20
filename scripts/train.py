"""Treina os modelos com CV no TRAIN e registra tudo no MLflow
(wrapper de ``python -m saude_fetal.train``).

Uso::

    python scripts/train.py
    python scripts/train.py --models dummy_most_frequent random_forest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saude_fetal.train import main  # noqa: E402

if __name__ == "__main__":
    main()
