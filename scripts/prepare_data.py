"""Cria o split TRAIN/TEST persistido (wrapper de ``python -m saude_fetal.data``).

Uso (pela raiz, com o ambiente conda ``saude_fetal`` ativo)::

    python scripts/prepare_data.py           # não recria se já existir
    python scripts/prepare_data.py --force   # recria o split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saude_fetal.data import main  # noqa: E402

if __name__ == "__main__":
    main()
