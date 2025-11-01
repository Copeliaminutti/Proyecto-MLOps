# -*- coding: utf-8 -*-
"""Configuración del proyecto.

- Modo recomendado: `load_config("train.yaml")` lee YAML desde /configs
- Modo legacy: `CONFIGS` (dict estático) se conserva para compatibilidad.
"""

from pathlib import Path
import yaml

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / 'pyproject.toml').exists() or (cur / '.git').exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[2]

ROOT = _find_repo_root(Path(__file__).parent)
CONFIG_DIR = ROOT / 'configs'

def load_config(name: str = 'train.yaml'):
    """Carga y devuelve un dict desde configs/<name> (YAML)."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f'No se encontró {path}. Asegura configs/{name}.')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


__all__ = ["load_config", "CONFIG_DIR", "ROOT"]
