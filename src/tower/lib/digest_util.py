from hashlib import sha256
from pathlib import Path
from typing import Optional


def create_digest(params) -> str:
    s = sha256()
    if isinstance(params, dict):
        seed = str(list(sorted(params.items()))).encode("utf8")
    elif isinstance(params, str):
        seed = params.encode()
    else:
        seed = params
    s.update(seed)
    return s.hexdigest()


def get_file_digest(path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    with path.open(mode="rb") as f:
        return create_digest(f.read())
