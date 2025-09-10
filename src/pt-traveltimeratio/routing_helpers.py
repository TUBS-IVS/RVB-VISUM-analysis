from __future__ import annotations

from pathlib import Path
from typing import Sequence, List
import os
import shutil
import time as _time

import requests


def resolve_java_bin() -> Path:
    env = os.environ.get("GH_JAVA_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cand = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if cand.exists():
            return cand
    if os.name == "nt":
        candidates: List[Path] = []
        pf_vars = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramW6432")]
        patterns = [
            "Eclipse Adoptium/jdk*/*/bin/java.exe",
            "Eclipse Adoptium/jdk*/bin/java.exe",
            "AdoptOpenJDK/jdk*/bin/java.exe",
            "Java/jdk*/bin/java.exe",
            "Zulu/zulu*/bin/java.exe",
        ]
        for base_str in pf_vars:
            if not base_str:
                continue
            base = Path(base_str)
            for pattern in patterns:
                for match in base.glob(pattern):
                    candidates.append(match)
        if candidates:
            def _sort_key(p: Path):
                return str(p.parent)
            best = sorted(candidates, key=_sort_key, reverse=True)[0]
            if best.exists():
                return best
    which = shutil.which("java")
    if which:
        return Path(which)
    return Path("java")


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    here = cwd
    for _ in range(8):
        if (here / "input" / "shapes" / "pt-data").exists():
            return here
        if here.parent == here:
            break
        here = here.parent
    raise RuntimeError("Could not find the repo root (input/shapes/pt-data missing). Please run this script from within the repository.")


def health_urls(port: int) -> Sequence[str]:
    base = f"http://localhost:{port}"
    return (f"{base}/health", f"{base}/actuator/health")


def is_backend_ready(port: int, timeout: float = 0.0) -> bool:
    end = _time.time() + max(0.0, timeout)
    while True:
        for url in health_urls(port):
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
        if timeout <= 0 or _time.time() >= end:
            return False
        _time.sleep(1.0)


def sanitize_time_for_filename(t: str) -> str:
    return t.replace(":", "-")
