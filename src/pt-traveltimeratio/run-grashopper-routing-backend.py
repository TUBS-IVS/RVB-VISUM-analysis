from pathlib import Path
import os
import shutil
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper
import socket
import time


def resolve_java_bin() -> Path:
    """Resolve a Java executable path flexibly:
    Priority:
      1. Environment variable GH_JAVA_BIN (if points to existing file)
      2. JAVA_HOME/bin/java(.exe)
      3. Common Windows install directories (Eclipse Adoptium, AdoptOpenJDK, etc.)
      4. java found on PATH (shutil.which)
      5. Fallback 'java' (rely on PATH at runtime)
    Returns a Path (may be 'java' if only fallback).
    """
    # 1) Explicit override
    env = os.environ.get("GH_JAVA_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p
    # 2) JAVA_HOME
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cand = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if cand.exists():
            return cand
    # 3) Common Windows locations
    if os.name == "nt":
        candidates: list[Path] = []
        pf_vars = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramW6432")]
        for base_str in pf_vars:
            if not base_str:
                continue
            base = Path(base_str)
            for pattern in [
                "Eclipse Adoptium/jdk*/*/bin/java.exe",
                "Eclipse Adoptium/jdk*/bin/java.exe",
                "AdoptOpenJDK/jdk*/bin/java.exe",
                "Java/jdk*/bin/java.exe",
                "Zulu/zulu*/bin/java.exe",
            ]:
                for match in base.glob(pattern):
                    candidates.append(match)
        # pick the newest by version-like sorting of parent folder names
        def _sort_key(p: Path):
            return str(p.parent)
        if candidates:
            best = sorted(candidates, key=_sort_key, reverse=True)[0]
            if best.exists():
                return best
    # 4) PATH lookup
    which = shutil.which("java")
    if which:
        return Path(which)
    # 5) Fallback symbolic Path
    return Path("java")


# Repository root (RVB-VISUM-analysis)
root = Path(__file__).resolve().parents[2]
# Ensure we run with repo root as working directory so relative paths inside GraphHopper config/logback remain valid
os.chdir(root)
print(f"Changed working directory to repo root: {root}")
input_dir = root / "input"
graphhopper_dir = input_dir / "graphhopper"
gtfs_dir = input_dir / "gtfs-data" / "2025(V10)"

java_bin = resolve_java_bin()
print(f"Using Java binary: {java_bin}")


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _wait_until_ready(port: int, host: str = "127.0.0.1", timeout: float = 180.0, poll: float = 2.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if _is_port_open(port):
            return True
        time.sleep(poll)
    return False


if _is_port_open(8989):
    print("GraphHopper already appears to be running on port 8989 – skipping start.")
else:
    cfg = AppConfig(
        project_root=root,
        graphhopper_dir=graphhopper_dir,
        gh_config_path=(graphhopper_dir / "config.yml").resolve(),
        gtfs_input_zip=(gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip").resolve(),
        output_base=(root / "output").resolve(),
        scenario_name="scenario_V10_2025",
        gh_jar_path=(graphhopper_dir / "graphhopper-web-10.2.jar").resolve(),
        gh_cache_dir=(graphhopper_dir / "graph-cache").resolve(),
        gh_port=8989,
        java_bin=java_bin,
        java_opts=[
        "-Xms16g",
        "-Xmx64g",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=200",
        "-XX:ActiveProcessorCount=12",
        "-Xss256k",
        "-XX:ReservedCodeCacheSize=256m",
        "-XX:MaxDirectMemorySize=512m",
        "-Dlogging.level.root=ERROR",
        "-Dlogging.level.com.graphhopper=ERROR",
        "-Dlogging.level.org.springframework=ERROR",
        ],
    )
    run_grashopper(cfg)
    print("Waiting for GraphHopper to become ready...")
    if _wait_until_ready(8989):
        print("GraphHopper backend is ready.")
    else:
        print("Warning: GraphHopper did not open port 8989 within timeout.")
