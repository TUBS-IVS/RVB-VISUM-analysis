from pathlib import Path
import os
import shutil
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper


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
input_dir = root / "input"
graphhopper_dir = input_dir / "graphhopper"
gtfs_dir = input_dir / "gtfs-data" / "2025(V10)"

java_bin = resolve_java_bin()
print(f"Using Java binary: {java_bin}")

cfg = AppConfig(
    project_root=root,
    graphhopper_dir=graphhopper_dir,
    gh_config_path=graphhopper_dir / "config.yml",
    gtfs_input_zip=gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip",
    output_base=root / "output",
    scenario_name="scenario_V10_2025",
    gh_jar_path=graphhopper_dir / "graphhopper-web-10.2.jar",
    gh_cache_dir=graphhopper_dir / "graph-cache",
    gh_port=8989,
    java_bin=java_bin,
    java_opts=["-Xms32g", "-Xmx110g"],
)

run_grashopper(cfg)
