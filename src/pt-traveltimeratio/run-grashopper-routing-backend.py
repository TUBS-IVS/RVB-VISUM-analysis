from pathlib import Path
from gtfs_routing.gtfs_setup import AppConfig, run_grashopper

# Repository root (RVB-VISUM-analysis)
root = Path(__file__).resolve().parents[2]

# All inputs live under the repo's input/ folder
input_dir = root / "input"
graphhopper_dir = input_dir / "graphhopper"
gtfs_dir = input_dir / "gtfs-data" / "2025(V10)"

cfg = AppConfig(
    project_root=root,
    graphhopper_dir=graphhopper_dir,
    gh_config_path=graphhopper_dir / "config.yml",
    gtfs_input_zip=gtfs_dir / "VM_RVB_Basisszenario2025_mDTicket_V10_init_LB GTFS_250827.zip",
    # write scenario outputs next to the repo (create if missing)
    output_base=root / "output",
    scenario_name="scenario_V10_2025",

    # central GraphHopper paths
    gh_jar_path=graphhopper_dir / "graphhopper-web-10.2.jar",
    gh_cache_dir=graphhopper_dir / "graph-cache",
    gh_port=8989,

    # central Java runtime configuration
    java_bin=Path(r"C:\Program Files\Eclipse Adoptium\jdk-21.0.3.9-hotspot\bin\java.exe"),
    java_opts=["-Xms32g", "-Xmx110g"],   # this is the max heap you asked to set at startup
)

run_grashopper(cfg)
