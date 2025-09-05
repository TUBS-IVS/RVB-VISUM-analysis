import argparse
import os
import signal
import subprocess
from pathlib import Path


def _guess_repo_root(start: Path) -> Path:
    """Walk up a few levels to find the repository root by locating input/graphhopper/config.yml."""
    here = start.resolve()
    for i in range(1, 7):
        try:
            cand = here.parents[i]
        except IndexError:
            break
        if (cand / "input" / "graphhopper" / "config.yml").exists():
            return cand
    return Path.cwd()


def _default_pid_path() -> Path:
    repo_root = _guess_repo_root(Path(__file__))
    return repo_root / "input" / "graphhopper" / "gh.pid"


def shutdown_graphhopper(pid_file: Path) -> None:
    pid_path = Path(pid_file)
    if not pid_path.exists():
        print(f"PID file not found: {pid_path}")
        return

    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception as e:
        print(f"Failed to read PID from {pid_path}: {e}")
        return

    print(f"Sending SIGINT to GraphHopper process (PID {pid})...")
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        print("No process found with that PID.")
        return
    except Exception as e:
        print(f"Failed to send SIGINT: {e}")

    # On Windows, SIGINT may not terminate the JVM; fall back to taskkill if still running.
    try:
        # Probe if process still alive
        os.kill(pid, 0)
        if os.name == "nt":
            print("Process still alive; invoking taskkill /F /T ...")
            subprocess.call(["taskkill", "/PID", str(pid), "/F", "/T"])
    except Exception:
        pass

    # Best-effort cleanup of PID file
    try:
        pid_path.unlink(missing_ok=True)
    except Exception:
        pass

    print("Shutdown procedure completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shut down a running GraphHopper process by PID file.")
    parser.add_argument(
        "--pid-file",
        type=str,
        default=None,
        help="Path to gh.pid. Defaults to <repo>/input/graphhopper/gh.pid",
    )
    parser.add_argument(
        "--graphhopper-dir",
        type=str,
        default=None,
        help="GraphHopper directory (containing gh.pid). Overrides default if provided.",
    )
    args = parser.parse_args()

    if args.graphhopper_dir:
        pid = Path(args.graphhopper_dir) / "gh.pid"
    elif args.pid_file:
        pid = Path(args.pid_file)
    else:
        pid = _default_pid_path()

    print(f"Using PID file: {pid}")
    shutdown_graphhopper(pid)
