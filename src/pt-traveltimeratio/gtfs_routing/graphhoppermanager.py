# -*- coding: utf-8 -*-
"""
GraphHopper process manager.

Responsibilities
1) Start or rebuild a GraphHopper web server using a given config.yml and JAR
2) Write a PID file for later shutdown
3) Poll the health endpoint until the server is ready
4) Provide a clean shutdown method with Windows and Linux support

Python 3.9 compatible. Comments in English.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, List

import requests


JavaMemType = Union[str, Sequence[str]]
PathLike = Union[str, Path]

class GraphHopperManager:
    """
    Manage a GraphHopper server process.
    """

    def __init__(
        self,
        *,
        config_path: PathLike,
        graph_cache_dir: Optional[PathLike] = None,
        jar_path: Optional[PathLike] = None,
        host: str = "localhost",
        port: int = 8989,
        java_mem: JavaMemType = "-Xmx110g",
        pid_file: Optional[PathLike] = None,
        java_bin: Optional[str] = None,
        gtfs_path: Optional[PathLike] = None,
        osm_path: Optional[PathLike] = None,
    ) -> None:
        # Critical paths derived from config.yml location
        self.config_path = Path(config_path)
        base_dir = self.config_path.parent
        self.graph_cache_dir = Path(graph_cache_dir) if graph_cache_dir is not None else (base_dir / "graph-cache")
        self.jar_path = Path(jar_path) if jar_path is not None else (base_dir / "graphhopper-web-10.2.jar")

        # Optional references
        self.gtfs_path = Path(gtfs_path) if gtfs_path is not None else None
        self.osm_path = Path(osm_path) if osm_path is not None else None

        # Runtime parameters
        if isinstance(java_mem, str):
            jm = java_mem.strip()
            self.java_opts = jm.split() if " " in jm else ([jm] if jm else [])
        elif isinstance(java_mem, (list, tuple)):
            self.java_opts = [str(x) for x in java_mem]
        else:
            raise TypeError("java_mem must be a string or a sequence of strings")
        self.host = host
        self.port = port
        self.pid_file = Path(pid_file) if pid_file is not None else (base_dir / "gh.pid")

        # Java executable resolution
        self.java_bin = self._resolve_java_bin(java_bin)
        self._process = None

    # -------------------------------------------------------------------------
    # Command builder and basic validators
    # -------------------------------------------------------------------------

    @property
    def _java_cmd(self) -> Sequence[str]:
        """Build the exact Java command used to start GraphHopper."""
        return [
            self.java_bin,
            *self.java_opts,          # expand multiple JVM flags correctly
            "-jar",
            str(self.jar_path),
            "server",
            str(self.config_path),
        ]

    def _assert_files(self) -> None:
        """Ensure required files exist before starting."""
        if not self.jar_path.exists():
            raise FileNotFoundError(f"GraphHopper JAR not found: {self.jar_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"GraphHopper config.yml not found: {self.config_path}")

    def _resolve_java_bin(self, provided: Optional[str]) -> str:
        """Resolve Java executable with priority: provided > JAVA_EXE > JAVA_HOME/bin/java(.exe) > 'java'."""
        if provided:
            return provided
        # 1) JAVA_EXE env var (explicit full path)
        env_java = os.environ.get("JAVA_EXE")
        if env_java:
            return env_java
        # 2) JAVA_HOME/bin/java(.exe)
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            cand = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
            if cand.exists():
                return str(cand)
        # 3) Fallback to PATH
        return "java"

    @property
    def health_urls(self) -> Sequence[str]:
        """Possible health endpoints used by different builds."""
        base = f"http://{self.host}:{self.port}"
        return (f"{base}/health", f"{base}/actuator/health")

    # -------------------------------------------------------------------------
    # PID helpers
    # -------------------------------------------------------------------------

    def _write_pid(self, pid: int) -> None:
        """Write the child PID to disk for later shutdown."""
        try:
            with open(self.pid_file, "w", encoding="utf-8") as f:
                f.write(str(pid))
            print(f"PID {pid} written to {self.pid_file}")
        except Exception as e:
            print(f"Warning: failed to write PID file {self.pid_file}: {e}")

    def _read_pid(self) -> Optional[int]:
        """Read the child PID from disk if available."""
        try:
            if not self.pid_file.exists():
                return None
            return int(self.pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Remove the graph-cache directory."""
        if self.graph_cache_dir.exists():
            shutil.rmtree(self.graph_cache_dir)
            print(f"Cleared graph-cache: {self.graph_cache_dir}")

    def rebuild(self) -> None:
        """
        Clear the graph cache and start a fresh server.
        Equivalent to a cold build after config or GTFS changes.
        """
        self._assert_files()
        print("Rebuilding GraphHopper graph...")
        self.clear_cache()
        self.start()

    def start(self, log_file: Optional[PathLike] = None, env: Optional[Dict[str, str]] = None) -> None:
        """
        Start the GraphHopper server using the configured Java command.

        Parameters
        ----------
        log_file : Optional[PathLike]
            If provided, stdout and stderr are redirected to this file.
        env : Optional[dict]
            Additional environment variables for the Java process.
        """
        self._assert_files()

        stdout_target = None
        stderr_target = None
        file_handle = None

        # Redirect logs to a file if requested
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handle = open(log_path, "a", encoding="utf-8")
            stdout_target = file_handle
            stderr_target = file_handle

        print(f"Starting GraphHopper: {self._java_cmd}")

        proc = subprocess.Popen(
            self._java_cmd,
            stdout=None,
            stderr=None,
            env={**os.environ, **(env or {})},
            creationflags=0,
        )
        self._process = proc
        self._write_pid(proc.pid)

        # Close file handle in parent if we opened one
        if file_handle is not None:
            # Keep the file descriptor open for the child, close only our handle
            try:
                file_handle.flush()
            except Exception:
                pass

    def is_ready(self, timeout: float = 0.0) -> bool:
        """
        Poll health endpoints once or for up to timeout seconds.
        Returns True as soon as HTTP 200 is observed on any known health URL.
        """
        end = time.time() + max(0.0, timeout)
        while True:
            for url in self.health_urls:
                try:
                    r = requests.get(url, timeout=2)
                    if r.status_code == 200:
                        return True
                except requests.RequestException:
                    pass
            if timeout <= 0 or time.time() >= end:
                return False
            time.sleep(1.5)

    def wait_until_ready(self, timeout: float = 180.0) -> bool:
        """
        Wait up to timeout seconds for the server to report healthy.
        """
        print("Waiting for GraphHopper to become ready...")
        ok = self.is_ready(timeout=timeout)
        print("GraphHopper is ready" if ok else "GraphHopper did not become ready in time")
        return ok

    def shutdown(self, graceful: bool = True) -> None:
        """
        Stop the running GraphHopper process.

        Parameters
        ----------
        graceful : bool
            If True, send SIGINT first and allow a short grace period, then
            terminate if the process is still alive.
        """
        pid = self._read_pid()

        # Prefer the live handle if we started the process in this session
        if self._process and self._process.poll() is None:
            pid = self._process.pid

        if pid is None:
            print("No PID found. Nothing to shut down.")
            return

        print(f"Shutting down GraphHopper (PID {pid})...")
        try:
            if graceful:
                # On Windows SIGINT is translated; if it fails we fall back to terminate.
                os.kill(pid, signal.SIGINT)
                time.sleep(3)

            # If still running, terminate forcefully
            try:
                os.kill(pid, 0)  # probe
                if os.name == "nt":
                    subprocess.call(["taskkill", "/PID", str(pid), "/F", "/T"])
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        except ProcessLookupError:
            print("Process already exited.")
        except Exception as e:
            print(f"Failed to shut down GraphHopper: {e}")

        # Clean up PID file
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except Exception:
            pass

        self._process = None
        print("Shutdown procedure completed.")
