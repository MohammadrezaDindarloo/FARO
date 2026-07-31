"""Meshcat port bookkeeping.

Meshcat's server scans upward from port 7000 and takes the first free one, and it
offers no way to request a specific port (`python -m meshcat.servers.zmqserver
--help` exposes only `--zmq-url`, `--open`, and TLS options).

That creates a nasty failure mode on a shared cluster node: a leftover server from
an earlier run keeps port 7000, your new script silently lands on 7001, and if you
forward 7000 you get someone else's *empty* viewer. It looks exactly like "the
visualization is broken" while every process involved is perfectly healthy.

This module makes that state visible before the viewer starts.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
from dataclasses import dataclass

MESHCAT_FIRST_PORT = 7000
MESHCAT_SCAN_RANGE = 12


@dataclass
class MeshcatServer:
    """A running meshcat server process.

    NOTE: one meshcat server listens on TWO ports -- a ZeroMQ port (6000+) for the
    Python client and an HTTP port (7000+) for the browser. `ports` holds both;
    `http_port` is the one you actually forward.
    """

    pid: int
    ppid: int
    ports: set[int]
    elapsed: str

    @property
    def http_port(self) -> int | None:
        """The browser-facing port, i.e. the one in meshcat's HTTP scan range."""
        in_range = [p for p in self.ports if MESHCAT_FIRST_PORT <= p < MESHCAT_FIRST_PORT + MESHCAT_SCAN_RANGE]
        return min(in_range) if in_range else None

    @property
    def is_orphan(self) -> bool:
        """True if the parent script has died and left this server behind.

        PPID 1 means it was re-parented to init: the Python process that spawned it
        is gone, so nothing is driving this viewer and it holds a port for nothing.
        Orphans are the ones that are safe to clean up; a server whose parent is
        alive belongs to a running script and must be left alone.
        """
        return self.ppid == 1


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Can we bind this port right now?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def next_free_port(start: int = MESHCAT_FIRST_PORT, count: int = MESHCAT_SCAN_RANGE) -> int | None:
    """The port meshcat will most likely pick, mirroring its upward scan."""
    for port in range(start, start + count):
        if port_is_free(port):
            return port
    return None


def _ports_by_pid() -> dict[int, set[int]]:
    """Map pid -> every port it listens on, via `ss`. Empty dict if unavailable.

    Must be one-pid-to-many-ports: a meshcat server holds both a ZeroMQ port and an
    HTTP port, so collapsing this to one port per pid loses the one we care about.
    """
    try:
        out = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    ports: dict[int, set[int]] = {}
    for line in out.splitlines():
        m_port = re.search(r":(\d{2,5})\s", line)
        m_pid = re.search(r"pid=(\d+)", line)
        if m_port and m_pid:
            ports.setdefault(int(m_pid.group(1)), set()).add(int(m_port.group(1)))
    return ports


def find_meshcat_servers() -> list[MeshcatServer]:
    """All meshcat zmqserver processes belonging to the current user."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,cmd"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    ports_by_pid = _ports_by_pid()

    servers: list[MeshcatServer] = []
    for line in out.splitlines():
        if "meshcat.servers.zmqserver" not in line or " grep " in line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, elapsed = int(parts[0]), int(parts[1]), parts[2]
        except ValueError:
            continue
        servers.append(
            MeshcatServer(pid=pid, ppid=ppid, ports=ports_by_pid.get(pid, set()), elapsed=elapsed)
        )
    return servers


def kill_stale_servers(dry_run: bool = False) -> list[MeshcatServer]:
    """Terminate ORPHANED meshcat servers only.

    Never touches a server whose parent process is still alive -- that one belongs
    to a script someone is actively using, quite possibly in another terminal.
    """
    stale = [s for s in find_meshcat_servers() if s.is_orphan]
    if dry_run:
        return stale
    for server in stale:
        try:
            os.kill(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return stale


def preflight_report(preferred: int = MESHCAT_FIRST_PORT) -> str:
    """Human-readable warning about port squatting, or '' if all is well."""
    if port_is_free(preferred):
        return ""

    servers = find_meshcat_servers()
    squatter = next((s for s in servers if preferred in s.ports), None)
    expected = next_free_port()

    lines = [
        f"WARNING: port {preferred} is already in use, so this viewer will NOT be on {preferred}.",
    ]
    if squatter is not None:
        kind = "an ORPHANED meshcat server (its parent script has exited)" if squatter.is_orphan \
            else "a meshcat server from another running script"
        lines.append(f"  Port {preferred} is held by {kind}, pid {squatter.pid}, up {squatter.elapsed}.")
        if squatter.is_orphan:
            lines.append("  It is showing an EMPTY scene -- if you forward 7000 you will see nothing.")
            lines.append("  Clear it with:   python scripts/01_load_and_visualize_robot.py --kill-stale")
    if expected is not None:
        lines.append(f"  This run will most likely use port {expected} instead -- forward THAT port.")
    return "\n".join(lines)
