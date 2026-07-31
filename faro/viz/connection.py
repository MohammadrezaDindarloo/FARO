"""Working out how to actually reach the Meshcat viewer from a browser.

The viewer always serves on `127.0.0.1:<port>` *of the machine running the script*.
Whether that is reachable from your browser depends entirely on where you are:

  * running on your own laptop        -> just open it
  * SSH'd into a cluster node         -> needs a tunnel, or direct access over VPN
  * VS Code Tunnels (`code tunnel`)   -> ports come out on *.devtunnels.ms, NOT localhost
  * under Slurm on a compute node     -> the node changes every allocation

Guessing wrong gives a blank page while every process is perfectly healthy, which
is indistinguishable from a broken visualizer. So instead of documenting one setup,
this module detects the situation and prints commands with the real hostname, user
and port already substituted -- ready to copy and paste.

Nothing here is FARO-specific; every later milestone's script uses it too.
"""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionInfo:
    """Where this process is running, and how a browser might reach it."""

    hostname: str                 # FQDN of the machine running the viewer
    username: str
    is_ssh: bool                  # we arrived over SSH
    ssh_entry_host: str | None    # the address the SSH connection came in on
    in_vscode: bool               # running inside a VS Code integrated terminal
    vscode_tunnel: bool           # a `code tunnel` process is running on this host
    slurm_job: str | None
    slurm_node: str | None
    has_display: bool             # an X/Wayland display is available
    ssh_alias: str | None         # $FARO_SSH_GATEWAY: how YOUR LAPTOP reaches this cluster

    @property
    def browser_is_local(self) -> bool:
        """True when the browser runs on this same machine.

        SSH and VS Code Tunnels both mean the human is elsewhere. A local VS Code
        window does NOT (VSCODE_IPC_HOOK_CLI is set for local terminals too), which
        is why `in_vscode` deliberately does not appear here.
        """
        return not (self.is_ssh or self.vscode_tunnel)

    @property
    def needs_two_hop(self) -> bool:
        """True when the host we SSH'd into is NOT the host running the viewer.

        Typical cluster layout: you SSH to a gateway/login node, and Slurm puts you
        on a compute node that is only reachable from inside. A plain
        `-L port:localhost:port` to the compute node then fails, because the laptop
        cannot resolve or reach it at all -- the tunnel has to go *through* the
        gateway.

        Compared by resolved IP, not by name: the entry point arrives as a bare
        address in SSH_CONNECTION, and the same machine routinely has several names.
        """
        if not self.ssh_entry_host:
            return False
        return not _same_host(self.ssh_entry_host, self.hostname)

    @property
    def is_directly_reachable(self) -> bool:
        """Can a browser plausibly hit this host head-on?

        False when the host resolves only to a private (RFC 1918) address, which is
        the normal case for cluster compute nodes: no VPN makes 192.168.x.x routable
        from a laptop, so offering a direct URL would just waste the user's time.
        """
        return not _resolves_private(self.hostname)

    @property
    def tunnel_target(self) -> str:
        """Hostname to use as the far end of an SSH tunnel."""
        return self.slurm_node or self.hostname.split(".")[0]

    @property
    def tunnel_gateway(self) -> str | None:
        """What to SSH into from the laptop, when this host is not reachable directly.

        $FARO_SSH_GATEWAY wins whenever it is set: an explicit `~/.ssh/config` alias
        beats anything inferrable from inside the cluster, because the address that
        works from outside frequently differs from any address visible in here.
        """
        if self.ssh_alias:
            return self.ssh_alias
        if not self.needs_two_hop:
            return None
        return _reverse_name(self.ssh_entry_host) or self.ssh_entry_host

    @property
    def gateway_needs_user(self) -> bool:
        """An ssh:// alias already carries its own User; a bare hostname does not."""
        return self.ssh_alias is None


def detect_session() -> SessionInfo:
    """Inspect the environment. Never raises -- unknown just becomes None/False."""
    hostname = socket.getfqdn() or socket.gethostname()

    ssh_conn = os.environ.get("SSH_CONNECTION", "")
    is_ssh = bool(ssh_conn or os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"))
    # SSH_CONNECTION is "client_ip client_port server_ip server_port"; the server
    # address is the one a tunnel from the laptop should target.
    parts = ssh_conn.split()
    ssh_entry_host = parts[2] if len(parts) >= 3 else None

    return SessionInfo(
        hostname=hostname,
        username=getpass.getuser(),
        is_ssh=is_ssh,
        ssh_entry_host=ssh_entry_host,
        in_vscode=bool(os.environ.get("VSCODE_IPC_HOOK_CLI") or os.environ.get("TERM_PROGRAM") == "vscode"),
        vscode_tunnel=_vscode_tunnel_running(),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        slurm_node=os.environ.get("SLURMD_NODENAME"),
        has_display=bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        # A cluster node cannot know how your laptop's ~/.ssh/config names it: the
        # address you SSH to from outside is often not the one visible from inside
        # (gateways, VPNs, ControlMaster aliases). Set this once and the generated
        # tunnel command becomes exactly right:
        #     conda env config vars set FARO_SSH_GATEWAY=mlp1 -n faro
        ssh_alias=os.environ.get("FARO_SSH_GATEWAY") or None,
    )


def _addresses(host: str) -> set[str]:
    """All IPs a hostname resolves to. Empty set if it does not resolve."""
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None)}
    except (socket.gaierror, UnicodeError, OSError):
        return set()


def _same_host(a: str, b: str) -> bool:
    """Do two host references point at the same machine?

    Resolves both and intersects, because one side is typically a bare IP from
    SSH_CONNECTION and the other a FQDN. Falls back to comparing short names when
    neither resolves (e.g. no DNS in a container).
    """
    addrs_a, addrs_b = _addresses(a), _addresses(b)
    if addrs_a and addrs_b:
        return bool(addrs_a & addrs_b)
    return a.split(".")[0].lower() == b.split(".")[0].lower()


def _resolves_private(host: str) -> bool:
    """True if every address for `host` is private/loopback (RFC 1918 et al.)."""
    import ipaddress

    addrs = _addresses(host)
    if not addrs:
        return False  # unknown -- do not claim unreachable on a DNS failure
    parsed = []
    for addr in addrs:
        try:
            parsed.append(ipaddress.ip_address(addr))
        except ValueError:
            return False
    return all(ip.is_private or ip.is_loopback or ip.is_link_local for ip in parsed)


def _reverse_name(address: str | None) -> str | None:
    """Friendly hostname for an IP, so generated commands read sensibly."""
    if not address:
        return None
    try:
        return socket.gethostbyaddr(address)[0]
    except (OSError, socket.herror):
        return None


def _vscode_tunnel_running() -> bool:
    """Is a `code tunnel` server running on this host?

    This is the distinction that matters most: with VS Code Tunnels, forwarded
    ports are served from a *.devtunnels.ms URL and are NOT reachable at
    127.0.0.1 on the laptop, unlike Remote-SSH.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return any(
        "code" in line and " tunnel" in line and "grep" not in line
        for line in out.splitlines()
    )


def open_browser_if_local(url: str, session: SessionInfo | None = None) -> bool:
    """Open the viewer automatically when the browser is on this machine.

    Returns True if a browser was launched. Deliberately a no-op when remote:
    launching a browser on a headless cluster node achieves nothing.
    """
    session = session or detect_session()
    if not session.browser_is_local:
        return False
    try:
        import webbrowser

        return webbrowser.open(url)
    except Exception:  # noqa: BLE001 -- never let a viewer convenience kill the run
        return False


def connection_help(port: int, session: SessionInfo | None = None, path: str = "/static/") -> str:
    """Copy-pasteable instructions for reaching the viewer, tailored to this session.

    Every command comes out with the real hostname, username and port already
    filled in, so there is nothing to edit by hand -- which matters because the
    Slurm node changes on every allocation.
    """
    session = session or detect_session()
    local_url = f"http://127.0.0.1:{port}{path}"

    lines: list[str] = []
    width = 78
    lines.append("=" * width)

    # ---------------------------------------------------------------- local case
    if session.browser_is_local:
        lines.append(f"  OPEN THIS:  {local_url}")
        lines.append("=" * width)
        lines.append("  Running locally -- no tunnel or port forwarding needed.")
        if not session.has_display:
            lines.append("  (No DISPLAY detected. If this machine really is headless, see the")
            lines.append("   remote options in SETUP.md section 4.)")
        return "\n".join(lines)

    # --------------------------------------------------------------- remote case
    host = session.hostname
    user = session.username
    lines.append(f"  The viewer is running on:  {host}")
    lines.append(f"  It serves {local_url} ON THAT MACHINE -- not on your laptop.")
    lines.append("=" * width)

    if session.slurm_job:
        lines.append(f"  Slurm job {session.slurm_job} on node {session.slurm_node}"
                     " (this changes every allocation --")
        lines.append("  that is why these commands are generated, not hard-coded).")
        lines.append("")

    option = 1
    # --- ssh tunnel: listed FIRST because it is the one that reliably works ----
    gateway = session.tunnel_gateway
    lines.append(f"  [{option}] SSH TUNNEL -- run this on YOUR LAPTOP and leave it open:")
    if gateway:
        # The viewer's host is not (or may not be) the host reachable from outside,
        # so forward through the gateway; the laptop often cannot even resolve the
        # compute node's name.
        target = f"{gateway}" if not session.gateway_needs_user else f"{user}@{gateway}"
        lines.append(f"        ssh -N -L {port}:{session.tunnel_target}:{port} {target}")
        lines.append(f"        (connects to {gateway}, then forwards on to {session.tunnel_target})")
    else:
        lines.append(f"        ssh -N -L {port}:localhost:{port} {user}@{host}")
    lines.append(f"     then open:  {local_url}")
    if not session.ssh_alias:
        lines.append("")
        lines.append("     If that host is not how YOUR laptop reaches this cluster, set your")
        lines.append("     ~/.ssh/config alias once and this command becomes exact:")
        lines.append("        conda env config vars set FARO_SSH_GATEWAY=<your-alias> -n faro")
    lines.append("")
    option += 1

    # --- direct, only when it can actually work -------------------------------
    if session.is_directly_reachable:
        lines.append(f"  [{option}] DIRECT, no tunnel (works if you are on the VPN):")
        lines.append(f"        http://{host}:{port}{path}")
        lines.append("")
        option += 1
    else:
        lines.append(f"  [{option}] Direct access will NOT work: {host} resolves only to a")
        lines.append("        private address, so no VPN makes it routable from your laptop.")
        lines.append("")
        option += 1

    # --- vs code --------------------------------------------------------------
    if session.vscode_tunnel:
        lines.append(f"  [{option}] VS Code TUNNELS is active on this host. In the PORTS panel,")
        lines.append(f"        forward port {port}, click the globe icon, and APPEND {path}")
        lines.append("        to the *.devtunnels.ms URL. Do NOT use 127.0.0.1 on your laptop --")
        lines.append("        with tunnels (unlike Remote-SSH) ports do not surface there.")
        lines.append("        If the page stays blank: right-click the port -> Port Visibility")
        lines.append("        -> Public (Meshcat needs a WebSocket).")
    elif session.in_vscode:
        lines.append(f"  [{option}] VS Code Remote-SSH: open the PORTS panel, forward port {port},")
        lines.append(f"        then open {local_url}")

    lines.append("-" * width)
    # Meshcat routes "/" to a WebSocket handler, so a browser GET on the bare origin
    # returns HTTP 400 with this message. It looks like a networking failure but is
    # purely a missing path -- and VS Code's globe icon opens the bare origin.
    lines.append('  If the page says:  Can "Upgrade" only to "WebSocket".')
    lines.append(f"  ...you reached the server but omitted the path. Append {path} to the URL.")
    lines.append("=" * width)
    return "\n".join(lines)


def print_connection_help(port: int, path: str = "/static/") -> SessionInfo:
    """Detect, print, and return the session info."""
    session = detect_session()
    print(connection_help(port, session, path=path), file=sys.stdout, flush=True)
    return session
