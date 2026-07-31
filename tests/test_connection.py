"""Tests for viewer-connection detection.

These encode a real debugging session: the viewer was healthy, but the generated
instructions pointed at a host the laptop could neither resolve nor reach, so the
browser showed nothing. The logic below is what distinguishes the cases.

    pytest tests/test_connection.py -v
"""

from __future__ import annotations

from unittest import mock

from faro.viz.connection import SessionInfo, connection_help, detect_session


def make_session(**overrides) -> SessionInfo:
    """A remote cluster session, matching the real setup this was debugged against."""
    defaults = dict(
        hostname="crannog04.inf.ed.ac.uk",
        username="s2816905",
        is_ssh=True,
        ssh_entry_host="129.215.18.137",
        in_vscode=True,
        vscode_tunnel=True,
        slurm_job="3577768",
        slurm_node="crannog04",
        has_display=False,
        ssh_alias=None,
    )
    defaults.update(overrides)
    return SessionInfo(**defaults)


# ------------------------------------------------------------------ local case
def test_local_session_needs_no_tunnel():
    """On a laptop the viewer URL is simply openable; say so and stop."""
    session = make_session(
        is_ssh=False, ssh_entry_host=None, vscode_tunnel=False,
        slurm_job=None, slurm_node=None, has_display=True, hostname="my-laptop",
    )
    assert session.browser_is_local

    text = connection_help(7000, session)
    assert "http://127.0.0.1:7000/static/" in text
    assert "no tunnel or port forwarding needed" in text.lower()
    assert "ssh -N -L" not in text  # must not clutter a local run with tunnel advice


def test_local_vscode_terminal_is_still_local():
    """VSCODE_IPC_HOOK_CLI is set for LOCAL VS Code terminals too.

    So `in_vscode` must not imply remoteness -- only SSH or an active
    `code tunnel` server does.
    """
    session = make_session(
        is_ssh=False, ssh_entry_host=None, in_vscode=True, vscode_tunnel=False,
        slurm_job=None, slurm_node=None, has_display=True,
    )
    assert session.browser_is_local


# ----------------------------------------------------------------- remote case
def test_ssh_session_is_not_local():
    assert not make_session().browser_is_local


def test_two_hop_detected_when_entry_host_differs():
    """Entry host `stanger` != viewer host `crannog04`, so the tunnel needs a hop.

    Compared by resolved IP, not name: SSH_CONNECTION gives a bare address.
    """
    session = make_session()
    with mock.patch("faro.viz.connection._same_host", return_value=False):
        assert session.needs_two_hop


def test_single_hop_when_ssh_target_is_the_viewer_host():
    session = make_session()
    with mock.patch("faro.viz.connection._same_host", return_value=True):
        assert not session.needs_two_hop


def test_tunnel_command_forwards_through_the_gateway():
    """The regression that caused the bug: `-L port:localhost:port` to the compute
    node tunnels to the wrong machine, or to one the laptop cannot resolve."""
    session = make_session(ssh_alias="mlp1")
    text = connection_help(7000, session)
    assert "ssh -N -L 7000:crannog04:7000 mlp1" in text
    # It must NOT emit the naive single-hop form for this topology.
    assert "-L 7000:localhost:7000" not in text


def test_ssh_alias_overrides_autodetection():
    """An explicit ~/.ssh/config alias always wins.

    The address reachable from outside routinely differs from anything visible
    from inside the cluster, so a user-provided alias beats inference.
    """
    session = make_session(ssh_alias="mlp1")
    assert session.tunnel_gateway == "mlp1"
    text = connection_help(7000, session)
    # An alias carries its own User, so no user@ prefix.
    assert "s2816905@mlp1" not in text
    assert "FARO_SSH_GATEWAY" not in text  # already configured: stop nagging


def test_missing_alias_explains_how_to_set_one():
    text = connection_help(7000, make_session())
    assert "FARO_SSH_GATEWAY" in text


def test_private_address_suppresses_the_direct_option():
    """Compute nodes on 192.168.x.x are unreachable no matter what VPN you use.

    Offering a direct URL there sends the user down a dead end, which is exactly
    what happened during debugging.
    """
    session = make_session()
    with mock.patch("faro.viz.connection._resolves_private", return_value=True):
        assert not session.is_directly_reachable
        text = connection_help(7000, session)
        assert "will NOT work" in text
        assert f"http://{session.hostname}:7000/static/" not in text


def test_public_address_offers_the_direct_option():
    session = make_session()
    with mock.patch("faro.viz.connection._resolves_private", return_value=False):
        assert session.is_directly_reachable
        assert "http://crannog04.inf.ed.ac.uk:7000/static/" in connection_help(7000, session)


def test_vscode_tunnel_warns_against_localhost():
    """With `code tunnel`, forwarded ports do NOT appear on the laptop's 127.0.0.1."""
    text = connection_help(7000, make_session(vscode_tunnel=True))
    assert "devtunnels.ms" in text
    assert "Do NOT use 127.0.0.1" in text


def test_slurm_node_appears_so_the_command_survives_reallocation():
    """The node changes every allocation; the command must name the current one."""
    text = connection_help(7000, make_session(slurm_node="gpu-node-17", ssh_alias="mlp1"))
    assert "gpu-node-17" in text
    assert "3577768" in text


def test_port_is_substituted_everywhere():
    """Meshcat may land on 7001+; stale ports in the instructions are the whole bug."""
    text = connection_help(7004, make_session(ssh_alias="mlp1"))
    assert "7004" in text
    assert ":7000" not in text


# ------------------------------------------------------------------- detection
def test_detect_session_runs_and_is_self_consistent():
    """Smoke test against the real environment -- must never raise."""
    session = detect_session()
    assert session.hostname
    assert session.username
    assert isinstance(session.browser_is_local, bool)
    # Instructions must render for whatever this machine happens to be.
    assert connection_help(7000, session)
