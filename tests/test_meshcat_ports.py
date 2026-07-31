"""Tests for meshcat port bookkeeping.

Motivated by a real failure: a leftover server held port 7000, the new run silently
landed on 7001, and forwarding 7000 showed an empty scene that looked exactly like
a broken visualizer.

    pytest tests/test_meshcat_ports.py -v
"""

from __future__ import annotations

import socket

from faro.viz.meshcat_ports import (
    MESHCAT_FIRST_PORT,
    MeshcatServer,
    next_free_port,
    port_is_free,
    preflight_report,
)


def _server(pid=1234, ppid=1, ports=(6000, 7000), elapsed="00:10") -> MeshcatServer:
    return MeshcatServer(pid=pid, ppid=ppid, ports=set(ports), elapsed=elapsed)


def test_http_port_ignores_the_zmq_port():
    """A meshcat server binds BOTH a ZeroMQ port (6000+) and an HTTP port (7000+).

    Regression test: an earlier version collapsed pid->port as a one-to-one map, so
    the zmq port overwrote the HTTP port and the diagnostic reported 6000 -- which
    also suppressed the most useful line of the warning.
    """
    assert _server(ports=(6000, 7000)).http_port == 7000
    assert _server(ports=(6001, 7003)).http_port == 7003


def test_http_port_is_none_when_no_port_in_range():
    assert _server(ports=(6000,)).http_port is None
    assert _server(ports=()).http_port is None


def test_orphan_detection_is_by_parent_pid():
    """PPID 1 means the spawning script died and left the server behind.

    Only orphans are safe to kill; a server with a live parent belongs to a script
    someone may be using in another terminal.
    """
    assert _server(ppid=1).is_orphan
    assert not _server(ppid=4242).is_orphan


def test_port_is_free_detects_a_bound_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))          # let the OS pick a free port
        sock.listen(1)
        port = sock.getsockname()[1]
        assert not port_is_free(port)
    # Once closed it should be bindable again (SO_REUSEADDR avoids TIME_WAIT issues).
    assert port_is_free(port)


def test_next_free_port_skips_occupied_ports():
    """Mirrors meshcat's own upward scan, so we can predict the port it will take."""
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", MESHCAT_FIRST_PORT))
        except OSError:
            import pytest

            pytest.skip("port 7000 already in use by something else")
        sock.listen(1)
        assert next_free_port() != MESHCAT_FIRST_PORT
        assert next_free_port() > MESHCAT_FIRST_PORT


def test_preflight_is_silent_when_the_port_is_free():
    """No warning noise in the normal case."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", MESHCAT_FIRST_PORT))
        except OSError:
            import pytest

            pytest.skip("port 7000 is in use; cannot test the clean path")
    assert preflight_report() == ""


def test_preflight_warns_when_the_port_is_taken():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", MESHCAT_FIRST_PORT))
        except OSError:
            import pytest

            pytest.skip("port 7000 already in use by something else")
        sock.listen(1)
        report = preflight_report()

    assert "WARNING" in report
    # The whole point is telling the user which port to forward instead.
    assert "forward" in report.lower()
