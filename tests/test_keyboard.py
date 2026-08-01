"""Tests for the single-keypress navigation used by the demo scripts.

Keyboard handling is easy to get wrong in ways that only show up interactively --
a terminal left with echo disabled, or a batch run that blocks forever waiting for
a key nobody will press. Both are covered here.

    pytest tests/test_keyboard.py -v
"""

from __future__ import annotations

import pytest

from faro.viz import keyboard
from faro.viz.keyboard import NEXT, PREVIOUS, QUIT, REPEAT, read_navigation


@pytest.fixture
def interactive(monkeypatch):
    """Pretend stdin is a terminal, and feed keys from a list."""
    monkeypatch.setattr(keyboard, "stdin_is_interactive", lambda: True)

    def feed(keys):
        pending = list(keys)
        monkeypatch.setattr(keyboard, "read_key", lambda: pending.pop(0))

    return feed


def test_enter_and_n_go_to_the_next_scenario(interactive):
    for key in ("\r", "\n", "n"):
        interactive([key])
        assert read_navigation() == NEXT


def test_space_and_r_replay_the_current_scenario(interactive):
    """SPACE is the whole point: watch the same sweep again against the equation."""
    for key in (" ", "r"):
        interactive([key])
        assert read_navigation() == REPEAT


def test_p_goes_back(interactive):
    interactive(["p"])
    assert read_navigation() == PREVIOUS


def test_p_on_the_first_scenario_replays_instead_of_underflowing(interactive):
    """There is no scenario before the first one, so 'p' must not walk off the end."""
    interactive(["p"])
    assert read_navigation(allow_previous=False) == REPEAT


def test_q_and_escape_quit(interactive):
    for key in ("q", "\x1b"):
        interactive([key])
        assert read_navigation() == QUIT


def test_unrecognised_keys_are_ignored_not_guessed(interactive):
    """A stray keypress must keep waiting rather than advancing unexpectedly."""
    interactive(["z", "!", "5", "\r"])
    assert read_navigation() == NEXT


def test_non_interactive_stdin_advances_instead_of_blocking(monkeypatch):
    """Piped input and batch jobs must play straight through.

    Without this, `--no-viz` in a script or CI run would hang forever on the first
    navigation prompt.
    """
    monkeypatch.setattr(keyboard, "stdin_is_interactive", lambda: False)
    assert read_navigation() == NEXT


def test_read_key_returns_none_when_not_a_terminal(monkeypatch):
    monkeypatch.setattr(keyboard, "stdin_is_interactive", lambda: False)
    assert keyboard.read_key() is None


def _fake_terminal(monkeypatch, *, reads, restored=None):
    """Fake stdin + termios/tty so read_key can be exercised off a real terminal."""
    import sys

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 0

        def read(self, _n):
            value = reads.pop(0)
            if isinstance(value, type) and issubclass(value, BaseException):
                raise value
            return value

    fake_termios = type("T", (), {
        "tcgetattr": staticmethod(lambda fd: "SAVED"),
        "tcsetattr": staticmethod(
            lambda fd, when, attrs: (restored.append(attrs) if restored is not None else None)
        ),
        "TCSADRAIN": 1,
    })
    fake_tty = type("Y", (), {"setraw": staticmethod(lambda fd: None)})

    monkeypatch.setitem(sys.modules, "termios", fake_termios)
    monkeypatch.setitem(sys.modules, "tty", fake_tty)
    monkeypatch.setattr(sys, "stdin", FakeStdin())


def test_ctrl_c_still_interrupts(monkeypatch):
    """In raw mode Ctrl-C arrives as \\x03 instead of raising, so it is re-raised.

    Otherwise the only way out of the loop would be the 'q' key, and habit says
    Ctrl-C should work.
    """
    _fake_terminal(monkeypatch, reads=["\x03"])
    with pytest.raises(KeyboardInterrupt):
        keyboard.read_key()


def test_terminal_settings_are_restored_after_ctrl_c(monkeypatch):
    """Even the Ctrl-C path must restore the terminal before propagating."""
    restored: list = []
    _fake_terminal(monkeypatch, reads=["\x03"], restored=restored)
    with pytest.raises(KeyboardInterrupt):
        keyboard.read_key()
    assert restored == ["SAVED"]


def test_terminal_settings_are_restored_after_an_exception(monkeypatch):
    """A crash inside read_key must not leave the shell with echo disabled.

    This is the failure users notice: the terminal stops echoing what they type.
    """
    class Boom(Exception):
        pass

    restored: list = []
    _fake_terminal(monkeypatch, reads=[Boom], restored=restored)

    with pytest.raises(Boom):
        keyboard.read_key()

    assert restored == ["SAVED"], "terminal settings were not restored"
