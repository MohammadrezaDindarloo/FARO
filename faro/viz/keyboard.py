"""Single-keypress input for the interactive demo scripts.

Reading one key without waiting for ENTER needs the terminal put into raw mode.
That works over SSH, which is where these scripts run -- but it must be undone
reliably, or the shell is left with echo off and no line editing after a crash.
Hence the try/finally around every use.

Degrades on purpose: when stdin is not a terminal (piped input, a batch job, a
pytest run) there is no key to read, so `read_key` returns None and callers fall
back to non-interactive behaviour rather than blocking forever.
"""

from __future__ import annotations

import sys

# Normalised key names, so callers never compare against raw escape bytes.
NEXT = "next"
REPEAT = "repeat"
PREVIOUS = "previous"
QUIT = "quit"


def stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def read_key() -> str | None:
    """Read exactly one keypress. Returns None if stdin is not a terminal.

    Ctrl-C is delivered as \\x03 in raw mode rather than raising, so it is
    converted back into a KeyboardInterrupt to keep the usual behaviour.
    """
    if not stdin_is_interactive():
        return None

    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        # Always restore, even on exception -- otherwise the user's shell is left
        # with echo disabled.
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    if char == "\x03":
        raise KeyboardInterrupt
    return char


def read_navigation(allow_previous: bool = True) -> str:
    """Block until the user presses a navigation key.

    ENTER / n  -> NEXT
    SPACE / r  -> REPEAT
    p          -> PREVIOUS
    q / ESC    -> QUIT

    Returns NEXT immediately when stdin is not a terminal, so piped and batch runs
    play straight through instead of hanging.
    """
    if not stdin_is_interactive():
        return NEXT

    while True:
        key = read_key()
        if key is None:
            return NEXT
        if key in ("\r", "\n", "n"):
            return NEXT
        if key in (" ", "r"):
            return REPEAT
        if key == "p":
            return PREVIOUS if allow_previous else REPEAT
        if key in ("q", "\x1b"):
            return QUIT
        # Anything else: keep waiting rather than guessing what was meant.
