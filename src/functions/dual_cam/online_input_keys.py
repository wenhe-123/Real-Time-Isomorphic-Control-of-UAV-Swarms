"""Global / queued key handling for online control (1 climb, SPACE arm/disarm, 0 L-move, q quit)."""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from functions.mode_switch.modes_runtime import clear_mode_rotation_freeze_latch

if TYPE_CHECKING:
    from functions.runtime.online_boot import OnlineBoot

# --- input keys ---
_KEY_SPACE = ord(" ")
_KEY_ENTER = 13
_KEY_Q = ord("q")
_warned_climb_while_gestured = False


def _hl_prearm_key1(ctx: OnlineKeyContext, ex: object) -> bool:
    """Advance the real/sim prearm state machine on key ``1``.

    Cycles vertical hold → formation → HL descend/land → takeoff.

    Args:
        ctx: Mutable hotkey context with prearm phase boxes.
        ex: Real or sim high-level executor.

    Returns:
        Always ``False`` (never requests quit).
    """
    phase = str(ctx.prearm_phase_box[0])
    leg = str(ctx.prearm_vertical_leg_box[0])
    if phase == "vertical" and leg == "climb":
        ctx.prearm_phase_box[0] = "formation"
        print(
            f"Hover formation ramp → z≈{ctx.prearm_hover_z:.2f}m (axswarm). "
            "Press 1 for in-place HL descend + land.",
            flush=True,
        )
    elif phase == "formation":
        ex.pause_setpoints_for_hl()
        print(
            f"Axswarm off; HL descend −{ctx.prearm_descend_m:.2f}m in place, then land...",
            flush=True,
        )
        ex.high_level_descend(ctx.prearm_descend_m)
        ex.high_level_land()
        ctx.prearm_phase_box[0] = "ground"
        ctx.prearm_climb_enabled[0] = False
        ctx.prearm_vertical_leg_box[0] = "climb"
        print("On ground. Press 1 to take off again, or quit.", flush=True)
    elif phase == "ground":
        ex.high_level_takeoff(ctx.prearm_takeoff_z)
        ctx.prearm_phase_box[0] = "vertical"
        ctx.prearm_vertical_leg_box[0] = "climb"
        ctx.prearm_climb_enabled[0] = True
        ctx.prearm_has_flown_box[0] = True
        print(
            f"Axswarm vertical hold at z≈{ctx.prearm_takeoff_z:.2f}m. "
            "Press 1: formation → in-place HL descend + land.",
            flush=True,
        )
    else:
        print(
            f"[WARN] Real prearm phase {phase!r}/{leg!r} ignored "
            "(wait for formation/descend/land to finish).",
            flush=True,
        )
    return False


@dataclass
class OnlineKeyContext:
    """Mutable hotkey state boxes + scalars shared by key handlers."""

    gesture_control_enabled: list
    prearm_climb_enabled: list
    prearm_phase_box: list
    prearm_vertical_leg_box: list
    prearm_has_flown_box: list
    prearm_takeoff_z: float
    prearm_hover_z: float
    prearm_descend_m: float
    left_pose_reset_req: list
    left_pose_runtime_armed: list
    left_pose_state: object
    mode_state: object | None
    left_unwind_s: float
    left_swarm_enabled: bool
    real_executor: object | None = None
    sim_executor: object | None = None

    @classmethod
    def from_boot(cls, boot: OnlineBoot) -> OnlineKeyContext:
        """Build key-handler context from an ``OnlineBoot`` runtime bundle.

        Args:
            boot: Initialized online boot object with pose and mode state.

        Returns:
            ``OnlineKeyContext`` wired to the boot object's mutable boxes.
        """
        return cls(
            gesture_control_enabled=boot.gesture_control_enabled_box,
            prearm_climb_enabled=boot.prearm_climb_enabled_box,
            prearm_phase_box=boot.prearm_phase_box,
            prearm_vertical_leg_box=boot.prearm_vertical_leg_box,
            prearm_has_flown_box=boot.prearm_has_flown_box,
            prearm_takeoff_z=float(boot.prearm_takeoff_z),
            prearm_hover_z=float(boot.prearm_hover_z),
            prearm_descend_m=float(boot.prearm_descend_m),
            left_pose_reset_req=boot.left_pose_reset_req_box,
            left_pose_runtime_armed=boot.left_pose_runtime_armed_box,
            left_pose_state=boot.left_pose_state,
            mode_state=boot.mode_state,
            left_unwind_s=float(boot.left_unwind_s),
            left_swarm_enabled=bool(boot.left_pose_state.enabled),
            real_executor=boot.real_executor,
            sim_executor=boot.sim_executor,
        )


def probe_global_hotkey_backends() -> dict:
    """Report whether global hotkey libraries are importable in this interpreter.

    Returns:
        Dict with keys ``pynput``, ``keyboard``, ``stdin_tty``, ``dev_tty``,
        ``python``, and ``errors`` (import failure messages).
    """
    out: dict = {
        "pynput": False,
        "keyboard": False,
        "stdin_tty": False,
        "dev_tty": False,
        "python": sys.executable,
        "errors": {},
    }
    try:
        from pynput import keyboard as _pk  # noqa: F401

        out["pynput"] = True
    except Exception as exc:
        out["errors"]["pynput"] = str(exc)
    try:
        import keyboard as _kb  # noqa: F401

        out["keyboard"] = True
    except Exception as exc:
        out["errors"]["keyboard"] = str(exc)
    out["stdin_tty"] = bool(sys.stdin.isatty())
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
        os.close(fd)
        out["dev_tty"] = True
    except OSError:
        out["dev_tty"] = False
    return out


def format_hotkey_install_hint(probe: dict | None = None) -> str:
    """Format a multi-line hint for installing or using global hotkey backends.

    Args:
        probe: Optional precomputed probe dict; probes when ``None``.

    Returns:
        Human-readable install and troubleshooting text.
    """
    p = probe if probe is not None else probe_global_hotkey_backends()
    lines = [
        f"Python: {p.get('python', sys.executable)}",
        "Global SPACE/0 need pynput or keyboard in THIS environment.",
        "  pixi:   cd iso_swarm/src && pixi install && pixi run online-dual",
        "  pip:    python -m pip install pynput keyboard   # use the Python path above",
        "Without them: focus the launch terminal, or click the Orbbec preview, then press keys.",
    ]
    errs = p.get("errors") or {}
    for name, msg in errs.items():
        lines.append(f"  import {name} failed: {msg}")
    return "\n".join(lines)


def try_install_hotkey_dependencies(*, quiet: bool = False) -> dict:
    """Install ``pynput`` and ``keyboard`` into ``sys.executable`` if imports fail.

    Args:
        quiet: If True, suppress install output and capture subprocess stdout.

    Returns:
        Updated probe dict from ``probe_global_hotkey_backends``.
    """
    probe = probe_global_hotkey_backends()
    if probe.get("pynput") or probe.get("keyboard"):
        return probe
    import subprocess

    if not quiet:
        print(
            f"[INFO] Installing pynput + keyboard into {probe.get('python', sys.executable)} ..."
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pynput>=1.7.6,<2",
            "keyboard>=0.13.5,<0.14",
        ],
        check=False,
        capture_output=bool(quiet),
    )
    return probe_global_hotkey_backends()


class OnlineKeyQueue:
    """Thread-safe key codes (``ord``) from global hotkeys and/or stdin."""

    def __init__(self) -> None:
        self._q: queue.Queue[int] = queue.Queue()
        self._running = False
        self._pynput_listener = None
        self._keyboard_hook = None
        self._stdin_thread: threading.Thread | None = None
        self._stdin_restore: Callable[[], None] | None = None
        self._stdin_fd: int | None = None
        self._stdin_owned_tty: bool = False
        self._mode = "off"

    @property
    def mode(self) -> str:
        """Active listener backend summary (e.g. ``"pynput+stdin"`` or ``"off"``).

        Returns:
            Plus-separated list of started backends.
        """
        return self._mode

    def _enqueue_char(self, ch: str | None) -> None:
        if not ch:
            return
        self._q.put(ord(ch))

    def start(self, *, use_global: bool = True, use_stdin: bool = True) -> None:
        """Start background listeners for global hotkeys and/or stdin.

        Args:
            use_global: Try ``pynput`` and ``keyboard`` global hooks.
            use_stdin: Start a cbreak stdin reader thread on a TTY.
        """
        if self._running:
            return
        self._running = True
        started: list[str] = []
        if use_global:
            try:
                from pynput import keyboard as pynput_kb

                def _on_press(key) -> None:
                    try:
                        if hasattr(key, "char") and key.char is not None:
                            self._enqueue_char(key.char)
                        elif key == pynput_kb.Key.space:
                            self._q.put(_KEY_SPACE)
                        elif key == pynput_kb.Key.enter:
                            self._q.put(_KEY_ENTER)
                        elif hasattr(key, "vk") and key.vk is not None:
                            vk = int(key.vk)
                            if vk in (48, 96):
                                self._q.put(ord("0"))
                            elif vk == ord("Q"):
                                self._q.put(_KEY_Q)
                    except Exception:
                        pass

                self._pynput_listener = pynput_kb.Listener(on_press=_on_press)
                self._pynput_listener.start()
                started.append("pynput")
            except Exception:
                self._pynput_listener = None

            try:
                import keyboard as kb

                def _kb_event(e) -> None:
                    try:
                        if getattr(e, "event_type", None) != kb.KEY_DOWN:
                            return
                        name = str(getattr(e, "name", "") or "").lower()
                        if name == "space":
                            self._q.put(_KEY_SPACE)
                        elif name in ("0", "num 0"):
                            self._q.put(ord("0"))
                        elif name == "q":
                            self._q.put(_KEY_Q)
                        elif name in ("enter", "return"):
                            self._q.put(_KEY_ENTER)
                        elif name == "z":
                            self._q.put(ord("z"))
                        elif name == "1":
                            self._q.put(ord("1"))
                    except Exception:
                        pass

                self._keyboard_hook = kb.hook(_kb_event)
                started.append("keyboard")
            except Exception:
                self._keyboard_hook = None

        if use_stdin:
            fd = self._resolve_stdin_fd()
            if fd is not None:
                self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
                self._stdin_thread.start()
                started.append("stdin")

        self._mode = "+".join(started) if started else "off"

    def _resolve_stdin_fd(self) -> int | None:
        if sys.stdin.isatty():
            return sys.stdin.fileno()
        try:
            self._stdin_fd = os.open("/dev/tty", os.O_RDONLY)
            self._stdin_owned_tty = True
            return int(self._stdin_fd)
        except OSError:
            self._stdin_fd = None
            self._stdin_owned_tty = False
            return None

    def _stdin_loop(self) -> None:
        fd = self._stdin_fd if self._stdin_fd is not None else sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            return

        def _restore() -> None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
            if self._stdin_owned_tty and self._stdin_fd is not None:
                try:
                    os.close(self._stdin_fd)
                except OSError:
                    pass
                self._stdin_fd = None
                self._stdin_owned_tty = False

        self._stdin_restore = _restore
        try:
            tty.setcbreak(fd)
            while self._running:
                r, _, _ = select.select([fd], [], [], 0.06)
                if not r:
                    continue
                ch = os.read(fd, 1)
                if not ch:
                    continue
                self._q.put(ord(ch.decode("utf-8", errors="ignore") or "\x00"))
        finally:
            _restore()

    def stop(self) -> None:
        """Stop all listeners and restore terminal settings.

        Returns:
            None.
        """
        self._running = False
        if self._pynput_listener is not None:
            try:
                self._pynput_listener.stop()
            except Exception:
                pass
            self._pynput_listener = None
        if self._keyboard_hook is not None:
            try:
                import keyboard as kb

                kb.unhook(self._keyboard_hook)
            except Exception:
                pass
            self._keyboard_hook = None
        if self._stdin_thread is not None and self._stdin_thread.is_alive():
            self._stdin_thread.join(timeout=0.5)
        self._stdin_thread = None
        if self._stdin_owned_tty and self._stdin_fd is not None:
            try:
                os.close(self._stdin_fd)
            except OSError:
                pass
            self._stdin_fd = None
            self._stdin_owned_tty = False
        if self._stdin_restore is not None:
            try:
                self._stdin_restore()
            except Exception:
                pass
            self._stdin_restore = None

    def poll(self) -> int | None:
        """Return one queued key code without blocking.

        Returns:
            Key code (``ord`` value), or ``None`` when the queue is empty.
        """
        try:
            return int(self._q.get_nowait())
        except queue.Empty:
            return None

    def drain(self) -> list[int]:
        """Remove and return all queued key codes.

        Returns:
            List of key codes in arrival order (may be empty).
        """
        out: list[int] = []
        while True:
            k = self.poll()
            if k is None:
                break
            out.append(k)
        return out

    def merge_cv_key(self, cv_key: int) -> int | None:
        """Prefer queued global/stdin keys over an OpenCV ``waitKey`` result.

        Args:
            cv_key: Raw key from ``cv2.waitKey`` (255 means no key).

        Returns:
            Merged key code, or ``None`` when no key is pending.
        """
        keys = self.drain()
        if keys:
            return keys[0]
        if cv_key is not None and int(cv_key) >= 0:
            ck = int(cv_key) & 0xFF
            if ck != 255:
                return ck
        return None


def process_online_control_keys(
    key_queue: OnlineKeyQueue | None,
    *,
    global_hotkeys: bool,
    ctx: OnlineKeyContext,
    cv_key: int | None = None,
) -> bool:
    """Drain all pending keys and dispatch them to control handlers.

    Args:
        key_queue: Optional ``OnlineKeyQueue`` for global/stdin hotkeys.
        global_hotkeys: If True, read from ``key_queue`` before ``cv_key``.
        ctx: Mutable hotkey context shared by handlers.
        cv_key: Fallback key from OpenCV ``waitKey``.

    Returns:
        True when a handler requests quitting the main loop.
    """
    keys: list[int] = []
    if global_hotkeys and key_queue is not None:
        keys = key_queue.drain()
        if not keys and cv_key is not None:
            ck = int(cv_key) & 0xFF
            if ck != 255:
                keys = [ck]
    elif cv_key is not None:
        ck = int(cv_key) & 0xFF
        if ck != 255:
            keys = [ck]

    for k in keys:
        if apply_online_control_key(k, ctx=ctx):
            return True
    return False


def apply_online_control_key(
    key: int | None,
    *,
    ctx: OnlineKeyContext,
) -> bool:
    """Handle one control key press (SPACE arm, 0 L-move, 1 climb, q quit).

    Args:
        key: Key code from ``ord`` or OpenCV ``waitKey``.
        ctx: Mutable hotkey context with pose and prearm state.

    Returns:
        True when the main loop should exit (q or Enter).
    """
    global _warned_climb_while_gestured
    if key is None:
        return False
    if key in (_KEY_Q, _KEY_ENTER):
        print("Quitting online control...", flush=True)
        return True
    if key == ord("1"):
        if ctx.gesture_control_enabled[0]:
            if not _warned_climb_while_gestured:
                print("Cannot toggle climb/ground: gesture control is armed (SPACE).")
                _warned_climb_while_gestured = True
            return False
        _warned_climb_while_gestured = False
        ex = ctx.real_executor if ctx.real_executor is not None else ctx.sim_executor
        if ex is not None:
            return _hl_prearm_key1(ctx, ex)
        return False
    if key == _KEY_SPACE:
        if ctx.gesture_control_enabled[0]:
            ctx.gesture_control_enabled[0] = False
            _warned_climb_while_gestured = False
            print(
                "Gesture control disarmed. Press 1 to land: formation → HL descend + land, "
                "then q to quit."
            )
            return False
        ctx.gesture_control_enabled[0] = True
        print(
            f"Gesture control armed: morph follows mode/open at hover z={ctx.prearm_hover_z:.2f}m."
        )
        return False
    if key == ord("z") and ctx.left_swarm_enabled and ctx.left_pose_runtime_armed[0]:
        ctx.left_pose_reset_req[0] = True
        print("Left swarm move: will re-zero reference on next valid left hand.")
        return False
    if key == ord("0"):
        if not ctx.left_swarm_enabled:
            print("Left swarm move ignored: left swarm pose disabled (yaml or --no-left-swarm-pose).")
            return False
        if ctx.left_pose_state.is_unwinding():
            ctx.left_pose_state.cancel_unwind()
            ctx.left_pose_runtime_armed[0] = True
            ctx.left_pose_reset_req[0] = True
            if ctx.mode_state is not None:
                clear_mode_rotation_freeze_latch(ctx.mode_state)
            print("Left swarm: restore cancelled — re-armed with current hand as baseline.")
        elif not ctx.left_pose_runtime_armed[0]:
            ctx.left_pose_runtime_armed[0] = True
            ctx.left_pose_reset_req[0] = True
            if ctx.mode_state is not None:
                clear_mode_rotation_freeze_latch(ctx.mode_state)
            print(
                "Left swarm move: ON — move hand to translate/rotate formation; "
                "press 0 again to smoothly return to morph-only frame."
            )
        else:
            ctx.left_pose_runtime_armed[0] = False
            ctx.left_pose_state.begin_unwind(ctx.left_unwind_s)
            if ctx.mode_state is not None:
                clear_mode_rotation_freeze_latch(ctx.mode_state)
            print(
                f"Left swarm move: restoring morph frame over ~{ctx.left_unwind_s:.1f}s "
                "(press 0 during restore to cancel and re-arm)."
            )
        return False
    return False