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
    left_pose_reset_req: list
    left_pose_runtime_armed: list
    left_pose_state: object
    mode_state: object | None
    left_unwind_s: float
    left_swarm_enabled: bool

    @classmethod
    def from_boot(cls, boot: OnlineBoot) -> OnlineKeyContext:
        return cls(
            gesture_control_enabled=boot.gesture_control_enabled_box,
            prearm_climb_enabled=boot.prearm_climb_enabled_box,
            prearm_phase_box=boot.prearm_phase_box,
            prearm_vertical_leg_box=boot.prearm_vertical_leg_box,
            prearm_has_flown_box=boot.prearm_has_flown_box,
            prearm_takeoff_z=float(boot.prearm_takeoff_z),
            prearm_hover_z=float(boot.prearm_hover_z),
            left_pose_reset_req=boot.left_pose_reset_req_box,
            left_pose_runtime_armed=boot.left_pose_runtime_armed_box,
            left_pose_state=boot.left_pose_state,
            mode_state=boot.mode_state,
            left_unwind_s=float(boot.left_unwind_s),
            left_swarm_enabled=bool(boot.left_pose_state.enabled),
        )


def probe_global_hotkey_backends() -> dict:
    """Report whether pynput/keyboard are importable in the **current** interpreter."""
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
    """``pip install pynput keyboard`` into ``sys.executable`` if imports fail."""
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
        return self._mode

    def _enqueue_char(self, ch: str | None) -> None:
        if not ch:
            return
        self._q.put(ord(ch))

    def start(self, *, use_global: bool = True, use_stdin: bool = True) -> None:
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
        try:
            return int(self._q.get_nowait())
        except queue.Empty:
            return None

    def drain(self) -> list[int]:
        out: list[int] = []
        while True:
            k = self.poll()
            if k is None:
                break
            out.append(k)
        return out

    def merge_cv_key(self, cv_key: int) -> int | None:
        """Prefer queued global/stdin keys; else OpenCV ``waitKey`` (may need window focus)."""
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
    """Drain all pending keys; return True to quit the main loop."""
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
    """Handle one key press. Returns True to quit the main loop."""
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
        phase = str(ctx.prearm_phase_box[0])
        if phase == "ground":
            ctx.prearm_phase_box[0] = "vertical"
            ctx.prearm_vertical_leg_box[0] = "climb"
            ctx.prearm_climb_enabled[0] = True
            ctx.prearm_has_flown_box[0] = True
            print(
                f"Vertical takeoff: axswarm-filtered climb to z={ctx.prearm_takeoff_z:.2f}m "
                "(ground XY fixed). Press 1 again for hover formation."
            )
        elif phase == "vertical":
            if str(ctx.prearm_vertical_leg_box[0]) == "climb":
                ctx.prearm_phase_box[0] = "formation"
                print(
                    f"Hover formation: direct 3D move to hover layout "
                    f"(z≈{ctx.prearm_hover_z:.2f}m, axswarm-filtered). "
                    "Press 1 to shrink back to vertical."
                )
            else:
                ctx.prearm_phase_box[0] = "ground"
                ctx.prearm_climb_enabled[0] = False
                print(
                    f"Axswarm-filtered descent to ground layout "
                    f"(from z≈{ctx.prearm_takeoff_z:.2f}m). Press 1 for next takeoff."
                )
        elif phase == "formation":
            ctx.prearm_phase_box[0] = "vertical"
            ctx.prearm_vertical_leg_box[0] = "descend"
            print(
                f"Hover formation: direct 3D return to vertical layout "
                f"(z≈{ctx.prearm_takeoff_z:.2f}m, axswarm-filtered). "
                "Press 1 again for ground."
            )
        return False
    if key == _KEY_SPACE:
        if ctx.gesture_control_enabled[0]:
            ctx.gesture_control_enabled[0] = False
            _warned_climb_while_gestured = False
            print(
                "Gesture control disarmed. Press 1 to descend: formation → vertical → ground, "
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
            print("Left swarm move ignored: --left-swarm-pose is disabled.")
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