"""Global / queued key handling for online_control."""

from shared.online_input_keys import OnlineKeyQueue, apply_online_control_key, process_online_control_keys


class _PoseStub:
    enabled = True

    def is_unwinding(self):
        return False

    def cancel_unwind(self):
        pass

    def begin_unwind(self, _s):
        pass


def test_key_queue_drain_order():
    q = OnlineKeyQueue()
    q._q.put(ord("q"))
    q._q.put(_KEY_SPACE := 32)
    got = q.drain()
    assert got == [ord("q"), 32]


def test_process_drains_multiple_keys():
    q = OnlineKeyQueue()
    armed = [False]
    reset = [False]
    runtime = [False]
    q._q.put(32)
    quit_ = process_online_control_keys(
        q,
        global_hotkeys=True,
        cv_key=None,
        gesture_control_enabled=armed,
        left_pose_reset_req=reset,
        left_pose_runtime_armed=runtime,
        left_pose_state=_PoseStub(),
        left_unwind_s=1.0,
        left_swarm_enabled=True,
    )
    assert not quit_
    assert armed[0] is True
    assert q.poll() is None
