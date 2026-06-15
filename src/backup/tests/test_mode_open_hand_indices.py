"""Mode vs open hand index assignment and open visibility gate."""

from types import SimpleNamespace

from functions.mode_switch.modes_runtime import ModeState, RightHandState, process_right_open, update_open_state
from functions.dual_cam.mp_hand_utils import resolve_mode_open_hand_indices


def _lm(x: float):
    return SimpleNamespace(x=float(x), y=0.5, z=0.0, visibility=0.9)


def _result(*wrist_x: float):
    hands = [[_lm(x) for _ in range(21)] for x in wrist_x]
    return SimpleNamespace(hand_landmarks=hands, handedness=[])


def test_single_hand_on_right_side_is_mode_only():
    idx_l, idx_r = resolve_mode_open_hand_indices(_result(0.72), swap_mp_hands=False)
    assert idx_l == 0
    assert idx_r is None


def test_single_hand_on_left_side_is_mode_only():
    idx_l, idx_r = resolve_mode_open_hand_indices(_result(0.28), swap_mp_hands=False)
    assert idx_l == 0
    assert idx_r is None


def test_single_hand_always_mode_only():
    idx_l, idx_r = resolve_mode_open_hand_indices(_result(0.72), swap_mp_hands=False)
    assert idx_l == 0
    assert idx_r is None


def test_two_hands_spatial_assignment():
    idx_l, idx_r = resolve_mode_open_hand_indices(_result(0.25, 0.75), swap_mp_hands=False)
    assert idx_r == 0
    assert idx_l == 1


def test_swap_flips_spatial_roles():
    idx_l, idx_r = resolve_mode_open_hand_indices(_result(0.25, 0.75), swap_mp_hands=True)
    assert idx_l == 0
    assert idx_r == 1


def test_open_held_when_right_visibility_low():
    st = RightHandState()
    st.last_open_out = 0.2
    st.last_right_pts = [(0.0, 0.0, 0.0)] * 21
    low_vis = SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=0.2, y=0.5, z=0.0, visibility=0.1)] * 21],
        handedness=[],
    )

    def _analyze(_pts):
        return {"morph_alpha": 1.0}

    _, open_out = process_right_open(
        [[(0.0, 0.0, 0.0)] * 21],
        0,
        st,
        mp_result=low_vis,
        open_vis_min=0.5,
    )
    assert open_out == 0.2


def test_update_open_state_holds_when_topology_none():
    st = RightHandState()
    st.last_open_out = 0.35
    out = update_open_state(
        [(0.0, 0.0, 0.0)] * 21,
        right_state=st,
        analyze_topology_fn=lambda _p: None,
        open_smooth=0.2,
        plane_snap_on=0.9,
        plane_snap_off=0.7,
        sphere_snap_on=0.2,
        sphere_snap_off=0.35,
    )
    assert out == 0.35
