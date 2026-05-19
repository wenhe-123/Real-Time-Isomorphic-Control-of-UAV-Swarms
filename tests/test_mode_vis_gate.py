"""Morph mode updates gated on MP visibility."""

from shared.modes_runtime import ModeState, update_mode_state


def _classify_always_5(_pts):
    return 5, 3, {}


def test_mode_held_when_visibility_low():
    st = ModeState()
    st.morph_mode = 2
    st.last_mode_raw = 2
    st.mode_raw_prev = 2
    st.mode_stable_frames = 99
    raw, _ = update_mode_state(
        [(0.0, 0.0, 0.0)] * 21,
        mode_state=st,
        classify_mode_fn=_classify_always_5,
        debounce_frames=1,
        mode_smooth=0.2,
        mode_vis_min=0.5,
        hand_visibility_min=0.2,
    )
    assert raw == 2
    assert st.morph_mode == 2


def test_mode_updates_when_visibility_high():
    st = ModeState()
    st.morph_mode = 1
    st.last_mode_raw = 1
    raw, _ = update_mode_state(
        [(0.0, 0.0, 0.0)] * 21,
        mode_state=st,
        classify_mode_fn=_classify_always_5,
        debounce_frames=1,
        mode_smooth=0.2,
        mode_vis_min=0.5,
        hand_visibility_min=0.8,
    )
    assert raw == 5
    assert st.morph_mode == 5
