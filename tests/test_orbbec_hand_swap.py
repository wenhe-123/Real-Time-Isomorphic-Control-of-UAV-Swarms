"""Orbbec MediaPipe L/R swap policy."""

from shared.mp_hand_utils import orbbec_resolve_swap_mp_hands


def test_webcam_never_swaps():
    assert not orbbec_resolve_swap_mp_hands(hand_swap="on", flip_horizontal=True, use_orbbec=False)


def test_auto_swaps_when_flip():
    assert orbbec_resolve_swap_mp_hands(hand_swap="auto", flip_horizontal=True, use_orbbec=True)
    assert not orbbec_resolve_swap_mp_hands(hand_swap="auto", flip_horizontal=False, use_orbbec=True)


def test_on_off():
    assert orbbec_resolve_swap_mp_hands(hand_swap="on", flip_horizontal=False, use_orbbec=True)
    assert not orbbec_resolve_swap_mp_hands(hand_swap="off", flip_horizontal=True, use_orbbec=True)


def test_invalid_hand_swap_treated_as_auto():
    assert orbbec_resolve_swap_mp_hands(hand_swap="bogus", flip_horizontal=True, use_orbbec=True)
