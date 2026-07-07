"""Morph plane extent helper for Lp visualization (horizontal radius vs openness)."""


def morph_plane_extent_radius(radius: float, open_alpha: float, a: float, b: float) -> float:
    """Compute horizontal morph extent R from radius and openness.

    Linear blend between closed scale ``a`` and open scale ``b``:
    ``R = radius * (a + b * open_alpha)``.

    Args:
        radius: Base superellipsoid radius in millimeters.
        open_alpha: Openness in ``[0, 1]``.
        a: Closed-shape plane radius scale factor.
        b: Open-shape plane radius scale factor.

    Returns:
        Effective horizontal extent radius in millimeters.
    """
    return float(radius) * (float(a) + float(b) * float(open_alpha))
