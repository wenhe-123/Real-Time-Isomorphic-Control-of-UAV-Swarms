"""Morph plane extent helper for Lp visualization (horizontal radius vs openness)."""


def morph_plane_extent_radius(radius: float, open_alpha: float, a: float, b: float) -> float:
    """Horizontal extent R for morph geometry and plane samples."""
    return float(radius) * (float(a) + float(b) * float(open_alpha))
