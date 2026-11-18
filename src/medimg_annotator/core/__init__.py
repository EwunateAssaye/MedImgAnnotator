"""
Core engines and utilities for MedImgAnnotator:
I/O, geometry, configuration, history, segmentation, interpolation, and plugins.
"""

from medimg_annotator.core.config import (
    DARK_QSS,
    DEFAULT_LABEL_COLORS,
    LABEL_COLOR_MAP,
    Settings,
    default_color_name_for_label,
)

__all__ = [
    "Settings",
    "LABEL_COLOR_MAP",
    "DEFAULT_LABEL_COLORS",
    "DARK_QSS",
    "default_color_name_for_label",
]
