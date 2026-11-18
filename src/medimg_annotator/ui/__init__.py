"""
Graphical user interface components for MedImgAnnotator.
"""

from medimg_annotator.ui.editor_canvas import EditorCanvas
from medimg_annotator.ui.main_window import NiftiMaskEditor
from medimg_annotator.ui.preview_canvas import PreviewCanvas
from medimg_annotator.ui.universal_input_dialog import UniversalInputDialog

__all__ = [
    "EditorCanvas",
    "PreviewCanvas",
    "UniversalInputDialog",
    "NiftiMaskEditor",
]
