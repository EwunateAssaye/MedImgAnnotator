"""
Application bootstrap and runtime entry point for MedImgAnnotator.
"""

from medimg_annotator.imports import ctk, ctypes, os, sys

from medimg_annotator.ui.main_window import NiftiMaskEditor

def resolve_asset_path(relative_path):
    """Resolve an asset path relative to PyInstaller _MEIPASS, repo root, or installed package."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.extend([
            os.path.join(meipass, "assets", relative_path),
            os.path.join(meipass, relative_path),
        ])
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.abspath(os.path.join(pkg_dir, "..", "..", "assets", relative_path)),
        os.path.abspath(os.path.join(pkg_dir, "..", "..", relative_path)),
        os.path.abspath(os.path.join(pkg_dir, "assets", relative_path)),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.abspath(relative_path)

def run_app():
    """Initialize dark theme appearance, configure taskbar AppUserModelID, and launch the GUI."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("medimg.annotator.2.0")
        except Exception:
            pass

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    app = NiftiMaskEditor()
    icon_path = resolve_asset_path("icon.ico")
    if os.path.exists(icon_path):
        try:
            app.iconbitmap(icon_path)
        except Exception:
            pass

    app.mainloop()
