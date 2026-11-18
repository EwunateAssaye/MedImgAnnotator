"""
Entry point launcher for MedImgAnnotator.
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from medimg_annotator.app import run_app


if __name__ == "__main__":
    run_app()
