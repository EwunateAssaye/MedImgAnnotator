"""
Centralized, lazy-cached import hub for MedImgAnnotator.

All package modules resolve standard-library, GUI, scientific, and optional
third-party dependencies through this single module to eliminate redundant
import boilerplate while deferring heavy library initialization until first use.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import csv
    import ctypes
    import datetime
    from datetime import datetime as DateTime
    import difflib
    import functools
    import getpass
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import string
    import subprocess
    import sys
    import tarfile
    import tempfile
    import threading
    import time
    import webbrowser
    import zipfile
    import zlib
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
    import customtkinter as ctk
    import cv2
    import dicom2nifti
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.path import Path as MplPath
    import nibabel as nib
    import nibabel.processing as nibabel_processing
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageTk
    import pydicom
    import scipy.ndimage as scipy_ndimage
    from scipy.ndimage import label as ndimage_label, rotate as ndimage_rotate
    import SimpleITK as sitk
    from skimage import color as skimage_color, feature as skimage_feature, io as skimage_io
    from skimage.filters import threshold_local, threshold_otsu
    import tifffile

_IMPORT_TABLE = {
    # Standard library
    "csv": ("csv", None),
    "ctypes": ("ctypes", None),
    "datetime": ("datetime", None),
    "DateTime": ("datetime", "datetime"),
    "difflib": ("difflib", None),
    "functools": ("functools", None),
    "getpass": ("getpass", None),
    "json": ("json", None),
    "os": ("os", None),
    "Path": ("pathlib", "Path"),
    "re": ("re", None),
    "shutil": ("shutil", None),
    "string": ("string", None),
    "subprocess": ("subprocess", None),
    "sys": ("sys", None),
    "tarfile": ("tarfile", None),
    "tempfile": ("tempfile", None),
    "threading": ("threading", None),
    "time": ("time", None),
    "webbrowser": ("webbrowser", None),
    "zipfile": ("zipfile", None),
    "zlib": ("zlib", None),
    # GUI frameworks
    "ctk": ("customtkinter", None),
    "tk": ("tkinter", None),
    "filedialog": ("tkinter.filedialog", None),
    "messagebox": ("tkinter.messagebox", None),
    "simpledialog": ("tkinter.simpledialog", None),
    "ttk": ("tkinter.ttk", None),
    # Scientific, imaging, and plotting libraries
    "cv2": ("cv2", None),
    "dicom2nifti": ("dicom2nifti", None),
    "Figure": ("matplotlib.figure", "Figure"),
    "FigureCanvasTkAgg": ("matplotlib.backends.backend_tkagg", "FigureCanvasTkAgg"),
    "MplPath": ("matplotlib.path", "Path"),
    "nib": ("nibabel", None),
    "nibabel_processing": ("nibabel.processing", None),
    "np": ("numpy", None),
    "pd": ("pandas", None),
    "Image": ("PIL.Image", None),
    "ImageTk": ("PIL.ImageTk", None),
    "pydicom": ("pydicom", None),
    "scipy_ndimage": ("scipy.ndimage", None),
    "ndimage_label": ("scipy.ndimage", "label"),
    "ndimage_rotate": ("scipy.ndimage", "rotate"),
    "sitk": ("SimpleITK", None),
    "skimage_color": ("skimage.color", None),
    "skimage_feature": ("skimage.feature", None),
    "skimage_io": ("skimage.io", None),
    "threshold_local": ("skimage.filters", "threshold_local"),
    "threshold_otsu": ("skimage.filters", "threshold_otsu"),
    "tifffile": ("tifffile", None),
}

_OPTIONAL_NAMES = frozenset({"nib", "sitk", "pydicom", "tifffile"})

__all__ = tuple(sorted(_IMPORT_TABLE))


def __getattr__(name: str):
    """Lazily import, optionally fall back to None, and cache requested symbols."""
    spec = _IMPORT_TABLE.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = spec
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name) if attr_name else module
    except ImportError:
        if name in _OPTIONAL_NAMES:
            value = None
        else:
            raise

    globals()[name] = value
    return value


def __dir__():
    """Expose all lazily available symbols for interactive inspection."""
    return sorted(set(globals()) | set(__all__))
