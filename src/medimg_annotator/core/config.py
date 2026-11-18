"""
Application configuration constants and atomic user settings persistence.
"""

from medimg_annotator.imports import json, os, shutil

# CustomTkinter native appearance is used; kept for backwards compatibility.
DARK_QSS = ""

# 40 perceptually separated high-contrast segmentation colors (min pairwise CIEDE2000 dE >= 15.2)
LABEL_COLOR_MAP = {
    "Red": (255, 0, 0),            # #FF0000
    "Green": (0, 255, 0),          # #00FF00
    "Blue": (0, 0, 255),           # #0000FF
    "Yellow": (255, 255, 0),       # #FFFF00
    "Cyan": (0, 255, 255),         # #00FFFF
    "Magenta": (255, 0, 255),      # #FF00FF
    "Orange": (255, 128, 0),       # #FF8000
    "Forest Green": (0, 120, 0),   # #007800
    "Azure": (0, 150, 255),        # #0096FF
    "Berry Rose": (210, 0, 105),   # #D20069
    "Gold": (210, 165, 0),         # #D2A500
    "Teal": (0, 150, 120),         # #009678
    "Periwinkle": (135, 120, 255), # #8778FF
    "Bronze": (165, 105, 0),       # #A56900
    "Coral": (255, 105, 120),      # #FF6978
    "Orchid": (240, 165, 255),     # #F0A5FF
    "Olive Gold": (135, 150, 15),  # #87960F
    "Purple": (165, 45, 195),      # #A52DC3
    "Emerald": (0, 180, 15),       # #00B40F
    "Mint": (15, 255, 180),        # #0FFFB4
    "Peach": (255, 165, 135),      # #FFA587
    "Cobalt": (45, 90, 210),       # #2D5AD2
    "Lime": (165, 210, 0),         # #A5D200
    "Burgundy": (105, 30, 45),     # #691E2D
    "Pine": (0, 90, 75),           # #005A4B
    "Brown": (90, 60, 15),         # #5A3C0F
    "Cerulean": (0, 180, 210),     # #00B4D2
    "Ice Blue": (180, 195, 255),   # #B4C3FF
    "Terracotta": (165, 105, 90),  # #A5695A
    "Rust": (150, 30, 0),          # #961E00
    "Cream": (240, 210, 165),      # #F0D2A5
    "Plum": (90, 30, 105),         # #5A1E69
    "Slate Violet": (120, 105, 150),  # #786996
    "Sage": (150, 195, 150),       # #96C396
    "Mauve": (195, 120, 165),      # #C378A5
    "Blush Pink": (255, 180, 195), # #FFB4C3
    "Ocean Blue": (0, 105, 135),   # #006987
    "Navy": (30, 60, 105),         # #1E3C69
    "Dark Olive": (45, 75, 0),     # #2D4B00
    "Moss": (105, 120, 75),        # #69784B
}

DEFAULT_LABEL_COLORS = list(LABEL_COLOR_MAP.keys())


def default_color_name_for_label(label_id):
    """Return the default high-contrast palette color name for a 1-based label ID."""
    idx = (max(1, int(label_id)) - 1) % len(DEFAULT_LABEL_COLORS)
    return DEFAULT_LABEL_COLORS[idx]

class Settings:
    """Atomic JSON-backed settings store persisted under ~/.medimgannotator/settings.json."""

    def __init__(self):
        user_dir = os.path.join(os.path.expanduser("~"), ".medimgannotator")
        try:
            os.makedirs(user_dir, exist_ok=True)
            self.file = os.path.join(user_dir, "settings.json")
        except Exception:
            self.file = "settings.json"

        if os.path.exists("settings.json") and self.file != "settings.json" and not os.path.exists(self.file):
            try:
                shutil.copy2("settings.json", self.file)
            except Exception:
                pass

        try:
            with open(self.file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def value(self, key, default=""):
        """Return a persisted setting value or default if absent."""
        return self.data.get(key, default)

    def setValue(self, key, val):
        """Persist a setting key atomically using a temporary file swap."""
        self.data[key] = val
        tmp_file = self.file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_file, self.file)
        except Exception:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
