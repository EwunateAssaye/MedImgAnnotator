"""
Filename utilities and navigation sidebar widget for the ImPrep Preprocessing Suite.
"""

from medimg_annotator.imports import ctk

from medimg_annotator.imprep.config import COLORS, FONTS

def get_out_name(fname):
    """Ensure consistent `.nii.gz` output filename extension."""
    base = fname
    if base.lower().endswith(".nii.gz"):
        base = base[:-7]
    elif base.lower().endswith(".nii"):
        base = base[:-4]
    return f"{base}.nii.gz"

class ImPrepSidebar(ctk.CTkScrollableFrame):
    """Vertical sidebar for navigating across the dashboard and 16 preprocessing tools."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg_sidebar"], width=230, corner_radius=0, **kwargs)
        self.controller = controller
        self.buttons = {}
        self.active_tool = None

    def add_nav_button(self, label, target_frame=None, command=None, key=None):
        """Register and pack a navigation button in the sidebar."""
        if key is None:
            key = label

        def cmd():
            self.set_active(key)
            if command:
                command()
            elif target_frame:
                self.controller.show(target_frame)

        btn = ctk.CTkButton(
            self,
            text=f"  {label}",
            anchor="w",
            font=FONTS["nav"],
            fg_color="transparent",
            text_color=COLORS["text_secondary"],
            hover_color=COLORS["bg_sidebar_hover"],
            corner_radius=6,
            height=32,
            command=cmd,
        )
        btn.pack(fill="x", padx=6, pady=2)
        self.buttons[key] = btn
        return btn

    def set_active(self, key):
        """Highlight the selected navigation button."""
        self.active_tool = key
        for k, b in self.buttons.items():
            if k == key:
                b.configure(
                    fg_color=COLORS["bg_sidebar_active"],
                    text_color=COLORS["text_primary"],
                    font=FONTS["body_bold"],
                )
            else:
                b.configure(
                    fg_color="transparent",
                    text_color=COLORS["text_secondary"],
                    font=FONTS["nav"],
                )
