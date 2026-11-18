# imprep/modules/app_06_compressor.py
from medimg_annotator.imports import ctk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class CompressorFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Losslessly compresses uncompressed NIfTI (.nii) files into gzip-compressed (.nii.gz) format."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(
            info_card,
            text="• Enforces standardized .nii.gz format.\n• Reduces storage requirements without data loss.\n• Preserves exact headers, orientation, and voxel data.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left"
        ).pack(anchor="w")

    def process_volume(self, data, affine):
        return data, affine
