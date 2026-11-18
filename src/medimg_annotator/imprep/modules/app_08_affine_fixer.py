# imprep/modules/app_08_affine_fixer.py
from medimg_annotator.imports import ctk, nib, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import FONTS

class AffineFixerFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Reorients volumes to canonical orthogonal coordinates (RAS) and removes oblique tilt shear angles."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Alignment Method:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.align_method = tk.StringVar(value="canonical")

        ctk.CTkRadioButton(
            info_card,
            text="Reorient to Canonical RAS (Standard)",
            variable=self.align_method,
            value="canonical",
            font=FONTS["body"],
            command=self.apply_realtime_change
        ).pack(anchor="w", pady=2)

        ctk.CTkRadioButton(
            info_card,
            text="Force Pure Diagonal Orthogonal Affine",
            variable=self.align_method,
            value="orthogonal",
            font=FONTS["body"],
            command=self.apply_realtime_change
        ).pack(anchor="w", pady=2)

    def process_volume(self, data, affine):
        if self.align_method.get() == "canonical":
            try:
                img = nib.Nifti1Image(data, affine)
                canonical_img = nib.as_closest_canonical(img)
                return canonical_img.get_fdata(dtype=np.float32), canonical_img.affine
            except:
                pass

        # Diagonal Orthogonal Fallback
        sp = np.linalg.norm(affine[:3, :3], axis=0)
        origin = affine[:3, 3]
        new_aff = np.eye(4)
        for i in range(3):
            new_aff[i, i] = sp[i] if affine[i, i] >= 0 else -sp[i]
        new_aff[:3, 3] = origin
        return data, new_aff
