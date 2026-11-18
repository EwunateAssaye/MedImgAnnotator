# imprep/modules/app_11_resampler.py
from medimg_annotator.imports import ctk, nib, nibabel_processing, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class ResampleFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Resamples volumes to standardized voxel spacing (isotropic or specified dimensions in mm)."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Target Voxel Spacing (mm):", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))

        row_sp = ctk.CTkFrame(info_card, fg_color="transparent")
        row_sp.pack(fill="x", pady=2)

        ctk.CTkLabel(row_sp, text="dx:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        self.vox_x = ctk.CTkEntry(row_sp, width=55, height=28)
        self.vox_x.insert(0, "1.0")
        self.vox_x.pack(side="left", padx=(0, 8))
        self.bind_realtime_entry(self.vox_x)

        ctk.CTkLabel(row_sp, text="dy:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        self.vox_y = ctk.CTkEntry(row_sp, width=55, height=28)
        self.vox_y.insert(0, "1.0")
        self.vox_y.pack(side="left", padx=(0, 8))
        self.bind_realtime_entry(self.vox_y)

        ctk.CTkLabel(row_sp, text="dz:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        self.vox_z = ctk.CTkEntry(row_sp, width=55, height=28)
        self.vox_z.insert(0, "1.0")
        self.vox_z.pack(side="left")
        self.bind_realtime_entry(self.vox_z)

        ctk.CTkLabel(info_card, text="Interpolation Order:", font=FONTS["sub_header"]).pack(anchor="w", pady=(8, 2))
        self.order_var = tk.StringVar(value="1")
        ctk.CTkRadioButton(info_card, text="Trilinear (order=1, Recommended for Images)", variable=self.order_var, value="1", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="Nearest Neighbor (order=0, Recommended for Masks)", variable=self.order_var, value="0", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)

        ctk.CTkButton(info_card, text="Compute Preview", height=30, fg_color=COLORS["accent_secondary"], hover_color=COLORS["accent_hover"], command=self.apply_realtime_change).pack(fill="x", pady=(8, 4))

    def process_volume(self, data, affine):
        try:
            dx = float(self.vox_x.get())
            dy = float(self.vox_y.get())
            dz = float(self.vox_z.get())
            order = int(self.order_var.get())

            img = nib.Nifti1Image(data, affine)
            res = nibabel_processing.resample_to_output(img, voxel_sizes=(dx, dy, dz), order=order)
            return res.get_fdata(dtype=np.float32), res.affine
        except:
            return data, affine
