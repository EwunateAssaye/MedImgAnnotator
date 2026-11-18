# imprep/modules/app_10_thresholding.py
from medimg_annotator.imports import ctk, np, threshold_local, threshold_otsu, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class ThresholdingFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Segmentation thresholding: Simple Range, Global Otsu, or Local Adaptive binarization."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Thresholding Strategy:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.thresh_mode = tk.StringVar(value="simple")

        ctk.CTkRadioButton(info_card, text="Range Threshold (Min - Max)", variable=self.thresh_mode, value="simple", font=FONTS["body"], command=self._update_opts).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="Automatic Otsu Threshold", variable=self.thresh_mode, value="otsu", font=FONTS["body"], command=self._update_opts).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="Adaptive Local Threshold", variable=self.thresh_mode, value="adaptive", font=FONTS["body"], command=self._update_opts).pack(anchor="w", pady=2)

        # Range Options Frame
        self.frame_simple = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        self.frame_simple.pack(fill="x", padx=8, pady=6)

        row_vals = ctk.CTkFrame(self.frame_simple, fg_color="transparent")
        row_vals.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(row_vals, text="Min:", font=FONTS["small"]).pack(side="left", padx=(0, 4))
        self.e_min = ctk.CTkEntry(row_vals, width=70, height=28)
        self.e_min.insert(0, "100")
        self.e_min.pack(side="left", padx=(0, 10))
        self.bind_realtime_entry(self.e_min)

        ctk.CTkLabel(row_vals, text="Max:", font=FONTS["small"]).pack(side="left", padx=(0, 4))
        self.e_max = ctk.CTkEntry(row_vals, width=70, height=28)
        self.e_max.insert(0, "2000")
        self.e_max.pack(side="left")
        self.bind_realtime_entry(self.e_max)

        self.bin_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.frame_simple, text="Output as Binary Mask (0 or 1)", variable=self.bin_var, font=FONTS["small"], command=self.apply_realtime_change).pack(anchor="w", padx=8, pady=4)

        ctk.CTkButton(self.frame_simple, text="Apply Range", height=28, fg_color=COLORS["accent_secondary"], hover_color=COLORS["accent_hover"], command=self.apply_realtime_change).pack(fill="x", padx=8, pady=(2, 6))

        # Adaptive Options Frame
        self.frame_adaptive = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        row_ad = ctk.CTkFrame(self.frame_adaptive, fg_color="transparent")
        row_ad.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(row_ad, text="Block Size (odd):", font=FONTS["small"]).pack(side="left", padx=(0, 6))
        self.e_blk = ctk.CTkEntry(row_ad, width=60, height=28)
        self.e_blk.insert(0, "35")
        self.e_blk.pack(side="left")
        self.bind_realtime_entry(self.e_blk)
        ctk.CTkButton(row_ad, text="Apply Adaptive", height=28, fg_color=COLORS["accent_secondary"], hover_color=COLORS["accent_hover"], command=self.apply_realtime_change).pack(side="right")

        self._update_opts()

    def _update_opts(self):
        m = self.thresh_mode.get()
        if m == "simple":
            self.frame_adaptive.pack_forget()
            self.frame_simple.pack(fill="x", padx=8, pady=6)
        elif m == "adaptive":
            self.frame_simple.pack_forget()
            self.frame_adaptive.pack(fill="x", padx=8, pady=6)
        else:
            self.frame_simple.pack_forget()
            self.frame_adaptive.pack_forget()

        if self.vol_in is not None:
            self.apply_realtime_change()

    def process_slice(self, slice_data):
        m = self.thresh_mode.get()
        if m == "simple":
            try:
                mn = float(self.e_min.get())
                mx = float(self.e_max.get())
                mk = (slice_data >= mn) & (slice_data <= mx)
                return (mk.astype(np.uint8) if self.bin_var.get() else np.where(mk, slice_data, 0))
            except Exception:
                return slice_data
        elif m == "otsu":
            try:
                thresh = threshold_otsu(slice_data)
                return (slice_data > thresh).astype(np.uint8)
            except Exception:
                return slice_data
        elif m == "adaptive":
            try:
                bs = int(self.e_blk.get())
                if bs % 2 == 0:
                    bs += 1
                bs = max(3, bs)
                mn_val, mx_val = slice_data.min(), slice_data.max()
                d = (slice_data - mn_val) / (mx_val - mn_val) if mx_val != mn_val else slice_data
                return (d > threshold_local(d, bs, offset=0.1)).astype(np.uint8)
            except Exception:
                return slice_data

        return slice_data

    def process_volume(self, data, affine):
        m = self.thresh_mode.get()
        if m == "simple":
            try:
                mn = float(self.e_min.get())
                mx = float(self.e_max.get())
                mk = (data >= mn) & (data <= mx)
                return (mk.astype(np.uint8) if self.bin_var.get() else np.where(mk, data, 0)), affine
            except:
                return data, affine
        elif m == "otsu":
            try:
                thresh = threshold_otsu(data)
                return (data > thresh).astype(np.uint8), affine
            except:
                return data, affine
        elif m == "adaptive":
            try:
                bs = int(self.e_blk.get())
                if bs % 2 == 0:
                    bs += 1
                out = np.zeros_like(data, dtype=np.uint8)
                d = (data - data.min()) / (data.max() - data.min()) if data.max() != data.min() else data
                for z in range(data.shape[2]):
                    out[:, :, z] = (d[:, :, z] > threshold_local(d[:, :, z], bs, offset=0.1)).astype(np.uint8)
                return out, affine
            except:
                return data, affine

        return data, affine
