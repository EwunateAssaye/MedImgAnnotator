# imprep/modules/app_09_intensity_processor.py
from medimg_annotator.imports import ctk, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class IntensityProcessorFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Windowing, HU clipping, band-pass intensity filtering, and dynamic range rescaling."
        )

    def add_custom_widgets(self, parent):
        # 1. Clipping
        grp_clip = ctk.CTkFrame(parent, fg_color="transparent")
        grp_clip.pack(fill="x", padx=8, pady=4)

        self.use_clip = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(grp_clip, text="Enable Value Clipping (HU / Intensity)", variable=self.use_clip, font=FONTS["sub_header"], command=self.apply_realtime_change).pack(anchor="w", pady=(0, 4))

        row_clip = ctk.CTkFrame(grp_clip, fg_color="transparent")
        row_clip.pack(fill="x", pady=2)

        ctk.CTkLabel(row_clip, text="Min:", font=FONTS["small"]).pack(side="left", padx=(0, 4))
        self.hu_min = ctk.CTkEntry(row_clip, width=70, height=28)
        self.hu_min.insert(0, "-1000")
        self.hu_min.pack(side="left", padx=(0, 10))
        self.bind_realtime_entry(self.hu_min)

        ctk.CTkLabel(row_clip, text="Max:", font=FONTS["small"]).pack(side="left", padx=(0, 4))
        self.hu_max = ctk.CTkEntry(row_clip, width=70, height=28)
        self.hu_max.insert(0, "1000")
        self.hu_max.pack(side="left", padx=(0, 10))
        self.bind_realtime_entry(self.hu_max)

        ctk.CTkButton(row_clip, text="Apply", width=60, height=28, fg_color=COLORS["accent_secondary"], hover_color=COLORS["accent_hover"], command=self.apply_realtime_change).pack(side="right")

        # 2. Band Pass
        grp_band = ctk.CTkFrame(parent, fg_color="transparent")
        grp_band.pack(fill="x", padx=8, pady=8)

        self.use_band = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(grp_band, text="Enable Band-Pass Filter", variable=self.use_band, font=FONTS["sub_header"], command=self.apply_realtime_change).pack(anchor="w", pady=(0, 4))

        self.lbl_band_low = ctk.CTkLabel(grp_band, text="Lower Cutoff: -1000", font=FONTS["small"])
        self.lbl_band_low.pack(anchor="w")
        self.band_low = ctk.CTkSlider(grp_band, from_=-1000, to=3000, command=self._on_slider_low)
        self.band_low.set(-1000)
        self.band_low.pack(fill="x", pady=(0, 6))

        self.lbl_band_high = ctk.CTkLabel(grp_band, text="Upper Cutoff: 3000", font=FONTS["small"])
        self.lbl_band_high.pack(anchor="w")
        self.band_high = ctk.CTkSlider(grp_band, from_=-1000, to=3000, command=self._on_slider_high)
        self.band_high.set(3000)
        self.band_high.pack(fill="x", pady=(0, 6))

        # 3. Rescaling Mode
        grp_scale = ctk.CTkFrame(parent, fg_color="transparent")
        grp_scale.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(grp_scale, text="Output Range Rescaling:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))

        self.scale_mode = tk.StringVar(value="none")
        ctk.CTkRadioButton(grp_scale, text="Keep Original Range", variable=self.scale_mode, value="none", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(grp_scale, text="Normalize to [0.0, 1.0]", variable=self.scale_mode, value="01", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(grp_scale, text="Rescale to 8-bit [0, 255]", variable=self.scale_mode, value="255", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)

    def _on_slider_low(self, val):
        self.lbl_band_low.configure(text=f"Lower Cutoff: {int(val)}")
        self.use_band.set(True)
        self.apply_realtime_change()

    def _on_slider_high(self, val):
        self.lbl_band_high.configure(text=f"Upper Cutoff: {int(val)}")
        self.use_band.set(True)
        self.apply_realtime_change()

    def _sync_band_sliders(self):
        if self.vol_in is not None:
            mn, mx = float(self.vol_in.min()), float(self.vol_in.max())
            buffer = (mx - mn) * 0.05 if mx != mn else 10.0
            s_min, s_max = mn - buffer, mx + buffer
            if s_max <= s_min:
                s_max = s_min + 1.0
            self.band_low.configure(from_=s_min, to=s_max)
            self.band_high.configure(from_=s_min, to=s_max)
            self.band_low.set(mn)
            self.band_high.set(mx)
            self.lbl_band_low.configure(text=f"Lower Cutoff: {int(mn)}")
            self.lbl_band_high.configure(text=f"Upper Cutoff: {int(mx)}")

    def init_slider(self):
        super().init_slider()
        self._sync_band_sliders()

    def load_preview(self, path):
        super().load_preview(path)
        self._sync_band_sliders()

    def process_slice(self, slice_data):
        d = np.copy(slice_data).astype(np.float32)
        if self.use_clip.get():
            try:
                mn_val = float(self.hu_min.get())
                mx_val = float(self.hu_max.get())
                d = np.clip(d, mn_val, mx_val)
            except Exception:
                pass

        if self.use_band.get():
            low = self.band_low.get()
            high = self.band_high.get()
            mask = (d >= low) & (d <= high)
            d = np.where(mask, d, 0)

        mode = self.scale_mode.get()
        if mode != "none":
            d_min, d_max = d.min(), d.max()
            if d_max > d_min:
                d = (d - d_min) / (d_max - d_min)
                if mode == "255":
                    d = (d * 255.0).astype(np.uint8)

        return d

    def process_volume(self, data, affine):
        d = np.copy(data).astype(np.float32)
        if self.use_clip.get():
            try:
                mn_val = float(self.hu_min.get())
                mx_val = float(self.hu_max.get())
                d = np.clip(d, mn_val, mx_val)
            except:
                pass

        if self.use_band.get():
            low = self.band_low.get()
            high = self.band_high.get()
            mask = (d >= low) & (d <= high)
            d = np.where(mask, d, 0)

        mode = self.scale_mode.get()
        if mode != "none":
            d_min, d_max = d.min(), d.max()
            if d_max > d_min:
                d = (d - d_min) / (d_max - d_min)
                if mode == "255":
                    d = (d * 255.0).astype(np.uint8)

        return d, affine
