# imprep/modules/app_14_normalization.py
from medimg_annotator.imports import ctk, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import FONTS

class NormalizationFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Statistical normalization for deep learning inputs: Min-Max [0,1], standard Z-score, or outlier-robust Median/IQR."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Normalization Algorithm:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.norm_type = tk.StringVar(value="minmax")

        ctk.CTkRadioButton(info_card, text="Min-Max Scaling [0.0, 1.0]", variable=self.norm_type, value="minmax", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=3)
        ctk.CTkRadioButton(info_card, text="Z-Score Standardization (Mean=0, Std=1)", variable=self.norm_type, value="zscore", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=3)
        ctk.CTkRadioButton(info_card, text="Robust Median / IQR (Resistant to artifacts)", variable=self.norm_type, value="median", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=3)

    def _get_stats(self, data):
        if getattr(self, '_stats_data_id', None) != id(data):
            d = data.astype(np.float32)
            mn, mx = float(d.min()), float(d.max())
            mean, std = float(d.mean()), float(d.std())
            sample = d.flat[::10] if d.size > 500000 else d
            q75, q25 = np.percentile(sample, [75, 25])
            iqr = float(q75 - q25)
            med = float(np.median(sample))
            self._cached_stats = (mn, mx, mean, std, med, iqr)
            self._stats_data_id = id(data)
        return self._cached_stats

    def process_slice(self, slice_data):
        if self.vol_in is None:
            return slice_data
        d = slice_data.astype(np.float32)
        mode = self.norm_type.get()
        mn, mx, mean, std, med, iqr = self._get_stats(self.vol_in)

        if mode == "minmax":
            if mx != mn:
                return (d - mn) / (mx - mn)
            return d
        elif mode == "zscore":
            if std != 0:
                return (d - mean) / std
            return d
        elif mode == "median":
            if iqr != 0:
                return (d - med) / iqr
            return d

        return d

    def process_volume(self, data, affine):
        d = data.astype(np.float32)
        mode = self.norm_type.get()
        mn, mx, mean, std, med, iqr = self._get_stats(data)

        if mode == "minmax":
            if mx != mn:
                return (d - mn) / (mx - mn), affine
            return d, affine

        elif mode == "zscore":
            if std != 0:
                return (d - mean) / std, affine
            return d, affine

        elif mode == "median":
            if iqr != 0:
                return (d - med) / iqr, affine
            return d, affine

        return data, affine
