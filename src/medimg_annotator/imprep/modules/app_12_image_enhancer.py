# imprep/modules/app_12_image_enhancer.py
from medimg_annotator.imports import ctk, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import FONTS

class ImageEnhancerFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Non-linear contrast enhancement: Power-law Gamma correction and Logarithmic dynamic range expansion."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Enhancement Function:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.m = tk.StringVar(value="gamma")

        ctk.CTkRadioButton(info_card, text="Power-Law (Gamma)", variable=self.m, value="gamma", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="Logarithmic Expansion", variable=self.m, value="log", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)

        # Coefficient (c)
        self.lbl_c = ctk.CTkLabel(info_card, text="Gain Coefficient (c): 1.0", font=FONTS["small"])
        self.lbl_c.pack(anchor="w", pady=(6, 0))
        self.slider_c = ctk.CTkSlider(info_card, from_=0.1, to=5.0, command=self._on_c)
        self.slider_c.set(1.0)
        self.slider_c.pack(fill="x", pady=(0, 6))

        # Power / Factor (p)
        self.lbl_p = ctk.CTkLabel(info_card, text="Exponent / Factor (p): 1.0", font=FONTS["small"])
        self.lbl_p.pack(anchor="w", pady=(2, 0))
        self.slider_p = ctk.CTkSlider(info_card, from_=0.1, to=5.0, command=self._on_p)
        self.slider_p.set(1.0)
        self.slider_p.pack(fill="x", pady=(0, 6))

    def _on_c(self, val):
        self.lbl_c.configure(text=f"Gain Coefficient (c): {val:.2f}")
        self.apply_realtime_change()

    def _on_p(self, val):
        self.lbl_p.configure(text=f"Exponent / Factor (p): {val:.2f}")
        self.apply_realtime_change()

    def process_slice(self, slice_data):
        mn, mx = slice_data.min(), slice_data.max()
        n = (slice_data - mn) / (mx - mn) if mx != mn else slice_data

        c_val = self.slider_c.get()
        p_val = self.slider_p.get()

        if self.m.get() == "gamma":
            e = c_val * np.power(np.clip(n, 0, 1), p_val)
        else:
            e = c_val * (np.log1p(p_val * np.clip(n, 0, 1)) / np.log1p(p_val))

        return (e * (mx - mn) + mn)

    def process_volume(self, data, affine):
        mn, mx = data.min(), data.max()
        n = (data - mn) / (mx - mn) if mx != mn else data

        c_val = self.slider_c.get()
        p_val = self.slider_p.get()

        if self.m.get() == "gamma":
            e = c_val * np.power(np.clip(n, 0, 1), p_val)
        else:
            e = c_val * (np.log1p(p_val * np.clip(n, 0, 1)) / np.log1p(p_val))

        return (e * (mx - mn) + mn), affine
