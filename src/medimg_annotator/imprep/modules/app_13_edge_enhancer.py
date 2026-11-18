# imprep/modules/app_13_edge_enhancer.py
from medimg_annotator.imports import ctk, skimage_feature as feature, np
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import FONTS

class EdgeEnhancerFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="High-frequency feature & structural boundary enhancement via multi-scale Canny edge filtration."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        # Alpha (Blend Weight)
        self.lbl_a = ctk.CTkLabel(info_card, text="Edge Blend Weight (Alpha): 1.0", font=FONTS["small"])
        self.lbl_a.pack(anchor="w", pady=(2, 0))
        self.slider_a = ctk.CTkSlider(info_card, from_=0.0, to=5.0, command=self._on_a)
        self.slider_a.set(1.0)
        self.slider_a.pack(fill="x", pady=(0, 6))

        # Sigma (Filter Kernel)
        self.lbl_s = ctk.CTkLabel(info_card, text="Gaussian Sigma: 1.0", font=FONTS["small"])
        self.lbl_s.pack(anchor="w", pady=(2, 0))
        self.slider_s = ctk.CTkSlider(info_card, from_=0.1, to=5.0, command=self._on_s)
        self.slider_s.set(1.0)
        self.slider_s.pack(fill="x", pady=(0, 6))

    def _on_a(self, val):
        self.lbl_a.configure(text=f"Edge Blend Weight (Alpha): {val:.2f}")
        self.apply_realtime_change()

    def _on_s(self, val):
        self.lbl_s.configure(text=f"Gaussian Sigma: {val:.2f}")
        self.apply_realtime_change()

    def process_slice(self, slice_data):
        mn, mx = slice_data.min(), slice_data.max()
        d = (slice_data - mn) / (mx - mn) if mx != mn else slice_data
        sig = self.slider_s.get()
        e = feature.canny(d, sigma=sig)
        result = d + (self.slider_a.get() * e)
        return (result * (mx - mn) + mn)

    def process_volume(self, data, affine):
        mn, mx = data.min(), data.max()
        d = (data - mn) / (mx - mn) if mx != mn else data
        e = np.zeros_like(d)

        sig = self.slider_s.get()
        for z in range(data.shape[2]):
            e[:, :, z] = feature.canny(d[:, :, z], sigma=sig)

        result = d + (self.slider_a.get() * e)
        return (result * (mx - mn) + mn), affine
