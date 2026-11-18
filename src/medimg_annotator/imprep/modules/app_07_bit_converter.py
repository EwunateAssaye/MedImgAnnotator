# imprep/modules/app_07_bit_converter.py
from medimg_annotator.imports import ctk, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import FONTS

class BitConverterFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Converts volume voxel bit-depth and data types (e.g. 8-bit uint8, 16-bit int16, 32-bit float32)."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Target Bit-Depth / Data Type:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.bit_mode = tk.StringVar(value="uint8")

        ctk.CTkRadioButton(info_card, text="8-bit unsigned integer (uint8: 0 - 255)", variable=self.bit_mode, value="uint8", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="16-bit signed integer (int16)", variable=self.bit_mode, value="int16", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(info_card, text="32-bit floating point (float32)", variable=self.bit_mode, value="float32", font=FONTS["body"], command=self.apply_realtime_change).pack(anchor="w", pady=2)

    def process_slice(self, slice_data):
        mode = self.bit_mode.get()
        mn, mx = slice_data.min(), slice_data.max()

        if mode == "uint8":
            if mx != mn:
                return ((slice_data - mn) / (mx - mn) * 255.0).astype(np.uint8)
            return np.zeros_like(slice_data, dtype=np.uint8)
        elif mode == "int16":
            return slice_data.astype(np.int16)
        elif mode == "float32":
            return slice_data.astype(np.float32)

        return slice_data

    def process_volume(self, data, affine):
        mode = self.bit_mode.get()
        mn, mx = data.min(), data.max()

        if mode == "uint8":
            if mx != mn:
                res = ((data - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                res = np.zeros_like(data, dtype=np.uint8)
            return res, affine
        elif mode == "int16":
            return data.astype(np.int16), affine
        elif mode == "float32":
            return data.astype(np.float32), affine

        return data, affine
