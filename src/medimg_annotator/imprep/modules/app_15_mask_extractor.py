# imprep/modules/app_15_mask_extractor.py
from medimg_annotator.imports import ctk, ndimage_label as label, np, tk
from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class MaskExtractorFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Isolate largest 3D anatomical connected components, extract specific label IDs, or merge multiple segmentation labels."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(info_card, text="Extraction Strategy:", font=FONTS["sub_header"]).pack(anchor="w", pady=(0, 4))
        self.op_mode = tk.StringVar(value="largest")

        ctk.CTkRadioButton(
            info_card,
            text="Largest Connected Component\n(Auto-remove small detached islands)",
            variable=self.op_mode,
            value="largest",
            font=FONTS["body"],
            command=self._update_ui_state
        ).pack(anchor="w", pady=3)

        ctk.CTkRadioButton(
            info_card,
            text="Extract Specific Label ID\n(Retain single class, discard others)",
            variable=self.op_mode,
            value="target",
            font=FONTS["body"],
            command=self._update_ui_state
        ).pack(anchor="w", pady=3)

        ctk.CTkRadioButton(
            info_card,
            text="Merge Specific Partitions\n(Combine multiple IDs into unified mask)",
            variable=self.op_mode,
            value="merge",
            font=FONTS["body"],
            command=self._update_ui_state
        ).pack(anchor="w", pady=3)

        # Dynamic Input Frame
        self.frame_input = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        self.lbl_instruction = ctk.CTkLabel(self.frame_input, text="", font=FONTS["small"])
        self.lbl_instruction.pack(anchor="w", padx=10, pady=(6, 2))

        self.ent_labels = ctk.CTkEntry(self.frame_input, font=FONTS["body"], height=30)
        self.ent_labels.pack(fill="x", padx=10, pady=(0, 10))
        self.bind_realtime_entry(self.ent_labels)

        self._update_ui_state()

    def _update_ui_state(self):
        mode = self.op_mode.get()
        if mode == "largest":
            self.frame_input.pack_forget()
        else:
            self.frame_input.pack(fill="x", padx=8, pady=6)
            if mode == "target":
                self.lbl_instruction.configure(text="Target Label ID (Integer):")
                if not self.ent_labels.get():
                    self.ent_labels.insert(0, "1")
            else:
                self.lbl_instruction.configure(text="Label IDs to Merge (e.g., 1, 2, 5):")
                if not self.ent_labels.get():
                    self.ent_labels.insert(0, "1, 2")

        if self.vol_in is not None:
            self.apply_realtime_change()

    def process_slice(self, slice_data):
        vol = np.round(slice_data).astype(int)
        mode = self.op_mode.get()

        if mode == "largest":
            lbls, num_features = label(vol > 0)
            if num_features == 0:
                return slice_data
            counts = np.bincount(lbls.flat)
            counts[0] = 0
            max_lbl = counts.argmax()
            return (lbls == max_lbl).astype(np.uint8)

        user_input = self.ent_labels.get().strip()
        if not user_input:
            return slice_data

        try:
            if mode == "target":
                target_id = int(user_input)
                return (vol == target_id).astype(np.uint8)
            elif mode == "merge":
                target_ids = [int(x.strip()) for x in user_input.replace(' ', '').split(',') if x.strip().isdigit()]
                return np.isin(vol, target_ids).astype(np.uint8)
        except Exception:
            pass

        return slice_data

    def process_volume(self, data, affine):
        vol = np.round(data).astype(int)
        mode = self.op_mode.get()

        if mode == "largest":
            lbls, num_features = label(vol > 0)
            if num_features == 0:
                return data, affine
            counts = np.bincount(lbls.flat)
            counts[0] = 0
            max_lbl = counts.argmax()
            return (lbls == max_lbl).astype(np.uint8), affine

        user_input = self.ent_labels.get().strip()
        if not user_input:
            return data, affine

        try:
            if mode == "target":
                target_id = int(user_input)
                result = (vol == target_id).astype(np.uint8)
                return result, affine
            elif mode == "merge":
                target_ids = [int(x.strip()) for x in user_input.replace(' ', '').split(',') if x.strip().isdigit()]
                mask = np.isin(vol, target_ids)
                return mask.astype(np.uint8), affine
        except:
            pass

        return data, affine
