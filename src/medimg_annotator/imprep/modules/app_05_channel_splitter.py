# imprep/modules/app_05_channel_splitter.py
from medimg_annotator.imports import ctk, messagebox, nib, np, os, threading

from medimg_annotator.imprep.base import BaseNiftiTool, get_out_name
from medimg_annotator.imprep.config import COLORS, FONTS

class ChannelSplitterFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Decomposes 4D / multi-channel NIfTI volumes into individual 3D volumes (one file per channel)."
        )

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(
            info_card,
            text="• Splits multi-sequence / 4D time-series into 3D files.\n• Output files named: <filename>_c0.nii.gz, etc.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left"
        ).pack(anchor="w")

    def run_logic(self):
        threading.Thread(target=self._split, daemon=True).start()

    def _split(self):
        out_folder = self.path_out.get()
        if not out_folder:
            messagebox.showerror("Missing Output", "Please specify an output folder.")
            return

        os.makedirs(out_folder, exist_ok=True)
        self.btn_run.configure(state="disabled", text="SPLITTING...")

        try:
            if self.mode.get() == "single":
                in_p = self.path_in.get()
                if not in_p or not os.path.isfile(in_p):
                    messagebox.showerror("Missing Input", "Please select a valid 4D NIfTI file.")
                    return
                files = [in_p]
            else:
                in_dir = self.path_in.get()
                if not in_dir or not os.path.isdir(in_dir):
                    messagebox.showerror("Missing Input", "Please select a valid input directory.")
                    return
                files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(('.nii', '.nii.gz'))]

            total_split = 0
            for f in files:
                img = nib.load(f)
                d = img.get_fdata(dtype=np.float32)
                if d.ndim == 4:
                    base_name = get_out_name(os.path.basename(f)).replace('.nii.gz', '')
                    for c in range(d.shape[3]):
                        out_path = os.path.join(out_folder, f"{base_name}_c{c}.nii.gz")
                        h = img.header.copy() if img.header is not None else None
                        if h is not None:
                            h.set_data_shape(d[..., c].shape)
                        nib.save(nib.Nifti1Image(d[..., c], img.affine, h), out_path)
                        total_split += 1
                else:
                    # Already 3D, copy as is
                    out_path = os.path.join(out_folder, get_out_name(os.path.basename(f)))
                    nib.save(nib.Nifti1Image(d, img.affine, img.header), out_path)
                    total_split += 1

            messagebox.showinfo("Done", f"Channel splitting completed.\nCreated {total_split} 3D volumes in:\n{out_folder}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_run.configure(state="normal", text="PROCESS & SAVE TO DISK")
