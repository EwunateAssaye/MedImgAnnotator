# imprep/modules/app_02_jpg_png_to_nifti.py
from medimg_annotator.imports import (
    skimage_color as color,
    ctk,
    filedialog,
    skimage_io as io,
    messagebox,
    nib,
    np,
    os,
    threading,
    tk,
)

from medimg_annotator.imprep.config import COLORS, FONTS

class PngToNiftiFrame(ctk.CTkFrame):
    def __init__(self, parent, title, controller):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.controller = controller

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.spacing_x = tk.StringVar(value="1.0")
        self.spacing_y = tk.StringVar(value="1.0")
        self.spacing_z = tk.StringVar(value="1.0")
        self.status_var = tk.StringVar(value="Ready")

        container = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_panel"], corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Header
        ctk.CTkLabel(container, text=title, font=FONTS["title"], text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 2))
        ctk.CTkLabel(
            container,
            text="Convert sequential 2D image slices (PNG, JPG, TIFF) into a unified 3D NIfTI volume.",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w"
        ).pack(fill="x", pady=(0, 15))

        # I/O Card
        io_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        io_card.pack(fill="x", pady=8, padx=2)
        ctk.CTkLabel(io_card, text="Directory Configuration", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 6))

        self._make_row(io_card, "Input Image Folder:", self.input_var, self.browse_input)
        self._make_row(io_card, "Output NIfTI Folder:", self.output_var, self.browse_output)

        # Spacing Card
        sp_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        sp_card.pack(fill="x", pady=8, padx=2)
        ctk.CTkLabel(sp_card, text="Voxel Spacing (mm)", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 6))

        sp_row = ctk.CTkFrame(sp_card, fg_color="transparent")
        sp_row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(sp_row, text="dx:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        ctk.CTkEntry(sp_row, textvariable=self.spacing_x, width=60, height=28).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(sp_row, text="dy:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        ctk.CTkEntry(sp_row, textvariable=self.spacing_y, width=60, height=28).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(sp_row, text="dz:", font=FONTS["small"]).pack(side="left", padx=(0, 2))
        ctk.CTkEntry(sp_row, textvariable=self.spacing_z, width=60, height=28).pack(side="left")

        # Action Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(fill="x", pady=15, padx=2)

        self.btn_run = ctk.CTkButton(
            btn_frame,
            text="CONVERT STACK TO NIFTI",
            font=FONTS["button"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=40,
            command=self.run
        )
        self.btn_run.pack(fill="x")

        self.lbl_status = ctk.CTkLabel(container, textvariable=self.status_var, font=FONTS["body"], text_color=COLORS["accent"], anchor="w")
        self.lbl_status.pack(fill="x", pady=5)

    def _make_row(self, parent, label, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label, width=150, anchor="w", font=FONTS["body"]).pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=30).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(row, text="Browse", width=70, height=30, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=cmd).pack(side="right")

    def browse_input(self):
        f = filedialog.askdirectory(title="Select Folder with 2D Images")
        if f:
            self.input_var.set(f)

    def browse_output(self):
        f = filedialog.askdirectory(title="Select Output Folder")
        if f:
            self.output_var.set(f)

    def run(self):
        threading.Thread(target=self._proc, daemon=True).start()

    def _proc(self):
        in_dir = self.input_var.get()
        out_dir = self.output_var.get()

        if not in_dir or not out_dir:
            messagebox.showerror("Missing Paths", "Please specify both input and output folders.")
            return

        try:
            self.btn_run.configure(state="disabled", text="CONVERTING...")
            self.status_var.set("Scanning image slices...")

            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(exts)])
            if not files:
                messagebox.showwarning("No Images", "No compatible 2D image files found in the input folder.")
                return

            self.status_var.set(f"Stacking {len(files)} slices...")
            slices = []
            for f in files:
                img = io.imread(os.path.join(in_dir, f))
                if img.ndim == 3:
                    if img.shape[2] == 4:
                        img = img[..., :3]
                    if img.shape[2] == 3:
                        img = color.rgb2gray(img)
                    elif img.shape[2] == 1:
                        img = img[..., 0]
                    else:
                        img = np.mean(img, axis=2)
                slices.append(img)

            # Vol shape: (H, W, D) -> transpose to (W, H, D) for standard radiological viewing
            vol = np.stack(slices, axis=-1)

            try:
                dx = float(self.spacing_x.get())
                dy = float(self.spacing_y.get())
                dz = float(self.spacing_z.get())
            except:
                dx, dy, dz = 1.0, 1.0, 1.0

            affine = np.diag([dx, dy, dz, 1.0])
            nii = nib.Nifti1Image(vol.astype(np.float32), affine)
            out_file = os.path.join(out_dir, "volume_from_stack.nii.gz")
            nib.save(nii, out_file)

            self.status_var.set("Conversion completed successfully!")
            messagebox.showinfo("Success", f"3D volume successfully created:\n{out_file}")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Conversion Failed", str(e))
        finally:
            self.btn_run.configure(state="normal", text="CONVERT STACK TO NIFTI")
