# imprep/modules/app_03_image_mask_aligner.py
from medimg_annotator.imports import (
    ctk,
    Figure,
    FigureCanvasTkAgg,
    filedialog,
    messagebox,
    nib,
    np,
    os,
    threading,
    tk,
)

from medimg_annotator.imprep.config import COLORS, FONTS
from medimg_annotator.imprep.base import get_out_name

class ImageMaskAlignerFrame(ctk.CTkFrame):
    def __init__(self, parent, title, controller):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.controller = controller

        self.vol_img, self.vol_msk = None, None
        self.aff_img, self.header_img = None, None
        self.aff_msk, self.header_msk = None, None
        self.fname_img, self.fname_msk = "image.nii.gz", "mask.nii.gz"

        # High-performance visualization caches
        self._im_img = None
        self._im_msk = None
        self._slider_timer = None
        self._current_z = -1

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=430)
        self.grid_columnconfigure(1, weight=1)

        # --- LEFT CONTROLS ---
        self.ctrl_frame = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_panel"], corner_radius=8, width=430)
        self.ctrl_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        ctk.CTkLabel(self.ctrl_frame, text=title, font=FONTS["title"], text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 2))
        ctk.CTkLabel(
            self.ctrl_frame,
            text="Visually align images and masks with flips, 90° rotations, and live overlay inspection.",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
            wraplength=380,
            justify="left"
        ).pack(fill="x", pady=(0, 10))

        # Mode
        mode_card = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        mode_card.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(mode_card, text="Operation Scope", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.mode = tk.StringVar(value="single")
        row_mode = ctk.CTkFrame(mode_card, fg_color="transparent")
        row_mode.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkRadioButton(row_mode, text="Single File (Visual Alignment)", variable=self.mode, value="single", font=FONTS["body"], command=self._update_ui_state).pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(row_mode, text="Batch Processing", variable=self.mode, value="batch", font=FONTS["body"], command=self._update_ui_state).pack(side="left")

        # I/O Card
        self.grp_io = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        self.grp_io.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(self.grp_io, text="Files & Output", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.path_img = tk.StringVar()
        self.path_msk = tk.StringVar()
        self.path_batch_in = tk.StringVar()
        self.path_out = tk.StringVar()

        self.row_img = self._make_row(self.grp_io, "Image:", self.path_img, self.load_img)
        self.row_msk = self._make_row(self.grp_io, "Mask:", self.path_msk, self.load_msk)
        self.row_batch = self._make_row(self.grp_io, "In Folder:", self.path_batch_in, lambda: self.path_batch_in.set(filedialog.askdirectory()))
        self.row_out = self._make_row(self.grp_io, "Out Folder:", self.path_out, lambda: self.path_out.set(filedialog.askdirectory()))

        # Transformations Card
        grp_trans = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        grp_trans.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(grp_trans, text="Transformation Operations", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.target_layer = tk.StringVar(value="img")
        self.row_target = ctk.CTkFrame(grp_trans, fg_color="transparent")
        self.row_target.pack(fill="x", padx=10, pady=(2, 6))

        ctk.CTkLabel(self.row_target, text="Target:", font=FONTS["small"]).pack(side="left", padx=(0, 8))
        ctk.CTkRadioButton(self.row_target, text="Image", variable=self.target_layer, value="img", font=FONTS["body"]).pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.row_target, text="Mask", variable=self.target_layer, value="msk", font=FONTS["body"]).pack(side="left")

        # Transform Buttons Grid
        btn_grid = ctk.CTkFrame(grp_trans, fg_color="transparent")
        btn_grid.pack(fill="x", padx=8, pady=(0, 8))

        self.btn_flip_x = ctk.CTkButton(btn_grid, text="Flip X (L/R)", font=FONTS["button"], height=32, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=lambda: self.trigger_action(0))
        self.btn_flip_x.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        self.btn_flip_y = ctk.CTkButton(btn_grid, text="Flip Y (A/P)", font=FONTS["button"], height=32, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=lambda: self.trigger_action(1))
        self.btn_flip_y.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.btn_flip_z = ctk.CTkButton(btn_grid, text="Flip Z (S/I)", font=FONTS["button"], height=32, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=lambda: self.trigger_action(2))
        self.btn_flip_z.grid(row=0, column=2, sticky="ew", padx=2, pady=2)

        self.btn_rot = ctk.CTkButton(btn_grid, text="Rotate 90°", font=FONTS["button"], height=32, fg_color=COLORS["accent_secondary"], hover_color=COLORS["accent_hover"], command=lambda: self.trigger_action('rot'))
        self.btn_rot.grid(row=1, column=0, columnspan=3, sticky="ew", padx=2, pady=4)

        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        btn_grid.columnconfigure(2, weight=1)

        # Single Actions
        self.btn_save = ctk.CTkButton(
            self.ctrl_frame,
            text="SAVE SELECTED LAYER",
            font=FONTS["button"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=36,
            command=self.save_single
        )
        self.btn_save.pack(fill="x", pady=(10, 4), padx=2)

        self.btn_wb = ctk.CTkButton(
            self.ctrl_frame,
            text="Push Image to Workbench RAM",
            font=FONTS["small"],
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            height=30,
            command=self.push_wb
        )
        self.btn_wb.pack(fill="x", pady=2, padx=2)

        # Slider Card
        self.slider_frame = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        self.slider_frame.pack(fill="x", pady=6, padx=2)

        self.lbl_slice = ctk.CTkLabel(self.slider_frame, text="Slice: 0 / 0", font=FONTS["sub_header"], text_color=COLORS["text_secondary"])
        self.lbl_slice.pack(anchor="w", padx=10, pady=(6, 2))

        self.slider = ctk.CTkSlider(self.slider_frame, from_=0, to=1, number_of_steps=1, command=self.on_slider)
        self.slider.pack(fill="x", padx=10, pady=(0, 10))

        # Batch Progress Card
        self.grp_progress = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        ctk.CTkLabel(self.grp_progress, text="Batch Progress", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=10, pady=(6, 2))
        self.status_var = tk.StringVar(value="Ready")
        self.lbl_status = ctk.CTkLabel(self.grp_progress, textvariable=self.status_var, font=FONTS["small"], text_color=COLORS["accent"], wraplength=350, justify="left")
        self.lbl_status.pack(anchor="w", padx=10, pady=2)
        self.pbar = ctk.CTkProgressBar(self.grp_progress, progress_color=COLORS["accent"])
        self.pbar.pack(fill="x", padx=10, pady=(4, 10))
        self.pbar.set(0)

        # --- RIGHT PANEL: VISUALIZATION ---
        vis = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"], corner_radius=8)
        vis.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=COLORS["plot_bg"])
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas_widget.bind("<MouseWheel>", lambda e: self.step_slice(1 if getattr(e, 'delta', 0) > 0 else -1))
        canvas_widget.bind("<Button-4>", lambda e: self.step_slice(1))
        canvas_widget.bind("<Button-5>", lambda e: self.step_slice(-1))

        self._update_ui_state()

        # Hover tracking for keyboard arrow updates on slider
        self.hovered_slider = None
        self._leave_timer = None
        self.register_slider_hover(self.slider)

    def _style_ax(self):
        self.ax.set_facecolor(COLORS["plot_axes"])
        self.ax.set_title("Image (Grayscale) + Mask (Overlay)", color=COLORS["plot_text"], fontsize=11, fontweight="bold", pad=8)
        self.ax.axis('off')

    def _make_row(self, parent, label, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=65, anchor="w", font=FONTS["small"]).pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=28).pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row, text="...", width=32, height=28, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=cmd).pack(side="right")
        return row

    def _update_ui_state(self):
        mode = self.mode.get()
        if mode == "single":
            self.row_batch.pack_forget()
            self.grp_progress.pack_forget()
            self.row_img.pack(fill="x", padx=8, pady=3)
            self.row_msk.pack(fill="x", padx=8, pady=3)
            self.row_target.pack(fill="x", padx=10, pady=(2, 6))
            self.btn_save.pack(fill="x", pady=(10, 4), padx=2)
            self.btn_wb.pack(fill="x", pady=2, padx=2)
            self.slider_frame.pack(fill="x", pady=6, padx=2)
        else:
            self.row_img.pack_forget()
            self.row_msk.pack_forget()
            self.row_batch.pack(fill="x", padx=8, pady=3)
            self.row_target.pack_forget()
            self.btn_save.pack_forget()
            self.btn_wb.pack_forget()
            self.slider_frame.pack_forget()
            self.grp_progress.pack(fill="x", pady=6, padx=2)
            self.ax.clear()
            self._style_ax()
            self.canvas.draw_idle()

    def load_img(self):
        f = filedialog.askopenfilename(title="Select NIfTI Image", filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if f:
            self.path_img.set(f)
            nii = nib.load(f)
            d = nii.get_fdata()
            if d.ndim > 3 and d.shape[2] == 1 and d.shape[3] > 1:
                d = np.squeeze(d, axis=2)
            while d.ndim > 3:
                d = d[..., 0]
            if d.ndim == 2:
                d = d[:, :, np.newaxis]
            self.vol_img, self.aff_img, self.header_img = d, nii.affine, nii.header
            self.fname_img = os.path.basename(f)
            self._im_img = None
            self._im_msk = None
            self.init_slider()
            self.update_vis()

    def load_msk(self):
        f = filedialog.askopenfilename(title="Select NIfTI Mask", filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if f:
            self.path_msk.set(f)
            nii = nib.load(f)
            d = nii.get_fdata()
            if d.ndim > 3 and d.shape[2] == 1 and d.shape[3] > 1:
                d = np.squeeze(d, axis=2)
            while d.ndim > 3:
                d = d[..., 0]
            if d.ndim == 2:
                d = d[:, :, np.newaxis]
            self.vol_msk, self.aff_msk, self.header_msk = d, nii.affine, nii.header
            self.fname_msk = os.path.basename(f)
            self._im_msk = None
            self.update_vis()

    def init_slider(self):
        if self.vol_img is not None and self.vol_img.ndim >= 3:
            max_s = max(0, self.vol_img.shape[2] - 1)
            self.slider.configure(to=max(1, max_s), from_=0, number_of_steps=max(1, max_s))
            mid = max_s // 2
            self.slider.set(mid)
            self.slider.configure(state="normal" if max_s > 0 else "disabled")
            self._current_z = mid
            self.lbl_slice.configure(text=f"Slice: {mid} / {max_s}")

    def on_slider(self, val):
        if self.vol_img is not None and self.vol_img.ndim >= 3:
            max_s = max(0, self.vol_img.shape[2] - 1)
            z = max(0, min(max_s, int(float(val))))
        else:
            z = int(float(val))
        if getattr(self, '_current_z', None) == z:
            return
        self._current_z = z
        if self.vol_img is not None and self.vol_img.ndim >= 3:
            self.lbl_slice.configure(text=f"Slice: {z} / {max(0, self.vol_img.shape[2] - 1)}")
        if getattr(self, '_slider_timer', None) is None:
            self._slider_timer = self.after(16, self._finish_slider)

    def _finish_slider(self):
        self._slider_timer = None
        self.update_vis()

    def step_slice(self, delta=1):
        if self.vol_img is None or self.vol_img.ndim < 3:
            return
        max_s = self.vol_img.shape[2] - 1
        cur = int(self.slider.get())
        target = max(0, min(max_s, cur + delta))
        if target != cur:
            self.slider.set(target)
            self.on_slider(target)

    def register_slider_hover(self, slider):
        """Binds hover and mouse-wheel events to track the active slider for keyboard arrow updates."""
        if not isinstance(slider, ctk.CTkSlider):
            return

        def on_enter(e):
            if hasattr(self, '_leave_timer') and self._leave_timer is not None:
                try:
                    self.after_cancel(self._leave_timer)
                except Exception:
                    pass
                self._leave_timer = None
            controller = getattr(self, 'controller', None)
            if controller:
                controller.hovered_slider = slider
            self.hovered_slider = slider

        def on_leave(e):
            def clear():
                self._leave_timer = None
                controller = getattr(self, 'controller', None)
                if controller and getattr(controller, 'hovered_slider', None) == slider:
                    controller.hovered_slider = None
                if getattr(self, 'hovered_slider', None) == slider:
                    self.hovered_slider = None

            try:
                self._leave_timer = self.after(60, clear)
            except Exception:
                clear()

        def on_wheel(e):
            if slider.cget("state") == "disabled":
                return "break"
            delta = 1 if getattr(e, 'delta', 0) > 0 or getattr(e, 'num', 0) == 4 else -1
            controller = getattr(self, 'controller', None)
            if controller and hasattr(controller, 'increment_slider'):
                controller.increment_slider(slider, delta)
            return "break"

        for target in (slider, getattr(slider, '_canvas', None)):
            if target is not None:
                target.bind("<Enter>", on_enter, add="+")
                target.bind("<Leave>", on_leave, add="+")
                target.bind("<MouseWheel>", on_wheel, add="+")
                target.bind("<Button-4>", on_wheel, add="+")
                target.bind("<Button-5>", on_wheel, add="+")

    def trigger_action(self, axis):
        if self.mode.get() == "batch":
            self._toggle_buttons("disabled")
            threading.Thread(target=lambda: self.batch_worker(axis), daemon=True).start()
        else:
            self.single_transform(axis)

    def _toggle_buttons(self, state):
        self.btn_flip_x.configure(state=state)
        self.btn_flip_y.configure(state=state)
        self.btn_flip_z.configure(state=state)
        self.btn_rot.configure(state=state)

    def single_transform(self, axis):
        target = self.target_layer.get()
        if target == "img":
            if self.vol_img is None:
                return
            if axis == 'rot':
                self.vol_img = np.rot90(self.vol_img, k=1, axes=(0, 1))
            else:
                self.vol_img = np.flip(self.vol_img, axis=axis)
            self._im_img = None
        else:
            if self.vol_msk is None:
                return
            if axis == 'rot':
                self.vol_msk = np.rot90(self.vol_msk, k=1, axes=(0, 1))
            else:
                self.vol_msk = np.flip(self.vol_msk, axis=axis)
            self._im_msk = None
        self.update_vis()

    def batch_worker(self, axis):
        in_dir = self.path_batch_in.get()
        out_dir = self.path_out.get()

        if not in_dir or not out_dir:
            messagebox.showerror("Missing Paths", "Select Input and Output folders for Batch mode.")
            self._toggle_buttons("normal")
            return

        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.nii', '.nii.gz'))]
        total_files = len(files)

        if total_files == 0:
            messagebox.showwarning("Empty", "No NIfTI files found in input folder.")
            self._toggle_buttons("normal")
            return

        success_count = 0
        for i, f in enumerate(files):
            self.status_var.set(f"Processing ({i+1}/{total_files}): {f}")
            try:
                path = os.path.join(in_dir, f)
                nii = nib.load(path)
                data = nii.get_fdata()
                affine = nii.affine

                if axis == 'rot':
                    data = np.rot90(data, k=1, axes=(0, 1))
                else:
                    data = np.flip(data, axis=axis)

                new_nii = nib.Nifti1Image(data, affine, nii.header)
                nib.save(new_nii, os.path.join(out_dir, get_out_name(f)))
                success_count += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")

            prog = (i + 1) / total_files
            self.pbar.set(prog)

        self.status_var.set(f"Done. Processed {success_count}/{total_files}.")
        self._toggle_buttons("normal")
        messagebox.showinfo("Batch Complete", f"Successfully processed {success_count} files.\nSaved to: {out_dir}")

    def update_vis(self, _=None):
        if self.vol_img is None or self.vol_img.ndim < 3:
            return
        z = int(self.slider.get())
        if z >= self.vol_img.shape[2]:
            z = self.vol_img.shape[2] - 1

        img_sl = np.flipud(np.rot90(self.vol_img[:, :, z]))

        if getattr(self, '_im_img', None) is None:
            self.ax.clear()
            self._im_img = self.ax.imshow(img_sl, cmap='gray', interpolation='bilinear')
            self._im_msk = None
            if self.vol_msk is not None and z < self.vol_msk.shape[2]:
                msk_sl = np.flipud(np.rot90(self.vol_msk[:, :, z]))
                masked = np.ma.masked_where(msk_sl < 0.5, msk_sl)
                self._im_msk = self.ax.imshow(masked, cmap='autumn', alpha=0.5, interpolation='nearest')
            self._style_ax()
        else:
            self._im_img.set_data(img_sl)
            vmin, vmax = float(np.min(img_sl)), float(np.max(img_sl))
            if vmin != vmax:
                self._im_img.set_clim(vmin=vmin, vmax=vmax)
            if self.vol_msk is not None and z < self.vol_msk.shape[2]:
                msk_sl = np.flipud(np.rot90(self.vol_msk[:, :, z]))
                masked = np.ma.masked_where(msk_sl < 0.5, msk_sl)
                if getattr(self, '_im_msk', None) is not None:
                    self._im_msk.set_data(masked)
                else:
                    self._im_msk = self.ax.imshow(masked, cmap='autumn', alpha=0.5, interpolation='nearest')
            elif getattr(self, '_im_msk', None) is not None:
                self._im_msk.set_data(np.ma.masked_all_like(img_sl))

        self.canvas.draw_idle()

    def save_single(self):
        out_folder = self.path_out.get()
        if not out_folder:
            messagebox.showwarning("Warning", "Select output folder.")
            return

        os.makedirs(out_folder, exist_ok=True)
        target = self.target_layer.get()
        try:
            if target == "img":
                if self.vol_img is None:
                    return
                out_name = get_out_name(self.fname_img)
                save_path = os.path.join(out_folder, out_name)
                nib.save(nib.Nifti1Image(self.vol_img, self.aff_img, self.header_img), save_path)
                messagebox.showinfo("Saved Image", f"Saved successfully:\n{save_path}")
            else:
                if self.vol_msk is None:
                    return
                out_name = get_out_name(self.fname_msk)
                save_path = os.path.join(out_folder, out_name)
                nib.save(nib.Nifti1Image(self.vol_msk, self.aff_msk, self.header_msk), save_path)
                messagebox.showinfo("Saved Mask", f"Saved successfully:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def push_wb(self):
        if self.vol_img is None:
            return
        self.controller.shared_vol = self.vol_img
        self.controller.shared_affine = self.aff_img
        self.controller.shared_header = self.header_img
        self.controller.shared_fname = self.fname_img
        messagebox.showinfo("Success", "Image successfully pushed to Workbench RAM.")
