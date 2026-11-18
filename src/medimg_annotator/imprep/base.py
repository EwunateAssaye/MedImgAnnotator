"""
Base classes for ImPrep NIfTI preprocessing modules, workbench RAM bridge, and batch runner.
"""

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

from medimg_annotator.imprep.base_vis import ImPrepVisMixin
from medimg_annotator.imprep.config import COLORS, FONTS
from medimg_annotator.imprep.sidebar import ImPrepSidebar, get_out_name

__all__ = ["get_out_name", "ImPrepSidebar", "BaseNiftiTool"]

class BaseNiftiTool(ImPrepVisMixin, ctk.CTkFrame):
    """
    Standard base class for NIfTI processing tools.
    Provides split control/visualization layout, workbench memory bridge,
    MedImgAnnotator bridge, single-file preview, and batch folder execution.
    """

    def __init__(self, parent, title, controller, description=""):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.controller = controller
        self.title_text = title
        self.description_text = description

        self.vol_in = None
        self.vol_out = None
        self.affine = None
        self.header = None
        self.current_fname = "output.nii.gz"

        self._im1 = None
        self._im2 = None
        self._preview_timer = None
        self._preview_thread_id = 0
        self._slider_timer = None
        self._current_slice_idx = -1

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=420)
        self.grid_columnconfigure(1, weight=1)

        self.ctrl_frame = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_panel"], corner_radius=8, width=430)
        self.ctrl_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        ctk.CTkLabel(
            self.ctrl_frame, text=title, font=FONTS["title"], text_color=COLORS["text_primary"], anchor="w"
        ).pack(fill="x", pady=(5, 2))

        if description:
            ctk.CTkLabel(
                self.ctrl_frame,
                text=description,
                font=FONTS["small"],
                text_color=COLORS["text_muted"],
                anchor="w",
                wraplength=380,
                justify="left",
            ).pack(fill="x", pady=(0, 10))

        self._build_bridge_card()
        self._build_mode_card()
        self._build_io_card()

        self.btn_run = ctk.CTkButton(
            self.ctrl_frame,
            text="PROCESS & SAVE TO DISK",
            font=FONTS["button"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=38,
            corner_radius=6,
            command=self.run_logic,
        )

        self.custom_opts_frame = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        self.custom_opts_frame.pack(fill="x", pady=6, padx=2)
        self.add_custom_widgets(self.custom_opts_frame)

        self.btn_run.pack(fill="x", pady=12, padx=2)
        self._build_slice_slider_card()
        self._build_vis_panel()

        self.hovered_slider = None
        self._leave_timer = None
        self.auto_bind_sliders()

    def _build_bridge_card(self):
        wb_card = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        wb_card.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(
            wb_card, text="Memory & Workstation Bridge", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=10, pady=(8, 4))

        row_wb = ctk.CTkFrame(wb_card, fg_color="transparent")
        row_wb.pack(fill="x", padx=6, pady=(0, 4))
        self.btn_load_wb = ctk.CTkButton(
            row_wb,
            text="⬇ Workbench",
            font=FONTS["small"],
            fg_color=COLORS["accent_secondary"],
            hover_color=COLORS["accent_hover"],
            height=26,
            command=self.load_from_workbench,
        )
        self.btn_load_wb.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_push_wb = ctk.CTkButton(
            row_wb,
            text="⬆ Workbench",
            font=FONTS["small"],
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            height=26,
            command=self.push_to_workbench,
        )
        self.btn_push_wb.pack(side="left", fill="x", expand=True, padx=2)

        row_bridge = ctk.CTkFrame(wb_card, fg_color="transparent")
        row_bridge.pack(fill="x", padx=6, pady=(0, 8))
        self.btn_load_annotator = ctk.CTkButton(
            row_bridge,
            text="📥 From Annotator",
            font=FONTS["small"],
            fg_color="#3a4b5c",
            hover_color="#2b3846",
            height=26,
            command=self.load_from_annotator,
        )
        self.btn_load_annotator.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_push_annotator = ctk.CTkButton(
            row_bridge,
            text="📤 To Annotator",
            font=FONTS["small"],
            fg_color="#2c6b54",
            hover_color="#1d4838",
            height=26,
            command=self.push_to_annotator,
        )
        self.btn_push_annotator.pack(side="left", fill="x", expand=True, padx=2)

    def _build_mode_card(self):
        mode_card = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        mode_card.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(
            mode_card, text="Operation Scope", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.mode = tk.StringVar(value="single")
        row_mode = ctk.CTkFrame(mode_card, fg_color="transparent")
        row_mode.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkRadioButton(
            row_mode, text="Single File (Preview)", variable=self.mode, value="single", font=FONTS["body"], command=self.toggle_mode
        ).pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(
            row_mode, text="Batch Folder", variable=self.mode, value="batch", font=FONTS["body"], command=self.toggle_mode
        ).pack(side="left")

    def _build_io_card(self):
        io_card = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        io_card.pack(fill="x", pady=6, padx=2)
        ctk.CTkLabel(io_card, text="File I/O", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(
            anchor="w", padx=10, pady=(6, 2)
        )
        self.path_in = tk.StringVar()
        self.path_out = tk.StringVar()
        self._make_io_row(io_card, "Input:", self.path_in, is_out=False)
        self._make_io_row(io_card, "Output:", self.path_out, is_out=True)

    def _build_slice_slider_card(self):
        self.slider_frame = ctk.CTkFrame(self.ctrl_frame, fg_color=COLORS["bg_card"], corner_radius=6)
        self.slider_frame.pack(fill="x", pady=4, padx=2)
        self.lbl_slice = ctk.CTkLabel(
            self.slider_frame, text="Slice: 0 / 0", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]
        )
        self.lbl_slice.pack(anchor="w", padx=10, pady=(6, 2))
        self.slider = ctk.CTkSlider(self.slider_frame, from_=0, to=1, number_of_steps=1, command=self.on_slider_change)
        self.slider.pack(fill="x", padx=10, pady=(0, 10))

    def _build_vis_panel(self):
        self.vis_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"], corner_radius=8)
        self.vis_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor=COLORS["plot_bg"])
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self._style_axes(self.ax1, "Input Volume")
        self._style_axes(self.ax2, "Processed Output")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas_widget.bind("<MouseWheel>", self._on_canvas_scroll)
        canvas_widget.bind("<Button-4>", lambda e: self.step_slice(1))
        canvas_widget.bind("<Button-5>", lambda e: self.step_slice(-1))
        canvas_widget.bind("<Button-1>", lambda e: canvas_widget.focus_set())
        self.reset_plot()

    def _make_io_row(self, parent, label, var, is_out):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, font=FONTS["small"], width=50, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=28).pack(
            side="left", fill="x", expand=True, padx=4
        )

        def browse():
            if is_out or self.mode.get() == "batch":
                res = filedialog.askdirectory(title=f"Select {label} Directory")
            else:
                res = filedialog.askopenfilename(
                    title=f"Select {label} NIfTI File",
                    filetypes=[("NIfTI Images", "*.nii *.nii.gz"), ("All Files", "*.*")],
                )
            if res:
                var.set(res)
                if not is_out and self.mode.get() == "single":
                    self.current_fname = os.path.basename(res)
                    self.load_preview(res)

        ctk.CTkButton(
            row, text="...", width=32, height=28, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=browse
        ).pack(side="right")

    def _load_volume_into_preview(self, vol_data, affine, header, fname):
        """Populate preview state from workbench RAM or MedImgAnnotator."""
        self.vol_in = vol_data
        self.affine = affine
        self.header = header
        self.current_fname = fname
        self.vol_out = None
        self._im1 = None
        self._im2 = None
        self.mode.set("single")
        self.toggle_mode()
        self.init_slider()
        self.update_vis()
        self.run_preview()

    def load_from_workbench(self):
        if getattr(self.controller, "shared_vol", None) is None:
            messagebox.showinfo(
                "Workbench Empty", "Workbench memory is currently empty. Process or push a volume first."
            )
            return
        fname = self.controller.shared_fname or "workbench_vol.nii.gz"
        self._load_volume_into_preview(
            self.controller.shared_vol, self.controller.shared_affine, self.controller.shared_header, fname
        )
        messagebox.showinfo("Loaded", f"Loaded '{self.current_fname}' from Workbench Memory.")

    def push_to_workbench(self):
        target_vol = self.vol_out if self.vol_out is not None else self.vol_in
        if target_vol is None:
            messagebox.showwarning("No Data", "No processed volume available to push to Workbench.")
            return
        self.controller.shared_vol = target_vol
        self.controller.shared_affine = self.affine
        self.controller.shared_header = self.header
        self.controller.shared_fname = self.current_fname
        messagebox.showinfo("Success", f"Volume '{self.current_fname}' pushed to Workbench RAM.")

    def load_from_annotator(self):
        """Load active image volume directly from MedImgAnnotator."""
        editor = getattr(self.controller, "editor", None)
        if editor is None or getattr(editor, "img_data", None) is None:
            messagebox.showwarning("No Annotator Data", "No active image volume loaded in MedImgAnnotator.")
            return

        raw = getattr(editor, "img_data_raw", None)
        vol = np.copy(raw if raw is not None else editor.img_data)
        aff = np.copy(editor.affine) if editor.affine is not None else np.eye(4)
        curr_idx = getattr(editor, "current_index", 0)
        file_list = getattr(editor, "file_list", [])
        if file_list and 0 <= curr_idx < len(file_list):
            item = file_list[curr_idx]
            fname = os.path.basename(item[0] if isinstance(item, (list, tuple)) else str(item))
        else:
            fname = "annotator_image.nii.gz"

        self._load_volume_into_preview(vol, aff, editor.header, fname)
        messagebox.showinfo("Loaded from Annotator", f"Successfully imported active volume '{self.current_fname}'.")

    def push_to_annotator(self):
        """Send processed volume directly to MedImgAnnotator as Main Image or Mask."""
        editor = getattr(self.controller, "editor", None)
        if editor is None:
            messagebox.showwarning("Bridge Unavailable", "MedImgAnnotator reference is not connected.")
            return

        target_vol = self.vol_out if self.vol_out is not None else self.vol_in
        if target_vol is None:
            messagebox.showwarning("No Data", "No volume to push to Annotator.")
            return

        ans = messagebox.askyesno(
            "Send to Annotator",
            "Do you want to send this volume to MedImgAnnotator as the MAIN IMAGE?\n\n(Click 'Yes' for Main Image, 'No' for Mask)",
        )
        as_mask = not ans
        if hasattr(editor, "set_external_volume"):
            editor.set_external_volume(
                target_vol,
                self.affine if self.affine is not None else np.eye(4),
                self.header,
                self.current_fname,
                as_mask=as_mask,
            )
            messagebox.showinfo(
                "Updated Annotator",
                f"Volume successfully sent to MedImgAnnotator as {'Mask' if as_mask else 'Main Image'}.",
            )
        else:
            messagebox.showerror("Error", "Annotator does not support set_external_volume method.")

    def add_custom_widgets(self, parent):
        """Override in subclasses to insert tool-specific controls."""

    def process_volume(self, data, affine):
        """Override in subclasses to implement volume transformation."""
        return data, affine

    def process_slice(self, slice_data):
        """Override in subclasses for instant 2D slice preview; return None if unsupported."""
        return None

    def run_logic(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        out_p = self.path_out.get()
        if not out_p:
            self.after(0, lambda: messagebox.showerror("Missing Output", "Please select an output directory."))
            return

        os.makedirs(out_p, exist_ok=True)
        self.after(0, lambda: self.btn_run.configure(state="disabled", text="PROCESSING..."))

        try:
            if self.mode.get() == "single":
                if self.vol_in is None:
                    self.after(0, lambda: messagebox.showwarning("No Input", "No input image loaded."))
                    return
                d, a = self.process_volume(self.vol_in, self.affine)
                self.vol_out = d
                save_path = os.path.join(out_p, get_out_name(self.current_fname))
                nib.save(nib.Nifti1Image(d, a, self.header), save_path)
                self.after(0, self.update_vis)
                self.after(
                    0, lambda p=save_path: messagebox.showinfo("Completed", f"Processed file successfully saved to:\n{p}")
                )
            else:
                in_p = self.path_in.get()
                if not in_p or not os.path.isdir(in_p):
                    self.after(0, lambda: messagebox.showerror("Invalid Input", "Please select a valid input directory."))
                    return
                files = [f for f in os.listdir(in_p) if f.endswith((".nii", ".nii.gz"))]
                if not files:
                    self.after(
                        0, lambda: messagebox.showwarning("Empty Folder", "No .nii or .nii.gz files found in input directory.")
                    )
                    return

                count = 0
                total_f = len(files)
                for i, f in enumerate(files):
                    self.after(0, lambda idx=i + 1, tot=total_f: self.btn_run.configure(text=f"PROCESSING ({idx}/{tot})..."))
                    try:
                        nii = nib.load(os.path.join(in_p, f))
                        d, a = self.process_volume(nii.get_fdata(dtype=np.float32), nii.affine)
                        nib.save(nib.Nifti1Image(d, a, nii.header), os.path.join(out_p, get_out_name(f)))
                        count += 1
                    except Exception as e:
                        print(f"Error processing {f}: {e}")

                self.after(
                    0, lambda c=count, t=total_f: messagebox.showinfo("Batch Complete", f"Successfully processed {c} / {t} files.")
                )
        except Exception as err:
            err_msg = str(err)
            self.after(0, lambda m=err_msg: messagebox.showerror("Processing Error", f"An unexpected error occurred:\n{m}"))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="PROCESS & SAVE TO DISK"))
