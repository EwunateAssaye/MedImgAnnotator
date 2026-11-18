"""
Main workstation toolbar builder, workspace layout, slider hover/wheel bindings, and keyboard shortcuts.
"""

from medimg_annotator.imports import ctk, messagebox, tk, webbrowser

from medimg_annotator.ui.editor_canvas import EditorCanvas
from medimg_annotator.ui.preview_canvas import PreviewCanvas

class ToolbarBuilderMixin:
    """Constructs the workstation rows, dual-canvas workspace, volume panel, and shortcuts."""

    def create_toolbar_group(self, parent):
        grp = ctk.CTkFrame(parent, fg_color="transparent")
        grp.pack(side=tk.LEFT, expand=True)
        return grp

    def init_ui(self):
        """Build the three toolbar rows, dual-canvas pane, volume statistics, and CSV controls."""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._build_row1_io(main_frame)
        self._build_row2_tools(main_frame)
        self._build_row3_labels(main_frame)
        self._build_main_workspace(main_frame)

        for slider in (self.slider, self.slider_opacity, self.sb_brightness, self.sb_contrast, self.sb_zoom):
            self.setup_slider_bindings(slider)

    def _build_row1_io(self, parent):
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill=tk.X, pady=(0, 4))

        g1_1 = self.create_toolbar_group(row1)
        ctk.CTkButton(
            g1_1, text="ImPrep", command=self.open_imprep, width=70, fg_color="#1f6aa5", hover_color="#144870"
        ).pack(side=tk.LEFT, padx=2)
        self.btn_input = ctk.CTkButton(g1_1, text="Input", command=self.select_folders, width=70)
        self.btn_input.pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(g1_1, text="Output", command=self.select_output_folder, width=70).pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(g1_1, text="Plugins", command=self.manage_plugins, width=70).pack(side=tk.LEFT, padx=2)

        g1_2 = self.create_toolbar_group(row1)
        ctk.CTkLabel(g1_2, text="Search:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=2)
        self.search_input = ctk.CTkEntry(g1_2, placeholder_text="Name/Idx", width=100)
        self.search_input.pack(side=tk.LEFT, padx=2)
        self.search_input.bind("<Return>", lambda e: self.perform_search())
        ctk.CTkButton(g1_2, text="Go", width=30, command=self.perform_search).pack(side=tk.LEFT, padx=2)

        g1_3 = self.create_toolbar_group(row1)
        self.btn_prev = ctk.CTkButton(g1_3, text="prev-vol", command=self.prev_image, state="disabled", width=65)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.lbl_file_info = ctk.CTkLabel(
            g1_3, text="No Data Loaded", text_color="#ffaa00", font=("Arial", 12, "bold")
        )
        self.lbl_file_info.pack(side=tk.LEFT, padx=10)
        self.btn_next = ctk.CTkButton(g1_3, text="Next-vol", command=self.next_image, state="disabled", width=65)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        g1_4 = self.create_toolbar_group(row1)
        self.save_container = ctk.CTkFrame(g1_4, fg_color="transparent")
        self.save_container.pack(side=tk.LEFT, padx=3)
        self.btn_quick_save = ctk.CTkButton(
            self.save_container, text="Save (S)", command=self.quick_save_mask, state="disabled", width=68
        )
        self.btn_quick_save.pack(side=tk.LEFT, padx=(0, 1))
        self.btn_save_drop = ctk.CTkButton(
            self.save_container,
            text="▾",
            command=self.show_save_menu,
            state="disabled",
            width=22,
            fg_color="#3a3d40",
            hover_color="#4f5256",
        )
        self.btn_save_drop.pack(side=tk.LEFT, padx=(0, 2))
        self.btn_quick_save.bind("<Button-3>", self.show_save_menu)
        self.btn_save_drop.bind("<Button-3>", self.show_save_menu)
        ctk.CTkButton(g1_4, text="Undo (Ctrl+Z)", command=self.undo_action, width=90).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(g1_4, text="Redo (Ctrl+Y)", command=self.redo_action, width=90).pack(side=tk.LEFT, padx=3)

    def _build_row2_tools(self, parent):
        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill=tk.X, pady=(0, 4))

        g2_1 = self.create_toolbar_group(row2)
        ctk.CTkLabel(g2_1, text="View:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=2)
        self.view_var = tk.IntVar(value=2)
        for txt, val in (("Ax", 2), ("Cor", 1), ("Sag", 0)):
            ctk.CTkRadioButton(
                g2_1, text=txt, variable=self.view_var, value=val, command=self.change_view_axis, width=45
            ).pack(side=tk.LEFT, padx=2)

        g2_2 = self.create_toolbar_group(row2)
        ctk.CTkLabel(g2_2, text="Scope:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=2)
        self.scope_var = tk.StringVar(value="2D")
        ctk.CTkRadioButton(g2_2, text="2D", variable=self.scope_var, value="2D", width=45).pack(side=tk.LEFT, padx=2)
        ctk.CTkRadioButton(g2_2, text="3D", variable=self.scope_var, value="3D", width=45).pack(side=tk.LEFT, padx=2)

        g2_3 = self.create_toolbar_group(row2)
        ctk.CTkLabel(g2_3, text="Tool:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=2)
        self.tool_var = tk.StringVar(value="brush")
        ctk.CTkRadioButton(
            g2_3, text="Brush", variable=self.tool_var, value="brush", command=self.change_tool_mode, width=55
        ).pack(side=tk.LEFT, padx=2)
        ctk.CTkRadioButton(
            g2_3, text="Bucket", variable=self.tool_var, value="bucket", command=self.change_tool_mode, width=65
        ).pack(side=tk.LEFT, padx=2)
        ctk.CTkRadioButton(
            g2_3, text="Poly", variable=self.tool_var, value="freehand", command=self.change_tool_mode, width=55
        ).pack(side=tk.LEFT, padx=2)

        self.poly_auto_var = tk.BooleanVar(value=False)
        self.check_poly_auto = ctk.CTkCheckBox(
            g2_3, text="(Auto)", variable=self.poly_auto_var, state="disabled", width=65
        )
        self.check_poly_auto.pack(side=tk.LEFT, padx=(0, 5))

        ctk.CTkRadioButton(
            g2_3, text="Wand", variable=self.tool_var, value="wand", command=self.change_tool_mode, width=55
        ).pack(side=tk.LEFT, padx=2)
        ctk.CTkLabel(g2_3, text="(Tol:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 1))
        self.spin_wand_tol_var = tk.StringVar(value=str(self.wand_tolerance))
        self.spin_wand_tol = ctk.CTkEntry(g2_3, textvariable=self.spin_wand_tol_var, width=35, state="disabled")
        self.spin_wand_tol.pack(side=tk.LEFT, padx=1)
        self.spin_wand_tol.bind("<Return>", self.change_wand_tol_event)
        self.spin_wand_tol.bind("<FocusOut>", self.change_wand_tol_event)

        ctk.CTkLabel(g2_3, text="Lim:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(2, 1))
        self.spin_wand_bound_var = tk.StringVar(value=str(self.wand_limit))
        self.spin_wand_bound = ctk.CTkEntry(g2_3, textvariable=self.spin_wand_bound_var, width=40, state="disabled")
        self.spin_wand_bound.pack(side=tk.LEFT, padx=1)
        self.spin_wand_bound.bind("<Return>", self.change_wand_bound_event)
        self.spin_wand_bound.bind("<FocusOut>", self.change_wand_bound_event)
        ctk.CTkLabel(g2_3, text=")", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 5))

        g2_4 = self.create_toolbar_group(row2)
        self.eraser_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(g2_4, text="Eraser", variable=self.eraser_var, command=self.toggle_eraser, width=65).pack(
            side=tk.LEFT, padx=2
        )
        ctk.CTkLabel(g2_4, text="Rad:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=2)
        self.spin_brush_var = tk.StringVar(value=str(self.brush_size))
        self.spin_brush = ctk.CTkEntry(g2_4, textvariable=self.spin_brush_var, width=45)
        self.spin_brush.pack(side=tk.LEFT, padx=2)
        self.spin_brush.bind("<Return>", self.change_brush_size_event)
        self.spin_brush.bind("<FocusOut>", self.change_brush_size_event)

        g2_5 = self.create_toolbar_group(row2)
        ctk.CTkLabel(g2_5, text="Slices:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(1, 1))
        self.line_slices = ctk.CTkEntry(g2_5, placeholder_text="10,15", width=62)
        self.line_slices.pack(side=tk.LEFT, padx=(1, 2))
        self.btn_interpolate = ctk.CTkButton(
            g2_5, text="Interpolate", command=self.apply_interpolation, fg_color="#2b5c8f", width=72
        )
        self.btn_interpolate.pack(side=tk.LEFT, padx=(1, 2))
        self.btn_obdet = ctk.CTkButton(
            g2_5, text="ObDet", command=self.toggle_obdet_mode, fg_color="#1b5e20", hover_color="#2e7d32", width=62
        )
        self.btn_obdet.pack(side=tk.LEFT, padx=(1, 2))

    def _build_row3_labels(self, parent):
        row3 = ctk.CTkFrame(parent, fg_color="transparent")
        row3.pack(fill=tk.X, pady=(0, 6))

        g3_1 = ctk.CTkFrame(row3, fg_color="transparent")
        g3_1.pack(side=tk.LEFT, expand=False, padx=(0, 10))
        self.multilabel_var = tk.BooleanVar(value=False)
        self.check_multilabel = ctk.CTkCheckBox(
            g3_1, text="Multi-label", variable=self.multilabel_var, command=self.toggle_multilabel, width=85
        )
        self.check_multilabel.pack(side=tk.LEFT, padx=2)
        ctk.CTkLabel(g3_1, text="Val:").pack(side=tk.LEFT, padx=2)
        self.combo_value = ctk.CTkComboBox(
            g3_1,
            values=[str(i) for i in range(1, 101)],
            command=self.on_value_selection,
            width=65,
            state="disabled",
        )
        self.combo_value.pack(side=tk.LEFT, padx=2)
        ctk.CTkLabel(g3_1, text="Color:").pack(side=tk.LEFT, padx=2)
        self.combo_color = ctk.CTkComboBox(
            g3_1,
            values=list(self.color_map.keys()),
            command=self.on_color_selection,
            width=126,
            state="disabled",
        )
        self.combo_color.set(self.value_to_color.get(1, next(iter(self.color_map))))
        self.combo_color.pack(side=tk.LEFT, padx=2)
        self.lbl_color_swatch = ctk.CTkLabel(
            g3_1,
            text="",
            width=18,
            height=18,
            corner_radius=4,
            fg_color="#ff0000",
        )
        self.lbl_color_swatch.pack(side=tk.LEFT, padx=(2, 4))
        self.line_name = ctk.CTkEntry(g3_1, placeholder_text="Name", width=90, state="disabled")
        self.line_name.pack(side=tk.LEFT, padx=(2, 10))
        self.line_name.bind("<KeyRelease>", self.on_label_name_changed)
        self.line_name.bind("<FocusOut>", self.on_label_name_changed)

        g3_2 = ctk.CTkFrame(row3, fg_color="transparent")
        g3_2.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.lbl_slice_info = ctk.CTkLabel(g3_2, text="Slice: 0/0", font=("Arial", 12, "bold"), width=80)
        self.lbl_slice_info.pack(side=tk.LEFT, padx=5)
        self.slider = ctk.CTkSlider(g3_2, from_=0, to=100, command=self.change_slice, state="disabled")
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        g3_3 = ctk.CTkFrame(row3, fg_color="transparent")
        g3_3.pack(side=tk.LEFT, expand=False, padx=(10, 0))
        ctk.CTkLabel(g3_3, text="Opacity:").pack(side=tk.LEFT, padx=2)
        self.slider_opacity = ctk.CTkSlider(g3_3, from_=0, to=255, command=self.change_opacity, width=120)
        self.slider_opacity.set(self.mask_opacity)
        self.slider_opacity.pack(side=tk.LEFT, padx=2)

    def _build_main_workspace(self, parent):
        view_frame = tk.PanedWindow(parent, orient=tk.HORIZONTAL, bg="#2b2d30", sashwidth=6, bd=0)
        view_frame.pack(fill=tk.BOTH, expand=True)

        left_container = ctk.CTkFrame(view_frame, width=1000)
        view_frame.add(left_container, minsize=400, width=1000, stretch="always")

        right_container = ctk.CTkFrame(view_frame, width=500)
        view_frame.add(right_container, minsize=180, width=500, stretch="never")

        overlay_header = ctk.CTkFrame(left_container, fg_color="transparent")
        overlay_header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(2, 2))
        ctk.CTkLabel(overlay_header, text="Overlay (Edit/Draw Here)", font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        def open_feedback():
            url = "https://ewunateassaye.github.io/developed-tools/medimgannotator.html"
            try:
                webbrowser.open(url)
            except Exception as ex:
                messagebox.showinfo("Feedback Link", f"Please visit:\n{url}\n\n(Notice: {ex})")

        self.btn_feedback = ctk.CTkButton(
            overlay_header,
            text="Give feedback about the tool",
            command=open_feedback,
            height=24,
            fg_color="#f0a500",
            hover_color="#d69200",
            text_color="#000000",
            font=("Arial", 11, "bold"),
        )
        self.btn_feedback.pack(side=tk.LEFT, padx=(14, 6))

        def open_paper_citation():
            paper_url = "https://arxiv.org/abs/2611.22450"
            if messagebox.askyesno(
                "Cite MedImgAnnotator",
                "If you use MedImgAnnotator in your work or research, please cite the paper as per standard citation style:\n\n"
                f"{paper_url}\n\n"
                "Open the paper in your browser now?",
            ):
                try:
                    webbrowser.open(paper_url)
                except Exception as ex:
                    messagebox.showinfo("Paper Link", f"Please visit:\n{paper_url}\n\n(Notice: {ex})")

        self.btn_cite_paper = ctk.CTkButton(
            overlay_header,
            text="Cite Paper (arXiv)",
            command=open_paper_citation,
            height=24,
            fg_color="#2b5c8f",
            hover_color="#1f4770",
            text_color="#ffffff",
            font=("Arial", 11, "bold"),
        )
        self.btn_cite_paper.pack(side=tk.LEFT, padx=(0, 8))

        overlay_container = ctk.CTkFrame(left_container, fg_color="transparent")
        overlay_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bc_layout = ctk.CTkFrame(overlay_container, fg_color="transparent", width=30)
        bc_layout.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ctk.CTkLabel(bc_layout, text="B", text_color="#4daafc", font=("Arial", 13, "bold")).pack(side=tk.TOP)
        self.sb_brightness = ctk.CTkSlider(
            bc_layout, from_=-500, to=500, orientation="vertical", command=self.update_views_event
        )
        self.sb_brightness.set(0)
        self.sb_brightness.pack(side=tk.TOP, fill=tk.Y, expand=True, pady=5)

        ctk.CTkLabel(bc_layout, text="C", text_color="#4daafc", font=("Arial", 13, "bold")).pack(side=tk.TOP)
        self.sb_contrast = ctk.CTkSlider(
            bc_layout, from_=1, to=1000, orientation="vertical", command=self.update_views_event
        )
        self.sb_contrast.set(100)
        self.sb_contrast.pack(side=tk.TOP, fill=tk.Y, expand=True, pady=5)

        self.lbl_overlay = EditorCanvas(overlay_container, editor=self)
        self.lbl_overlay.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        zoom_layout = ctk.CTkFrame(overlay_container, fg_color="transparent", width=30)
        zoom_layout.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

        ctk.CTkLabel(zoom_layout, text="Z", text_color="#4daafc", font=("Arial", 13, "bold")).pack(side=tk.TOP)
        self.sb_zoom = ctk.CTkSlider(zoom_layout, from_=10, to=500, orientation="vertical", command=self.change_zoom_event)
        self.sb_zoom.set(100)
        self.sb_zoom.pack(side=tk.TOP, fill=tk.Y, expand=True, pady=5)

        self._build_right_panel(right_container)

    def _build_right_panel(self, right_container):
        right_container.grid_rowconfigure(0, weight=55)
        right_container.grid_rowconfigure(1, weight=45)
        right_container.grid_columnconfigure(0, weight=1)

        mask_frame = ctk.CTkFrame(right_container, fg_color="transparent")
        mask_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(2, 2))
        ctk.CTkLabel(mask_frame, text="Mask Only", font=("Arial", 12, "bold")).pack(
            side=tk.TOP, anchor="w", padx=4, pady=(0, 2)
        )
        self.lbl_mask_only = PreviewCanvas(mask_frame)
        self.lbl_mask_only.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.vol_panel = ctk.CTkFrame(
            right_container, fg_color="#181a1c", corner_radius=6, border_width=1, border_color="#33373b"
        )
        self.vol_panel.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))

        vol_header = ctk.CTkFrame(self.vol_panel, fg_color="transparent")
        vol_header.pack(fill=tk.X, padx=8, pady=(6, 2))
        ctk.CTkLabel(vol_header, text="Label Volumes", font=("Arial", 12, "bold"), text_color="#ffffff").pack(
            side=tk.LEFT
        )
        self.lbl_vol_voxinfo = ctk.CTkLabel(
            vol_header, text="Voxel: 1.00×1.00×1.00 mm", font=("Arial", 10), text_color="#8c9299"
        )
        self.lbl_vol_voxinfo.pack(side=tk.RIGHT)

        self.lbl_vol_total = ctk.CTkLabel(
            self.vol_panel,
            text="Total: 0 voxels (0.00 mm³)",
            font=("Arial", 11, "bold"),
            text_color="#4daafc",
            anchor="w",
        )
        self.lbl_vol_total.pack(fill=tk.X, padx=10, pady=(0, 4))

        self.vol_scroll_frame = ctk.CTkScrollableFrame(self.vol_panel, fg_color="transparent")
        self.vol_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        self.csv_panel = ctk.CTkFrame(
            self.vol_panel, fg_color="#1e2124", corner_radius=6, border_width=1, border_color="#33373b"
        )
        self.csv_panel.pack(fill=tk.X, padx=4, pady=(2, 4))

        timer_row = ctk.CTkFrame(self.csv_panel, fg_color="transparent")
        timer_row.pack(fill=tk.X, padx=6, pady=(3, 1))
        self.lbl_active_timer = ctk.CTkLabel(
            timer_row, text="Active: ⏱ 00:00", font=("Arial", 11, "bold"), text_color="#f0a500", anchor="w"
        )
        self.lbl_active_timer.pack(side=tk.LEFT, padx=2)

        self.btn_open_csv = ctk.CTkButton(
            timer_row,
            text="Open CSV",
            width=68,
            height=22,
            font=("Arial", 10, "bold"),
            fg_color="#30363d",
            hover_color="#3d444d",
            command=self.open_time_log_csv,
        )
        self.btn_open_csv.pack(side=tk.RIGHT, padx=2)

        self.btn_export_csv = ctk.CTkButton(
            timer_row,
            text="Export CSV",
            width=76,
            height=22,
            font=("Arial", 10, "bold"),
            fg_color="#238636",
            hover_color="#2ea043",
            command=lambda: self.export_time_log_csv(show_msg=True),
        )
        self.btn_export_csv.pack(side=tk.RIGHT, padx=2)

        path_row = ctk.CTkFrame(self.csv_panel, fg_color="transparent")
        path_row.pack(fill=tk.X, padx=6, pady=(1, 4))
        ctk.CTkLabel(path_row, text="CSV Path:", font=("Arial", 10, "bold"), text_color="#8c9299").pack(
            side=tk.LEFT, padx=(2, 4)
        )
        self.entry_csv_path = ctk.CTkEntry(
            path_row, height=22, font=("Arial", 9), fg_color="#16181a", border_color="#33373b"
        )
        self.entry_csv_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.update_csv_path_display()

    def setup_slider_bindings(self, slider):
        """Attach hover tracking and scroll-wheel adjustment to a CTkSlider."""

        def on_enter(e):
            self.hovered_slider = slider

        def on_leave(e):
            if getattr(self, "hovered_slider", None) == slider:
                self.hovered_slider = None

        def on_wheel(e):
            if slider.cget("state") == "disabled":
                return "break"
            direction = 0
            if hasattr(e, "num") and e.num == 4:
                direction = 1
            elif hasattr(e, "num") and e.num == 5:
                direction = -1
            elif hasattr(e, "delta"):
                if e.delta > 0:
                    direction = 1
                elif e.delta < 0:
                    direction = -1
            if direction != 0:
                self.increment_slider(slider, direction)
            return "break"

        slider.bind("<Enter>", on_enter)
        slider.bind("<Leave>", on_leave)
        slider.bind("<MouseWheel>", on_wheel)
        slider.bind("<Button-4>", on_wheel)
        slider.bind("<Button-5>", on_wheel)

    def _is_typing_in_entry(self):
        focused = self.focus_get()
        if focused is not None:
            if isinstance(focused, (tk.Entry, tk.Text, ctk.CTkEntry)):
                return True
            if hasattr(focused, "master") and isinstance(focused.master, (ctk.CTkEntry, ctk.CTkComboBox)):
                return True
            try:
                if focused.winfo_class().lower() in ("entry", "text"):
                    return True
            except Exception:
                pass
        return False

    def setup_shortcuts(self):
        """Bind application keyboard shortcuts for undo/redo, save, ObDet, and navigation."""
        self.bind_all("<Key>", lambda e: self.record_user_activity(), add="+")
        self.bind("<Control-z>", lambda e: self.undo_action())
        self.bind("<Control-Z>", lambda e: self.undo_action())
        self.bind("<Control-y>", lambda e: self.redo_action())
        self.bind("<Control-Y>", lambda e: self.redo_action())
        self.bind("<Control-Shift-z>", lambda e: self.redo_action())
        self.bind("<Control-Shift-Z>", lambda e: self.redo_action())
        self.bind("s", lambda e: None if self._is_typing_in_entry() else self.quick_save_mask())
        self.bind("S", lambda e: None if self._is_typing_in_entry() else self.quick_save_mask())
        self.bind("<Return>", lambda e: None if self._is_typing_in_entry() else self.handle_return_key(e))
        self.bind("<Delete>", lambda e: None if self._is_typing_in_entry() else self.delete_active_obdet_box(e))
        self.bind("<BackSpace>", lambda e: None if self._is_typing_in_entry() else self.delete_active_obdet_box(e))

        self.bind("<Up>", lambda e: self.handle_arrow_keys(e, 1))
        self.bind("<Right>", lambda e: self.handle_arrow_keys(e, 1))
        self.bind("<Down>", lambda e: self.handle_arrow_keys(e, -1))
        self.bind("<Left>", lambda e: self.handle_arrow_keys(e, -1))
        self.bind("<Prior>", lambda e: self.handle_arrow_keys(e, 1))
        self.bind("<Next>", lambda e: self.handle_arrow_keys(e, -1))

    def handle_arrow_keys(self, event, direction):
        """Route arrow keys to hovered slider or default slice navigation."""
        if self._is_typing_in_entry():
            return
        self.record_user_activity()
        if hasattr(self, "hovered_slider") and self.hovered_slider is not None:
            self.increment_slider(self.hovered_slider, direction)
        else:
            self.increment_slider(self.slider, direction)
        return "break"

    def increment_slider(self, slider, direction):
        """Increment or decrement a slider by 1 slice or 5% range and invoke its command."""
        if slider.cget("state") == "disabled":
            return
        self.record_user_activity()
        val = slider.get()
        f = slider.cget("from_")
        t = slider.cget("to")

        if slider == self.slider:
            step = 1
        else:
            step = abs(t - f) * 0.05 or 1

        new_val = max(min(f, t), min(max(f, t), val + step * direction))
        slider.set(new_val)
        if slider._command:
            slider._command(new_val)
