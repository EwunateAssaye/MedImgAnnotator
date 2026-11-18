"""
Unified single-panel dialog allowing selection of either a folder (batch) or a single image/mask file.
"""

from medimg_annotator.imports import ctk, datetime, os, sys, tk, ttk

from medimg_annotator.core import image_io as uio

class UniversalInputDialog(ctk.CTkToplevel):
    """
    Universal File & Folder Browser Dialog for MedImgAnnotator.
    Allows browsing both folders and files in a unified view:
      - Double-click on a folder: opens and navigates inside the folder.
      - Single-click on a folder: selects that folder (loads entire folder for batch processing on 'Select').
      - Double-click on a file: immediately selects and loads that single image.
      - Single-click on a file: selects that single image for processing.
      - If no item is selected, clicking 'Select' loads the currently opened folder for batch processing.
    """

    def __init__(self, parent, initial_dir=None, title="Select Input Folder or Image"):
        super().__init__(parent)
        self.title(title)
        self.geometry("820x560")
        self.minsize(680, 420)

        self.parent = parent
        self.result = None

        if initial_dir and os.path.exists(initial_dir):
            if os.path.isfile(initial_dir):
                self.current_dir = os.path.abspath(os.path.dirname(initial_dir))
            else:
                self.current_dir = os.path.abspath(initial_dir)
        else:
            self.current_dir = os.path.abspath(os.path.expanduser("~"))

        self.selected_target = None
        self.is_folder_selected = True
        self.is_mask_mode = "mask" in str(title).lower()
        self.supported_exts = tuple(uio.get_supported_extensions())

        self._init_ui()
        self._setup_style()
        self.navigate_to(self.current_dir)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.transient(parent)
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))
        self.grab_set()
        self.focus_force()

        self.update_idletasks()
        try:
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            w, h = 820, 560
            x = px + max(0, (pw - w) // 2)
            y = py + max(0, (ph - h) // 2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

        self.wait_window(self)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Custom.Treeview",
            background="#1e1e1e",
            foreground="#e0e0e0",
            fieldbackground="#1e1e1e",
            borderwidth=0,
            rowheight=26,
            font=("Segoe UI", 10),
        )
        style.map(
            "Custom.Treeview",
            background=[("selected", "#1f6aa5")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#2b2d30",
            foreground="#ffffff",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Custom.Treeview.Heading", background=[("active", "#3a3d40")])

    def _init_ui(self):
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

        ctk.CTkButton(
            top_frame, text="▲ Up", width=60, command=self.go_up, fg_color="#3a3d40", hover_color="#4f5256"
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.entry_path = ctk.CTkEntry(top_frame, font=("Segoe UI", 11))
        self.entry_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.entry_path.bind("<Return>", lambda e: self.on_path_entered())

        ctk.CTkButton(
            top_frame, text="Go", width=45, command=self.on_path_entered, fg_color="#3a3d40", hover_color="#4f5256"
        ).pack(side=tk.LEFT, padx=(0, 6))

        ctk.CTkButton(
            top_frame,
            text="⟳",
            width=36,
            command=lambda: self.navigate_to(self.current_dir),
            fg_color="#3a3d40",
            hover_color="#4f5256",
        ).pack(side=tk.LEFT)

        mid_frame = ctk.CTkFrame(self, fg_color="transparent")
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        sidebar = ctk.CTkFrame(mid_frame, width=150, fg_color="#252526", corner_radius=6)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Quick Access", font=("Segoe UI", 11, "bold"), text_color="#aaaaaa").pack(
            anchor="w", padx=10, pady=(8, 4)
        )

        quick_places = [
            ("📁 Desktop", os.path.expanduser("~/Desktop")),
            ("📥 Downloads", os.path.expanduser("~/Downloads")),
            ("📄 Documents", os.path.expanduser("~/Documents")),
            ("🖼️ Pictures", os.path.expanduser("~/Pictures")),
        ]

        if sys.platform == "win32":
            from medimg_annotator.imports import string

            for drive in string.ascii_uppercase:
                dp = f"{drive}:\\"
                if os.path.exists(dp):
                    quick_places.append((f"💾 Drive ({drive}:)", dp))

        for label, path in quick_places:
            if os.path.exists(path):
                btn = ctk.CTkButton(
                    sidebar,
                    text=label,
                    anchor="w",
                    fg_color="transparent",
                    hover_color="#333333",
                    text_color="#dddddd",
                    height=28,
                    command=lambda p=path: self.navigate_to(p),
                )
                btn.pack(fill=tk.X, padx=4, pady=1)

        table_frame = ctk.CTkFrame(mid_frame, fg_color="#1e1e1e", corner_radius=6)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("name", "type", "size", "date")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", style="Custom.Treeview", selectmode="browse"
        )

        self.tree.heading("name", text="Name", anchor="w")
        self.tree.heading("type", text="Type", anchor="w")
        self.tree.heading("size", text="Size", anchor="e")
        self.tree.heading("date", text="Date Modified", anchor="w")

        self.tree.column("name", width=330, anchor="w")
        self.tree.column("type", width=120, anchor="w")
        self.tree.column("size", width=80, anchor="e")
        self.tree.column("date", width=130, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)

        self.tree.bind("<Double-Button-1>", self.on_double_click)
        self.tree.bind("<<TreeviewSelect>>", self.on_item_selected)

        bot_frame = ctk.CTkFrame(self, fg_color="transparent")
        bot_frame.pack(fill=tk.X, padx=10, pady=(6, 10))

        sel_row = ctk.CTkFrame(bot_frame, fg_color="transparent")
        sel_row.pack(fill=tk.X, pady=(0, 6))

        ctk.CTkLabel(sel_row, text="Selection:", font=("Segoe UI", 11, "bold"), width=70, anchor="w").pack(side=tk.LEFT)
        self.entry_selection = ctk.CTkEntry(sel_row, font=("Segoe UI", 11))
        self.entry_selection.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 8))
        self.entry_selection.bind("<Return>", lambda e: self.confirm_selection())

        self.lbl_status_hint = ctk.CTkLabel(
            bot_frame,
            text="Tip: Single-click folder to select entire folder; double-click folder to open inside.",
            font=("Segoe UI", 10),
            text_color="#00e5ff",
            anchor="w",
        )
        self.lbl_status_hint.pack(fill=tk.X, pady=(0, 6))

        btn_row = ctk.CTkFrame(bot_frame, fg_color="transparent")
        btn_row.pack(fill=tk.X)

        self.filter_var = tk.StringVar(value="All Supported Images & Volumes")
        ctk.CTkOptionMenu(
            btn_row,
            values=["All Supported Images & Volumes", "All Files (*.*)"],
            variable=self.filter_var,
            command=lambda v: self.refresh_items(),
            width=230,
            fg_color="#3a3d40",
        ).pack(side=tk.LEFT)

        ctk.CTkButton(
            btn_row, text="Cancel", width=80, command=self.cancel, fg_color="#444444", hover_color="#555555"
        ).pack(side=tk.RIGHT, padx=(6, 0))

        default_select_text = "Select Mask Folder" if self.is_mask_mode else "Select Folder"
        self.btn_select = ctk.CTkButton(
            btn_row,
            text=default_select_text,
            width=120,
            command=self.confirm_selection,
            fg_color="#1f6aa5",
            hover_color="#144870",
        )
        self.btn_select.pack(side=tk.RIGHT, padx=(6, 0))

        use_folder_txt = "📁 Use Current Mask Folder" if self.is_mask_mode else "📁 Use Current Folder (Batch)"
        self.btn_use_cur_dir = ctk.CTkButton(
            btn_row,
            text=use_folder_txt,
            width=190,
            command=self.confirm_current_folder,
            fg_color="#2b5b84",
            hover_color="#1d3f5d",
        )
        self.btn_use_cur_dir.pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda e: self.cancel())

    def navigate_to(self, target_dir):
        """Change active directory and default current selection to that folder."""
        if not target_dir or not os.path.exists(target_dir):
            return
        if os.path.isfile(target_dir):
            target_dir = os.path.dirname(target_dir)

        self.current_dir = os.path.abspath(target_dir)
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, self.current_dir)

        self.selected_target = self.current_dir
        self.is_folder_selected = True
        self.entry_selection.delete(0, tk.END)
        self.entry_selection.insert(0, os.path.basename(self.current_dir) or self.current_dir)
        action_desc = "as Mask Folder" if self.is_mask_mode else "for Batch Processing"
        btn_action = "Select Mask Folder" if self.is_mask_mode else "Select Folder"
        self.btn_select.configure(text=btn_action)
        self.lbl_status_hint.configure(
            text=f"📁 Opened Folder: {os.path.basename(self.current_dir) or self.current_dir} (Click '{btn_action}' to use {action_desc})",
            text_color="#00e5ff",
        )

        self.refresh_items()

    def go_up(self):
        """Navigate to the parent directory."""
        parent_dir = os.path.dirname(self.current_dir)
        if parent_dir and parent_dir != self.current_dir and os.path.exists(parent_dir):
            self.navigate_to(parent_dir)

    def on_path_entered(self):
        """Navigate directly to an address bar path entered by the user."""
        entered = self.entry_path.get().strip()
        if os.path.exists(entered):
            self.navigate_to(entered)
        else:
            self.lbl_status_hint.configure(text=f"⚠️ Path not found: {entered}", text_color="#ff5555")

    def refresh_items(self):
        """Scan current_dir and populate the Treeview respecting active extension filter."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        show_all = self.filter_var.get() == "All Files (*.*)"
        folders = []
        files = []

        try:
            with os.scandir(self.current_dir) as entries:
                for entry in entries:
                    try:
                        name = entry.name
                        if name.startswith("$") or name.startswith("System Volume Information"):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stat = entry.stat()
                            mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                            folders.append((name, entry.path, mtime))
                        elif entry.is_file(follow_symlinks=False):
                            lower_name = name.lower()
                            is_supp = any(lower_name.endswith(ext) for ext in self.supported_exts)
                            if show_all or is_supp:
                                stat = entry.stat()
                                size_str = self._format_size(stat.st_size)
                                mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                                ftype = self._get_file_type(name)
                                files.append((name, entry.path, ftype, size_str, mtime))
                    except Exception:
                        continue
        except Exception as e:
            self.lbl_status_hint.configure(text=f"⚠️ Access error: {str(e)}", text_color="#ff5555")
            return

        folders.sort(key=lambda x: x[0].lower())
        files.sort(key=lambda x: x[0].lower())

        for name, fullpath, mtime in folders:
            item_id = self.tree.insert("", tk.END, values=(f"📁  {name}", "File folder", "", mtime))
            self.tree.item(item_id, tags=("folder", fullpath))

        for name, fullpath, ftype, size_str, mtime in files:
            icon = (
                "🖼️  "
                if any(ftype.startswith(k) for k in ("NIfTI", "PNG", "JPEG", "TIFF", "DICOM", "Bitmap", "WebP"))
                else "📄  "
            )
            item_id = self.tree.insert("", tk.END, values=(f"{icon}{name}", ftype, size_str, mtime))
            self.tree.item(item_id, tags=("file", fullpath))

    def _format_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        if size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def _get_file_type(self, filename):
        lower = filename.lower()
        for suffixes, label in (
            ((".nii.gz", ".nii"), "NIfTI Image"),
            ((".dcm",), "DICOM Image"),
            ((".png",), "PNG Image"),
            ((".jpg", ".jpeg"), "JPEG Image"),
            ((".tif", ".tiff"), "TIFF Image"),
            ((".bmp",), "Bitmap Image"),
            ((".webp",), "WebP Image"),
        ):
            if any(lower.endswith(s) for s in suffixes):
                return label
        ext = os.path.splitext(filename)[1].upper()
        return f"{ext[1:]} File" if ext else "File"

    def on_item_selected(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        tags = self.tree.item(sel[0], "tags")
        if not tags or len(tags) < 2:
            return

        item_type, full_path = tags[0], tags[1]
        self.selected_target = full_path
        base_name = os.path.basename(full_path)
        self.entry_selection.delete(0, tk.END)
        self.entry_selection.insert(0, base_name)

        if item_type == "folder":
            self.is_folder_selected = True
            action_desc = "as Mask Folder" if self.is_mask_mode else "for Batch Processing"
            self.lbl_status_hint.configure(
                text=f"📁 Selected Folder: {base_name} (Click 'Select' to use {action_desc}, or double-click to open)",
                text_color="#00ff66",
            )
            self.btn_select.configure(text="Select Folder")
        else:
            self.is_folder_selected = False
            item_desc = "this mask file" if self.is_mask_mode else "this single image"
            btn_txt = "Select Mask" if self.is_mask_mode else "Select Image"
            self.lbl_status_hint.configure(
                text=f"📄 Selected File: {base_name} (Click 'Select' or double-click to choose {item_desc})",
                text_color="#00e5ff",
            )
            self.btn_select.configure(text=btn_txt)

    def on_double_click(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        tags = self.tree.item(sel[0], "tags")
        if not tags or len(tags) < 2:
            return

        item_type, full_path = tags[0], tags[1]
        if item_type == "folder":
            self.navigate_to(full_path)
        else:
            self.selected_target = full_path
            self.is_folder_selected = False
            self.confirm_selection()

    def confirm_selection(self):
        user_text = self.entry_selection.get().strip()
        candidate = None

        if user_text:
            if user_text == self.current_dir or user_text == os.path.basename(self.current_dir):
                candidate = self.current_dir
            elif self.selected_target and user_text == os.path.basename(self.selected_target):
                candidate = self.selected_target
            elif os.path.exists(user_text):
                candidate = os.path.abspath(user_text)
            else:
                direct_candidate = os.path.join(self.current_dir, user_text)
                if os.path.exists(direct_candidate):
                    candidate = direct_candidate

        if candidate is None:
            candidate = self.selected_target or self.current_dir

        if not os.path.exists(candidate):
            self.lbl_status_hint.configure(text=f"⚠️ Selection does not exist: {candidate}", text_color="#ff5555")
            return

        self.result = {"path": os.path.abspath(candidate), "is_folder": os.path.isdir(candidate)}
        self._close_dialog()

    def confirm_current_folder(self):
        if self.current_dir and os.path.exists(self.current_dir) and os.path.isdir(self.current_dir):
            self.result = {"path": os.path.abspath(self.current_dir), "is_folder": True}
            self._close_dialog()

    def cancel(self):
        self.result = None
        self._close_dialog()

    def _close_dialog(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def get_result(self):
        """Return {'path': str, 'is_folder': bool} or None if cancelled."""
        return self.result
