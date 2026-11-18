"""
Plugin Manager dialog UI built on top of `medimg_annotator.core.plugin_runner`.
"""

from medimg_annotator.imports import (
    ctk,
    filedialog,
    messagebox,
    os,
    shutil,
    simpledialog,
    subprocess,
    threading,
    tk,
)

from medimg_annotator.core import plugin_runner as pr

class PluginManagerMixin:
    """Provides the Plugin Manager dialog for installing, managing dependencies, and running plugins."""

    def _build_plugin_context_env(self):
        """Collect current image/mask/output workstation paths for child plugin processes."""
        ctx = {}
        if self.file_list and 0 <= self.current_index < len(self.file_list):
            current_fname = self.file_list[self.current_index]
            ctx["MIA_ACTIVE_IMAGE"] = os.path.join(self.image_dir, current_fname) if self.image_dir else ""
            if getattr(self, "specific_mask_file", None) and os.path.exists(self.specific_mask_file):
                ctx["MIA_ACTIVE_MASK"] = self.specific_mask_file
            elif self.mask_dir and os.path.exists(self.mask_dir):
                ctx["MIA_ACTIVE_MASK"] = os.path.join(self.mask_dir, current_fname)
            ctx["MIA_OUTPUT_DIR"] = self.output_dir or ""
            ctx["MIA_CURRENT_SLICE"] = str(self.current_slice + 1)
            ctx["MIA_CURRENT_AXIS"] = str(self.axis_index)
        return ctx

    def manage_plugins(self):
        """Open or focus the workstation Plugin Manager dialog."""
        if hasattr(self, "plugin_dialog") and self.plugin_dialog is not None and self.plugin_dialog.winfo_exists():
            self.plugin_dialog.deiconify()
            self.plugin_dialog.lift()
            self.plugin_dialog.focus()
            if hasattr(self.plugin_dialog, "_refresh_func"):
                self.plugin_dialog._refresh_func()
            return

        plugin_dir = pr.get_plugin_dir()

        dialog = ctk.CTkToplevel(self)
        self.plugin_dialog = dialog
        dialog.title("Plugin Manager")
        dialog.geometry("720x480")
        dialog.transient(self)

        ctk.CTkLabel(dialog, text="Installed Workstation Plugins", font=("Arial", 14, "bold")).pack(pady=(12, 4))
        lbl_status = ctk.CTkLabel(dialog, text="Ready", font=("Arial", 10), text_color="#888888")
        lbl_status.pack(pady=(0, 6))

        scroll_frame = ctk.CTkScrollableFrame(dialog, width=590, height=270)
        scroll_frame.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)

        def get_interpreter_cached():
            cached = getattr(self, "_cached_python_exec", None)
            resolved = pr.find_python_interpreter(cached)
            self._cached_python_exec = resolved
            return resolved

        def get_plugin_interp(target_path):
            return pr.get_plugin_interpreter(target_path, default_interpreter=get_interpreter_cached())

        def prompt_requirements_install(req_file, target_dir, source_label):
            if req_file and dialog.winfo_exists():
                def _ask():
                    if dialog.winfo_exists():
                        ans = messagebox.askyesno(
                            "Install Required Libraries?",
                            f"Plugin {source_label} specifies library requirements in 'requirements.txt'.\n\n"
                            "Would you like to install the required Python dependencies now using pip?",
                            parent=dialog,
                        )
                        if ans:
                            pr.launch_pip_install(req_file, get_plugin_interp(target_dir), cwd=os.path.dirname(req_file))

                dialog.after(350, _ask)

        def populate_ui(runnables, status_override=None):
            if not dialog.winfo_exists():
                return
            for widget in scroll_frame.winfo_children():
                widget.destroy()

            if status_override:
                lbl_status.configure(text=status_override, text_color="#2fa572")
            else:
                lbl_status.configure(text=f"Found {len(runnables)} plugin(s) ready", text_color="#888888")

            if not runnables:
                ctk.CTkLabel(
                    scroll_frame,
                    text="No plugins installed.\nUse 'Install File / Zip' or 'Install Folder' to add runnable tools.",
                    text_color="gray",
                ).pack(pady=40)
                return

            for file_name, full_path, rel_path in runnables:
                item_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
                item_frame.pack(fill=tk.X, pady=4, padx=5)

                loc_info = os.path.dirname(rel_path)
                lbl_text = f"{file_name} [{loc_info if loc_info else 'Root'}]"
                ctk.CTkLabel(item_frame, text=lbl_text, anchor="w", font=("Segoe UI", 11)).pack(
                    side=tk.LEFT, fill=tk.X, expand=True
                )

                btn_run = ctk.CTkButton(
                    item_frame, text="Run", width=55, height=26, fg_color="#1f6aa5", hover_color="#144870"
                )
                btn_run.pack(side=tk.RIGHT, padx=3)

                plugin_folder = os.path.dirname(full_path)
                candidate_req = None
                for check_dir in (plugin_folder, os.path.dirname(plugin_folder)):
                    if (
                        check_dir
                        and os.path.isdir(check_dir)
                        and os.path.commonpath([check_dir, plugin_dir]) == plugin_dir
                    ):
                        rf = os.path.join(check_dir, "requirements.txt")
                        if os.path.isfile(rf):
                            candidate_req = rf
                            break

                if candidate_req:
                    btn_deps = ctk.CTkButton(
                        item_frame, text="Install Deps", width=80, height=26, fg_color="#455a64", hover_color="#37474f"
                    )
                    btn_deps.pack(side=tk.RIGHT, padx=3)
                    btn_deps.configure(
                        command=lambda rf=candidate_req, tp=full_path: pr.launch_pip_install(
                            rf, get_plugin_interp(tp), cwd=os.path.dirname(rf)
                        )
                    )

                def make_run_cmd(target_p=full_path, b=btn_run):
                    def _launch():
                        b.configure(text="Running...", state="disabled")

                        def _worker():
                            try:
                                ctx = self._build_plugin_context_env()
                                pr.launch_plugin_process(target_p, get_plugin_interp(target_p), context_vars=ctx)
                            except Exception as err:
                                if dialog.winfo_exists():
                                    dialog.after(
                                        0,
                                        lambda e=err: messagebox.showerror(
                                            "Run Failure", f"Could not launch plugin:\n{str(e)}", parent=dialog
                                        ),
                                    )
                            finally:
                                if dialog.winfo_exists():
                                    dialog.after(
                                        800,
                                        lambda: b.configure(text="Run", state="normal")
                                        if dialog.winfo_exists()
                                        else None,
                                    )

                        threading.Thread(target=_worker, daemon=True).start()

                    return _launch

                btn_run.configure(command=make_run_cmd())

        def refresh_list(status_override=None):
            os.makedirs(plugin_dir, exist_ok=True)
            if not status_override:
                lbl_status.configure(text="Scanning plugins...", text_color="#1f6aa5")
            for widget in scroll_frame.winfo_children():
                widget.destroy()
            ctk.CTkLabel(scroll_frame, text="Loading plugins...", text_color="gray").pack(pady=40)

            def _scan_worker():
                try:
                    runnables = pr.scan_plugins(plugin_dir)
                except Exception as e:
                    print(f"Plugin scan error: {e}")
                    runnables = []
                if dialog.winfo_exists():
                    dialog.after(0, lambda: populate_ui(runnables, status_override))

            threading.Thread(target=_scan_worker, daemon=True).start()

        dialog._refresh_func = refresh_list

        def install_file():
            selected_file = filedialog.askopenfilename(
                title="Select Plugin File or Archive",
                filetypes=[
                    ("Plugin Files & Archives", "*.zip;*.py;*.exe;*.bat;*.tar.gz;*.tgz"),
                    ("Zip Archives (*.zip)", "*.zip"),
                    ("Python Scripts (*.py)", "*.py"),
                    ("Executables (*.exe;*.bat)", "*.exe;*.bat"),
                    ("All Files", "*.*"),
                ],
                parent=dialog,
            )
            if not selected_file:
                return

            selected_file = os.path.normpath(os.path.abspath(selected_file))
            base_name = os.path.basename(selected_file)
            target_dest = os.path.normpath(os.path.abspath(os.path.join(plugin_dir, base_name)))
            if selected_file == target_dest:
                refresh_list(status_override=f"'{base_name}' is already in plugins folder.")
                return

            lbl_status.configure(text=f"Installing {base_name}...", text_color="#1f6aa5")

            def _worker():
                try:
                    os.makedirs(plugin_dir, exist_ok=True)
                    detected_req = pr.install_plugin_archive_or_file(selected_file, plugin_dir)
                    if dialog.winfo_exists():
                        dialog.after(0, lambda: refresh_list(status_override=f"✓ Installed '{base_name}' successfully!"))
                        if detected_req:
                            prompt_requirements_install(detected_req, os.path.dirname(detected_req), f"'{base_name}'")
                except Exception as ex:
                    if dialog.winfo_exists():
                        err_msg = str(ex)
                        dialog.after(0, lambda: lbl_status.configure(text=f"Install error: {err_msg[:45]}", text_color="#e74c3c"))

            threading.Thread(target=_worker, daemon=True).start()

        def install_folder():
            selected_dir = filedialog.askdirectory(
                title="Select Plugin Folder (Contains Scripts, Models, etc.)", parent=dialog
            )
            if not selected_dir:
                return

            selected_dir = os.path.normpath(os.path.abspath(selected_dir))
            folder_name = os.path.basename(selected_dir)
            target_dest = os.path.normpath(os.path.abspath(os.path.join(plugin_dir, folder_name)))
            if selected_dir == target_dest:
                refresh_list(status_override=f"Folder '{folder_name}' is already in plugins folder.")
                return

            lbl_status.configure(text=f"Installing folder '{folder_name}'...", text_color="#1f6aa5")

            def _worker():
                try:
                    os.makedirs(plugin_dir, exist_ok=True)
                    shutil.copytree(selected_dir, target_dest, dirs_exist_ok=True)
                    detected_req = pr.find_requirements_in_tree(target_dest)
                    if dialog.winfo_exists():
                        dialog.after(0, lambda: refresh_list(status_override=f"✓ Installed folder '{folder_name}' successfully!"))
                        if detected_req:
                            prompt_requirements_install(detected_req, target_dest, f"folder '{folder_name}'")
                except Exception as ex:
                    if dialog.winfo_exists():
                        err_msg = str(ex)
                        dialog.after(0, lambda: lbl_status.configure(text=f"Install error: {err_msg[:45]}", text_color="#e74c3c"))

            threading.Thread(target=_worker, daemon=True).start()

        def open_folder():
            try:
                os.makedirs(plugin_dir, exist_ok=True)
                if hasattr(os, "startfile"):
                    os.startfile(plugin_dir)
                else:
                    subprocess.Popen(["xdg-open", plugin_dir])
            except Exception:
                pass

        def prompt_install_package():
            pkg = simpledialog.askstring(
                "Install Python Library",
                "Enter library / package name to install (e.g. monai, einops, nnunetv2):\n\nOr provide full path to a requirements.txt file:",
                parent=dialog,
            )
            if pkg and pkg.strip():
                pkg_str = pkg.strip()
                pr.launch_pip_install(pkg_str, get_interpreter_cached(), cwd=plugin_dir)
                lbl_status.configure(text=f"Launched pip installer for '{pkg_str}'", text_color="#2fa572")

        bottom_layout = ctk.CTkFrame(dialog, fg_color="transparent")
        bottom_layout.pack(fill=tk.X, pady=12, padx=10)

        ctk.CTkButton(bottom_layout, text="Install File / Zip", command=install_file, width=120, height=32).pack(
            side=tk.LEFT, padx=3, expand=True
        )
        ctk.CTkButton(bottom_layout, text="Install Folder", command=install_folder, width=105, height=32).pack(
            side=tk.LEFT, padx=3, expand=True
        )
        ctk.CTkButton(
            bottom_layout,
            text="Install Pip Library",
            command=prompt_install_package,
            width=125,
            height=32,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
        ).pack(side=tk.LEFT, padx=3, expand=True)
        ctk.CTkButton(
            bottom_layout,
            text="Refresh",
            command=lambda: refresh_list(),
            width=65,
            height=32,
            fg_color="#3a3f45",
            hover_color="#2a2e32",
        ).pack(side=tk.LEFT, padx=3, expand=True)
        ctk.CTkButton(
            bottom_layout,
            text="Open Folder",
            command=open_folder,
            width=95,
            height=32,
            fg_color="#3a3f45",
            hover_color="#2a2e32",
        ).pack(side=tk.LEFT, padx=3, expand=True)
        ctk.CTkButton(
            bottom_layout,
            text="Close",
            command=dialog.destroy,
            width=60,
            height=32,
            fg_color="#3a3f45",
            hover_color="#2a2e32",
        ).pack(side=tk.LEFT, padx=3, expand=True)

        refresh_list()
