"""
Input/output workflows: universal file/folder loading, 3-tier pairing, saving, and ImPrep bridge.
"""

from medimg_annotator.imports import (
    difflib,
    filedialog,
    Image,
    messagebox,
    nib,
    np,
    os,
    re,
    threading,
    tk,
)

from medimg_annotator.core import image_io as uio
from medimg_annotator.core import obdet_io
from medimg_annotator.core.config import Settings
from medimg_annotator.ui.universal_input_dialog import UniversalInputDialog

class IOControllerMixin:
    """Manages input loading, fuzzy mask pairing, saving masks/JPEG overlays, and ImPrep bridge."""

    def show_save_menu(self, event=None):
        """Show context menu for saving segmentation masks, slice JPEG overlays, or ObDet boxes."""
        if self.img_data is None:
            return
        self.record_user_activity()
        menu = tk.Menu(
            self, tearoff=0, bg="#2b2d30", fg="#ffffff", activebackground="#1f6aa5", activeforeground="#ffffff", bd=1
        )
        menu.add_command(label="Save Mask (Default)  [S]", command=self.quick_save_mask)
        menu.add_command(label="Save Current Slice with Overlay (JPEG)...", command=self.save_current_slice_overlay_jpeg)
        if getattr(self, "obdet_active", False) and getattr(self, "obdet_bbox", None) is not None:
            menu.add_command(label="Save ObDet Bounding Box Image", command=self.save_obdet_image)

        if event is not None and hasattr(event, "x_root") and hasattr(event, "y_root"):
            x, y = event.x_root, event.y_root
        else:
            x = self.btn_quick_save.winfo_rootx()
            y = self.btn_quick_save.winfo_rooty() + self.btn_quick_save.winfo_height() + 2
        menu.tk_popup(x, y)

    def save_current_slice_overlay_jpeg(self):
        """Export the current anatomical slice with overlaid colored masks as a JPEG image."""
        if self.img_data is None:
            messagebox.showwarning("No Image", "No image is currently loaded.")
            return

        self.record_user_activity()
        canvas = self.lbl_overlay
        if canvas._img_slice_8 is not None:
            src_h, src_w = canvas._img_slice_8.shape[:2]
            composite_pil = canvas._render_scaled_composite(src_w, src_h)
        elif canvas.current_image is not None:
            composite_pil = canvas.current_image
        else:
            messagebox.showwarning("Render Error", "Could not render current slice image.")
            return

        if getattr(self, "obdet_active", False) and getattr(self, "obdet_boxes", None):
            cur_boxes = [
                b
                for b in self.obdet_boxes
                if b.get("slice", 0) == self.current_slice and b.get("axis", 2) == self.axis_index
            ]
            if cur_boxes:
                comp_rgb = np.array(composite_pil.convert("RGB"))
                comp_rgb, _ = obdet_io.burn_bounding_box_on_image(comp_rgb, slice_idx=0, axis=2, boxes=cur_boxes)
                composite_pil = Image.fromarray(comp_rgb)

        cur_fname = (
            self.file_list[self.current_index]
            if (self.file_list and 0 <= self.current_index < len(self.file_list))
            else "image"
        )
        stem = cur_fname
        for ext in (".nii.gz", ".nii", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".nrrd", ".mha"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break

        ax_str = {0: "sag", 1: "cor", 2: "ax"}.get(self.axis_index, "slice")
        default_name = f"{stem}_{ax_str}{self.current_slice + 1}_overlay.jpg"
        init_dir = (
            self.output_dir
            if (self.output_dir and os.path.exists(self.output_dir))
            else (self.image_dir if (self.image_dir and os.path.exists(self.image_dir)) else os.path.expanduser("~"))
        )

        save_path = filedialog.asksaveasfilename(
            title="Save Current Slice with Overlaid Mask as JPEG",
            initialdir=init_dir,
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG Image (*.jpg;*.jpeg)", "*.jpg;*.jpeg"), ("All Files (*.*)", "*.*")],
        )
        if not save_path:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            composite_pil.convert("RGB").save(save_path, "JPEG", quality=95)
            messagebox.showinfo("Saved", f"Current slice with overlaid colored masks saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save JPEG:\n{str(e)}")

    def _build_header_snapshot(self, mask_dtype):
        """Clone NIfTI header safely for saving mask arrays."""
        header_snap = self.header.copy() if (self.header is not None and hasattr(self.header, "copy")) else nib.Nifti1Header()
        if hasattr(header_snap, "set_data_dtype"):
            header_snap.set_data_dtype(mask_dtype)
        try:
            header_snap.set_slope_inter(None, None)
        except Exception:
            pass
        return header_snap

    def quick_save_mask(self):
        """Save current segmentation mask (or ObDet boxes when ObDet mode is active)."""
        if getattr(self, "obdet_active", False) and (
            bool(getattr(self, "obdet_boxes", [])) or getattr(self, "obdet_bbox", None) is not None
        ):
            self.save_obdet_image()
            return

        if self.mask_data is None:
            return
        filename = (
            self.file_list[self.current_index]
            if (self.file_list and 0 <= self.current_index < len(self.file_list))
            else "mask.nii.gz"
        )
        if self.output_dir and os.path.exists(self.output_dir) and self.file_list:
            path = os.path.join(self.output_dir, filename)
            self.btn_quick_save.configure(text="Saving...", fg_color="#c97116", text_color="white")
            mask_snap = np.copy(self.mask_data)
            affine_snap = np.copy(self.affine) if self.affine is not None else np.eye(4)
            header_snap = self._build_header_snapshot(mask_snap.dtype)

            def _save_worker():
                try:
                    uio.write_universal_image(
                        mask_snap,
                        path,
                        affine=affine_snap,
                        header=header_snap,
                        format_info=self.current_file_format_info,
                    )
                    self.after(0, self.set_saved_status)
                    try:
                        self.after(0, lambda: self.export_time_log_csv(show_msg=False))
                    except Exception:
                        pass
                except Exception as e:
                    def _fail():
                        self.btn_quick_save.configure(text="Save Failed", fg_color="#bf3434", text_color="white")
                        messagebox.showerror("Save Error", f"Failed to save:\n{str(e)}")

                    self.after(0, _fail)

            threading.Thread(target=_save_worker, daemon=True).start()
        else:
            base_path = (
                self.mask_dir
                if (self.mask_dir and os.path.exists(self.mask_dir))
                else (self.image_dir if (self.image_dir and os.path.exists(self.image_dir)) else os.path.expanduser("~"))
            )
            path = filedialog.asksaveasfilename(title="Save Mask As", initialdir=base_path, initialfile=filename)
            if path:
                try:
                    m_snap = np.copy(self.mask_data)
                    aff_snap = np.copy(self.affine) if self.affine is not None else np.eye(4)
                    h_snap = self._build_header_snapshot(m_snap.dtype)
                    uio.write_universal_image(
                        m_snap,
                        path,
                        affine=aff_snap,
                        header=h_snap,
                        format_info=self.current_file_format_info,
                    )
                    self.set_saved_status()
                    try:
                        self.export_time_log_csv(show_msg=False)
                    except Exception:
                        pass
                except Exception as e:
                    messagebox.showerror("Error", str(e))

    def select_folders(self):
        """Open unified dialog to choose either an image folder (batch) or a single image."""
        self.record_user_activity()
        settings = Settings()
        init_dir = self.image_dir or settings.value("last_image_dir", "")
        if not init_dir or not os.path.exists(init_dir):
            init_dir = os.path.expanduser("~")

        dialog = UniversalInputDialog(self, initial_dir=init_dir, title="Select Input Image or Folder")
        result = dialog.get_result()
        if not result or not result.get("path"):
            return

        chosen_path = result["path"]
        if result.get("is_folder", False):
            self.image_dir = chosen_path
            settings.setValue("last_image_dir", chosen_path)
            self._prompt_and_load_masks(chosen_path, start_file=None, single_image=False)
        else:
            img_dir = os.path.dirname(chosen_path)
            start_file = os.path.basename(chosen_path)
            self.image_dir = img_dir
            settings.setValue("last_image_dir", img_dir)
            self._prompt_and_load_masks(img_dir, start_file=start_file, single_image=True)

    def select_batch_folder_direct(self):
        """Prompt natively for a batch image directory."""
        settings = Settings()
        init_dir = self.image_dir or settings.value("last_image_dir", "")
        img_dir = filedialog.askdirectory(
            title="Select Batch Folder for Processing",
            initialdir=init_dir if (init_dir and os.path.exists(init_dir)) else None,
        )
        if not img_dir:
            return
        self.image_dir = img_dir
        settings.setValue("last_image_dir", img_dir)
        self._prompt_and_load_masks(img_dir, start_file=None, single_image=False)

    def select_single_image_direct(self):
        """Prompt natively for a single input image file."""
        settings = Settings()
        init_dir = self.image_dir or settings.value("last_image_dir", "")
        ext_patterns = " ".join([f"*{ext}" for ext in uio.get_supported_extensions()])
        filetypes = [("All Supported Images & Volumes", ext_patterns), ("All Files (*.*)", "*.*")]
        chosen_path = filedialog.askopenfilename(
            title="Select Single Image for Processing",
            initialdir=init_dir if (init_dir and os.path.exists(init_dir)) else None,
            filetypes=filetypes,
        )
        if not chosen_path:
            return
        img_dir = os.path.dirname(chosen_path)
        start_file = os.path.basename(chosen_path)
        self.image_dir = img_dir
        settings.setValue("last_image_dir", img_dir)
        self._prompt_and_load_masks(img_dir, start_file=start_file, single_image=True)

    def _prompt_and_load_masks(self, img_dir, start_file=None, single_image=False):
        """Prompt optionally for existing masks (folder or file) and trigger volume scan."""
        settings = Settings()
        self.single_image_mode = bool(single_image)
        self.specific_mask_file = None
        self.initial_start_file = start_file

        if messagebox.askyesno("Existing Masks?", "Do you have existing masks?"):
            init_mask = self.mask_dir or settings.value("last_mask_dir", "") or img_dir
            if not init_mask or not os.path.exists(init_mask):
                init_mask = os.path.expanduser("~")

            dialog = UniversalInputDialog(self, initial_dir=init_mask, title="Select Existing Mask Folder or File")
            result = dialog.get_result()

            if result and result.get("path"):
                chosen_mask_path = result["path"]
                if result.get("is_folder", False):
                    self.mask_dir = chosen_mask_path
                    self.specific_mask_file = None
                    settings.setValue("last_mask_dir", chosen_mask_path)
                else:
                    self.mask_dir = os.path.dirname(chosen_mask_path)
                    self.specific_mask_file = chosen_mask_path
                    settings.setValue("last_mask_dir", self.mask_dir)
            else:
                self.mask_dir = None
                self.specific_mask_file = None
        else:
            self.mask_dir = None
            self.specific_mask_file = None

        self.scan_pairs(start_file=start_file, single_image=single_image)
        self.update_csv_path_display()

    def select_output_folder(self):
        """Choose output destination folder for masks, ObDet exports, and time logs."""
        settings = Settings()
        init_out = self.output_dir or settings.value("last_output_dir", "") or self.image_dir
        out_dir = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=init_out if (init_out and os.path.exists(init_out)) else None,
        )
        if out_dir:
            self.output_dir = out_dir
            settings.setValue("last_output_dir", out_dir)
            self.update_csv_path_display()
            messagebox.showinfo(
                "Output Set", f"Output folder set to:\n{out_dir}\n\nYou can now quick-save using 'S'."
            )

    def get_best_mask_match(self, img_name, mask_files):
        """Match an image filename to a mask via exact, numeric-ID, or fuzzy sequence similarity."""
        if img_name in mask_files:
            return img_name
        img_nums = re.findall(r"\d+", img_name)
        for m_name in mask_files:
            if img_nums and img_nums == re.findall(r"\d+", m_name):
                return m_name
        best_match = None
        best_ratio = 0.0
        for m_name in mask_files:
            ratio = difflib.SequenceMatcher(None, img_name, m_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = m_name
        return best_match if best_ratio > 0.6 else None

    def scan_pairs(self, start_file=None, single_image=False):
        """Enumerate supported volumes in image_dir and load the initial case."""
        try:
            img_files = os.listdir(self.image_dir)
            valid_exts = uio.get_supported_extensions()
            img_set = {
                f
                for f in img_files
                if os.path.isfile(os.path.join(self.image_dir, f)) and any(f.lower().endswith(ext) for ext in valid_exts)
            }
            common_files = sorted(list(img_set))
        except Exception as e:
            messagebox.showerror("Error", f"Read failed: {e}")
            return

        if not common_files:
            messagebox.showwarning("No Data", "No matching files found.")
            self.file_list = []
            self.update_nav_buttons()
            return

        if single_image and start_file:
            self.file_list = [start_file]
            start_idx = 0
        else:
            self.file_list = common_files
            settings = Settings()
            last_file = settings.value("last_file", "")

            start_idx = 0
            if start_file and start_file in self.file_list:
                start_idx = self.file_list.index(start_file)
            elif self.output_dir and os.path.exists(self.output_dir):
                for i, fname in enumerate(self.file_list):
                    if not os.path.exists(os.path.join(self.output_dir, fname)):
                        start_idx = i
                        break
            elif last_file in self.file_list:
                start_idx = self.file_list.index(last_file)

        self.current_index = start_idx
        self.load_current_pair()

    def perform_search(self):
        """Jump to an image volume by 1-based index or case-insensitive substring match."""
        if not self.file_list:
            return
        query = self.search_input.get().strip()
        if not query:
            return
        found_index = -1
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(self.file_list):
                found_index = idx
        if found_index == -1:
            for i, fname in enumerate(self.file_list):
                if query.lower() in fname.lower():
                    found_index = i
                    break
        if found_index != -1:
            self.current_index = found_index
            self.load_current_pair()
            self.search_input.delete(0, tk.END)
        else:
            messagebox.showwarning("Not Found", f"Could not find match for:\n'{query}'")

    @staticmethod
    def _coerce_mask_dtype(data_mask):
        """Normalize loaded mask values to uint8 or uint16."""
        cleaned = np.nan_to_num(data_mask, copy=False)
        max_label = int(np.max(cleaned)) if np.any(cleaned) else 0
        dtype = np.uint16 if max_label > 255 else np.uint8
        return np.round(cleaned).astype(dtype)

    def load_current_pair(self):
        """Load the current image volume, resolve its companion mask, and refresh UI state."""
        if not self.file_list or self.current_index < 0:
            return
        self.undo_stack.clear()

        self.sb_brightness.set(0)
        self.sb_contrast.set(100)
        self.sb_zoom.set(100)
        self.lbl_overlay.reset_pan()

        filename = self.file_list[self.current_index]
        img_path = os.path.join(self.image_dir, filename)
        Settings().setValue("last_file", filename)

        try:
            data_img, self.affine, self.header, self.current_file_format_info = uio.read_universal_image(img_path)
            self.img_data_raw = np.nan_to_num(data_img, copy=False)

            _min, _max = float(np.min(self.img_data_raw)), float(np.max(self.img_data_raw))
            if (_max - _min) > 1e-12:
                scale8 = 255.0 / (_max - _min)
                self.img_data = np.clip((self.img_data_raw - _min) * scale8, 0, 255).astype(np.uint8)
            else:
                self.img_data = np.zeros(self.img_data_raw.shape, dtype=np.uint8)

            mask_loaded = False
            if getattr(self, "specific_mask_file", None) and os.path.exists(self.specific_mask_file):
                spec_base = os.path.basename(self.specific_mask_file)
                use_specific = (
                    getattr(self, "single_image_mode", False)
                    or spec_base == filename
                    or bool(self.get_best_mask_match(filename, [spec_base]))
                    or filename == getattr(self, "initial_start_file", None)
                )
                if use_specific:
                    try:
                        data_mask, _, _, _ = uio.read_universal_image(self.specific_mask_file)
                        if data_mask.shape == self.img_data_raw.shape:
                            self.mask_data = self._coerce_mask_dtype(data_mask)
                            mask_loaded = True
                    except Exception as spec_err:
                        print(f"[Specific Mask Load Notice] Could not load specific mask {self.specific_mask_file}: {spec_err}")

            if not mask_loaded and self.mask_dir and os.path.exists(self.mask_dir):
                try:
                    valid_exts = uio.get_supported_extensions()
                    mask_files = [
                        f
                        for f in os.listdir(self.mask_dir)
                        if os.path.isfile(os.path.join(self.mask_dir, f))
                        and any(f.lower().endswith(ext) for ext in valid_exts)
                    ]
                    best_mask = self.get_best_mask_match(filename, mask_files)
                    if best_mask:
                        data_mask, _, _, _ = uio.read_universal_image(os.path.join(self.mask_dir, best_mask))
                        if data_mask.shape == self.img_data_raw.shape:
                            self.mask_data = self._coerce_mask_dtype(data_mask)
                            mask_loaded = True
                except Exception as mask_err:
                    print(f"[Optional Mask Load Notice] Could not load mask for {filename}: {mask_err}")
                    mask_loaded = False

            if not mask_loaded:
                self.mask_data = np.zeros(self.img_data.shape, dtype=np.uint8)

            self.obdet_active = False
            self.obdet_boxes = []
            self.active_obdet_box_idx = None
            self.obdet_center_2d = None
            if hasattr(self, "btn_obdet") and self.btn_obdet.winfo_exists():
                self.btn_obdet.configure(text="ObDet", fg_color="#1b5e20")

            self.setup_dimensions()
            self.update_views()
            self.update_volume_stats()
            self.update_nav_buttons()
            self.set_loaded_status()
            self.record_user_activity()
            self.update_csv_path_display()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed: {e}")

    def update_nav_buttons(self):
        total = len(self.file_list)
        if total == 0:
            self.lbl_file_info.configure(text="No Data")
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            return
        self.lbl_file_info.configure(text=f"File {self.current_index + 1} of {total}:  {self.file_list[self.current_index]}")
        self.btn_prev.configure(state="normal" if self.current_index > 0 else "disabled")
        self.btn_next.configure(state="normal" if self.current_index < total - 1 else "disabled")

    def _confirm_save_before_nav(self):
        if getattr(self, "has_unsaved_changes", False):
            ans = messagebox.askyesnocancel(
                "Unsaved Changes", "You have unsaved mask changes.\n\nSave changes before proceeding?"
            )
            if ans is None:
                return False
            if ans is True:
                self.quick_save_mask()
        try:
            self.export_time_log_csv(show_msg=False)
        except Exception:
            pass
        return True

    def next_image(self):
        if self.current_index < len(self.file_list) - 1 and self._confirm_save_before_nav():
            self.current_index += 1
            self.load_current_pair()

    def prev_image(self):
        if self.current_index > 0 and self._confirm_save_before_nav():
            self.current_index -= 1
            self.load_current_pair()

    def open_imprep(self):
        """Open or focus the integrated ImPrep preprocessing suite."""
        try:
            from medimg_annotator.imprep.imprep_window import ImPrepWindow

            if self.imprep_window is None or not self.imprep_window.winfo_exists():
                self.imprep_window = ImPrepWindow(parent=self)
            else:
                self.imprep_window.deiconify()
                self.imprep_window.lift()
                self.imprep_window.focus()
        except Exception as e:
            messagebox.showerror("ImPrep Error", f"Failed to open ImPrep Suite:\n{str(e)}")

    def get_current_volume(self):
        """Return currently loaded volume data, affine, header, and filename for ImPrep."""
        if self.img_data is None:
            return None, None, None, ""
        fname = ""
        if self.file_list and 0 <= self.current_index < len(self.file_list):
            item = self.file_list[self.current_index]
            fname = os.path.basename(item[0] if isinstance(item, (list, tuple)) else str(item))
        vol_to_return = self.img_data_raw if self.img_data_raw is not None else self.img_data
        return vol_to_return, self.affine, self.header, fname

    def set_external_volume(self, vol_data, affine, header, name, as_mask=False):
        """Receive a processed 3D volume or mask from ImPrep and display it."""
        if vol_data is None:
            return
        if vol_data.ndim > 3 and vol_data.shape[2] == 1 and vol_data.shape[3] > 1:
            vol_data = np.squeeze(vol_data, axis=2)
        while vol_data.ndim > 3:
            vol_data = vol_data[..., 0]
        if vol_data.ndim == 2:
            vol_data = vol_data[:, :, np.newaxis]

        if as_mask:
            if self.img_data is not None and vol_data.shape != self.img_data.shape:
                messagebox.showerror(
                    "Dimension Mismatch", f"Mask shape {vol_data.shape} does not match image shape {self.img_data.shape}."
                )
                return
            self.mask_data = np.nan_to_num(vol_data, copy=False).round().astype(np.uint8)
            self.set_unsaved_status()
            self.update_views()
            self.update_volume_stats()
            return

        self.img_data_raw = np.nan_to_num(np.asarray(vol_data, dtype=np.float32), copy=False)
        _min, _max = float(np.min(self.img_data_raw)), float(np.max(self.img_data_raw))
        if (_max - _min) > 1e-12:
            scale = 255.0 / (_max - _min)
            self.img_data = np.clip((self.img_data_raw - _min) * scale, 0, 255).astype(np.uint8)
        else:
            self.img_data = np.zeros(self.img_data_raw.shape, dtype=np.uint8)

        self.affine = np.copy(affine) if affine is not None else np.eye(4)
        self.header = header
        if self.mask_data is None or self.mask_data.shape != self.img_data.shape:
            self.mask_data = np.zeros(self.img_data.shape, dtype=np.uint8)

        self.setup_dimensions()
        self.set_loaded_status()
        self.lbl_file_info.configure(text=f"[ImPrep] {name[:18]}", text_color="#00ffcc")
        self.update_views()
        self.update_volume_stats()
