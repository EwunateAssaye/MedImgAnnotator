"""
Multi-label selection, real-time physical volume statistics, active annotation timer, and CSV logging.
"""

from medimg_annotator.imports import csv, ctk, datetime, messagebox, np, os, time, tk

from medimg_annotator.core.config import LABEL_COLOR_MAP, Settings, default_color_name_for_label
from medimg_annotator.core.geometry import get_voxel_spacing_mm

class StatsLoggerMixin:
    """Manages multi-label controls, volume statistics panel, active timer, and CSV export."""

    def _resolve_label_color(self, label_id, is_multi):
        """Return `(color_name, rgb_tuple)` for a label ID in single- or multi-label mode."""
        if not is_multi:
            return "Single", (255, 0, 0)
        default_cname = default_color_name_for_label(label_id)
        color_name = self.value_to_color.get(label_id, default_cname)
        rgb = self.color_map.get(color_name, LABEL_COLOR_MAP[default_cname])
        return color_name, rgb

    def _update_color_swatch(self):
        """Sync the toolbar color preview swatch with active mode and selected label."""
        swatch = getattr(self, "lbl_color_swatch", None)
        if swatch is None or not swatch.winfo_exists():
            return
        is_multi = bool(self.multilabel_var.get())
        try:
            val = max(1, int(self.combo_value.get()))
        except Exception:
            val = 1
        _, rgb = self._resolve_label_color(val, is_multi)
        swatch.configure(fg_color=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    def toggle_multilabel(self):
        state = "normal" if self.multilabel_var.get() else "disabled"
        self.combo_value.configure(state=state)
        self.combo_color.configure(state=state)
        self.line_name.configure(state=state)
        self._update_color_swatch()
        self._last_vol_stats_state = None
        self.update_views()
        self.update_volume_stats()

    def on_value_selection(self, text):
        self.record_user_activity()
        try:
            val = max(1, int(text))
        except ValueError:
            val = 1
        self.combo_color.set(self.value_to_color.get(val, default_color_name_for_label(val)))
        self._update_color_swatch()
        current_name = self.label_names.get(val, "")
        self.line_name.delete(0, tk.END)
        if current_name:
            self.line_name.insert(0, current_name)

    def on_label_name_changed(self, event=None):
        self.record_user_activity()
        if not self.multilabel_var.get():
            return
        try:
            val = max(1, int(self.combo_value.get()))
        except ValueError:
            val = 1
        self.label_names[val] = self.line_name.get().strip()
        self._last_vol_stats_state = None
        self.update_volume_stats()

    def on_color_selection(self, text):
        self.record_user_activity()
        try:
            val = max(1, int(self.combo_value.get()))
        except ValueError:
            val = 1
        self.value_to_color[val] = text
        self._update_color_swatch()
        self._color_lut_dirty = True
        self._last_vol_stats_state = None
        self.update_views()
        self.update_volume_stats()

    @staticmethod
    def _bincount_positive_labels(mask_data):
        """Count voxels for every positive foreground label via fast O(N) bincount."""
        try:
            flat_mask = mask_data.ravel()
            if not np.issubdtype(flat_mask.dtype, np.integer):
                flat_mask = np.nan_to_num(flat_mask).round().astype(np.int32)
            elif flat_mask.dtype.kind in ("i", "f") and np.any(flat_mask < 0):
                flat_mask = np.maximum(flat_mask, 0)
            counts = np.bincount(flat_mask)
            labels_present = (np.flatnonzero(counts[1:]) + 1).tolist() if len(counts) > 1 else []
            return counts, labels_present
        except Exception:
            unique_vals, u_counts = np.unique(mask_data, return_counts=True)
            counts_map = dict(zip(unique_vals.tolist(), u_counts.tolist()))
            labels_present = [int(v) for v in unique_vals if v > 0]
            counts = np.zeros(max(labels_present, default=0) + 1, dtype=np.int64)
            for l_val in labels_present:
                counts[l_val] = counts_map.get(l_val, 0)
            return counts, labels_present

    def update_volume_stats(self):
        """Calculate and render physical label volumes (mm³ / cm³) and active time per label."""
        if not hasattr(self, "vol_scroll_frame"):
            return

        if self.mask_data is None:
            self.lbl_vol_voxinfo.configure(text="Voxel: N/A")
            self.lbl_vol_total.configure(text="Total: No Data Loaded")
            self._clear_vol_scroll_frame()
            self._last_vol_stats_state = None
            return

        dx, dy, dz = get_voxel_spacing_mm(self.header, self.affine)
        voxel_vol_mm3 = dx * dy * dz
        self.lbl_vol_voxinfo.configure(text=f"Voxel: {dx:.2f}×{dy:.2f}×{dz:.2f} mm")

        counts, labels_present = self._bincount_positive_labels(self.mask_data)
        is_multi = bool(self.multilabel_var.get())

        state_sig = (
            id(self.mask_data),
            tuple(
                (
                    int(l),
                    int(counts[l]),
                    self.label_names.get(int(l), ""),
                    self._resolve_label_color(int(l), is_multi)[0],
                )
                for l in labels_present
            ),
            voxel_vol_mm3,
        )
        if getattr(self, "_last_vol_stats_state", None) == state_sig:
            return
        self._last_vol_stats_state = state_sig

        if not labels_present:
            self.lbl_vol_total.configure(text="Total: 0 voxels (0.00 mm³)")
            self._clear_vol_scroll_frame()
            if not getattr(self, "_vol_empty_label", None) or not self._vol_empty_label.winfo_exists():
                self._vol_empty_label = ctk.CTkLabel(
                    self.vol_scroll_frame,
                    text="No foreground labels segmented yet.\nDraw with Brush, Bucket, Poly, or Wand.",
                    font=("Arial", 11),
                    text_color="#6e7681",
                    justify="center",
                )
                self._vol_empty_label.pack(expand=True, fill=tk.BOTH, pady=25)
            return

        if getattr(self, "_vol_empty_label", None) and self._vol_empty_label.winfo_exists():
            try:
                self._vol_empty_label.destroy()
            except Exception:
                pass
            self._vol_empty_label = None

        total_vox = sum(int(counts[l]) for l in labels_present)
        total_mm3 = total_vox * voxel_vol_mm3
        total_cm3 = total_mm3 / 1000.0
        total_str = (
            f"Total: {total_vox:,} voxels ({total_cm3:.2f} cm³)"
            if total_cm3 >= 1.0
            else f"Total: {total_vox:,} voxels ({total_mm3:.1f} mm³)"
        )
        self.lbl_vol_total.configure(text=total_str)

        if not hasattr(self, "_vol_row_widgets"):
            self._vol_row_widgets = {}

        current_lids = set(labels_present)
        for old_lid in list(self._vol_row_widgets.keys()):
            if old_lid not in current_lids:
                try:
                    self._vol_row_widgets[old_lid]["frame"].destroy()
                except Exception:
                    pass
                del self._vol_row_widgets[old_lid]

        cur_img = (
            self.file_list[self.current_index]
            if (self.file_list and 0 <= self.current_index < len(self.file_list))
            else ""
        )
        cur_img_name = os.path.basename(cur_img) if cur_img else ""

        for l_id in labels_present:
            color_name, rgb = self._resolve_label_color(l_id, is_multi)
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            luminance = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
            badge_text_color = "#000000" if luminance > 140 else "#ffffff"

            custom_name = self.label_names.get(l_id, "")
            disp_name = custom_name if custom_name else (f"Label {l_id}" if is_multi else "Mask")

            n_vox = int(counts[l_id])
            vol_mm3 = n_vox * voxel_vol_mm3
            vol_cm3 = vol_mm3 / 1000.0
            v_str = f"{vol_cm3:.2f} cm³" if vol_cm3 >= 1.0 else f"{vol_mm3:.1f} mm³"
            stats_str = f"{n_vox:,} vox  ({v_str})"

            active_time_sec = self.active_annotation_times.get(cur_img_name, {}).get(l_id, 0.0) if cur_img_name else 0.0
            time_str = self.format_active_time(active_time_sec)

            if l_id in self._vol_row_widgets and self._vol_row_widgets[l_id]["frame"].winfo_exists():
                w_dict = self._vol_row_widgets[l_id]
                w_dict["name"].configure(text=disp_name)
                w_dict["stats"].configure(text=stats_str)
                if "time" in w_dict and w_dict["time"].winfo_exists():
                    w_dict["time"].configure(text=f"⏱ {time_str}")
                if w_dict.get("color_name") != color_name:
                    w_dict["badge"].configure(fg_color=hex_color, text_color=badge_text_color)
                    w_dict["color_name"] = color_name
            else:
                row = ctk.CTkFrame(self.vol_scroll_frame, fg_color="#212427", corner_radius=5)
                row.pack(fill=tk.X, padx=2, pady=2)

                badge = ctk.CTkLabel(
                    row,
                    text=f" {l_id} ",
                    fg_color=hex_color,
                    text_color=badge_text_color,
                    font=("Arial", 10, "bold"),
                    corner_radius=4,
                    width=24,
                    height=20,
                )
                badge.pack(side=tk.LEFT, padx=(6, 8), pady=4)

                name_label = ctk.CTkLabel(
                    row, text=disp_name, font=("Arial", 11, "bold"), anchor="w", text_color="#e6edf3"
                )
                name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

                time_label = ctk.CTkLabel(
                    row, text=f"⏱ {time_str}", font=("Arial", 10), text_color="#e5c07b", anchor="e"
                )
                time_label.pack(side=tk.RIGHT, padx=(2, 6))

                stats_label = ctk.CTkLabel(
                    row, text=stats_str, font=("Arial", 10), text_color="#7ee787", anchor="e"
                )
                stats_label.pack(side=tk.RIGHT, padx=(2, 4))

                self._vol_row_widgets[l_id] = {
                    "frame": row,
                    "badge": badge,
                    "name": name_label,
                    "stats": stats_label,
                    "time": time_label,
                    "color_name": color_name,
                }

    def _clear_vol_scroll_frame(self):
        for w in getattr(self, "_vol_row_widgets", {}).values():
            try:
                w["frame"].destroy()
            except Exception:
                pass
        self._vol_row_widgets = {}
        if getattr(self, "_vol_empty_label", None):
            try:
                self._vol_empty_label.destroy()
            except Exception:
                pass
            self._vol_empty_label = None
        for child in self.vol_scroll_frame.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

    def record_user_activity(self, event=None):
        """Record active interaction timestamp to reset idle timeout."""
        self._last_interaction_time = time.time()

    def get_active_label_id(self):
        """Resolve the currently active foreground label ID (1, 2, ...) being annotated."""
        if bool(self.multilabel_var.get()):
            try:
                return max(1, int(self.combo_value.get()))
            except Exception:
                return 1
        return 1

    def _schedule_active_timer(self):
        if getattr(self, "_active_timer_job", None) is not None:
            try:
                self.after_cancel(self._active_timer_job)
            except Exception:
                pass
            self._active_timer_job = None
        try:
            if self.winfo_exists():
                self._active_timer_job = self.after(self._timer_interval_ms, self._on_active_timer_tick)
        except Exception:
            self._active_timer_job = None

    def _on_active_timer_tick(self):
        """Accrue active annotation seconds per label while interaction is within idle timeout."""
        try:
            if not self.winfo_exists():
                self._active_timer_job = None
                return
        except Exception:
            self._active_timer_job = None
            return

        now = time.time()
        if self.file_list and 0 <= self.current_index < len(self.file_list) and self.img_data is not None:
            img_name = os.path.basename(self.file_list[self.current_index])
            elapsed = now - getattr(self, "_last_interaction_time", now)
            if elapsed <= self._idle_timeout_seconds:
                active_lbl = self.get_active_label_id()
                img_dict = self.active_annotation_times.setdefault(img_name, {})
                img_dict[active_lbl] = img_dict.get(active_lbl, 0.0) + (self._timer_interval_ms / 1000.0)
                time_fmt = self.format_active_time(img_dict[active_lbl])

                if hasattr(self, "lbl_active_timer") and self.lbl_active_timer.winfo_exists():
                    lbl_name = (
                        self.label_names.get(active_lbl, f"Label {active_lbl}")
                        if bool(self.multilabel_var.get())
                        else "Mask"
                    )
                    self.lbl_active_timer.configure(text=f"Active: ⏱ {time_fmt} ({lbl_name})")

                if hasattr(self, "_vol_row_widgets") and active_lbl in self._vol_row_widgets:
                    w = self._vol_row_widgets[active_lbl]
                    if "time" in w and w["time"].winfo_exists():
                        w["time"].configure(text=f"⏱ {time_fmt}")
            elif hasattr(self, "lbl_active_timer") and self.lbl_active_timer.winfo_exists():
                self.lbl_active_timer.configure(text="Active: ⏸ Paused (Idle)")

        self._schedule_active_timer()

    @staticmethod
    def format_active_time(seconds):
        """Format floating-point seconds as MM:SS or HH:MM:SS."""
        total_sec = max(0, int(round(seconds)))
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def get_csv_log_path(self):
        """Resolve destination path for annotation_time_log.csv."""
        base_dir = self.output_dir if (getattr(self, "output_dir", None) and os.path.isdir(self.output_dir)) else None
        if not base_dir:
            base_dir = self.mask_dir if (getattr(self, "mask_dir", None) and os.path.isdir(self.mask_dir)) else None
        if not base_dir and self.file_list and 0 <= self.current_index < len(self.file_list):
            img_p = os.path.join(self.image_dir, self.file_list[self.current_index])
            if os.path.exists(img_p):
                base_dir = os.path.dirname(os.path.abspath(img_p))
        if not base_dir:
            settings = Settings()
            base_dir = settings.value("last_output_dir", "") or settings.value("last_image_dir", "")
        if not base_dir or not os.path.isdir(base_dir):
            base_dir = os.path.expanduser("~")
        return os.path.normpath(os.path.join(base_dir, "annotation_time_log.csv"))

    def update_csv_path_display(self):
        if hasattr(self, "entry_csv_path") and self.entry_csv_path.winfo_exists():
            p = self.get_csv_log_path()
            if self.entry_csv_path.get() != p:
                self.entry_csv_path.delete(0, tk.END)
                self.entry_csv_path.insert(0, p)

    def load_existing_active_times_from_csv(self):
        """Preload previously logged active annotation times for current user from CSV."""
        csv_path = self.get_csv_log_path()
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                if not next(reader, None):
                    return
                for r in reader:
                    if len(r) >= 6 and r[0].strip() == self.current_user:
                        try:
                            lid = int(r[3].strip())
                            sec = float(r[5].strip())
                            self.active_annotation_times.setdefault(r[2].strip(), {})[lid] = sec
                        except Exception:
                            pass
        except Exception:
            pass

    def export_time_log_csv(self, show_msg=True):
        """Upsert active annotation times and label volume metrics into annotation_time_log.csv."""
        csv_path = self.get_csv_log_path()
        today_str = datetime.date.today().isoformat()
        user_name = self.current_user

        header = [
            "User Name",
            "Date of Annotation",
            "Image Name",
            "Label ID",
            "Label Name",
            "Active Annotation Time (s)",
            "Active Annotation Time (formatted)",
            "Voxel Count",
            "Volume (cm³)",
        ]

        rows_dict = {}
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for r in reader:
                        if len(r) >= 5:
                            k = (r[0].strip(), r[2].strip(), str(r[3]).strip())
                            while len(r) < len(header):
                                r.append("")
                            rows_dict[k] = r
            except Exception:
                pass

        dx, dy, dz = get_voxel_spacing_mm(self.header, self.affine)
        voxel_vol_cm3 = (dx * dy * dz) / 1000.0

        cur_img = (
            self.file_list[self.current_index]
            if (self.file_list and 0 <= self.current_index < len(self.file_list))
            else ""
        )
        cur_counts = {}
        if self.mask_data is not None:
            bc, positive_ids = self._bincount_positive_labels(self.mask_data)
            for lid in positive_ids:
                cur_counts[lid] = int(bc[lid])

        if cur_img:
            cur_img_name = os.path.basename(cur_img)
            active_img_times = self.active_annotation_times.setdefault(cur_img_name, {})
            for lid in sorted(set(cur_counts.keys()) | set(active_img_times.keys())):
                sec = float(active_img_times.get(lid, 0.0))
                vox = cur_counts.get(lid, 0)
                lname = self.label_names.get(lid, "") or (
                    f"Label {lid}" if bool(self.multilabel_var.get()) else "Mask"
                )
                rows_dict[(user_name, cur_img_name, str(lid))] = [
                    user_name,
                    today_str,
                    cur_img_name,
                    str(lid),
                    lname,
                    f"{sec:.1f}",
                    self.format_active_time(sec),
                    str(vox),
                    f"{vox * voxel_vol_cm3:.4f}",
                ]

        for img_name, lids in self.active_annotation_times.items():
            if cur_img and img_name == os.path.basename(cur_img):
                continue
            for lid, sec in lids.items():
                k = (user_name, img_name, str(lid))
                if k in rows_dict:
                    rows_dict[k][5] = f"{sec:.1f}"
                    rows_dict[k][6] = self.format_active_time(sec)
                else:
                    lname = self.label_names.get(lid, f"Label {lid}")
                    rows_dict[k] = [
                        user_name,
                        today_str,
                        img_name,
                        str(lid),
                        lname,
                        f"{sec:.1f}",
                        self.format_active_time(sec),
                        "0",
                        "0.0000",
                    ]

        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            tmp_path = csv_path + ".tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                def sort_key(item):
                    try:
                        lid_int = int(item[2])
                    except Exception:
                        lid_int = 0
                    return (item[1].lower(), lid_int)

                for k in sorted(rows_dict.keys(), key=sort_key):
                    writer.writerow(rows_dict[k])
            os.replace(tmp_path, csv_path)
            self.update_csv_path_display()
            if show_msg:
                messagebox.showinfo(
                    "CSV Exported",
                    f"Active annotation time log successfully exported to:\n\n{csv_path}\n\nTotal records logged: {len(rows_dict)}",
                )
        except Exception as e:
            if show_msg:
                messagebox.showerror("Export Failed", f"Could not write CSV file:\n{str(e)}\n\nTarget path:\n{csv_path}")

    def open_time_log_csv(self):
        csv_path = self.get_csv_log_path()
        if not os.path.exists(csv_path):
            self.export_time_log_csv(show_msg=False)
        if os.path.exists(csv_path):
            try:
                os.startfile(csv_path)
            except Exception as e:
                messagebox.showinfo(
                    "Open CSV", f"Could not launch default spreadsheet application: {e}\n\nFile is located at:\n{csv_path}"
                )
        else:
            messagebox.showwarning(
                "Log Not Found", f"No CSV log file found at:\n{csv_path}\n\nAnnotate a label and click 'Export CSV'."
            )
