"""
Object Detection (ObDet) multi-box interaction, handle resizing, and YOLO/JSON export controller.
"""

from medimg_annotator.imports import filedialog, messagebox, os

from medimg_annotator.core import obdet_io
from medimg_annotator.core.config import Settings

class ObDetControllerMixin:
    """Manages Object Detection bounding boxes, corner handle drags, wheel scaling, and export."""

    @property
    def obdet_bbox(self):
        """Return the active bounding box (x1, y1, x2, y2) or None."""
        if (
            self.obdet_boxes
            and self.active_obdet_box_idx is not None
            and 0 <= self.active_obdet_box_idx < len(self.obdet_boxes)
        ):
            return tuple(self.obdet_boxes[self.active_obdet_box_idx]["bbox"])
        return None

    @obdet_bbox.setter
    def obdet_bbox(self, val):
        """Update the active bounding box or append a new box if none is active."""
        if val is None:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in val]
        if (
            self.obdet_boxes
            and self.active_obdet_box_idx is not None
            and 0 <= self.active_obdet_box_idx < len(self.obdet_boxes)
        ):
            self.obdet_boxes[self.active_obdet_box_idx]["bbox"] = [x1, y1, x2, y2]
        else:
            new_box = {
                "id": len(self.obdet_boxes) + 1,
                "bbox": [x1, y1, x2, y2],
                "slice": getattr(self, "current_slice", 0),
                "axis": getattr(self, "axis_index", 2),
                "label": self.get_active_label_name(),
                "label_id": self.get_active_label_id(),
            }
            self.obdet_boxes.append(new_box)
            self.active_obdet_box_idx = len(self.obdet_boxes) - 1

    @staticmethod
    def _clamp_scaled_box(new_w, new_h, cur_w, cur_h, ox1, oy1, ox2, oy2, pix_w, pix_h):
        """Clamp resized bounding box strictly within image bounds while preserving aspect ratio."""
        if new_w > pix_w or new_h > pix_h:
            max_scale = min(float(pix_w) / cur_w, float(pix_h) / cur_h)
            new_w = cur_w * max_scale
            new_h = cur_h * max_scale

        if new_w < 8.0 or new_h < 8.0:
            min_scale = max(8.0 / cur_w, 8.0 / cur_h)
            new_w = cur_w * min_scale
            new_h = cur_h * min_scale

        cx = (ox1 + ox2) / 2.0
        cy = (oy1 + oy2) / 2.0
        half_w = new_w / 2.0
        half_h = new_h / 2.0

        x1, x2 = cx - half_w, cx + half_w
        y1, y2 = cy - half_h, cy + half_h

        if x1 < 0:
            shift = -x1
            x1, x2 = x1 + shift, x2 + shift
        if x2 > pix_w - 1:
            shift = x2 - (pix_w - 1)
            x1, x2 = x1 - shift, x2 - shift
        if y1 < 0:
            shift = -y1
            y1, y2 = y1 + shift, y2 + shift
        if y2 > pix_h - 1:
            shift = y2 - (pix_h - 1)
            y1, y2 = y1 - shift, y2 - shift

        return [
            max(0, min(pix_w - 1, int(round(x1)))),
            max(0, min(pix_h - 1, int(round(y1)))),
            max(0, min(pix_w - 1, int(round(x2)))),
            max(0, min(pix_h - 1, int(round(y2)))),
        ]

    def on_obdet_zoom_slider(self, value):
        """Resize active bounding box via the zoom slider while keeping the viewport static."""
        self.record_user_activity()
        if not self.obdet_boxes or self.active_obdet_box_idx is None or self.lbl_overlay.current_image is None:
            return
        if not (0 <= self.active_obdet_box_idx < len(self.obdet_boxes)):
            return

        box = self.obdet_boxes[self.active_obdet_box_idx]
        pix_w, pix_h = self.lbl_overlay.current_image.size
        ox1, oy1, ox2, oy2 = box["bbox"]
        cur_w = float(max(6, ox2 - ox1))
        cur_h = float(max(6, oy2 - oy1))
        aspect_ratio = cur_w / cur_h

        scale = max(0.1, float(value) / 100.0)
        base_w = getattr(self, "_obdet_base_w", 64.0)
        new_w = base_w * scale
        new_h = new_w / aspect_ratio

        box["bbox"] = self._clamp_scaled_box(new_w, new_h, cur_w, cur_h, ox1, oy1, ox2, oy2, pix_w, pix_h)
        self.obdet_bbox_w = box["bbox"][2] - box["bbox"][0]
        self.obdet_bbox_h = box["bbox"][3] - box["bbox"][1]
        self.lbl_overlay.update_overlays_only()

    def handle_return_key(self, event=None):
        """Save active ObDet annotations when pressing Enter/Return."""
        if getattr(self, "obdet_active", False) and (
            getattr(self, "obdet_boxes", None) or getattr(self, "obdet_bbox", None) is not None
        ):
            self.save_obdet_image()

    def get_active_label_name(self):
        """Return display class name for the active label, defaulting to 'ObDet'."""
        active_label = "ObDet"
        if hasattr(self, "multilabel_var") and self.multilabel_var.get():
            try:
                val = int(self.combo_value.get())
                custom_name = self.label_names.get(val, "")
                active_label = custom_name if custom_name else f"Label {val}"
            except Exception:
                pass
        return active_label

    def find_obdet_corner_at(self, screen_x, screen_y, canvas_widget):
        """Return corner handle key ('tl', 'tr', 'bl', 'br') near screen coords, or None."""
        if not self.obdet_active or not self.obdet_boxes or self.active_obdet_box_idx is None:
            return None
        if not (0 <= self.active_obdet_box_idx < len(self.obdet_boxes)):
            return None

        box = self.obdet_boxes[self.active_obdet_box_idx]
        if box.get("slice", 0) != self.current_slice or box.get("axis", 2) != self.axis_index:
            return None

        rx, ry, rw, rh = canvas_widget.get_draw_rect()
        if rw <= 0 or rh <= 0 or canvas_widget.current_image is None:
            return None
        pix_w, pix_h = canvas_widget.current_image.size
        sx_scale = float(rw) / float(pix_w)
        sy_scale = float(rh) / float(pix_h)

        x1, y1, x2, y2 = box["bbox"]
        sx1, sy1 = rx + x1 * sx_scale, ry + y1 * sy_scale
        sx2, sy2 = rx + x2 * sx_scale, ry + y2 * sy_scale

        tol = 10.0
        for c_name, (cx, cy) in (("tl", (sx1, sy1)), ("tr", (sx2, sy1)), ("bl", (sx1, sy2)), ("br", (sx2, sy2))):
            if (screen_x - cx) ** 2 + (screen_y - cy) ** 2 <= tol * tol:
                return c_name
        return None

    def find_obdet_box_at(self, ix, iy):
        """Find index of box on current slice/axis containing (ix, iy), prioritizing active box."""
        if not self.obdet_boxes:
            return None

        if self.active_obdet_box_idx is not None and 0 <= self.active_obdet_box_idx < len(self.obdet_boxes):
            b = self.obdet_boxes[self.active_obdet_box_idx]
            if b.get("slice", 0) == self.current_slice and b.get("axis", 2) == self.axis_index:
                x1, y1, x2, y2 = b["bbox"]
                if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
                    return self.active_obdet_box_idx

        for idx in reversed(range(len(self.obdet_boxes))):
            if idx == self.active_obdet_box_idx:
                continue
            b = self.obdet_boxes[idx]
            if b.get("slice", 0) == self.current_slice and b.get("axis", 2) == self.axis_index:
                x1, y1, x2, y2 = b["bbox"]
                if min(x1, x2) <= ix <= max(x1, x2) and min(y1, y2) <= iy <= max(y1, y2):
                    return idx
        return None

    def get_obdet_cursor_at(self, screen_x, screen_y, canvas_widget):
        """Return Tkinter cursor shape appropriate for corner resize or box move."""
        corner = self.find_obdet_corner_at(screen_x, screen_y, canvas_widget)
        if corner in ("tl", "br"):
            return "size_nw_se"
        if corner in ("tr", "bl"):
            return "size_ne_sw"

        ix, iy = self.get_slice_2d_coords_from_screen(screen_x, screen_y, canvas_widget)
        if ix is not None and iy is not None and self.find_obdet_box_at(ix, iy) is not None:
            return "fleur"
        return ""

    def delete_active_obdet_box(self, event=None):
        """Remove the currently selected ObDet bounding box."""
        if not getattr(self, "obdet_active", False):
            return
        if self.obdet_boxes and self.active_obdet_box_idx is not None and 0 <= self.active_obdet_box_idx < len(self.obdet_boxes):
            self.record_user_activity()
            self.obdet_boxes.pop(self.active_obdet_box_idx)
            if not self.obdet_boxes:
                self.active_obdet_box_idx = None
                self.obdet_center_2d = None
            else:
                self.active_obdet_box_idx = min(self.active_obdet_box_idx, len(self.obdet_boxes) - 1)
                b = self.obdet_boxes[self.active_obdet_box_idx]
                self.obdet_center_2d = ((b["bbox"][0] + b["bbox"][2]) // 2, (b["bbox"][1] + b["bbox"][3]) // 2)
                self.obdet_slice = b.get("slice", self.current_slice)
                self.obdet_axis = b.get("axis", self.axis_index)
            self.lbl_overlay.update_overlays_only()
            return "break"

    def get_slice_2d_coords_from_screen(self, screen_x, screen_y, canvas_widget):
        """Convert screen canvas coords into 2D slice pixel coordinates (ix, iy)."""
        rx, ry, rw, rh = canvas_widget.get_draw_rect()
        if rw <= 0 or rh <= 0 or canvas_widget.current_image is None:
            return None, None
        if not (rx <= screen_x <= rx + rw and ry <= screen_y <= ry + rh):
            return None, None
        pix_w, pix_h = canvas_widget.current_image.size
        sx = (screen_x - rx) * (pix_w / float(rw))
        sy = (screen_y - ry) * (pix_h / float(rh))
        return int(round(max(0, min(sx, pix_w - 1)))), int(round(max(0, min(sy, pix_h - 1))))

    def toggle_obdet_mode(self):
        """Toggle Object Detection annotation mode ON or OFF."""
        self.record_user_activity()
        if self.obdet_active:
            self.obdet_active = False
            self.btn_obdet.configure(text="ObDet", fg_color="#1b5e20")
            if hasattr(self, "_pre_obdet_zoom") and hasattr(self, "sb_zoom"):
                self.sb_zoom.set(self._pre_obdet_zoom)
            self.lbl_overlay.repaint()
            return

        self.obdet_active = True
        if hasattr(self, "sb_zoom"):
            self._pre_obdet_zoom = self.sb_zoom.get()
        self.btn_obdet.configure(text="ObDet [ON]", fg_color="#00a846")
        if self.lbl_overlay.current_image is not None:
            pix_w, pix_h = self.lbl_overlay.current_image.size
            has_box_on_slice = any(
                b.get("slice", 0) == self.current_slice and b.get("axis", 2) == self.axis_index
                for b in self.obdet_boxes
            )
            if not has_box_on_slice:
                cx, cy = pix_w // 2, pix_h // 2
                self.obdet_center_2d = (cx, cy)
                self.obdet_slice = self.current_slice
                self.obdet_axis = self.axis_index
                half_w = max(10, self.obdet_bbox_w // 2)
                half_h = max(10, self.obdet_bbox_h // 2)
                new_box = {
                    "id": len(self.obdet_boxes) + 1,
                    "bbox": [
                        max(0, cx - half_w),
                        max(0, cy - half_h),
                        min(pix_w - 1, cx + half_w),
                        min(pix_h - 1, cy + half_h),
                    ],
                    "slice": self.current_slice,
                    "axis": self.axis_index,
                    "label": self.get_active_label_name(),
                    "label_id": self.get_active_label_id(),
                }
                self.obdet_boxes.append(new_box)
                self.active_obdet_box_idx = len(self.obdet_boxes) - 1
        self.lbl_overlay.repaint()

    def on_obdet_canvas_click(self, screen_x, screen_y, canvas_widget):
        """Select/move an existing box, grab a corner handle, or spawn a new box."""
        self.record_user_activity()
        if canvas_widget.current_image is None:
            return

        corner = self.find_obdet_corner_at(screen_x, screen_y, canvas_widget)
        if corner is not None:
            self.obdet_drag_mode = "resize_corner"
            self.obdet_drag_corner = corner
            self.obdet_orig_bbox = list(self.obdet_boxes[self.active_obdet_box_idx]["bbox"])
            return

        ix, iy = self.get_slice_2d_coords_from_screen(screen_x, screen_y, canvas_widget)
        if ix is None or iy is None:
            return

        hit_idx = self.find_obdet_box_at(ix, iy)
        if hit_idx is not None:
            self.active_obdet_box_idx = hit_idx
            b = self.obdet_boxes[hit_idx]
            self.obdet_center_2d = ((b["bbox"][0] + b["bbox"][2]) // 2, (b["bbox"][1] + b["bbox"][3]) // 2)
            self.obdet_slice = b.get("slice", self.current_slice)
            self.obdet_axis = b.get("axis", self.axis_index)
            self.obdet_drag_mode = "move"
            self.obdet_drag_start = (ix, iy)
            self.obdet_orig_bbox = list(b["bbox"])
            canvas_widget.update_overlays_only()
            return

        pix_w, pix_h = canvas_widget.current_image.size
        self.obdet_center_2d = (ix, iy)
        self.obdet_slice = self.current_slice
        self.obdet_axis = self.axis_index
        half_w = max(10, self.obdet_bbox_w // 2)
        half_h = max(10, self.obdet_bbox_h // 2)

        new_box = {
            "id": len(self.obdet_boxes) + 1,
            "bbox": [
                int(max(0, ix - half_w)),
                int(max(0, iy - half_h)),
                int(min(pix_w - 1, ix + half_w)),
                int(min(pix_h - 1, iy + half_h)),
            ],
            "slice": self.current_slice,
            "axis": self.axis_index,
            "label": self.get_active_label_name(),
            "label_id": self.get_active_label_id(),
        }
        self.obdet_boxes.append(new_box)
        self.active_obdet_box_idx = len(self.obdet_boxes) - 1
        self.obdet_drag_mode = None
        self.obdet_drag_corner = None
        canvas_widget.update_overlays_only()

    def on_obdet_drag(self, screen_x, screen_y, canvas_widget):
        """Handle corner handle resizing or translation of the active box."""
        self.record_user_activity()
        if not self.obdet_boxes or self.active_obdet_box_idx is None:
            return
        if not (0 <= self.active_obdet_box_idx < len(self.obdet_boxes)):
            return
        if canvas_widget.current_image is None:
            return

        box = self.obdet_boxes[self.active_obdet_box_idx]
        ix, iy = self.get_slice_2d_coords_from_screen(screen_x, screen_y, canvas_widget)
        if ix is None or iy is None:
            return
        pix_w, pix_h = canvas_widget.current_image.size

        if self.obdet_drag_mode == "resize_corner" and self.obdet_orig_bbox is not None:
            ox1, oy1, ox2, oy2 = self.obdet_orig_bbox
            corner = self.obdet_drag_corner

            if corner == "br":
                x1, y1, x2, y2 = ox1, oy1, max(ox1 + 8, min(pix_w - 1, ix)), max(oy1 + 8, min(pix_h - 1, iy))
            elif corner == "tl":
                x1, y1, x2, y2 = min(ox2 - 8, max(0, ix)), min(oy2 - 8, max(0, iy)), ox2, oy2
            elif corner == "tr":
                x1, y1, x2, y2 = ox1, min(oy2 - 8, max(0, iy)), max(ox1 + 8, min(pix_w - 1, ix)), oy2
            elif corner == "bl":
                x1, y1, x2, y2 = min(ox2 - 8, max(0, ix)), oy1, ox2, max(oy1 + 8, min(pix_h - 1, iy))
            else:
                return

            box["bbox"] = [int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))]
            self.obdet_bbox_w = abs(box["bbox"][2] - box["bbox"][0])
            self.obdet_bbox_h = abs(box["bbox"][3] - box["bbox"][1])
            self.obdet_center_2d = ((box["bbox"][0] + box["bbox"][2]) // 2, (box["bbox"][1] + box["bbox"][3]) // 2)
            canvas_widget.update_overlays_only()

        elif self.obdet_drag_mode == "move" and self.obdet_drag_start is not None and self.obdet_orig_bbox is not None:
            dx = ix - self.obdet_drag_start[0]
            dy = iy - self.obdet_drag_start[1]
            ox1, oy1, ox2, oy2 = self.obdet_orig_bbox
            bw, bh = ox2 - ox1, oy2 - oy1

            nx1 = max(0, min(pix_w - 1 - bw, ox1 + dx))
            ny1 = max(0, min(pix_h - 1 - bh, oy1 + dy))
            nx2, ny2 = nx1 + bw, ny1 + bh

            box["bbox"] = [int(nx1), int(ny1), int(nx2), int(ny2)]
            self.obdet_center_2d = ((int(nx1) + int(nx2)) // 2, (int(ny1) + int(ny2)) // 2)
            canvas_widget.update_overlays_only()

    def on_obdet_release(self):
        """Reset transient drag state after releasing mouse button."""
        self.obdet_drag_mode = None
        self.obdet_drag_corner = None
        self.obdet_drag_start = None
        self.obdet_orig_bbox = None

    def on_obdet_wheel(self, direction, canvas_widget):
        """Scale the active bounding box via mouse wheel while preserving aspect ratio."""
        self.record_user_activity()
        if not self.obdet_boxes or self.active_obdet_box_idx is None:
            return
        if not (0 <= self.active_obdet_box_idx < len(self.obdet_boxes)):
            return
        if canvas_widget.current_image is None:
            return

        box = self.obdet_boxes[self.active_obdet_box_idx]
        pix_w, pix_h = canvas_widget.current_image.size
        ox1, oy1, ox2, oy2 = box["bbox"]
        cur_w = float(max(6, ox2 - ox1))
        cur_h = float(max(6, oy2 - oy1))
        aspect_ratio = cur_w / cur_h

        scale_factor = 1.10 if direction > 0 else 0.90
        new_w = cur_w * scale_factor
        new_h = new_w / aspect_ratio

        if int(round(new_w)) == int(cur_w):
            new_w = (cur_w + 4.0) if direction > 0 else max(6.0, cur_w - 4.0)
            new_h = new_w / aspect_ratio

        box["bbox"] = self._clamp_scaled_box(new_w, new_h, cur_w, cur_h, ox1, oy1, ox2, oy2, pix_w, pix_h)
        self.obdet_bbox_w = box["bbox"][2] - box["bbox"][0]
        self.obdet_bbox_h = box["bbox"][3] - box["bbox"][1]
        self.obdet_center_2d = ((box["bbox"][0] + box["bbox"][2]) // 2, (box["bbox"][1] + box["bbox"][3]) // 2)
        canvas_widget.update_overlays_only()

    def save_obdet_image(self):
        """Burn ObDet boxes onto the image slice and write image + YOLO/JSON annotations."""
        if self.img_data is None:
            messagebox.showwarning("No Image", "No image is currently loaded.")
            return
        if not self.obdet_boxes:
            messagebox.showinfo("No Bounding Box", "Please click on the image to place a bounding box first.")
            return

        if not self.output_dir or not os.path.exists(self.output_dir):
            settings = Settings()
            init_out = settings.value("last_output_dir", "") or self.image_dir
            out_dir = filedialog.askdirectory(
                title="Select Output Folder to Save ObDet Result",
                initialdir=init_out if (init_out and os.path.exists(init_out)) else None,
            )
            if not out_dir:
                return
            self.output_dir = out_dir
            settings.setValue("last_output_dir", out_dir)
            self.update_csv_path_display()

        filename = (
            self.file_list[self.current_index]
            if (self.file_list and 0 <= self.current_index < len(self.file_list))
            else "obdet_result.png"
        )
        active_label = self.get_active_label_name()

        try:
            self.btn_obdet.configure(text="Saving...", fg_color="#c97116")
            img_to_burn = self.img_data_raw if self.img_data_raw is not None else self.img_data
            obdet_io.save_obdet_result(
                image_data=img_to_burn,
                bbox=self.obdet_bbox,
                slice_idx=self.obdet_slice,
                axis=self.obdet_axis,
                output_dir=self.output_dir,
                filename=filename,
                affine=self.affine,
                header=self.header,
                format_info=self.current_file_format_info,
                label_text=active_label,
                displayed_slice=getattr(self, "_cached_img_slice_8", None),
                boxes=self.obdet_boxes,
            )
            self.btn_obdet.configure(text="ObDet [Saved!]", fg_color="#2e7d32")
            self.btn_quick_save.configure(text="Saved ObDet!", fg_color="#3b9148", text_color="white")
            self.lbl_file_info.configure(text=f"ObDet Saved: {filename}")

            def _reset_btn():
                if getattr(self, "obdet_active", False):
                    self.btn_obdet.configure(text="ObDet [ON]", fg_color="#00a846")
                else:
                    self.btn_obdet.configure(text="ObDet", fg_color="#1b5e20")

            self.after(2000, _reset_btn)

            try:
                self.export_time_log_csv(show_msg=False)
            except Exception:
                pass
        except Exception as err:
            self.btn_obdet.configure(text="ObDet [Error]", fg_color="#bf3434")
            messagebox.showerror("ObDet Save Error", f"Failed to save ObDet bounding box image:\n{str(err)}")
