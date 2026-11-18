"""
Slice navigation, throttled viewport rendering, LUT colorization, and tool parameter state.
"""

from medimg_annotator.imports import Image, np

from medimg_annotator.core.config import LABEL_COLOR_MAP, default_color_name_for_label

class ViewControllerMixin:
    """Manages slice axes, throttled redraws, LUT compositing, and active tool settings."""

    def update_views_event(self, value=None):
        """Schedule a throttled (~60fps) view update."""
        if self._is_updating:
            self._pending_update = True
            return
        self._is_updating = True
        self.update_views()
        self._update_job = self.after(16, self._finish_update)

    def _finish_update(self):
        try:
            if not self.winfo_exists():
                self._is_updating = False
                self._update_job = None
                return
        except Exception:
            self._is_updating = False
            self._update_job = None
            return
        if self._pending_update:
            self._pending_update = False
            self.update_views()
            self._update_job = self.after(16, self._finish_update)
        else:
            self._is_updating = False
            self._update_job = None

    def change_slice(self, value):
        """Handle slice slider adjustments with frame-rate throttling."""
        val = max(0, min(self.max_slice, int(round(float(value)))))
        if val == self.current_slice:
            return
        self.current_slice = val
        self.lbl_slice_info.configure(text=f"Slice: {self.current_slice + 1}/{self.max_slice + 1}")

        if self._is_slicing:
            self._pending_slice = True
            return
        self._is_slicing = True
        self._apply_slice_change()
        self._slicing_job = self.after(16, self._finish_slicing)

    def _apply_slice_change(self):
        self.lbl_overlay.raw_points = []
        self.lbl_overlay.control_points = []
        self.lbl_overlay.drawing_new = False
        self.update_views()

    def _finish_slicing(self):
        try:
            if not self.winfo_exists():
                self._is_slicing = False
                self._slicing_job = None
                return
        except Exception:
            self._is_slicing = False
            self._slicing_job = None
            return
        if self._pending_slice:
            self._pending_slice = False
            self._apply_slice_change()
            self._slicing_job = self.after(16, self._finish_slicing)
        else:
            self._is_slicing = False
            self._slicing_job = None

    def change_opacity(self, value):
        self.mask_opacity = int(value)
        self.update_views_event()

    def change_zoom_event(self, value):
        if getattr(self, "obdet_active", False):
            if getattr(self, "obdet_bbox", None) is not None and getattr(self, "obdet_center_2d", None) is not None:
                self.on_obdet_zoom_slider(float(value))
            return
        self.lbl_overlay.repaint()

    def set_unsaved_status(self):
        self.has_unsaved_changes = True
        self._mask_labels_dirty = True
        self.btn_quick_save.configure(text="Save (S)", fg_color="#c97116", text_color="white")
        if hasattr(self, "btn_save_drop"):
            self.btn_save_drop.configure(state="normal")

    def set_loaded_status(self):
        self.has_unsaved_changes = False
        self._mask_labels_dirty = True
        self.btn_quick_save.configure(state="normal", text="Save (S)", fg_color="#4daafc", text_color="#222222")
        if hasattr(self, "btn_save_drop"):
            self.btn_save_drop.configure(state="normal")

    def set_saved_status(self):
        self.has_unsaved_changes = False
        self.btn_quick_save.configure(text="Saved!", fg_color="#3b9148", text_color="white")
        if hasattr(self, "btn_save_drop"):
            self.btn_save_drop.configure(state="normal")

    def setup_dimensions(self):
        """Reconfigure slice slider bounds for the active orthogonal view axis."""
        if self.img_data is None:
            return
        self.max_slice = max(0, self.img_data.shape[self.axis_index] - 1)
        self.current_slice = self.max_slice // 2
        if self.max_slice > 0:
            self.slider.configure(from_=0, to=self.max_slice, number_of_steps=self.max_slice)
            self.slider.set(self.current_slice)
            self.slider.configure(state="normal")
        else:
            self.slider.configure(from_=0, to=1, number_of_steps=1)
            self.slider.set(0)
            self.slider.configure(state="disabled")
        self.lbl_slice_info.configure(text=f"Slice: {self.current_slice + 1}/{self.max_slice + 1}")

    def change_view_axis(self):
        if self.img_data is None:
            return
        self.axis_index = self.view_var.get()
        self.setup_dimensions()
        self.update_views()

    def toggle_eraser(self):
        self.record_user_activity()
        self.is_eraser = self.eraser_var.get()

    def change_brush_size_event(self, event=None):
        self.record_user_activity()
        try:
            val = float(self.spin_brush_var.get())
            if val > 0:
                self.brush_size = val
                self.lbl_overlay.update_overlays_only()
        except ValueError:
            pass

    def change_wand_tol_event(self, event=None):
        self.record_user_activity()
        try:
            val = float(self.spin_wand_tol_var.get())
            if val >= 0:
                self.wand_tolerance = val
        except ValueError:
            pass

    def change_wand_bound_event(self, event=None):
        self.record_user_activity()
        try:
            val = int(self.spin_wand_bound_var.get())
            if val > 0:
                self.wand_limit = val
        except ValueError:
            pass

    def change_tool_mode(self):
        """Switch segmentation tool and toggle parameter controls accordingly."""
        self.record_user_activity()
        self.tool_mode = self.tool_var.get()
        self.check_poly_auto.configure(state="normal" if self.tool_mode == "freehand" else "disabled")
        wand_state = "normal" if self.tool_mode == "wand" else "disabled"
        self.spin_wand_tol.configure(state=wand_state)
        self.spin_wand_bound.configure(state=wand_state)
        self.lbl_overlay.update_overlays_only()

    def update_views(self):
        """Update main overlay canvas and preview mask-only canvas with LUT caching."""
        if self.img_data is None:
            return

        b = self.sb_brightness.get()
        c = self.sb_contrast.get() / 100.0
        state_signature = (self.current_index, self.current_slice, self.axis_index, b, c)

        if not hasattr(self, "_cached_view_state") or self._cached_view_state != state_signature:
            src_data = self.img_data
            if self.axis_index == 2:
                slice_raw = src_data[:, :, self.current_slice]
            elif self.axis_index == 1:
                slice_raw = src_data[:, self.current_slice, :]
            else:
                slice_raw = src_data[self.current_slice, :, :]

            if c != 1.0 or b != 0:
                lut = np.arange(256, dtype=np.float32) * c + b
                lut_8 = np.clip(lut, 0, 255).astype(np.uint8)
                adjusted = lut_8[slice_raw]
            else:
                adjusted = slice_raw

            self._cached_img_slice = np.ascontiguousarray(np.flipud(np.rot90(adjusted)))
            if self._cached_img_slice.dtype != np.uint8:
                s8 = (
                    (self._cached_img_slice >> 8).astype(np.uint8)
                    if self._cached_img_slice.dtype == np.uint16
                    else np.clip(self._cached_img_slice, 0, 255).astype(np.uint8)
                )
            else:
                s8 = self._cached_img_slice
            self._cached_img_slice_8 = s8
            self._cached_base_pil = Image.fromarray(s8)
            self._cached_view_state = state_signature

        img_slice_8 = self._cached_img_slice_8
        base_pil = self._cached_base_pil

        if self.axis_index == 2:
            mask_slice_raw = self.mask_data[:, :, self.current_slice]
        elif self.axis_index == 1:
            mask_slice_raw = self.mask_data[:, self.current_slice, :]
        else:
            mask_slice_raw = self.mask_data[self.current_slice, :, :]

        mask_slice = np.ascontiguousarray(np.flipud(np.rot90(mask_slice_raw)))
        h, w = img_slice_8.shape

        if not np.any(mask_slice):
            self.lbl_overlay.set_image(base_pil, img_slice_8=img_slice_8, mask_slice=None)
            if not getattr(self, "_mask_only_is_empty", False):
                if not hasattr(self, "_empty_mask_img") or self._empty_mask_img.size != (w, h):
                    self._empty_mask_img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
                self.lbl_mask_only.set_image(self._empty_mask_img)
                self._mask_only_is_empty = True
            return

        self._mask_only_is_empty = False
        alpha = self.mask_opacity
        inv_alpha = 255 - alpha
        is_multi = bool(self.multilabel_var.get())

        counts = np.bincount(mask_slice.ravel())
        available_labels = (np.flatnonzero(counts[1:]) + 1).tolist()
        max_label_val = available_labels[-1] if available_labels else 0

        if is_multi:
            needed_size = max(256, max_label_val + 1)
            if not hasattr(self, "_color_lut") or getattr(self, "_color_lut_dirty", True) or len(self._color_lut) < needed_size:
                lut = np.zeros((needed_size, 3), dtype=np.uint16)
                for val in range(1, needed_size):
                    default_cname = default_color_name_for_label(val)
                    cname = self.value_to_color.get(val, default_cname)
                    lut[val] = self.color_map.get(cname, LABEL_COLOR_MAP[default_cname])
                self._color_lut = lut
                self._color_lut_dirty = False
        else:
            if not hasattr(self, "_lut_blend_alpha") or self._lut_blend_alpha != alpha:
                lut_vals = np.arange(256, dtype=np.uint16)
                self._lut_blend_red = ((lut_vals * inv_alpha + 255 * alpha) // 255).astype(np.uint8)
                self._lut_blend_dim = ((lut_vals * inv_alpha) // 255).astype(np.uint8)
                self._lut_blend_alpha = alpha

        self.lbl_overlay.set_image(
            base_pil,
            img_slice_8=img_slice_8,
            mask_slice=mask_slice,
            color_lut=getattr(self, "_color_lut", None),
            alpha=alpha,
            is_multilabel=is_multi,
            lut_blend_red=getattr(self, "_lut_blend_red", None),
            lut_blend_dim=getattr(self, "_lut_blend_dim", None),
        )

        lut_size = max(256, max_label_val + 1)
        lut_mask = np.zeros(lut_size, dtype=np.uint8)
        if len(available_labels) == 1:
            lut_mask[available_labels[0]] = 255
        elif len(available_labels) > 1:
            n_labels = len(available_labels)
            min_g = max(60, int(round(255.0 / n_labels)))
            grays = np.linspace(min_g, 255, n_labels)
            for i, v in enumerate(available_labels):
                lut_mask[v] = int(round(grays[i]))

        self.lbl_mask_only.set_image(Image.fromarray(lut_mask[mask_slice]))
