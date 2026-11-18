"""
Interactive primary canvas handling slice zooming, panning, painting, and mouse wheel events.
"""

from medimg_annotator.imports import ImageTk, tk

from medimg_annotator.core.geometry import resample_points
from medimg_annotator.ui.canvas_overlays import CanvasOverlayMixin

class EditorCanvas(CanvasOverlayMixin, tk.Canvas):
    """Primary interactive 2D annotation canvas supporting zoom, pan, and segmentation tools."""

    def __init__(self, parent, editor=None, **kwargs):
        super().__init__(
            parent,
            bg="#16181a",
            highlightthickness=1,
            highlightbackground="#3c3f41",
            **kwargs,
        )
        self.editor = editor
        self.cursor_pos = None
        self.raw_points = []
        self.control_points = []
        self.drawing_new = False
        self.dragging_idx = None

        self.pan_x = 0
        self.pan_y = 0
        self.last_pan_pos = None
        self.current_image = None
        self.photo_image = None
        self._image_id = None

        # Dual-interpolation layer state
        self._img_slice_8 = None
        self._mask_slice = None
        self._color_lut = None
        self._mask_alpha = 75
        self._is_multilabel = False
        self._lut_blend_red = None
        self._lut_blend_dim = None
        self._render_token = 0
        self._last_render_token = -1
        self._cached_img_key = None
        self._cached_img_r = None
        self._cached_img_rgb = None
        self._cached_img_pil = None

        self._cursor_circle = None
        self._cursor_cross_h = None
        self._cursor_cross_v = None
        self._motion_pending = False
        self._last_brush_3d = None

        self.bind("<Enter>", self.on_enter)
        self.bind("<Motion>", self.on_mouse_move)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_mouse_press_left)
        self.bind("<B1-Motion>", self.on_mouse_drag_left)
        self.bind("<ButtonRelease-1>", self.on_mouse_release_left)

        self.bind("<ButtonPress-3>", self.on_mouse_press_right)
        self.bind("<B3-Motion>", self.on_mouse_drag_right)
        self.bind("<ButtonRelease-3>", self.on_mouse_release_right)

        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.bind("<Button-4>", self.on_mouse_wheel)
        self.bind("<Button-5>", self.on_mouse_wheel)

        self.bind("<Configure>", self.on_resize)

    def reset_pan(self):
        """Reset canvas translation offset back to center."""
        self.pan_x = 0
        self.pan_y = 0
        self.last_pan_pos = None
        self.repaint()

    def set_image(
        self,
        pil_image,
        img_slice_8=None,
        mask_slice=None,
        color_lut=None,
        alpha=75,
        is_multilabel=False,
        lut_blend_red=None,
        lut_blend_dim=None,
    ):
        """Update slice image/mask layers and composite rendering parameters."""
        self.current_image = pil_image
        self._img_slice_8 = img_slice_8
        self._mask_slice = mask_slice
        self._color_lut = color_lut
        self._mask_alpha = int(alpha)
        self._is_multilabel = bool(is_multilabel)
        self._lut_blend_red = lut_blend_red
        self._lut_blend_dim = lut_blend_dim
        self._render_token += 1
        self.repaint()

    def get_draw_rect(self):
        """Compute image draw rectangle (x, y, w, h) incorporating zoom and pan offsets."""
        if self.current_image is None or self.editor is None:
            return 0, 0, 0, 0
        pix_w, pix_h = self.current_image.size
        c_w = self.winfo_width()
        c_h = self.winfo_height()
        if c_w <= 1 or c_h <= 1:
            return 0, 0, 0, 0

        if getattr(self.editor, "obdet_active", False):
            zoom_factor = 1.0
        else:
            zoom_factor = self.editor.sb_zoom.get() / 100.0
        ratio = min(c_w / pix_w, c_h / pix_h)
        new_w = int(pix_w * ratio * zoom_factor)
        new_h = int(pix_h * ratio * zoom_factor)
        if getattr(self.editor, "obdet_active", False):
            x_offset = (c_w - new_w) // 2
            y_offset = (c_h - new_h) // 2
        else:
            x_offset = (c_w - new_w) // 2 + self.pan_x
            y_offset = (c_h - new_h) // 2 + self.pan_y
        return x_offset, y_offset, new_w, new_h

    def repaint(self):
        """Repaint image texture if resized/updated and refresh tool/cursor overlays."""
        if self.current_image is None or self.editor is None:
            self.delete("all")
            self._image_id = None
            self._cursor_circle = None
            self._cursor_cross_h = None
            self._cursor_cross_v = None
            return
        x, y, w, h = self.get_draw_rect()
        if w <= 0 or h <= 0:
            return

        if (
            not hasattr(self, "_last_size")
            or self._last_size != (w, h)
            or getattr(self, "_last_image", None) is not self.current_image
            or getattr(self, "_last_render_token", -1) != self._render_token
        ):
            resized = self._render_scaled_composite(w, h)
            self.photo_image = ImageTk.PhotoImage(resized)
            self._last_size = (w, h)
            self._last_image = self.current_image
            self._last_render_token = self._render_token

        if self._image_id is None:
            self.delete("all")
            self._cursor_circle = None
            self._cursor_cross_h = None
            self._cursor_cross_v = None
            self._image_id = self.create_image(x, y, anchor=tk.NW, image=self.photo_image)
        else:
            self.coords(self._image_id, x, y)
            self.itemconfig(self._image_id, image=self.photo_image)

        self._ensure_cursor_items()
        self.tag_raise("tool_overlay")
        self.tag_raise("cursor_overlay")
        self.update_overlays_only()

    def handle_interaction(self, screen_x, screen_y, is_drag=False):
        """Dispatch brush, bucket, or wand actions at mapped 3D voxel coordinates."""
        if self.editor is None or self.editor.img_data is None:
            return
        if hasattr(self.editor, "record_user_activity"):
            self.editor.record_user_activity()
        cx, cy, cz = self.editor.get_3d_coords_from_screen((screen_x, screen_y), self)
        if cx is None:
            return

        mode = self.editor.tool_mode
        if mode == "brush":
            if is_drag and getattr(self, "_last_brush_3d", None) is not None:
                last_cx, last_cy, last_cz = self._last_brush_3d
                if (last_cx, last_cy, last_cz) != (None, None, None):
                    self.editor.apply_brush_line(last_cx, last_cy, last_cz, cx, cy, cz)
            else:
                self.editor.apply_brush(cx, cy, cz)
            self._last_brush_3d = (cx, cy, cz)
        elif mode == "bucket":
            self.editor.apply_bucket(cx, cy, cz)
        elif mode == "wand":
            self.editor.apply_wand(cx, cy, cz)

    def on_resize(self, event):
        self.repaint()

    def on_enter(self, event):
        self.focus_set()

    def on_leave(self, event):
        self.cursor_pos = None
        self._hide_cursor()

    def on_mouse_move(self, event):
        self.cursor_pos = (event.x, event.y)
        if not self._motion_pending:
            self._motion_pending = True
            self.after_idle(self._flush_mouse_move)

    def _flush_mouse_move(self):
        self._motion_pending = False
        if self.cursor_pos and self.editor and hasattr(self.editor, "record_user_activity"):
            x, y, rw, rh = self.get_draw_rect()
            if rw > 0 and rh > 0 and (x <= self.cursor_pos[0] <= x + rw and y <= self.cursor_pos[1] <= y + rh):
                self.editor.record_user_activity()

        if self.editor and getattr(self.editor, "obdet_active", False) and self.cursor_pos:
            cursor = self.editor.get_obdet_cursor_at(self.cursor_pos[0], self.cursor_pos[1], self)
            self.config(cursor=cursor)
        elif not getattr(self, "last_pan_pos", None):
            self.config(cursor="")
        self.update_cursor_only()

    def on_mouse_press_left(self, event):
        if self.editor and hasattr(self.editor, "record_user_activity"):
            self.editor.record_user_activity()
        if self.editor and getattr(self.editor, "obdet_active", False):
            self.editor.on_obdet_canvas_click(event.x, event.y, self)
            return
        if self.editor.tool_mode == "freehand":
            if self.control_points:
                min_dist, min_idx = 999999.0, None
                for i, pt in enumerate(self.control_points):
                    dx, dy = pt[0] - event.x, pt[1] - event.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < 8.0:
                        min_dist, min_idx = dist, i
                if min_dist < 8.0:
                    self.dragging_idx = min_idx
                    return
            self.raw_points = [(event.x, event.y)]
            self.drawing_new = True
            self.control_points = []
            self.update_overlays_only()
        else:
            if self.editor:
                self.editor.push_undo()
            self._last_brush_3d = self.editor.get_3d_coords_from_screen((event.x, event.y), self)
            self.handle_interaction(event.x, event.y, is_drag=False)
            self.update_cursor_only()

    def on_mouse_drag_left(self, event):
        if self.editor and hasattr(self.editor, "record_user_activity"):
            self.editor.record_user_activity()
        self.cursor_pos = (event.x, event.y)
        if self.editor and getattr(self.editor, "obdet_active", False):
            self.editor.on_obdet_drag(event.x, event.y, self)
            return
        if self.editor.tool_mode == "freehand":
            if self.drawing_new:
                self.raw_points.append((event.x, event.y))
            elif self.dragging_idx is not None:
                self.control_points[self.dragging_idx] = (event.x, event.y)
            self.update_overlays_only()
        elif self.editor.tool_mode == "brush":
            self.handle_interaction(event.x, event.y, is_drag=True)
            self.update_cursor_only()

    def on_mouse_release_left(self, event):
        if self.editor and getattr(self.editor, "obdet_active", False):
            if hasattr(self.editor, "on_obdet_release"):
                self.editor.on_obdet_release()
            return
        if self.editor.tool_mode == "freehand":
            if self.drawing_new:
                self.drawing_new = False
                if len(self.raw_points) > 5:
                    self.control_points = resample_points(self.raw_points, n_pts=25)
                else:
                    self.raw_points, self.control_points = [], []
            if self.dragging_idx is not None:
                self.dragging_idx = None
            self.update_overlays_only()

        self._last_brush_3d = None
        if self.editor and hasattr(self.editor, "update_volume_stats"):
            self.editor.update_volume_stats()

    def on_mouse_press_right(self, event):
        self.last_pan_pos = (event.x, event.y)
        self.config(cursor="fleur")

    def on_mouse_drag_right(self, event):
        if self.last_pan_pos is not None:
            dx = event.x - self.last_pan_pos[0]
            dy = event.y - self.last_pan_pos[1]
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_pos = (event.x, event.y)
            if self.control_points:
                self.control_points = [(p[0] + dx, p[1] + dy) for p in self.control_points]
            if self.raw_points:
                self.raw_points = [(p[0] + dx, p[1] + dy) for p in self.raw_points]
            self.repaint()

    def on_mouse_release_right(self, event):
        self.last_pan_pos = None
        self.config(cursor="")

    def on_mouse_wheel(self, event):
        if self.editor is None:
            return "break"
        if hasattr(self.editor, "record_user_activity"):
            self.editor.record_user_activity()

        if event.num == 4:
            direction = 1
        elif event.num == 5:
            direction = -1
        else:
            direction = 1 if event.delta > 0 else -1

        if self.editor and getattr(self.editor, "obdet_active", False):
            if getattr(self.editor, "obdet_boxes", None) and getattr(self.editor, "active_obdet_box_idx", None) is not None:
                self.editor.on_obdet_wheel(direction, self)
                return "break"

        shift_held = (event.state & 0x0001) != 0
        ctrl_held = (event.state & 0x0004) != 0

        if shift_held:
            sb = getattr(self.editor, "sb_brightness", None)
            if sb is not None:
                step = abs(sb.cget("to") - sb.cget("from_")) * 0.05 or 1
                min_val = min(sb.cget("from_"), sb.cget("to"))
                max_val = max(sb.cget("from_"), sb.cget("to"))
                new_val = max(min_val, min(max_val, sb.get() + direction * step))
                sb.set(new_val)
                if sb._command:
                    sb._command(new_val)
            return "break"

        if ctrl_held:
            if hasattr(self.editor, "brush_size") and hasattr(self.editor, "spin_brush_var"):
                new_r = max(1, min(100, self.editor.brush_size + direction * 1))
                self.editor.brush_size = new_r
                self.editor.spin_brush_var.set(str(new_r))
                self.update_cursor_only()
            return "break"

        slider = getattr(self.editor, "slider", None)
        if slider is not None and str(slider.cget("state")) != "disabled":
            try:
                f_val = float(slider.cget("from_"))
            except Exception:
                try:
                    f_val = float(slider.cget("from"))
                except Exception:
                    f_val = 0.0
            try:
                t_val = float(slider.cget("to"))
            except Exception:
                try:
                    t_val = float(slider.cget("to_"))
                except Exception:
                    t_val = 100.0
            min_val, max_val = min(f_val, t_val), max(f_val, t_val)
            cur_val = float(slider.get())
            new_val = max(min_val, min(max_val, cur_val + direction))
            if cur_val != new_val:
                slider.set(new_val)
                cmd = getattr(slider, "_command", None) or getattr(slider, "command", None)
                if callable(cmd):
                    cmd(new_val)

        return "break"
