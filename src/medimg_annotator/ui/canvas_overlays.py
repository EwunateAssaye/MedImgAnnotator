"""
Canvas cursor, dual-interpolation compositing, and vector/ObDet overlay helpers.
"""

from medimg_annotator.imports import cv2, Image, np

from medimg_annotator.core.geometry import chaikins_smooth

class CanvasOverlayMixin:
    """Provides composite image scaling, persistent cursor overlays, and ObDet overlays."""

    def _ensure_cursor_items(self):
        """Ensure persistent cursor items exist on the canvas above image layer."""
        if self._cursor_circle is None or not self.find_withtag("cursor_circle"):
            self._cursor_circle = self.create_oval(
                0, 0, 0, 0, outline="#ffff00", width=2, state="hidden", tags=("cursor_overlay", "cursor_circle")
            )
            self._cursor_cross_h = self.create_line(
                0, 0, 0, 0, fill="#00ffff", width=2, state="hidden", tags=("cursor_overlay", "cursor_cross")
            )
            self._cursor_cross_v = self.create_line(
                0, 0, 0, 0, fill="#00ffff", width=2, state="hidden", tags=("cursor_overlay", "cursor_cross")
            )

    def _hide_cursor(self):
        """Hide cursor items without canvas item recreation or background damage."""
        if self._cursor_circle is not None and self.find_withtag("cursor_circle"):
            self.itemconfigure(self._cursor_circle, state="hidden")
            self.itemconfigure(self._cursor_cross_h, state="hidden")
            self.itemconfigure(self._cursor_cross_v, state="hidden")

    def update_cursor_only(self):
        """Ultra-fast cursor positioning using canvas coords."""
        if not self.cursor_pos or self.editor is None or self.editor.img_data is None:
            self._hide_cursor()
            return

        x, y, rw, rh = self.get_draw_rect()
        if rw <= 0 or rh <= 0 or self.current_image is None:
            self._hide_cursor()
            return

        cx, cy = self.cursor_pos
        if not (x <= cx <= x + rw and y <= cy <= y + rh):
            self._hide_cursor()
            return

        self._ensure_cursor_items()
        pix_w, _ = self.current_image.size
        scale = rw / pix_w if pix_w else 1.0
        radius = float(self.editor.brush_size) * scale

        x1 = int(round(cx - radius))
        y1 = int(round(cy - radius))
        x2 = int(round(cx + radius))
        y2 = int(round(cy + radius))

        if self.editor.tool_mode == "brush":
            self.coords(self._cursor_circle, x1, y1, x2, y2)
            self.itemconfigure(self._cursor_circle, outline="#ffff00", width=2, state="normal")
            self.itemconfigure(self._cursor_cross_h, state="hidden")
            self.itemconfigure(self._cursor_cross_v, state="hidden")
        else:
            self.coords(self._cursor_circle, x1, y1, x2, y2)
            self.itemconfigure(self._cursor_circle, outline="#00ffff", width=2, state="normal")

            cx_i, cy_i = int(round(cx)), int(round(cy))
            self.coords(self._cursor_cross_h, cx_i - 5, cy_i, cx_i + 5, cy_i)
            self.coords(self._cursor_cross_v, cx_i, cy_i - 5, cx_i, cy_i + 5)
            self.itemconfigure(self._cursor_cross_h, fill="#00ffff", width=2, state="normal")
            self.itemconfigure(self._cursor_cross_v, fill="#00ffff", width=2, state="normal")

        self.tag_raise("cursor_overlay")

    def update_overlays_only(self):
        """Update active tool overlays (polygons, handles) and position cursor."""
        self.delete("tool_overlay")
        x, y, w, h = self.get_draw_rect()
        if w > 0 and h > 0:
            self.draw_tool_overlays(x, y, w, h)
        self.update_cursor_only()

    def _render_scaled_composite(self, w, h):
        """
        Render the viewport at target dimensions (w, h) using:
          - Image display: INTER_AREA when downsampling, INTER_LINEAR when upsampling.
          - Mask labels: INTER_NEAREST to preserve crisp discrete boundaries.
        """
        if self._img_slice_8 is not None:
            src_h, src_w = self._img_slice_8.shape[:2]
            img_key = (id(self._img_slice_8), w, h)
            if self._cached_img_key == img_key and self._cached_img_r is not None:
                img_r = self._cached_img_r
            else:
                if (w, h) == (src_w, src_h):
                    img_r = self._img_slice_8
                elif w < src_w and h < src_h:
                    img_r = cv2.resize(self._img_slice_8, (w, h), interpolation=cv2.INTER_AREA)
                else:
                    img_r = cv2.resize(self._img_slice_8, (w, h), interpolation=cv2.INTER_LINEAR)
                self._cached_img_key = img_key
                self._cached_img_r = img_r
                self._cached_img_rgb = None
                self._cached_img_pil = None

            if self._mask_slice is None or self._mask_alpha <= 0:
                if self._cached_img_pil is None:
                    self._cached_img_pil = Image.fromarray(img_r)
                return self._cached_img_pil

            mask_u8 = (self._mask_slice > 0).astype(np.uint8)
            bx, by, bw, bh = cv2.boundingRect(mask_u8)
            if bw <= 0 or bh <= 0:
                if self._cached_img_pil is None:
                    self._cached_img_pil = Image.fromarray(img_r)
                return self._cached_img_pil

            if self._cached_img_rgb is None:
                self._cached_img_rgb = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB)
            rgb = self._cached_img_rgb.copy()

            if (w, h) == (src_w, src_h):
                mask_r = self._mask_slice
            else:
                mask_r = cv2.resize(self._mask_slice, (w, h), interpolation=cv2.INTER_NEAREST)

            sx, sy = w / float(src_w), h / float(src_h)
            x0 = max(0, int(bx * sx) - 2)
            y0 = max(0, int(by * sy) - 2)
            x1 = min(w, int(np.ceil((bx + bw) * sx)) + 2)
            y1 = min(h, int(np.ceil((by + bh) * sy)) + 2)

            sub_mask = mask_r[y0:y1, x0:x1]
            m_act = sub_mask > 0
            if np.any(m_act):
                sub_img = img_r[y0:y1, x0:x1]
                sub_rgb = rgb[y0:y1, x0:x1]
                if self._is_multilabel and self._color_lut is not None and len(self._color_lut) > 0:
                    ap = sub_img[m_act].astype(np.uint16)
                    lut_idx = np.minimum(sub_mask[m_act], len(self._color_lut) - 1)
                    ac = self._color_lut[lut_idx]
                    alpha = self._mask_alpha
                    inv_alpha = 255 - alpha
                    sub_rgb[m_act, 0] = ((ap * inv_alpha + ac[:, 0] * alpha) // 255).astype(np.uint8)
                    sub_rgb[m_act, 1] = ((ap * inv_alpha + ac[:, 1] * alpha) // 255).astype(np.uint8)
                    sub_rgb[m_act, 2] = ((ap * inv_alpha + ac[:, 2] * alpha) // 255).astype(np.uint8)
                elif self._lut_blend_red is not None and self._lut_blend_dim is not None:
                    ap = sub_img[m_act]
                    sub_rgb[m_act, 0] = self._lut_blend_red[ap]
                    sub_rgb[m_act, 1] = self._lut_blend_dim[ap]
                    sub_rgb[m_act, 2] = self._lut_blend_dim[ap]
            return Image.fromarray(rgb)

        pix_w, pix_h = self.current_image.size
        resample_mode = Image.BOX if (w < pix_w and h < pix_h) else Image.BILINEAR
        return self.current_image.resize((w, h), resample_mode)

    def _draw_polygon_overlay(self):
        """Draw active freehand raw curve or Chaikin-smoothed control polygon."""
        if self.editor.tool_mode not in ("freehand", "bucket", "wand"):
            return

        if self.drawing_new and len(self.raw_points) > 1:
            flat_pts = []
            for p in self.raw_points:
                flat_pts.extend([int(p[0]), int(p[1])])
            self.create_line(flat_pts, fill="#00ffff", width=2, tags=("overlay", "tool_overlay"))
        elif not self.drawing_new and len(self.control_points) > 2:
            smooth_pts = chaikins_smooth(self.control_points, 4)
            flat_pts = []
            for p in smooth_pts:
                flat_pts.extend([int(p[0]), int(p[1])])
            flat_pts.extend([int(smooth_pts[0][0]), int(smooth_pts[0][1])])
            self.create_line(flat_pts, fill="#00ffff", width=2, tags=("overlay", "tool_overlay"))
            if self.editor.tool_mode == "freehand":
                for pt in self.control_points:
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    self.create_oval(
                        px - 3, py - 3, px + 3, py + 3,
                        fill="#ffa500", outline="#ffa500", tags=("overlay", "tool_overlay")
                    )

    def _draw_obdet_overlays(self, rx, ry, rw, rh):
        """Draw ObDet bounding boxes, corner grab handles, and compact class badges."""
        if not getattr(self.editor, "obdet_active", False):
            return

        boxes = getattr(self.editor, "obdet_boxes", [])
        active_idx = getattr(self.editor, "active_obdet_box_idx", None)
        pix_w, pix_h = self.current_image.size if self.current_image else (0, 0)
        if pix_w <= 0 or pix_h <= 0 or rw <= 0 or rh <= 0:
            return

        scale_x = rw / float(pix_w)
        scale_y = rh / float(pix_h)
        for idx, box in enumerate(boxes):
            if box.get("slice", 0) != self.editor.current_slice or box.get("axis", 2) != self.editor.axis_index:
                continue
            x1, y1, x2, y2 = box["bbox"]
            is_active = idx == active_idx
            box_color = "#00ff66" if is_active else "#00e5ff"
            border_w = 2 if is_active else 1

            sx1 = rx + x1 * scale_x
            sy1 = ry + y1 * scale_y
            sx2 = rx + x2 * scale_x
            sy2 = ry + y2 * scale_y

            self.create_rectangle(
                sx1, sy1, sx2, sy2, outline=box_color, width=border_w,
                tags=("overlay", "tool_overlay", "obdet_overlay")
            )

            if is_active:
                hs = 3.0
                for hx, hy in ((sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)):
                    self.create_rectangle(
                        hx - hs, hy - hs, hx + hs, hy + hs,
                        fill="#ffffff", outline="#00cc55", width=1,
                        tags=("overlay", "tool_overlay", "obdet_overlay")
                    )

            lbl_tag = str(box.get("label", "ObDet"))
            badge_text = f"{lbl_tag} #{idx + 1}" if len(boxes) > 1 else lbl_tag
            tag_w = max(24, len(badge_text) * 7 + 8)
            tag_h = 14

            if sy1 - tag_h - 1 >= ry:
                bg_y1, bg_y2, txt_y = sy1 - tag_h - 1, sy1, sy1 - (tag_h // 2) - 1
            elif sy2 + tag_h + 1 <= ry + rh:
                bg_y1, bg_y2, txt_y = sy2, sy2 + tag_h + 1, sy2 + (tag_h // 2) + 1
            else:
                bg_y1, bg_y2, txt_y = sy1, sy1 + tag_h + 1, sy1 + (tag_h // 2) + 1

            self.create_rectangle(
                sx1, bg_y1, sx1 + tag_w, bg_y2,
                fill="#1b1f23", outline=box_color, width=1,
                tags=("overlay", "tool_overlay", "obdet_overlay")
            )
            self.create_text(
                sx1 + 4, txt_y, text=badge_text, fill=box_color,
                font=("Arial", 8, "bold"), anchor="w",
                tags=("overlay", "tool_overlay", "obdet_overlay")
            )

    def draw_tool_overlays(self, rx, ry, rw, rh):
        """Render freehand polygon handles and ObDet boxes above the composite image."""
        if self.editor is None or self.editor.img_data is None:
            return
        self._draw_polygon_overlay()
        self._draw_obdet_overlays(rx, ry, rw, rh)
        self.tag_raise("overlay")
        self.tag_raise("cursor_overlay")
