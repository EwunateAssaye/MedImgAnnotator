"""
Interactive 2D/3D segmentation algorithms:
Smart Brush, Multi-Mode Flood Fill Bucket, Freehand Polygon with IFT Watershed
edge-snapping, and Bounded Magic Wand region growing.
"""

from medimg_annotator.imports import cv2, functools, messagebox, np, MplPath as Path, scipy_ndimage

from medimg_annotator.core.geometry import chaikins_smooth

def snap_mask_to_edges(mask_2d, image_2d, max_iters=5):
    """
    Refine a 2D binary mask onto high-contrast anatomical boundaries using
    ROI-cropped Image Foresting Transform (IFT) seeded watershed (`scipy.ndimage.watershed_ift`).
    """
    if not np.any(mask_2d) or np.all(mask_2d):
        return mask_2d

    ys, xs = np.where(mask_2d)
    h, w = mask_2d.shape
    pad = max_iters + 4
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)

    sub_mask = mask_2d[y0:y1, x0:x1]
    sub_img = image_2d[y0:y1, x0:x1].astype(np.float32)

    core = scipy_ndimage.binary_erosion(sub_mask, iterations=max_iters)
    if not np.any(core):
        dist = scipy_ndimage.distance_transform_edt(sub_mask)
        core = dist == np.max(dist)

    bound = scipy_ndimage.binary_dilation(sub_mask, iterations=max_iters)

    markers = np.zeros(sub_mask.shape, dtype=np.int16)
    markers[core] = 1
    markers[~bound] = 2

    grad = scipy_ndimage.gaussian_gradient_magnitude(sub_img, sigma=1.0)
    grad_max = np.max(grad)
    if grad_max == 0:
        return mask_2d

    grad_norm = (grad / grad_max * 255).astype(np.uint8)
    labels = scipy_ndimage.watershed_ift(grad_norm, markers)

    out_mask = np.zeros_like(mask_2d, dtype=bool)
    out_mask[y0:y1, x0:x1] = labels == 1
    return out_mask

def _slice_pair_and_uv(img_vol, mask_vol, axis_index, cx, cy, cz):
    """Return `(img_slice_2d, mask_slice_2d, u, v)` matching active plane orientation."""
    if axis_index == 2:
        return (img_vol[:, :, cz] if img_vol is not None else None), mask_vol[:, :, cz], cx, cy
    if axis_index == 1:
        return (img_vol[:, cy, :] if img_vol is not None else None), mask_vol[:, cy, :], cx, cz
    return (img_vol[cx, :, :] if img_vol is not None else None), mask_vol[cx, :, :], cy, cz

def _polygon_to_slice_coords(canvas_widget, rx, ry, rw, rh):
    """Map smoothed canvas polygon control points into 2D image slice coordinates."""
    smooth_pts = chaikins_smooth(canvas_widget.control_points, iterations=4)
    smooth_pts.append(smooth_pts[0])
    pix_w, pix_h = canvas_widget.current_image.size
    scale_x = pix_w / rw
    scale_y = pix_h / rh

    poly_2d = []
    for pt in smooth_pts:
        vis_x = int((pt[0] - rx) * scale_x)
        vis_y = int((pt[1] - ry) * scale_y)
        poly_2d.append((max(0, min(vis_x, pix_w - 1)), max(0, min(vis_y, pix_h - 1))))
    return poly_2d

class SegmentationMixin:
    """Mixin implementing interactive 2D/3D Brush, Bucket, Polygon, and Magic Wand tools."""

    def get_paint_value(self):
        """Return integer paint value (0 for eraser, or active label ID)."""
        if self.is_eraser:
            return 0
        if self.multilabel_var.get():
            try:
                return max(1, int(self.combo_value.get()))
            except ValueError:
                return 1
        return 1

    def get_3d_coords_from_screen(self, pos, canvas_widget):
        """Convert screen canvas (x, y) pixel coordinates into 3D volume (x, y, z) indices."""
        rx, ry, rw, rh = canvas_widget.get_draw_rect()
        if rw == 0 or rh == 0:
            return None, None, None

        pos_x, pos_y = pos
        if not (rx <= pos_x <= rx + rw and ry <= pos_y <= ry + rh):
            return None, None, None

        pix_w, pix_h = canvas_widget.current_image.size
        vis_x = max(0, min(int((pos_x - rx) * (pix_w / rw)), pix_w - 1))
        vis_y = max(0, min(int((pos_y - ry) * (pix_h / rh)), pix_h - 1))

        if self.axis_index == 2:
            return vis_x, vis_y, self.current_slice
        if self.axis_index == 1:
            return vis_x, self.current_slice, vis_y
        return self.current_slice, vis_x, vis_y

    def snap_to_edges(self, mask_2d, image_2d, max_iters=5):
        """Snap a 2D region mask to anatomical boundaries using IFT watershed."""
        return snap_mask_to_edges(mask_2d, image_2d, max_iters=max_iters)

    @functools.lru_cache(maxsize=32)
    def _get_brush_kernel_2d(self, r):
        ir = int(round(r))
        ky, kx = np.ogrid[-ir : ir + 1, -ir : ir + 1]
        return ir, (kx * kx + ky * ky) <= (r * r)

    @functools.lru_cache(maxsize=16)
    def _get_brush_kernel_3d(self, r):
        ir = int(round(r))
        kz, ky, kx = np.ogrid[-ir : ir + 1, -ir : ir + 1, -ir : ir + 1]
        return ir, (kx * kx + ky * ky + kz * kz) <= (r * r)

    def _apply_brush_core(self, cx, cy, cz):
        r = float(self.brush_size)
        val = self.get_paint_value()

        if self.scope_var.get() == "2D":
            ir, kernel = self._get_brush_kernel_2d(r)
            _, slice_view, u, v = _slice_pair_and_uv(None, self.mask_data, self.axis_index, cx, cy, cz)
            max_u, max_v = slice_view.shape
            u_min, u_max = max(0, u - ir), min(max_u, u + ir + 1)
            v_min, v_max = max(0, v - ir), min(max_v, v + ir + 1)
            if u_min < u_max and v_min < v_max:
                ku_min = u_min - (u - ir)
                kv_min = v_min - (v - ir)
                sub_k = kernel[ku_min : ku_min + (u_max - u_min), kv_min : kv_min + (v_max - v_min)]
                slice_view[u_min:u_max, v_min:v_max][sub_k] = val
        else:
            ir, kernel = self._get_brush_kernel_3d(r)
            max_x, max_y, max_z = self.mask_data.shape
            x_min, x_max = max(0, cx - ir), min(max_x, cx + ir + 1)
            y_min, y_max = max(0, cy - ir), min(max_y, cy + ir + 1)
            z_min, z_max = max(0, cz - ir), min(max_z, cz + ir + 1)
            if x_min < x_max and y_min < y_max and z_min < z_max:
                kx_min = x_min - (cx - ir)
                ky_min = y_min - (cy - ir)
                kz_min = z_min - (cz - ir)
                sub_k = kernel[
                    kx_min : kx_min + (x_max - x_min),
                    ky_min : ky_min + (y_max - y_min),
                    kz_min : kz_min + (z_max - z_min),
                ]
                self.mask_data[x_min:x_max, y_min:y_max, z_min:z_max][sub_k] = val

    def apply_brush(self, cx, cy, cz):
        """Stamp the 2D or 3D brush kernel at (cx, cy, cz)."""
        self._apply_brush_core(cx, cy, cz)
        self.update_views_event()

    def apply_brush_line(self, x0, y0, z0, x1, y1, z1):
        """Interpolate continuous brush stamps along a drag stroke to eliminate gaps."""
        dist = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        if dist == 0:
            self._apply_brush_core(x1, y1, z1)
            self.update_views_event()
            return

        r = float(self.brush_size)
        step_dist = max(1.0, r * 0.45)
        steps = max(1, int(np.ceil(dist / step_dist)))

        xs = np.linspace(x0, x1, steps + 1)
        ys = np.linspace(y0, y1, steps + 1)
        zs = np.linspace(z0, z1, steps + 1)

        for x, y, z in zip(xs, ys, zs):
            self._apply_brush_core(int(round(x)), int(round(y)), int(round(z)))

        self.update_views_event()

    def _try_fill_active_polygon(self, cx, cy, cz, paint_val):
        """Fill or erase inside the closed freehand polygon if clicked inside it. Returns True if handled."""
        if not (self.lbl_overlay.control_points and self.lbl_overlay.current_image is not None):
            return False
        rx, ry, rw, rh = self.lbl_overlay.get_draw_rect()
        if rw <= 0 or rh <= 0:
            return False

        poly_2d = _polygon_to_slice_coords(self.lbl_overlay, rx, ry, rw, rh)
        path = Path(poly_2d)
        slice_img, slice_mask, seed_u, seed_v = _slice_pair_and_uv(
            self.img_data_raw, self.mask_data, self.axis_index, cx, cy, cz
        )
        if not path.contains_point((seed_u, seed_v)):
            return False

        h, w = slice_mask.shape
        pts_cv = np.array([[p[1], p[0]] for p in poly_2d], dtype=np.int32)
        poly_canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_canvas, [pts_cv], 1)
        mask_poly = poly_canvas > 0

        if self.poly_auto_var.get() and not self.is_eraser:
            mask_poly = self.snap_to_edges(mask_poly, slice_img, max_iters=8)

        fill_val = 0 if self.is_eraser else paint_val
        slice_mask[mask_poly] = fill_val
        self.update_views_event()
        self.update_volume_stats()
        return True

    def apply_bucket(self, cx, cy, cz):
        """Apply 2D/3D flood-fill, polygon fill, or connected-component erase."""
        is_3d_op = self.scope_var.get() == "3D"
        paint_val = self.get_paint_value()

        if self._try_fill_active_polygon(cx, cy, cz, paint_val):
            return

        self.config(cursor="watch")
        self.update()
        try:
            if not self.is_eraser:
                if is_3d_op:
                    val_at_seed = self.mask_data[cx, cy, cz]
                    if val_at_seed == paint_val:
                        return
                    labeled_volume, _ = scipy_ndimage.label(self.mask_data == val_at_seed)
                    target_id = labeled_volume[cx, cy, cz]
                    if val_at_seed == 0:
                        corner_labels = {
                            labeled_volume[0, 0, 0],
                            labeled_volume[-1, 0, 0],
                            labeled_volume[0, -1, 0],
                            labeled_volume[0, 0, -1],
                            labeled_volume[-1, -1, -1],
                        }
                        if target_id in corner_labels:
                            messagebox.showwarning(
                                "Safety Guard",
                                "3D Bucket fill is disabled on the open exterior background to prevent accidental volume overwrites.",
                            )
                            return
                    if target_id > 0:
                        self.mask_data[labeled_volume == target_id] = paint_val
                else:
                    _, slice_2d, seed_u, seed_v = _slice_pair_and_uv(
                        None, self.mask_data, self.axis_index, cx, cy, cz
                    )
                    val_at_seed = slice_2d[seed_u, seed_v]
                    if val_at_seed == paint_val:
                        return
                    labeled_slice, _ = scipy_ndimage.label(slice_2d == val_at_seed)
                    target_id = labeled_slice[seed_u, seed_v]
                    if target_id > 0:
                        slice_2d[labeled_slice == target_id] = paint_val
            else:
                r = float(self.brush_size)
                if is_3d_op:
                    labeled_array, num_features = scipy_ndimage.label(self.mask_data > 0)
                    if num_features == 0:
                        return
                    x_min, x_max = max(0, int(cx - r)), min(self.mask_data.shape[0], int(cx + r + 1))
                    y_min, y_max = max(0, int(cy - r)), min(self.mask_data.shape[1], int(cy + r + 1))
                    z_min, z_max = max(0, int(cz - r)), min(self.mask_data.shape[2], int(cz + r + 1))
                    gx, gy, gz = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
                    dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
                    local_labels = labeled_array[x_min:x_max, y_min:y_max, z_min:z_max]
                    intersecting = np.unique(local_labels[dist_sq <= r * r])
                    labels_list = [lbl for lbl in intersecting if lbl != 0]
                    if labels_list:
                        self.mask_data[np.isin(labeled_array, labels_list)] = 0
                else:
                    _, slice_2d, u, v = _slice_pair_and_uv(None, self.mask_data, self.axis_index, cx, cy, cz)
                    labeled_slice, num_features = scipy_ndimage.label(slice_2d > 0)
                    if num_features == 0:
                        return
                    u_min, u_max = max(0, int(u - r)), min(slice_2d.shape[0], int(u + r + 1))
                    v_min, v_max = max(0, int(v - r)), min(slice_2d.shape[1], int(v + r + 1))
                    gu, gv = np.ogrid[u_min:u_max, v_min:v_max]
                    dist_sq = (gu - u) ** 2 + (gv - v) ** 2
                    local_labels = labeled_slice[u_min:u_max, v_min:v_max]
                    intersecting = np.unique(local_labels[dist_sq <= r * r])
                    labels_list = [lbl for lbl in intersecting if lbl != 0]
                    if labels_list:
                        slice_2d[np.isin(labeled_slice, labels_list)] = 0

            self.update_views_event()
        except Exception:
            pass
        finally:
            self.config(cursor="")
            self.update_volume_stats()

    def apply_wand(self, cx, cy, cz):
        """Apply bounded intensity-similarity Magic Wand region growing in 2D or 3D."""
        is_3d_op = self.scope_var.get() == "3D"
        paint_val = self.get_paint_value()
        tolerance = self.wand_tolerance
        bound = self.wand_limit

        self.config(cursor="watch")
        self.update()
        try:
            seed_intensity = self.img_data_raw[cx, cy, cz]

            if is_3d_op:
                max_x, max_y, max_z = self.img_data_raw.shape
                x_min, x_max = max(0, cx - bound), min(max_x, cx + bound + 1)
                y_min, y_max = max(0, cy - bound), min(max_y, cy + bound + 1)
                z_min, z_max = max(0, cz - bound), min(max_z, cz + bound + 1)

                sub_img = self.img_data_raw[x_min:x_max, y_min:y_max, z_min:z_max]
                valid_mask = np.abs(sub_img.astype(np.float32) - float(seed_intensity)) <= tolerance
                labels, _ = scipy_ndimage.label(valid_mask)
                target_id = labels[cx - x_min, cy - y_min, cz - z_min]

                if target_id > 0:
                    mask_update = labels == target_id
                    sub_mask = self.mask_data[x_min:x_max, y_min:y_max, z_min:z_max]
                    sub_mask[mask_update] = 0 if self.is_eraser else paint_val
            else:
                slice_img, slice_mask, u, v = _slice_pair_and_uv(
                    self.img_data_raw, self.mask_data, self.axis_index, cx, cy, cz
                )
                max_u, max_v = slice_img.shape
                u_min, u_max = max(0, u - bound), min(max_u, u + bound + 1)
                v_min, v_max = max(0, v - bound), min(max_v, v + bound + 1)

                sub_img = slice_img[u_min:u_max, v_min:v_max]
                valid_mask = np.abs(sub_img.astype(np.float32) - float(seed_intensity)) <= tolerance

                if self.lbl_overlay.control_points and self.lbl_overlay.current_image is not None:
                    rx, ry, rw, rh = self.lbl_overlay.get_draw_rect()
                    if rw > 0 and rh > 0:
                        poly_2d = _polygon_to_slice_coords(self.lbl_overlay, rx, ry, rw, rh)
                        if Path(poly_2d).contains_point((u, v)):
                            h, w = sub_img.shape
                            sub_poly = [(p[0] - u_min, p[1] - v_min) for p in poly_2d]
                            pts_cv = np.array([[p[1], p[0]] for p in sub_poly], dtype=np.int32)
                            poly_canvas = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(poly_canvas, [pts_cv], 1)
                            valid_mask = valid_mask & (poly_canvas > 0)

                labels, _ = scipy_ndimage.label(valid_mask)
                target_id = labels[u - u_min, v - v_min]

                if target_id > 0:
                    mask_update = labels == target_id
                    sub_mask_view = slice_mask[u_min:u_max, v_min:v_max]
                    sub_mask_view[mask_update] = 0 if self.is_eraser else paint_val

        except Exception as e:
            print(e)
        finally:
            self.config(cursor="")
            self.update_views_event()
            self.update_volume_stats()
