"""
Hybrid Deformable Registration (Fast Symmetric Forces Demons) + Co-Moving
Signed Distance Map (SDM) 3D Multi-Slice Mask Interpolation Engine.
"""

from medimg_annotator.imports import cv2, messagebox, np, re, scipy_ndimage

def _get_centroid(sdm_arr):
    """Return (row, col) centroid of positive SDM region or peak distance point."""
    m = sdm_arr > 0
    if not np.any(m):
        idx = np.unravel_index(np.argmax(sdm_arr), sdm_arr.shape)
        return np.array([float(idx[0]), float(idx[1])])
    ys, xs = np.where(m)
    return np.array([float(ys.mean()), float(xs.mean())])

def _shift2d(arr, dy_dx, rw, rh, border_mode=cv2.BORDER_REPLICATE):
    """Translate a 2D array by (dy, dx) using bilinear OpenCV warping."""
    M = np.float32([[1.0, 0.0, float(dy_dx[1])], [0.0, 1.0, float(dy_dx[0])]])
    return cv2.warpAffine(arr, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=0.0)

class SliceInterpolationMixin:
    """Mixin implementing multi-slice Hybrid Demons + SDM mask interpolation."""

    def _to_sdm(self, mask_2d, spacing=(1.0, 1.0)):
        """Convert a binary 2D mask to an anisotropic Signed Distance Map (SDM)."""
        m = mask_2d.astype(bool)
        if not np.any(m):
            return -np.ones(m.shape, dtype=np.float32) * 1e4
        if np.all(m):
            return np.ones(m.shape, dtype=np.float32) * 1e4
        d_in = scipy_ndimage.distance_transform_edt(m, sampling=(spacing[1], spacing[0]))
        d_out = scipy_ndimage.distance_transform_edt(~m, sampling=(spacing[1], spacing[0]))
        return (d_in - d_out).astype(np.float32)

    def _find_translation_ncc(self, fixed_arr, moving_arr, cur_sdm_arr, max_shift=16):
        """Estimate coarse translational organ shift via masked Normalized Cross-Correlation."""
        roi = cur_sdm_arr > -4.0
        if not np.any(roi):
            return 0.0, 0.0
        ys, xs = np.where(roi)
        h, w = fixed_arr.shape
        y0, y1 = max(0, ys.min() - 4), min(h, ys.max() + 5)
        x0, x1 = max(0, xs.min() - 4), min(w, xs.max() + 5)
        if (y1 - y0) < 5 or (x1 - x0) < 5:
            return 0.0, 0.0
        tmpl = moving_arr[y0:y1, x0:x1]
        if np.std(tmpl) < 1e-4:
            return 0.0, 0.0
        shift_lim = max(max_shift, int(max(h, w) * 0.14))
        sy0, sy1 = max(0, y0 - shift_lim), min(h, y1 + shift_lim)
        sx0, sx1 = max(0, x0 - shift_lim), min(w, x1 + shift_lim)
        search = fixed_arr[sy0:sy1, sx0:sx1]
        if search.shape[0] <= tmpl.shape[0] or search.shape[1] <= tmpl.shape[1] or np.std(search) < 1e-4:
            return 0.0, 0.0
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.25:
            return 0.0, 0.0
        best_x, best_y = max_loc
        dy = float(y0 - (sy0 + best_y))
        dx = float(x0 - (sx0 + best_x))
        return dy, dx

    def _register_slices(self, f_arr, m_arr, cur_sdm_arr=None, max_reg_dim=112):
        """Coarse-to-fine Fast Symmetric Forces Demons deformable registration in 2D."""
        h, w = f_arr.shape
        if np.std(f_arr) < 1e-5 or np.std(m_arr) < 1e-5:
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        dy_px, dx_px = 0.0, 0.0
        if cur_sdm_arr is not None:
            dy_px, dx_px = self._find_translation_ncc(f_arr, m_arr, cur_sdm_arr)

        if cur_sdm_arr is not None:
            M_shift = np.float32([[1.0, 0.0, -dx_px], [0.0, 1.0, -dy_px]])
            shifted_sdm = cv2.warpAffine(cur_sdm_arr, M_shift, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            weight_f = cv2.GaussianBlur(np.clip((shifted_sdm + 18.0) * (1.0 / 8.0), 0.0, 1.0), (0, 0), 3.0)
            weight_m = cv2.GaussianBlur(np.clip((cur_sdm_arr + 18.0) * (1.0 / 8.0), 0.0, 1.0), (0, 0), 3.0)
            bg_f, bg_m = float(np.median(f_arr)), float(np.median(m_arr))
            f_g = bg_f + weight_f * (f_arr - bg_f)
            m_g = bg_m + weight_m * (m_arr - bg_m)
        else:
            f_g, m_g = f_arr, m_arr

        std_f, std_m = float(np.std(f_g)), float(np.std(m_g))
        if std_f > 1e-5 and std_m > 1e-5:
            f_g = (f_g - float(np.mean(f_g))) * (100.0 / std_f)
            m_g = (m_g - float(np.mean(m_g))) * (100.0 / std_m)

        scale = min(1.0, float(max_reg_dim) / float(max(h, w)))
        if scale < 0.95:
            nh, nw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
            f_w = cv2.resize(f_g, (nw, nh), interpolation=cv2.INTER_AREA)
            m_w = cv2.resize(m_g, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = float(nw) / float(w), float(nh) / float(h)
        else:
            nh, nw = h, w
            f_w, m_w = f_g, m_g
            sx, sy = 1.0, 1.0

        grid_x, grid_y = np.meshgrid(np.arange(nw, dtype=np.float32), np.arange(nh, dtype=np.float32))
        ux = np.full((nh, nw), dx_px * sx, dtype=np.float32)
        uy = np.full((nh, nw), dy_px * sy, dtype=np.float32)

        for img_sigma, reg_sigma, n_iters in [(2.0, 2.0, 22), (0.0, 1.2, 20)]:
            if img_sigma > 0.0:
                f_cur = cv2.GaussianBlur(f_w, (0, 0), img_sigma)
                m_cur = cv2.GaussianBlur(m_w, (0, 0), img_sigma)
            else:
                f_cur, m_cur = f_w, m_w
            gfx = cv2.Sobel(f_cur, cv2.CV_32F, 1, 0, ksize=3) * 0.125
            gfy = cv2.Sobel(f_cur, cv2.CV_32F, 0, 1, ksize=3) * 0.125
            for _ in range(n_iters):
                mw = cv2.remap(m_cur, grid_x + ux, grid_y + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                gmx = cv2.Sobel(mw, cv2.CV_32F, 1, 0, ksize=3) * 0.125
                gmy = cv2.Sobel(mw, cv2.CV_32F, 0, 1, ksize=3) * 0.125
                gx = 0.5 * (gfx + gmx)
                gy = 0.5 * (gfy + gmy)
                diff = f_cur - mw
                denom = gx * gx + gy * gy + 0.5 * diff * diff + 1e-6
                dux = np.clip(diff * gx / denom, -2.0, 2.0)
                duy = np.clip(diff * gy / denom, -2.0, 2.0)
                ux = cv2.GaussianBlur(ux + dux, (0, 0), reg_sigma)
                uy = cv2.GaussianBlur(uy + duy, (0, 0), reg_sigma)

        if scale < 0.95:
            ux = cv2.resize(ux, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / sx)
            uy = cv2.resize(uy, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / sy)
        return ux, uy

    def _band_ncc(self, fixed_arr, warped_arr, band):
        """Compute local boundary-band normalized cross-correlation confidence."""
        a = fixed_arr[band]
        b = warped_arr[band]
        if a.size < 20:
            return 0.5
        std_a, std_b = float(np.std(a)), float(np.std(b))
        if std_a < 1e-4 or std_b < 1e-4:
            return 0.5
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        denom = np.sqrt(float((a0 * a0).sum()) * float((b0 * b0).sum())) + 1e-8
        return max(0.05, float((a0 * b0).sum() / denom))

    def _propagate_sdm(self, imgs, sdm0_arr, order, grid_x, grid_y, band_mm=5.0):
        """Propagate an SDM along a slice sequence using chained Demons deformation."""
        out = {order[0]: (sdm0_arr, 1.0)}
        cur_arr, prev, conf = sdm0_arr, order[0], 1.0
        for step_idx, k in enumerate(order[1:]):
            ux, uy = self._register_slices(imgs[k], imgs[prev], cur_sdm_arr=cur_arr)
            map_x, map_y = grid_x + ux, grid_y + uy
            warped_img = cv2.remap(
                imgs[prev], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )
            cur_arr = cv2.remap(
                cur_arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-50.0
            )
            band = np.abs(cur_arr) < band_mm
            conf *= self._band_ncc(imgs[k], warped_img, band)
            out[k] = (cur_arr, conf)
            prev = k
            if (step_idx & 3) == 0:
                try:
                    self.update_idletasks()
                except Exception:
                    pass
        return out

    def _fast_sdm_interp(self, m1, m2, z1, z2):
        """Perform pure Signed Distance Map shape interpolation between two key slices."""

        def get_sdt(mask_2d):
            m = mask_2d.astype(bool) if isinstance(mask_2d, np.ndarray) else np.asarray(mask_2d, dtype=bool)
            if not np.any(m):
                return -np.ones(m.shape, dtype=float) * 1e4
            if np.all(m):
                return np.ones(m.shape, dtype=float) * 1e4
            return scipy_ndimage.distance_transform_edt(m) - scipy_ndimage.distance_transform_edt(~m)

        sdt1 = get_sdt(m1)
        sdt2 = get_sdt(m2)
        res = {}
        poly_auto = getattr(self, "poly_auto_var", None)
        use_filter = poly_auto is not None and hasattr(poly_auto, "get") and not poly_auto.get()
        for z in range(z1 + 1, z2):
            w = (z - z1) / float(z2 - z1)
            sdt_interp = (1.0 - w) * sdt1 + w * sdt2
            if use_filter:
                sdt_interp = scipy_ndimage.gaussian_filter(sdt_interp, sigma=0.8)
            res[z] = sdt_interp > 0
        return res

    def _get_2d_slice_spacing(self):
        """Resolve in-plane physical pixel spacing (sp_col, sp_row) for SDM transforms."""
        try:
            zooms = self.header.get_zooms() if (hasattr(self, "header") and self.header is not None) else (1.0, 1.0, 1.0)
            if self.axis_index == 2:
                sp_row, sp_col = float(zooms[0]), float(zooms[1])
            elif self.axis_index == 1:
                sp_row, sp_col = float(zooms[0]), float(zooms[2])
            else:
                sp_row, sp_col = float(zooms[1]), float(zooms[2])
            if sp_row > 0 and sp_col > 0 and not np.isnan(sp_row) and not np.isnan(sp_col):
                return (sp_col, sp_row)
        except Exception:
            pass
        return (1.0, 1.0)

    def _hybrid_demons_sdm_interp(self, slice_imgs, m1, m2, z1, z2, gating_dice=0.98):
        """
        Interpolate intermediate slices between z1 and z2 via bi-directional
        Demons propagation with co-moving endpoint drift and residual correction.
        """
        if not np.any(m1) or not np.any(m2):
            return self._fast_sdm_interp(m1, m2, z1, z2)

        inter = np.sum(m1 & m2)
        dice_val = 2.0 * inter / (np.sum(m1) + np.sum(m2) + 1e-8)
        if dice_val >= gating_dice:
            return self._fast_sdm_interp(m1, m2, z1, z2)

        union_mask = m1 | m2
        if not np.any(union_mask):
            return {z: np.zeros_like(m1) for z in range(z1 + 1, z2)}

        y_idx, x_idx = np.where(union_mask)
        pad = 32
        h, w = m1.shape
        y_min, y_max = max(0, y_idx.min() - pad), min(h, y_idx.max() + pad)
        x_min, x_max = max(0, x_idx.min() - pad), min(w, x_idx.max() + pad)

        try:
            sitk_spacing = self._get_2d_slice_spacing()
            sub_imgs = {k: slice_imgs[k][y_min:y_max, x_min:x_max].astype(np.float32) for k in range(z1, z2 + 1)}
            sub_m1 = m1[y_min:y_max, x_min:x_max]
            sub_m2 = m2[y_min:y_max, x_min:x_max]
            rh, rw = y_max - y_min, x_max - x_min
            grid_x, grid_y = np.meshgrid(np.arange(rw, dtype=np.float32), np.arange(rh, dtype=np.float32))

            sdm_a = self._to_sdm(sub_m1, spacing=sitk_spacing)
            sdm_b = self._to_sdm(sub_m2, spacing=sitk_spacing)

            fwd = self._propagate_sdm(sub_imgs, sdm_a, list(range(z1, z2 + 1)), grid_x, grid_y)
            bwd = self._propagate_sdm(sub_imgs, sdm_b, list(range(z2, z1 - 1, -1)), grid_x, grid_y)

            c_a, c_b = _get_centroid(sdm_a), _get_centroid(sdm_b)
            drift_f = c_b - _get_centroid(fwd[z2][0])
            drift_b = c_a - _get_centroid(bwd[z1][0])

            fwd_end_aligned = _shift2d(fwd[z2][0], drift_f, rw, rh, cv2.BORDER_REPLICATE)
            bwd_end_aligned = _shift2d(bwd[z1][0], drift_b, rw, rh, cv2.BORDER_REPLICATE)
            err_f_end = sdm_b - fwd_end_aligned
            err_b_end = sdm_a - bwd_end_aligned

            full_result = {}
            smooth_flag = not self.poly_auto_var.get()
            for z in range(z1 + 1, z2):
                t = (z - z1) / float(z2 - z1)
                F_shifted = _shift2d(fwd[z][0], t * drift_f, rw, rh, cv2.BORDER_REPLICATE)
                B_shifted = _shift2d(bwd[z][0], (1.0 - t) * drift_b, rw, rh, cv2.BORDER_REPLICATE)
                err_f_z = _shift2d(err_f_end, _get_centroid(F_shifted) - c_b, rw, rh, cv2.BORDER_CONSTANT)
                err_b_z = _shift2d(err_b_end, _get_centroid(B_shifted) - c_a, rw, rh, cv2.BORDER_CONSTANT)
                F = F_shifted + t * err_f_z
                B = B_shifted + (1.0 - t) * err_b_z
                wf, wb = (1.0 - t) * fwd[z][1], t * bwd[z][1]
                denom = wf + wb
                fused_sdm = (0.5 * F + 0.5 * B) if denom < 1e-6 else ((wf * F + wb * B) / denom)

                if smooth_flag:
                    fused_sdm = cv2.GaussianBlur(fused_sdm, (0, 0), 0.8)

                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[y_min:y_max, x_min:x_max] = fused_sdm > 0
                full_result[z] = full_mask

            return full_result
        except Exception as ex:
            print(f"[Hybrid Interpolation] Registration notice ({ex}), using SDM fallback.")
            return self._fast_sdm_interp(m1, m2, z1, z2)

    def apply_interpolation(self):
        """Interpolate active label masks between user-specified comma-separated key slices."""
        paint_val = self.get_paint_value()
        if paint_val == 0:
            messagebox.showwarning(
                "Warning",
                "Please select a valid label color to interpolate. You cannot interpolate the Eraser (0).",
            )
            return

        raw_text = self.line_slices.get()
        try:
            parts = re.split(r"[,\s]+", raw_text.strip())
            slice_indices = sorted(list({int(p) - 1 for p in parts if p}))
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Please enter valid slice numbers separated by commas (e.g., 10, 15).",
            )
            return

        valid_slices = [s for s in slice_indices if 0 <= s <= self.max_slice]
        if len(valid_slices) < 2:
            messagebox.showwarning(
                "Invalid Input",
                f"Please provide at least two valid slice numbers. Current valid range: 1 to {self.max_slice + 1}.",
            )
            return

        self.push_undo(force_3d=True)
        self.config(cursor="watch")
        self.update()
        try:
            for i in range(len(valid_slices) - 1):
                z1, z2 = valid_slices[i], valid_slices[i + 1]
                if z2 - z1 <= 1:
                    continue

                m1 = np.take(self.mask_data, z1, axis=self.axis_index) == paint_val
                m2 = np.take(self.mask_data, z2, axis=self.axis_index) == paint_val
                if not np.any(m1) and not np.any(m2):
                    continue

                slice_imgs = {}
                for z in range(z1, z2 + 1):
                    if self.axis_index == 2:
                        slice_imgs[z] = self.img_data_raw[:, :, z]
                    elif self.axis_index == 1:
                        slice_imgs[z] = self.img_data_raw[:, z, :]
                    else:
                        slice_imgs[z] = self.img_data_raw[z, :, :]

                interpolated_slices = self._hybrid_demons_sdm_interp(slice_imgs, m1, m2, z1, z2)

                for z in range(z1 + 1, z2):
                    interp_mask = interpolated_slices.get(z, None)
                    if interp_mask is None:
                        continue

                    if self.axis_index == 2:
                        slice_data, slice_img = self.mask_data[:, :, z], self.img_data_raw[:, :, z]
                    elif self.axis_index == 1:
                        slice_data, slice_img = self.mask_data[:, z, :], self.img_data_raw[:, z, :]
                    else:
                        slice_data, slice_img = self.mask_data[z, :, :], self.img_data_raw[z, :, :]

                    if np.any(interp_mask) and self.poly_auto_var.get():
                        snapped_mask = self.snap_to_edges(interp_mask, slice_img, max_iters=8)
                    else:
                        snapped_mask = interp_mask

                    slice_data[slice_data == paint_val] = 0
                    slice_data[snapped_mask] = paint_val
                    if (z & 3) == 0:
                        try:
                            self.update_idletasks()
                        except Exception:
                            pass

            self.set_unsaved_status()
        except Exception as e:
            print(f"Interpolation error: {e}")
            messagebox.showerror("Interpolation Error", f"Failed during interpolation: {e}")
        finally:
            self.config(cursor="")
            self.update_views_event()
            self.update_volume_stats()
