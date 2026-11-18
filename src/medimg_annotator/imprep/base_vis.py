"""
Matplotlib dual-slice preview, slider hover tracking, and live 2D/3D debouncing for ImPrep tools.
"""

from medimg_annotator.imports import ctk, messagebox, nib, np, threading

from medimg_annotator.imprep.config import COLORS

class ImPrepVisMixin:
    """Mixin supplying dark Matplotlib slice comparison and slider hover bindings."""

    def _style_axes(self, ax, title):
        ax.set_facecolor(COLORS["plot_axes"])
        ax.set_title(title, color=COLORS["plot_text"], fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=COLORS["plot_text"])
        ax.axis("off")

    def _render_axis_slice(self, ax, im_attr, slice_2d, title):
        """Render or incrementally update a Matplotlib image artist without recreating axes."""
        artist = getattr(self, im_attr, None)
        if artist is None:
            ax.clear()
            setattr(self, im_attr, ax.imshow(slice_2d, cmap="gray"))
            self._style_axes(ax, title)
            return
        artist.set_data(slice_2d)
        vmin, vmax = float(np.min(slice_2d)), float(np.max(slice_2d))
        if vmin != vmax:
            artist.set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(title, color=COLORS["plot_text"], fontsize=11, fontweight="bold", pad=8)

    def bind_realtime_entry(self, entry):
        """Bind KeyRelease to trigger real-time updates while editing parameters."""
        entry.bind("<KeyRelease>", lambda e: self.apply_realtime_change())

    def apply_realtime_change(self):
        """Render instant 2D slice output and schedule debounced 3D background preview."""
        if self.vol_in is None:
            return

        if self.vol_in.ndim >= 3:
            z = max(0, min(int(self.slider.get()), self.vol_in.shape[2] - 1))
            slice_in = np.flipud(np.rot90(self.vol_in[:, :, z]))
            try:
                slice_out = self.process_slice(slice_in)
                if slice_out is not None:
                    self._render_axis_slice(self.ax2, "_im2", slice_out, f"Output Slice (z={z}) [Live]")
                    self.canvas.draw_idle()
            except Exception:
                pass

        self.run_preview()

    def _on_canvas_scroll(self, event):
        delta = 1 if getattr(event, "delta", 0) > 0 else -1
        self.step_slice(delta)

    def step_slice(self, delta=1):
        """Advance or rewind preview slice by delta."""
        if self.vol_in is None or self.vol_in.ndim < 3:
            return
        max_s = self.vol_in.shape[2] - 1
        cur = int(self.slider.get())
        target = max(0, min(max_s, cur + delta))
        if target != cur:
            self.slider.set(target)
            self.on_slider_change(target)

    def register_slider_hover(self, slider):
        """Bind hover and mouse-wheel events so arrow keys control hovered sliders."""
        if not isinstance(slider, ctk.CTkSlider):
            return

        def on_enter(e):
            if getattr(self, "_leave_timer", None) is not None:
                try:
                    self.after_cancel(self._leave_timer)
                except Exception:
                    pass
                self._leave_timer = None
            controller = getattr(self, "controller", None)
            if controller:
                controller.hovered_slider = slider
            self.hovered_slider = slider

        def on_leave(e):
            def clear():
                self._leave_timer = None
                controller = getattr(self, "controller", None)
                if controller and getattr(controller, "hovered_slider", None) == slider:
                    controller.hovered_slider = None
                if getattr(self, "hovered_slider", None) == slider:
                    self.hovered_slider = None

            try:
                self._leave_timer = self.after(60, clear)
            except Exception:
                clear()

        def on_wheel(e):
            if slider.cget("state") == "disabled":
                return "break"
            delta = 1 if getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4 else -1
            controller = getattr(self, "controller", None)
            if controller and hasattr(controller, "increment_slider"):
                controller.increment_slider(slider, delta)
            return "break"

        for target in (slider, getattr(slider, "_canvas", None)):
            if target is not None:
                target.bind("<Enter>", on_enter, add="+")
                target.bind("<Leave>", on_leave, add="+")
                target.bind("<MouseWheel>", on_wheel, add="+")
                target.bind("<Button-4>", on_wheel, add="+")
                target.bind("<Button-5>", on_wheel, add="+")

    def auto_bind_sliders(self, root_widget=None):
        """Recursively bind hover/wheel tracking to all child CTkSliders."""
        if root_widget is None:
            root_widget = self
        if isinstance(root_widget, ctk.CTkSlider):
            self.register_slider_hover(root_widget)
            return
        try:
            for child in root_widget.winfo_children():
                if isinstance(child, ctk.CTkSlider):
                    self.register_slider_hover(child)
                else:
                    self.auto_bind_sliders(child)
        except Exception:
            pass

    def toggle_mode(self):
        if self.mode.get() == "batch":
            self.slider_frame.pack_forget()
            self.reset_plot()
        else:
            self.slider_frame.pack(fill="x", pady=4, padx=2)
            self.update_vis()

    def load_preview(self, path):
        """Load a single NIfTI volume from disk and start interactive preview."""
        try:
            nii = nib.load(path)
            d = nii.get_fdata()
            if d.ndim > 3 and d.shape[2] == 1 and d.shape[3] > 1:
                d = np.squeeze(d, axis=2)
            while d.ndim > 3:
                d = d[..., 0]
            if d.ndim == 2:
                d = d[:, :, np.newaxis]
            self.vol_in = d
            self.affine = nii.affine
            self.header = nii.header
            self.vol_out = None
            self._im1 = None
            self._im2 = None
            self.init_slider()
            self.update_vis()
            self.run_preview()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load NIfTI:\n{str(e)}")

    def init_slider(self):
        if self.vol_in is not None and self.vol_in.ndim >= 3:
            max_s = max(0, self.vol_in.shape[2] - 1)
            self.slider.configure(to=max(1, max_s), from_=0, number_of_steps=max(1, max_s))
            mid = max_s // 2
            self.slider.set(mid)
            self.slider.configure(state="normal" if max_s > 0 else "disabled")
            self._current_slice_idx = mid
            self.lbl_slice.configure(text=f"Slice: {mid} / {max_s}")

    def on_slider_change(self, val):
        if self.vol_in is not None and self.vol_in.ndim >= 3:
            max_s = max(0, self.vol_in.shape[2] - 1)
            z = max(0, min(max_s, int(float(val))))
        else:
            max_s = 0
            z = int(float(val))
        if getattr(self, "_current_slice_idx", None) == z:
            return
        self._current_slice_idx = z
        if self.vol_in is not None and self.vol_in.ndim >= 3:
            self.lbl_slice.configure(text=f"Slice: {z} / {max_s}")

        if getattr(self, "_slider_timer", None) is None:
            self._slider_timer = self.after(16, self._finish_slider_change)

    def _finish_slider_change(self):
        self._slider_timer = None
        self.update_vis()

    def run_preview(self, _=None):
        """Debounce full 3D volume computation by 150ms during interactive slider adjustments."""
        if self.vol_in is None:
            return
        if getattr(self, "_preview_timer", None) is not None:
            self.after_cancel(self._preview_timer)
        self._preview_timer = self.after(150, self._do_run_preview)

    def _do_run_preview(self):
        self._preview_timer = None
        if self.vol_in is None:
            return

        self._preview_thread_id = getattr(self, "_preview_thread_id", 0) + 1
        current_tid = self._preview_thread_id

        def worker(tid):
            try:
                out_vol, _ = self.process_volume(self.vol_in, self.affine)
                if getattr(self, "_preview_thread_id", 0) == tid:
                    self.vol_out = out_vol
                    self.after_idle(self.update_vis)
            except Exception as e:
                print(f"Preview computation error: {e}")

        threading.Thread(target=worker, args=(current_tid,), daemon=True).start()

    def update_vis(self, _=None):
        """Refresh side-by-side input vs output slice views."""
        if self.vol_in is None or self.vol_in.ndim < 3:
            return

        z = max(0, min(int(self.slider.get()), self.vol_in.shape[2] - 1))
        slice_in = np.flipud(np.rot90(self.vol_in[:, :, z]))
        self._render_axis_slice(self.ax1, "_im1", slice_in, f"Input Slice (z={z})")

        if self.vol_out is not None and self.vol_out.ndim >= 3:
            z_out = min(z, self.vol_out.shape[2] - 1)
            slice_out = np.flipud(np.rot90(self.vol_out[:, :, z_out]))
            self._render_axis_slice(self.ax2, "_im2", slice_out, f"Output Slice (z={z_out})")
        else:
            slice_out = None
            try:
                slice_out = self.process_slice(slice_in)
            except Exception:
                pass

            if slice_out is not None:
                self._render_axis_slice(self.ax2, "_im2", slice_out, f"Output Slice (z={z}) [Live]")
            elif getattr(self, "_im2", None) is not None:
                self.ax2.clear()
                self._im2 = None
                self._style_axes(self.ax2, "Output Pending")

        self.canvas.draw_idle()

    def reset_plot(self):
        self._im1 = None
        self._im2 = None
        self.ax1.clear()
        self.ax2.clear()
        self._style_axes(self.ax1, "Input Volume")
        self._style_axes(self.ax2, "Processed Output")
        self.canvas.draw_idle()
