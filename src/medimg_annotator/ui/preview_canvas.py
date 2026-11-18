"""
Lightweight read-only preview canvas for displaying binary/multi-label mask slices.
"""

from medimg_annotator.imports import Image, ImageTk, tk

class PreviewCanvas(tk.Canvas):
    """Aspect-ratio-preserving canvas for secondary mask-only preview rendering."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg="#16181a",
            highlightthickness=1,
            highlightbackground="#3c3f41",
            **kwargs,
        )
        self.current_image = None
        self.photo_image = None
        self._image_id = None
        self._last_rect = None
        self._last_size = None
        self._last_image = None

        self.bind("<Configure>", self.on_resize)

    def set_image(self, pil_image):
        """Assign a new PIL image and trigger an incremental repaint."""
        self.current_image = pil_image
        self.repaint()

    def get_draw_rect(self):
        """Compute centered bounding box (x, y, w, h) fitted to the viewport."""
        if self.current_image is None:
            return 0, 0, 0, 0
        pix_w, pix_h = self.current_image.size
        c_w = self.winfo_width()
        c_h = self.winfo_height()
        if c_w <= 1 or c_h <= 1:
            return 0, 0, 0, 0

        ratio = min(c_w / pix_w, c_h / pix_h)
        new_w = int(pix_w * ratio)
        new_h = int(pix_h * ratio)
        x_offset = (c_w - new_w) // 2
        y_offset = (c_h - new_h) // 2
        return x_offset, y_offset, new_w, new_h

    def repaint(self):
        """Redraw the preview image using nearest-neighbor scaling when geometry changes."""
        if self.current_image is None:
            self.delete("all")
            self._image_id = None
            self._last_rect = None
            return

        x, y, w, h = self.get_draw_rect()
        if w <= 0 or h <= 0:
            return

        if self._last_image is self.current_image and self._last_rect == (x, y, w, h):
            return
        self._last_rect = (x, y, w, h)

        if self._last_size != (w, h) or self._last_image is not self.current_image:
            resized = self.current_image.resize((w, h), Image.NEAREST)
            self.photo_image = ImageTk.PhotoImage(resized)
            self._last_size = (w, h)
            self._last_image = self.current_image

        if self._image_id is None:
            self.delete("all")
            self._image_id = self.create_image(x, y, anchor=tk.NW, image=self.photo_image)
        else:
            self.coords(self._image_id, x, y)
            self.itemconfig(self._image_id, image=self.photo_image)

    def on_resize(self, event):
        """Re-fit the preview image when the canvas container resizes."""
        self.repaint()
