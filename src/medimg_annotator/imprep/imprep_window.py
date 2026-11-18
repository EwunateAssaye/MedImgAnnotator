"""
ImPrep: Integrated 16-tool medical image preprocessing workbench window.
"""

from medimg_annotator.imports import ctk, tk

from medimg_annotator.imprep.config import COLORS, FONTS
from medimg_annotator.imprep.sidebar import ImPrepSidebar
from medimg_annotator.imprep.modules.app_01_dicom_to_nifti import DicomConverterFrame
from medimg_annotator.imprep.modules.app_02_jpg_png_to_nifti import PngToNiftiFrame
from medimg_annotator.imprep.modules.app_03_image_mask_aligner import ImageMaskAlignerFrame
from medimg_annotator.imprep.modules.app_04_modality_sorter import ModalitySorterFrame
from medimg_annotator.imprep.modules.app_05_channel_splitter import ChannelSplitterFrame
from medimg_annotator.imprep.modules.app_06_compressor import CompressorFrame
from medimg_annotator.imprep.modules.app_07_bit_converter import BitConverterFrame
from medimg_annotator.imprep.modules.app_08_affine_fixer import AffineFixerFrame
from medimg_annotator.imprep.modules.app_09_intensity_processor import IntensityProcessorFrame
from medimg_annotator.imprep.modules.app_10_thresholding import ThresholdingFrame
from medimg_annotator.imprep.modules.app_11_resampler import ResampleFrame
from medimg_annotator.imprep.modules.app_12_image_enhancer import ImageEnhancerFrame
from medimg_annotator.imprep.modules.app_13_edge_enhancer import EdgeEnhancerFrame
from medimg_annotator.imprep.modules.app_14_normalization import NormalizationFrame
from medimg_annotator.imprep.modules.app_15_mask_extractor import MaskExtractorFrame
from medimg_annotator.imprep.modules.app_16_dataset_curator import DatasetCuratorFrame

class ImPrepWindow(ctk.CTkToplevel):
    """Top-level Image Preprocessing Suite window linked to MedImgAnnotator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editor = parent
        self.title("ImPrep - Image Preprocessing Suite")

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        w = min(1400, screen_w - 60)
        h = min(880, screen_h - 60)
        self.geometry(f"{w}x{h}+30+30")
        self.minsize(980, 620)

        self.shared_vol = None
        self.shared_affine = None
        self.shared_header = None
        self.shared_fname = "workbench_vol.nii.gz"
        self.hovered_slider = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=240)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = ImPrepSidebar(self, controller=self)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        self.main_container = ctk.CTkFrame(self, fg_color=COLORS["bg_dark"], corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.home_frame = self._build_dashboard()
        self.home_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar.add_nav_button("📊 Dashboard", self.home_frame, key="Dashboard")

        self.tool_list = [
            ("1. DICOM to NIfTI", DicomConverterFrame),
            ("2. JPG/PNG to NIfTI", PngToNiftiFrame),
            ("3. Image-Mask Aligner", ImageMaskAlignerFrame),
            ("4. Modality Sorter", ModalitySorterFrame),
            ("5. Channel Splitter", ChannelSplitterFrame),
            ("6. Compressor (.nii.gz)", CompressorFrame),
            ("7. Bit Converter", BitConverterFrame),
            ("8. Affine Fixer (RAS)", AffineFixerFrame),
            ("9. Intensity Processor", IntensityProcessorFrame),
            ("10. Thresholding", ThresholdingFrame),
            ("11. Resampler (Spacing)", ResampleFrame),
            ("12. Image Enhancer", ImageEnhancerFrame),
            ("13. Edge Enhancer", EdgeEnhancerFrame),
            ("14. Normalization", NormalizationFrame),
            ("15. Mask Extractor", MaskExtractorFrame),
            ("16. Dataset Curator", DatasetCuratorFrame),
        ]

        self.frames = {}
        self._current_frame = self.home_frame

        for name, Cls in self.tool_list:
            self.sidebar.add_nav_button(name, command=lambda c=Cls, n=name: self.open_tool(c, n), key=name)

        self.sidebar.set_active("Dashboard")
        self.show(self.home_frame)

        for key in ("<Left>", "<Right>", "<Up>", "<Down>", "<Prior>", "<Next>"):
            self.bind(key, self._on_key_nav)
            self.main_container.bind(key, self._on_key_nav)

    def _build_dashboard(self):
        dash = ctk.CTkScrollableFrame(self.main_container, fg_color=COLORS["bg_dark"], corner_radius=0)

        hero = ctk.CTkFrame(dash, fg_color=COLORS["bg_panel"], corner_radius=10)
        hero.pack(fill="x", padx=25, pady=(20, 15))
        ctk.CTkLabel(
            hero, text="ImPrep Medical Suite", font=("Segoe UI", 22, "bold"), text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=20, pady=(16, 2))
        ctk.CTkLabel(
            hero,
            text="Comprehensive image preprocessing, alignment, normalization, and dataset curation workbench directly integrated with MedImgAnnotator.",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            wraplength=800,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            dash, text="Functional Tool Categories", font=FONTS["title"], text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=25, pady=(10, 8))

        cards_container = ctk.CTkFrame(dash, fg_color="transparent")
        cards_container.pack(fill="x", padx=20, pady=5)
        cards_container.grid_columnconfigure((0, 1, 2, 3), weight=1)

        groups = [
            ("Converters & IO", ["1. DICOM to NIfTI", "2. JPG/PNG to NIfTI", "4. Modality Sorter", "6. Compressor (.nii.gz)"]),
            ("Geometry & Align", ["3. Image-Mask Aligner", "8. Affine Fixer (RAS)", "11. Resampler (Spacing)"]),
            ("Intensity & Quality", ["9. Intensity Processor", "12. Image Enhancer", "13. Edge Enhancer", "14. Normalization", "7. Bit Converter"]),
            ("Segmentation & Data", ["5. Channel Splitter", "10. Thresholding", "15. Mask Extractor", "16. Dataset Curator"]),
        ]

        for col, (grp_name, tools) in enumerate(groups):
            card = ctk.CTkFrame(cards_container, fg_color=COLORS["bg_card"], corner_radius=8)
            card.grid(row=0, column=col, sticky="nsew", padx=6, pady=6)
            ctk.CTkLabel(card, text=grp_name, font=FONTS["sub_header"], text_color=COLORS["accent"]).pack(
                anchor="w", padx=12, pady=(10, 6)
            )
            for t_name in tools:
                def make_cmd(target_name=t_name):
                    for name, cls in self.tool_list:
                        if name == target_name:
                            self.open_tool(cls, name)
                            break

                ctk.CTkButton(
                    card,
                    text=t_name,
                    anchor="w",
                    font=FONTS["small"],
                    fg_color=COLORS["bg_panel"],
                    hover_color=COLORS["accent"],
                    height=28,
                    command=make_cmd,
                ).pack(fill="x", padx=10, pady=3)
            ctk.CTkFrame(card, fg_color="transparent", height=6).pack()

        bridge_card = ctk.CTkFrame(dash, fg_color=COLORS["bg_card"], corner_radius=8)
        bridge_card.pack(fill="x", padx=25, pady=20)
        ctk.CTkLabel(
            bridge_card, text="Workstation Bridge Status", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]
        ).pack(anchor="w", padx=15, pady=(12, 4))
        has_editor = self.editor is not None and getattr(self.editor, "img_data", None) is not None
        status_txt = f"Connected to MedImgAnnotator ({'Active volume loaded in canvas' if has_editor else 'Canvas is currently idle'})"
        ctk.CTkLabel(
            bridge_card,
            text=f"• {status_txt}",
            font=FONTS["body"],
            text_color=COLORS["success"] if has_editor else COLORS["text_muted"],
        ).pack(anchor="w", padx=15, pady=(0, 12))

        return dash

    def open_tool(self, Cls, name):
        """Lazily instantiate tool frame on first selection and display it."""
        if Cls not in self.frames:
            self.frames[Cls] = Cls(self.main_container, name, self)
        self.sidebar.set_active(name)
        self.show(self.frames[Cls])

    def show(self, frame):
        """Switch active frame using grid_remove() to keep hidden views lightweight."""
        if getattr(self, "_current_frame", None) is not None and self._current_frame != frame:
            self._current_frame.grid_remove()
        self._current_frame = frame
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

    def increment_slider(self, slider, direction, is_shift=False):
        """Step either the active slice slider or hovered parameter slider."""
        if slider is None or slider.cget("state") == "disabled":
            return

        active_tool = getattr(self, "_current_frame", None)
        slice_slider = getattr(active_tool, "slider", None)
        if slider == slice_slider:
            if active_tool and hasattr(active_tool, "step_slice"):
                active_tool.step_slice(direction * (5 if is_shift else 1))
            return

        try:
            val = float(slider.get())
            f = float(slider.cget("from_"))
            t = float(slider.cget("to"))
            total_range = abs(t - f)

            num_steps = slider.cget("number_of_steps")
            if num_steps and num_steps > 0:
                step = total_range / num_steps
            elif total_range <= 10.0:
                step = 0.25 if is_shift else 0.05
            elif total_range <= 100.0:
                step = 5.0 if is_shift else 1.0
            else:
                step = 100.0 if is_shift else 25.0

            new_val = max(min(f, t), min(max(f, t), val + step * direction))
            slider.set(new_val)
            cmd = slider.cget("command")
            if cmd:
                cmd(new_val)
        except Exception as e:
            print(f"Error incrementing slider: {e}")

    def _on_key_nav(self, event):
        focused = self.focus_get()
        if focused is not None:
            if isinstance(focused, (tk.Entry, tk.Text, ctk.CTkEntry)):
                return
            if hasattr(focused, "master") and isinstance(focused.master, ctk.CTkEntry):
                return
            try:
                if focused.winfo_class().lower() in ("entry", "text"):
                    return
            except Exception:
                pass

        keysym = event.keysym
        is_shift = bool(event.state & 0x1)
        if keysym in ("Right", "Up", "Next"):
            direction = 1
        elif keysym in ("Left", "Down", "Prior"):
            direction = -1
        else:
            return
        if keysym in ("Next", "Prior"):
            is_shift = True

        target_slider = getattr(self, "hovered_slider", None)
        active_tool = getattr(self, "_current_frame", None)
        if target_slider is None and active_tool is not None:
            target_slider = getattr(active_tool, "hovered_slider", None)

        if target_slider is None:
            try:
                x, y = self.winfo_pointerxy()
                under = self.winfo_containing(x, y)
                if under is not None:
                    if isinstance(under, ctk.CTkSlider):
                        target_slider = under
                    elif hasattr(under, "master") and isinstance(under.master, ctk.CTkSlider):
                        target_slider = under.master
            except Exception:
                pass

        if target_slider is None and focused is not None:
            if isinstance(focused, ctk.CTkSlider):
                target_slider = focused
            elif hasattr(focused, "master") and isinstance(focused.master, ctk.CTkSlider):
                target_slider = focused.master

        if target_slider is not None:
            self.increment_slider(target_slider, direction, is_shift=is_shift)
            return "break"

        if active_tool is not None and hasattr(active_tool, "step_slice"):
            active_tool.step_slice(direction * (5 if is_shift else 1))
            return "break"

        if active_tool is not None:
            canvas = getattr(active_tool, "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_scroll(-1 if direction > 0 else 1, "units")
                return "break"

    def on_close(self):
        """Withdraw rather than destroy so cached modules remain reusable."""
        self.withdraw()
