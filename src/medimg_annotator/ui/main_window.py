"""
Main workstation window (`NiftiMaskEditor`) composing modular core and UI mixins.
"""

from medimg_annotator.imports import ctk, getpass, os, time

from medimg_annotator.core.config import (
    DEFAULT_LABEL_COLORS,
    LABEL_COLOR_MAP,
    Settings,
    default_color_name_for_label,
)
from medimg_annotator.core.history import HistoryMixin
from medimg_annotator.core.interpolation import SliceInterpolationMixin
from medimg_annotator.core.segmentation import SegmentationMixin
from medimg_annotator.ui.io_controller import IOControllerMixin
from medimg_annotator.ui.obdet_controller import ObDetControllerMixin
from medimg_annotator.ui.plugin_manager import PluginManagerMixin
from medimg_annotator.ui.stats_logger import StatsLoggerMixin
from medimg_annotator.ui.toolbar_builder import ToolbarBuilderMixin
from medimg_annotator.ui.view_controller import ViewControllerMixin

class NiftiMaskEditor(
    HistoryMixin,
    SegmentationMixin,
    SliceInterpolationMixin,
    ToolbarBuilderMixin,
    ViewControllerMixin,
    ObDetControllerMixin,
    IOControllerMixin,
    StatsLoggerMixin,
    PluginManagerMixin,
    ctk.CTk,
):
    """Primary desktop application window for 2D/3D medical image segmentation and ObDet annotation."""

    def __init__(self):
        super().__init__()
        self.title("MedImgAnnotator")
        self.imprep_window = None

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{min(1600, screen_w - 50)}x{min(950, screen_h - 50)}")

        # Volume data
        self.img_data_raw = None
        self.img_data = None
        self.mask_data = None
        self.affine = None
        self.header = None

        # Batch and session paths
        settings = Settings()
        self.image_dir = ""
        self.mask_dir = None
        self.specific_mask_file = None
        self.single_image_mode = False
        self.initial_start_file = None
        saved_out = settings.value("last_output_dir", "")
        self.output_dir = saved_out if (saved_out and os.path.exists(saved_out)) else ""
        self.file_list = []
        self.current_index = -1

        # Interactive annotation parameters
        self.current_slice = 0
        self.max_slice = 0
        self.brush_size = 5.0
        self.wand_tolerance = 30.0
        self.wand_limit = 100
        self.is_eraser = False
        self.mask_opacity = 75
        self.tool_mode = "brush"

        self.color_map = dict(LABEL_COLOR_MAP)
        self.value_to_color = {i: default_color_name_for_label(i) for i in range(1, 101)}
        self.label_names = {}
        self._last_vol_stats_state = None
        self.has_unsaved_changes = False
        self._cached_mask_labels = []
        self._mask_labels_dirty = True

        self.undo_stack = []
        self.redo_stack = []
        self.axis_index = 2
        self.hovered_slider = None

        # Frame-rate throttling flags
        self._is_updating = False
        self._pending_update = False
        self._is_slicing = False
        self._pending_slice = False

        # Active annotation timer and CSV audit log
        try:
            self.current_user = getpass.getuser()
        except Exception:
            self.current_user = os.environ.get("USERNAME", "User")
        self.active_annotation_times = {}
        self._last_interaction_time = time.time()
        self._idle_timeout_seconds = 60.0
        self._timer_interval_ms = 1000
        self._active_timer_job = None

        # Object Detection (ObDet) state
        self.obdet_active = False
        self.obdet_boxes = []
        self.active_obdet_box_idx = None
        self.obdet_center_2d = None
        self.obdet_bbox_w = 64
        self.obdet_bbox_h = 64
        self.obdet_slice = 0
        self.obdet_axis = 2
        self.obdet_drag_mode = None
        self.obdet_drag_corner = None
        self.obdet_drag_start = None
        self.obdet_orig_bbox = None
        self.current_file_format_info = None

        self.init_ui()
        self.setup_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._schedule_active_timer()
        self.load_existing_active_times_from_csv()

    def on_closing(self):
        """Export active annotation times before closing the application window."""
        try:
            if getattr(self, "_active_timer_job", None) is not None:
                self.after_cancel(self._active_timer_job)
                self._active_timer_job = None
        except Exception:
            pass
        try:
            self.export_time_log_csv(show_msg=False)
        except Exception:
            pass
        self.destroy()

    def destroy(self):
        """Cancel pending scheduled jobs cleanly on window teardown."""
        for job_attr in ("_active_timer_job", "_update_job", "_slicing_job"):
            job = getattr(self, job_attr, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, job_attr, None)
        super().destroy()
