# MedImgAnnotator (with Integrated ImPrep Suite)

A high-performance desktop medical image annotation, object detection, and preprocessing workstation for radiology and pathology workflows (NIfTI, DICOM, TIFF, PNG, JPEG, BMP, WebP).

If you use **MedImgAnnotator** in your research or work, please cite the paper as per your citation style. [![arXiv](https://img.shields.io/badge/arXiv-2611.22450-b31b1b.svg)](https://arxiv.org/abs/2611.22450)

---

## Key Highlights

1. **Unified Workstation**: Integrates an interactive 2D/3D annotation canvas with the 16-module **ImPrep** image preprocessing suite.
2. **Unified File & Folder Input**: Single unified browser (`Input`) supporting both single-file selection and entire folder cohorts for images and masks, with a 3-tier fuzzy pairing engine and optional blank-mask fallback.
3. **Hybrid Deformable Registration + SDM Slice Interpolation**: Combines Fast Symmetric Forces Demons deformable registration with Co-Moving Signed Distance Map (SDM) residual fusion for anatomically consistent 3D multi-slice interpolation.
4. **Dual-Tier 8-Bit LUT Display & 32-Bit Quantitative Engine**: Real-time 8-bit (`uint8`) compositing with dual-layer scaling (`INTER_AREA` / `INTER_LINEAR` image scaling + `INTER_NEAREST` mask labels) paired with 32-bit (`float32`) intensity fidelity for Magic Wand, IFT Watershed Edge-Snapping, and Registration.
5. **Object Detection (ObDet) & Active Time Logging**: Multi-box ObDet mode with aspect-preserving resize, YOLO `.txt` + JSON export, real-time physical label volume statistics ($\text{mm}^3$ / $\text{cm}^3$), and idle-aware per-label CSV logging.
6. **Bidirectional ImPrep Bridge**: Pass volumes or masks seamlessly between ImPrep and the main annotation canvas, or chain modules via Workbench RAM memory.

---

## Repository Structure

```text
MedImgAnnotator/
├── main.py                              # Application launcher
├── pyproject.toml                       # Package metadata
├── requirements.txt                     # Python dependencies
├── assets/
│   └── icon.ico                         # Application icon
└── src/
│   └── medimg_annotator/
│       ├── imports.py                   # Centralized lazy-cached dependency import hub
│       ├── app.py                       # GUI startup bootstrap
│       ├── core/                        # Config, geometry, I/O, ObDet, history, segmentation, interpolation, plugins
│       ├── ui/                          # Main window, canvases, toolbars, controllers, input dialog, plugin manager
│       └── imprep/                      # 16-module ImPrep preprocessing workbench & modules
```

---

## Quick Start (Download, Setup & Run)

### Prerequisites
- **Operating System:** Windows 10/11 (64-bit), macOS, or Linux (X11/Wayland with Tkinter)
- **Python:** Python 3.10–3.12 recommended (`python --version`; `requirements.txt` pins `numpy<2.0.0`, and macOS uses Homebrew `python@3.11` with `python-tk@3.11`)
- **Git:** Optional on Windows (if downloading ZIP); recommended on macOS/Linux (`git --version`)

### Windows Setup & Launch
1. **Download the Repository:**
   - **Option A — Clone with Git:**
     ```bash
     git clone https://github.com/EwunateAssaye/MedImgAnnotator.git
     cd MedImgAnnotator
     ```
   - **Option B — Download ZIP (No Git required):**
     1. Open `https://github.com/EwunateAssaye/MedImgAnnotator` in your browser.
     2. Click **Code → Download ZIP** and extract the archive.
     3. Open PowerShell or Command Prompt inside the extracted `MedImgAnnotator` folder.

2. **Create a Python Environment & Install Dependencies:**
   ```powershell
   python -m venv .venv
   # PowerShell:
   .venv\Scripts\Activate.ps1
   # Command Prompt (cmd.exe):
   # .venv\Scripts\activate.bat

   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Optional)* To install as an editable package (registers the `medimg-annotator` launch command):
   ```bash
   pip install -e .
   ```

3. **Launch MedImgAnnotator:**
   ```bash
   python main.py
   # Or, if installed via `pip install -e .`:
   medimg-annotator
   ```

### macOS Setup & Launch
- **Step 0: Install Homebrew (one-time)**
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- **Step 1: Install Python 3.11 with Tkinter (one-time)**
  ```bash
  brew install python@3.11 python-tk@3.11
  ```

- **Step 2: Clone the repository from GitHub**
  ```bash
  cd ~
  git clone https://github.com/EwunateAssaye/MedImgAnnotator.git
  cd MedImgAnnotator
  ```

- **Step 3: Create and activate `.venv` (and confirm Tkinter)**
  ```bash
  "$(brew --prefix)/bin/python3.11" -m venv .venv
  source .venv/bin/activate
  python -c "import tkinter; print(tkinter.TkVersion)"
  ```

- **Step 4: Install dependencies**
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Step 5: Run MedImgAnnotator**
  ```bash
  python main.py
  ```

- **Step 6: Next time you want to run it**
  ```bash
  cd ~/MedImgAnnotator
  source .venv/bin/activate
  python main.py
  ```

### Linux Setup & Launch
*(The Windows `.venv\Scripts\Activate.ps1` step does not apply on Linux; use standard POSIX virtual environment activation and install Tkinter via your distro package manager.)*

1. **Install Python, venv, and Tkinter (Ubuntu/Debian example):**
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-tk git
   ```
2. **Clone, create `.venv`, and install dependencies** *(Python 3.10–3.12 recommended; if default `python3` is 3.13+, use `python3.11` or `python3.12`)*:
   ```bash
   cd ~
   git clone https://github.com/EwunateAssaye/MedImgAnnotator.git
   cd MedImgAnnotator
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run MedImgAnnotator (and next time):**
   ```bash
   python main.py
   ```
   *(To relaunch later: `cd ~/MedImgAnnotator && source .venv/bin/activate && python main.py`.)*

> **Cross-Platform Notes (macOS / Linux):** All segmentation tools, ObDet, ImPrep preprocessing modules, slice scrolling (`<MouseWheel>` and Linux X11 `<Button-4>`/`<Button-5>`), and file browsing run cross-platform. Clicking **Export CSV** writes `annotation_time_log.csv` normally on all OSes; clicking **Open CSV** on macOS/Linux falls back to showing the CSV path dialog (`os.startfile` is Windows-only), and **Open Plugins Folder** uses `xdg-open` on Linux.

### Basic Usage Workflow
1. Click **Input** to select either an entire folder cohort or a single scan (`.nii`, `.nii.gz`, `.dcm`, `.png`, `.jpg`, `.tif`, `.bmp`, `.webp`). Choose **No** at the mask prompt to start with blank masks, or **Yes** to load existing masks.
2. Click **Output** to set the save directory for masks, overlays, YOLO/JSON exports, and `annotation_time_log.csv`.
3. Select the plane (`Ax` / `Cor` / `Sag`), scope (`2D` / `3D`), and tool (`Brush`, `Bucket`, `Poly` with optional `Auto` watershed snap, `Wand`, `Eraser`, `Interpolate`, or `ObDet`).
4. Press **S** (or **Save (S)**) to save; use **Ctrl+Z** / **Ctrl+Y** for up to 50 undo/redo steps.
   - **Navigation & View Controls:** `Mouse Wheel` (or Arrow / `PageUp` / `PageDown` keys) steps slices; `Shift + Wheel` adjusts Brightness; `Ctrl + Wheel` adjusts Contrast; `Alt + Wheel` zooms at cursor; `Right-Click Drag` pans.

---

## Toolbar & Annotation Overview

- **Top Toolbar (Row 1)**: `ImPrep`, `Input`, `Output`, `Plugins`, `Search` / `Go`, `prev-vol` / `Next-vol`, split `Save (S)` (`.nii.gz` / JPEG slice overlay / ObDet YOLO+JSON), `Undo (Ctrl+Z)` / `Redo (Ctrl+Y)`.
- **Annotation Modes (Rows 2–3)**: Axial/Coronal/Sagittal views, `2D`/`3D` scope, Smart Brush, Flood Fill Bucket, Freehand Polygon (Chaikin smoothing + ROI IFT Watershed `Auto` snap), Bounded Magic Wand (`Tol`/`Lim`), Eraser, Hybrid Demons + SDM `Interpolate`, `ObDet`, up to 100-class Multi-label selector, Brightness (`B`), Contrast (`C`), Zoom (`Z`), Opacity.

### Annotation Tools & Workstation Functions

| # | Tool / Function | Description |
|---|-----------------|-------------|
| 1 | **Smart Brush (`Brush`, `Rad`)** | Freehand 2D circular slice painting or 3D spherical volume painting (when `3D` scope is enabled) with continuous linear point interpolation (`apply_brush_line`) to prevent dotted gaps during fast strokes. |
| 2 | **Flood Fill Bucket (`Bucket`)** | Connected-component labeling fill (`scipy.ndimage.label`): fills closed `Poly` contours, flood-fills enclosed 2D/3D cavities (with an exterior background safety guard in 3D), recolors clicked mask islands, or deletes intersecting islands when `Eraser` is active. |
| 3 | **Freehand Polygon (`Poly`, `(Auto)`)** | Draws closed contours smoothed by 4-iteration Chaikin subdivision (`chaikins_smooth`) with draggable orange vertex handles; fill interiors with `Bucket` or constrain `Wand`, and optionally check **(Auto)** for ROI-cropped IFT seeded watershed edge-snapping (`watershed_ift`). |
| 4 | **Magic Wand (`Wand`, `Tol`, `Lim`)** | Intensity-guided 2D/3D region growing on raw 32-bit (`float32`) intensities: extracts the connected component around the click seed within `±Tol` intensity difference and spatial radius `Lim` voxels (respecting active `Poly` contours in 2D). |
| 5 | **Precision Eraser (`Eraser`)** | Switches any active drawing tool (`Brush`, `Bucket`, `Poly`, or `Wand`) to background label `0` to erase strokes, clear polygon interiors, or remove connected mask islands. |
| 6 | **Multi-Slice Interpolation (`Slices`, `Interpolate`)** | Propagates sparse keyframe slice annotations (e.g., `10, 25`) across intermediate slices using Fast Symmetric Forces Demons deformable registration fused with Co-Moving Signed Distance Map (SDM) residual interpolation (plus optional `(Auto)` edge snapping). |
| 7 | **Object Detection (`ObDet`)** | Multi-box bounding-box mode: left-click to place centered boxes, scroll mouse wheel to scale while preserving aspect ratio within image bounds, drag corners or body to resize/reposition, press `Delete`/`Backspace` to remove, and export format-preserving images + YOLO `.txt` + JSON metadata. |
| 8 | **Multi-Label Manager (`Multi-label`, `Val`, `Color`, `Name`)** | Assigns up to 100 discrete integer classes (`1–100`) with 40 high-contrast overlay LUT colors and custom anatomical structure names. |
| 9 | **Multi-Planar & Viewport Controls (`Ax`/`Cor`/`Sag`, `B`/`C`/`Z`, `Opacity`)** | Switches Axial, Coronal, and Sagittal views (`2D`/`3D` scope) with 8-bit Brightness/Contrast windowing (`Shift`/`Ctrl + Wheel`), cursor-centered Zoom (`Alt + Wheel` with `INTER_AREA`/`INTER_LINEAR` grayscale and `INTER_NEAREST` masks), Right-Click Pan, Slice scrolling, and isolated **Mask Only** preview. |
| 10 | **Volumetry & Active Time Logger (`Open CSV`, `Export CSV`)** | Displays voxel counts and real-time physical label volumes ($\text{mm}^3$ / $\text{cm}^3$) alongside idle-aware active annotation timers logged per label to `annotation_time_log.csv`. |
| 11 | **History & Split Export (`Undo`/`Redo`, `Save (S) ▾`)** | 50-step `zlib`-compressed 2D/3D Undo (`Ctrl+Z`) and Redo (`Ctrl+Y`) history, plus quick-save (`S`) and dropdown exports for `.nii.gz` masks, JPEG slice overlays, and ObDet YOLO+JSON outputs. |
| 12 | **Unified I/O & AI Plugins (`Input`, `Output`, `Search`, `Plugins`, `ImPrep`)** | Loads single scans or cohort folders with 3-tier image-mask pairing, searches volumes by name or index, executes external `.py`/`.zip`/folder AI plugins with active session paths, and exchanges volumes/masks with **ImPrep**. |

---

## ImPrep Preprocessing Modules

| # | Tool | Description |
|---|------|-------------|
| 1 | **DICOM to NIfTI** | Recursive DICOM converter with MRI In-Phase / Out-Phase echo separation. |
| 2 | **JPG/PNG to NIfTI** | Converts 2D image stacks into 3D NIfTI volumes with custom voxel spacing. |
| 3 | **Image-Mask Aligner** | Visual 2D/3D overlay inspection with axis flips and 90° rotations. |
| 4 | **Modality Sorter** | Organizes image files into modality folders using CSV/XLSX metadata. |
| 5 | **Channel Splitter** | Splits 4D multi-channel/temporal volumes into separate 3D `.nii.gz` files. |
| 6 | **Compressor** | Converts uncompressed `.nii` files into `.nii.gz` archives. |
| 7 | **Bit Converter** | Converts volume bit-depths (`uint8`, `int16`, `float32`). |
| 8 | **Affine Fixer** | Canonicalizes orientation to orthogonal RAS coordinates. |
| 9 | **Intensity Processor** | Hounsfield Unit windowing, clipping, and band-pass thresholding. |
| 10 | **Thresholding** | Min/Max, Otsu, and adaptive thresholding mask generation. |
| 11 | **Resampler** | Resamples volumes/masks to target isotropic or anisotropic spacing. |
| 12 | **Image Enhancer** | Tunable Gamma power-law and logarithmic contrast enhancement. |
| 13 | **Edge Enhancer** | Multi-scale Canny edge emphasis overlay. |
| 14 | **Normalization** | Min-Max, Z-score, and Median/IQR intensity normalization. |
| 15 | **Mask Extractor** | Extracts label IDs, merges labels, or keeps largest 3D component. |
| 16 | **Dataset Curator** | Token-based 1-to-1 pairing generating nnU-Net dataset layouts. |

---

## Citation

If you use **MedImgAnnotator** in your research, clinical workflow, or publications, please cite the paper as per your citation style:

- **Paper Link:** https://arxiv.org/abs/2611.22450
