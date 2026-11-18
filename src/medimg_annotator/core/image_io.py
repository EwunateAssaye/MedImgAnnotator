"""
Universal Medical and Standard 2D/3D Image I/O Engine.

Supports reading and writing NIfTI (.nii, .nii.gz), DICOM (.dcm, .ima, .dicom),
NRRD (.nrrd, .nhdr), MetaImage (.mha, .mhd), standard 2D images (PNG, JPG, TIFF,
BMP, WebP), and NumPy arrays (.npy, .npz).
"""

from medimg_annotator.imports import Image, cv2, nib, np, os, pydicom, sitk, tifffile


SUPPORTED_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".ima",
    ".dicom",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".nrrd",
    ".nhdr",
    ".mha",
    ".mhd",
    ".npy",
    ".npz",
)


def get_supported_extensions():
    """Return tuple of all supported file extensions for directory traversal."""
    return SUPPORTED_EXTENSIONS


def _has_dicom_magic(filepath):
    """Check whether an extensionless file contains the DICOM 'DICM' preamble."""
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 132:
        try:
            with open(filepath, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except Exception:
            pass
    return False


def is_supported_image(filepath):
    """Check if file has a supported image extension or valid DICOM header."""
    if not filepath:
        return False
    lower = filepath.lower()
    if any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        return True
    return _has_dicom_magic(filepath)


def detect_file_format(filepath):
    """Detect image format string based on extension and content."""
    lower = filepath.lower()
    if lower.endswith(".nii.gz"):
        return "nifti_gz"
    if lower.endswith(".nii"):
        return "nifti"
    if lower.endswith((".dcm", ".ima", ".dicom")):
        return "dicom"
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return "standard_2d"
    if lower.endswith((".tif", ".tiff")):
        return "tiff"
    if lower.endswith((".nrrd", ".nhdr")):
        return "nrrd"
    if lower.endswith((".mha", ".mhd")):
        return "metaimage"
    if lower.endswith((".npy", ".npz")):
        return "numpy"
    if _has_dicom_magic(filepath):
        return "dicom"
    return "unknown"


class SyntheticHeader:
    """Lightweight proxy header mimicking Nifti1Header for non-NIfTI files."""

    def __init__(self, shape=(1, 1, 1), zooms=(1.0, 1.0, 1.0), dtype=np.uint8):
        self._shape = tuple(shape)
        self._zooms = tuple(zooms)
        self._dtype = np.dtype(dtype)

    def get_zooms(self):
        return self._zooms

    def get_data_dtype(self):
        return self._dtype

    def set_data_dtype(self, dtype):
        self._dtype = np.dtype(dtype)

    def set_slope_inter(self, slope, inter):
        pass

    def copy(self):
        return SyntheticHeader(self._shape, self._zooms, self._dtype)


def _read_with_sitk(filepath, format_info):
    """Read medical image/volume via SimpleITK into canonical (H, W, D) float32 array."""
    sitk_img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(sitk_img)
    if arr.ndim == 2:
        data_raw = np.transpose(arr, (1, 0))[:, :, np.newaxis].astype(np.float32)
        format_info["is_2d"] = True
    elif arr.ndim == 3:
        data_raw = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    else:
        data_raw = arr.astype(np.float32)
    sp = sitk_img.GetSpacing()
    affine = np.diag([sp[0], sp[1], sp[2] if len(sp) > 2 else 1.0, 1.0])
    header = SyntheticHeader(data_raw.shape, zooms=sp[:3], dtype=arr.dtype)
    return data_raw, affine, header, format_info


def read_universal_image(filepath):
    """
    Read any supported image into a standardized tuple:
      (data_raw, affine, header, format_info)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = detect_file_format(filepath)
    ext = ".nii.gz" if filepath.lower().endswith(".nii.gz") else os.path.splitext(filepath)[1].lower()

    format_info = {
        "format": fmt,
        "ext": ext,
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "is_2d": False,
        "color_mode": "gray",
        "rgb_data": None,
        "original_dtype": np.float32,
        "original_shape": None,
        "metadata": {},
    }

    if fmt in ("nifti", "nifti_gz") and nib is not None:
        nii = nib.load(filepath)
        format_info["original_shape"] = nii.shape
        format_info["original_dtype"] = nii.get_data_dtype()

        if nii.ndim > 3:
            if nii.shape[2] == 1 and nii.shape[3] > 1:
                data = np.squeeze(nii.get_fdata(dtype=np.float32), axis=2)
            else:
                data = np.asanyarray(nii.dataobj[..., 0], dtype=np.float32)
        else:
            data = nii.get_fdata(dtype=np.float32)

        while data.ndim > 3:
            data = data[..., 0]
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
            format_info["is_2d"] = True

        is_writeable = bool(getattr(getattr(data, "flags", None), "writeable", True))
        data_raw = np.nan_to_num(data, copy=not is_writeable).astype(np.float32)
        affine = nii.affine.copy() if nii.affine is not None else np.eye(4)
        header = nii.header.copy() if nii.header is not None else SyntheticHeader(data_raw.shape)
        return data_raw, affine, header, format_info

    if fmt == "standard_2d":
        pil_img = Image.open(filepath)
        format_info["is_2d"] = True
        format_info["mode"] = pil_img.mode
        format_info["original_shape"] = (pil_img.height, pil_img.width)

        if pil_img.mode in ("RGB", "RGBA"):
            rgb_arr = np.array(pil_img)
            format_info["rgb_data"] = rgb_arr
            format_info["color_mode"] = pil_img.mode.lower()
            rgb_only = rgb_arr[:, :, :3] if pil_img.mode == "RGBA" else rgb_arr
            gray = cv2.cvtColor(rgb_only, cv2.COLOR_RGB2GRAY)
            data_raw = np.transpose(gray, (1, 0))[:, :, np.newaxis].astype(np.float32)
        else:
            arr = np.array(pil_img)
            if arr.ndim == 2:
                data_raw = np.transpose(arr, (1, 0))[:, :, np.newaxis].astype(np.float32)
            else:
                data_raw = arr.astype(np.float32)

        affine = np.eye(4)
        header = SyntheticHeader(data_raw.shape, zooms=(1.0, 1.0, 1.0), dtype=np.uint8)
        return data_raw, affine, header, format_info

    if fmt == "tiff":
        arr = tifffile.imread(filepath) if tifffile is not None else np.array(Image.open(filepath))
        format_info["original_shape"] = arr.shape
        format_info["original_dtype"] = arr.dtype

        if arr.ndim == 2:
            data_raw = np.transpose(arr, (1, 0))[:, :, np.newaxis].astype(np.float32)
            format_info["is_2d"] = True
        elif arr.ndim == 3:
            if arr.shape[2] in (3, 4):
                format_info["is_2d"] = True
                format_info["rgb_data"] = arr
                gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
                data_raw = np.transpose(gray, (1, 0))[:, :, np.newaxis].astype(np.float32)
            else:
                data_raw = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        else:
            while arr.ndim > 3:
                arr = arr[0]
            data_raw = arr.astype(np.float32)

        affine = np.eye(4)
        header = SyntheticHeader(data_raw.shape, zooms=(1.0, 1.0, 1.0), dtype=arr.dtype)
        return data_raw, affine, header, format_info

    if fmt == "dicom":
        if pydicom is not None:
            try:
                ds = pydicom.dcmread(filepath)
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                if slope != 1.0 or intercept != 0.0:
                    arr = arr * slope + intercept

                dx, dy = 1.0, 1.0
                if hasattr(ds, "PixelSpacing"):
                    try:
                        dy, dx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
                    except Exception:
                        pass
                dz = float(getattr(ds, "SliceThickness", 1.0))

                if arr.ndim == 2:
                    data_raw = np.transpose(arr, (1, 0))[:, :, np.newaxis].astype(np.float32)
                    format_info["is_2d"] = True
                elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                    format_info["is_2d"] = True
                    format_info["rgb_data"] = arr.astype(np.uint8)
                    gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    data_raw = np.transpose(gray, (1, 0))[:, :, np.newaxis].astype(np.float32)
                else:
                    data_raw = arr

                affine = np.diag([dx, dy, dz, 1.0])
                header = SyntheticHeader(data_raw.shape, zooms=(dx, dy, dz), dtype=arr.dtype)
                format_info["dicom_dataset"] = ds
                return data_raw, affine, header, format_info
            except Exception as e:
                print(f"[pydicom failed, trying SimpleITK]: {e}")

        if sitk is not None:
            return _read_with_sitk(filepath, format_info)

    if fmt in ("nrrd", "metaimage") and sitk is not None:
        return _read_with_sitk(filepath, format_info)

    if fmt == "numpy":
        arr = np.load(filepath)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.keys())[0]]
        if arr.ndim == 2:
            data_raw = np.transpose(arr, (1, 0))[:, :, np.newaxis].astype(np.float32)
            format_info["is_2d"] = True
        else:
            data_raw = arr.astype(np.float32)
        affine = np.eye(4)
        header = SyntheticHeader(data_raw.shape, zooms=(1.0, 1.0, 1.0), dtype=arr.dtype)
        return data_raw, affine, header, format_info

    # Fallback to PIL Image
    pil_img = Image.open(filepath)
    arr = np.array(pil_img)
    if arr.ndim == 2:
        data_raw = arr[:, :, np.newaxis].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        format_info["rgb_data"] = arr
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        data_raw = gray[:, :, np.newaxis].astype(np.float32)
    else:
        data_raw = arr.astype(np.float32)
    format_info["is_2d"] = True
    affine = np.eye(4)
    header = SyntheticHeader(data_raw.shape, zooms=(1.0, 1.0, 1.0), dtype=arr.dtype)
    return data_raw, affine, header, format_info


def _untranspose_2d(arr, format_info):
    """Restore (H, W) orientation from transposed (W, H, 1) editor array when saving 2D formats."""
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return arr
    if arr.ndim == 3:
        out = arr[:, :, 0]
        if format_info and format_info.get("is_2d", False):
            return np.transpose(out, (1, 0))
        return out
    if arr.ndim == 2 and format_info and format_info.get("is_2d", False) and format_info.get("original_shape") is not None:
        orig_shape = format_info["original_shape"]
        if arr.shape == (orig_shape[1], orig_shape[0]) and orig_shape[0] != orig_shape[1]:
            return np.transpose(arr, (1, 0))
    return arr


def write_universal_image(data, output_path, affine=None, header=None, format_info=None):
    """Write data back to disk matching the original input file format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    lower = output_path.lower()

    if lower.endswith((".nii", ".nii.gz")):
        if nib is None:
            raise ImportError("nibabel required to save NIfTI files.")
        aff = affine if affine is not None else np.eye(4)
        hdr = header.copy() if (header is not None and hasattr(header, "copy")) else nib.Nifti1Header()
        hdr.set_data_dtype(data.dtype)
        try:
            hdr.set_slope_inter(None, None)
        except Exception:
            pass
        nib.save(nib.Nifti1Image(data, aff, hdr), output_path)
        return output_path

    if any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        save_arr = _untranspose_2d(data, format_info)
        if save_arr.dtype not in (np.uint8, np.uint16):
            if np.max(save_arr) <= 1.0 and np.min(save_arr) >= 0.0:
                save_arr = (save_arr * 255.0).astype(np.uint8)
            else:
                save_arr = np.clip(save_arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(save_arr)
        if lower.endswith((".jpg", ".jpeg")):
            img.convert("RGB").save(output_path, quality=95)
        else:
            img.save(output_path)
        return output_path

    if lower.endswith((".tif", ".tiff")):
        if data.ndim == 3 and data.shape[2] not in (1, 3, 4):
            save_arr = np.transpose(data, (2, 0, 1))
        else:
            save_arr = _untranspose_2d(data, format_info)

        if tifffile is not None:
            tifffile.imwrite(output_path, save_arr)
        else:
            Image.fromarray(save_arr).save(output_path)
        return output_path

    if any(lower.endswith(ext) for ext in (".nrrd", ".nhdr", ".mha", ".mhd")) and sitk is not None:
        if data.ndim == 3 and data.shape[2] > 1:
            sitk_arr = np.transpose(data, (2, 0, 1))
        elif data.ndim == 3 and data.shape[2] == 1:
            sitk_arr = data[:, :, 0]
        else:
            sitk_arr = data
        sitk_img = sitk.GetImageFromArray(sitk_arr)
        if header is not None and hasattr(header, "get_zooms"):
            zooms = header.get_zooms()
            sitk_img.SetSpacing([float(z) for z in zooms[: len(sitk_img.GetSize())]])
        sitk.WriteImage(sitk_img, output_path)
        return output_path

    if lower.endswith(".npy"):
        np.save(output_path, data)
        return output_path

    if nib is not None:
        nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), output_path)
    else:
        Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(output_path)
    return output_path
