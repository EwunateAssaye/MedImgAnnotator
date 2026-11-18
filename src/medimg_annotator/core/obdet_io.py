"""
Object Detection (ObDet) bounding-box rendering and YOLO/JSON export helpers.
"""

from medimg_annotator.imports import cv2, Image, json, np, os

from medimg_annotator.core.image_io import write_universal_image

def _normalize_boxes(boxes, bbox, slice_idx, axis, label_text):
    """Normalize single-box and multi-box inputs into a list of box dicts."""
    if boxes is None or (bbox is not None and not boxes):
        if bbox is not None:
            return [{"bbox": bbox, "slice": slice_idx, "axis": axis, "label": label_text, "id": 0}]
        return []
    return boxes

def _extract_raw_slice(image_data, slice_idx, axis):
    """Extract the 2D slice array matching the given volume axis and slice index."""
    if image_data.ndim == 2:
        return image_data.copy()
    if image_data.ndim == 3:
        if axis == 2:
            s_idx = max(0, min(slice_idx, image_data.shape[2] - 1))
            return image_data[:, :, s_idx].copy()
        if axis == 1:
            s_idx = max(0, min(slice_idx, image_data.shape[1] - 1))
            return image_data[:, s_idx, :].copy()
        s_idx = max(0, min(slice_idx, image_data.shape[0] - 1))
        return image_data[s_idx, :, :].copy()
    return image_data[..., 0].copy()

def burn_bounding_box_on_image(
    image_data,
    bbox=None,
    slice_idx=0,
    axis=2,
    format_info=None,
    label_text="ObDet",
    color=(0, 255, 0),
    thickness=2,
    displayed_slice=None,
    boxes=None,
):
    """
    Burn bounding box(es) directly onto an image slice or volume with exact display alignment.

    Returns:
      - annotated_slice_rgb: 2D uint8 RGB image with bounding box(es) burned in.
      - annotated_volume: 3D or 2D array with bounding box(es) burned in on the active slice.
    """
    boxes = _normalize_boxes(boxes, bbox, slice_idx, axis, label_text)
    is_2d = (
        (format_info and format_info.get("is_2d", False))
        or (image_data.ndim == 2)
        or (image_data.ndim == 3 and image_data.shape[2] == 1)
    )

    slice_raw = None
    if is_2d and format_info and format_info.get("rgb_data") is not None:
        canvas_bgr = format_info["rgb_data"].copy()
        if canvas_bgr.ndim == 3 and canvas_bgr.shape[2] == 4:
            canvas_bgr = cv2.cvtColor(canvas_bgr, cv2.COLOR_RGBA2RGB)
    elif displayed_slice is not None:
        if displayed_slice.ndim == 2:
            canvas_bgr = cv2.cvtColor(displayed_slice.copy(), cv2.COLOR_GRAY2RGB)
        else:
            canvas_bgr = displayed_slice.copy()
    else:
        slice_raw = _extract_raw_slice(image_data, slice_idx, axis)
        disp_slice = np.ascontiguousarray(np.flipud(np.rot90(slice_raw)))
        s_min, s_max = float(np.min(disp_slice)), float(np.max(disp_slice))
        if s_max - s_min > 1e-6:
            u8 = np.clip((disp_slice - s_min) * (255.0 / (s_max - s_min)), 0, 255).astype(np.uint8)
        else:
            u8 = np.zeros(disp_slice.shape, dtype=np.uint8)
        canvas_bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)

    canvas_bgr = np.ascontiguousarray(canvas_bgr)
    h, w = canvas_bgr.shape[:2]

    for box in boxes:
        b_slice = box.get("slice", slice_idx)
        b_axis = box.get("axis", axis)
        if not is_2d and (b_slice != slice_idx or b_axis != axis):
            continue

        b_coords = box.get("bbox", None)
        if b_coords is None:
            continue
        bx1, by1, bx2, by2 = [int(round(v)) for v in b_coords]
        bx1, bx2 = max(0, min(min(bx1, bx2), w - 1)), max(0, min(max(bx1, bx2), w - 1))
        by1, by2 = max(0, min(min(by1, by2), h - 1)), max(0, min(max(by1, by2), h - 1))

        cv2.rectangle(canvas_bgr, (bx1, by1), (bx2, by2), color, thickness)

        b_label = str(box.get("label", label_text))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        (tw, th), _ = cv2.getTextSize(b_label, font, font_scale, 1)

        if by1 - th - 6 >= 0:
            banner_y1, banner_y2, text_y = by1 - th - 5, by1, by1 - 3
        elif by2 + th + 6 <= h:
            banner_y1, banner_y2, text_y = by2, by2 + th + 5, by2 + th + 2
        else:
            banner_y1, banner_y2, text_y = by1, by1 + th + 5, by1 + th + 2

        cv2.rectangle(canvas_bgr, (bx1, banner_y1), (min(w - 1, bx1 + tw + 6), banner_y2), (20, 20, 20), -1)
        cv2.rectangle(canvas_bgr, (bx1, banner_y1), (min(w - 1, bx1 + tw + 6), banner_y2), color, 1)
        cv2.putText(canvas_bgr, b_label, (bx1 + 3, text_y), font, font_scale, color, 1, cv2.LINE_AA)

    annotated_volume = image_data.copy()
    if not is_2d and annotated_volume.ndim == 3:
        slice_gray_disp = cv2.cvtColor(canvas_bgr, cv2.COLOR_RGB2GRAY)
        slice_raw_burned = np.ascontiguousarray(np.rot90(np.flipud(slice_gray_disp), -1))

        if slice_raw is None:
            slice_raw = _extract_raw_slice(image_data, slice_idx, axis)

        s_orig_min, s_orig_max = float(np.min(slice_raw)), float(np.max(slice_raw))
        if s_orig_max - s_orig_min > 1.0:
            burned_val = s_orig_min + (slice_raw_burned.astype(np.float32) / 255.0) * (s_orig_max - s_orig_min)
        else:
            burned_val = slice_raw_burned

        burned_typed = burned_val.astype(image_data.dtype)
        if axis == 2:
            annotated_volume[:, :, slice_idx] = burned_typed
        elif axis == 1:
            annotated_volume[:, slice_idx, :] = burned_typed
        else:
            annotated_volume[slice_idx, :, :] = burned_typed
    else:
        annotated_volume = canvas_bgr

    return canvas_bgr, annotated_volume

def save_obdet_result(
    image_data,
    bbox=None,
    slice_idx=0,
    axis=2,
    output_dir=".",
    filename="result.png",
    affine=None,
    header=None,
    format_info=None,
    label_text="ObDet",
    displayed_slice=None,
    boxes=None,
):
    """
    Save the image with bounding box(es) burned in to output_dir, alongside YOLO (.txt) and JSON metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_img_path = os.path.join(output_dir, filename)
    stem = filename[:-7] if filename.lower().endswith(".nii.gz") else os.path.splitext(filename)[0]

    boxes = _normalize_boxes(boxes, bbox, slice_idx, axis, label_text)

    burned_slice_rgb, burned_volume = burn_bounding_box_on_image(
        image_data,
        bbox=bbox,
        slice_idx=slice_idx,
        axis=axis,
        format_info=format_info,
        label_text=label_text,
        displayed_slice=displayed_slice,
        boxes=boxes,
    )

    is_2d = (
        (format_info and format_info.get("is_2d", False))
        or (image_data.ndim == 2)
        or (image_data.ndim == 3 and image_data.shape[2] == 1)
    )

    if is_2d:
        write_universal_image(burned_slice_rgb, out_img_path, affine=affine, header=header, format_info=format_info)
    else:
        write_universal_image(burned_volume, out_img_path, affine=affine, header=header, format_info=format_info)
        png_preview = os.path.join(output_dir, f"{stem}_obdet_slice{slice_idx}.png")
        Image.fromarray(burned_slice_rgb).save(png_preview)

    h, w = burned_slice_rgb.shape[:2]

    txt_path = os.path.join(output_dir, f"{stem}_bbox.txt")
    box_records = []
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, b in enumerate(boxes):
            bx1, by1, bx2, by2 = b["bbox"]
            bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
            by1, by2 = min(by1, by2), max(by1, by2)
            cx_norm = ((bx1 + bx2) / 2.0) / float(w)
            cy_norm = ((by1 + by2) / 2.0) / float(h)
            bw_norm = abs(bx2 - bx1) / float(w)
            bh_norm = abs(by2 - by1) / float(h)
            cls_id = int(b.get("label_id", 0))
            f.write(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

            box_records.append(
                {
                    "id": b.get("id", i + 1),
                    "label": b.get("label", label_text),
                    "label_id": cls_id,
                    "slice_index": int(b.get("slice", slice_idx)),
                    "axis": int(b.get("axis", axis)),
                    "bounding_box_pixel": {"x1": int(bx1), "y1": int(by1), "x2": int(bx2), "y2": int(by2)},
                    "bounding_box_normalized": {
                        "cx": float(cx_norm),
                        "cy": float(cy_norm),
                        "w": float(bw_norm),
                        "h": float(bh_norm),
                    },
                }
            )

    json_path = os.path.join(output_dir, f"{stem}_bbox.json")
    json_data = {
        "image_filename": filename,
        "slice_index": int(slice_idx),
        "axis": int(axis),
        "image_dimensions": {"width": int(w), "height": int(h)},
        "total_boxes": len(box_records),
        "boxes": box_records,
    }
    if box_records:
        json_data["bounding_box_pixel"] = box_records[0]["bounding_box_pixel"]
        json_data["bounding_box_normalized"] = box_records[0]["bounding_box_normalized"]
        json_data["label"] = box_records[0]["label"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    return out_img_path
