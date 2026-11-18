"""
50-step zlib-compressed 2D/3D Undo and Redo history management.
"""

from medimg_annotator.imports import np, zlib

def _slice_view(mask_data, axis, slice_idx):
    """Return a 2D slice view of the 3D mask along the specified axis."""
    if axis == 2:
        return mask_data[:, :, slice_idx]
    if axis == 1:
        return mask_data[:, slice_idx, :]
    return mask_data[slice_idx, :, :]

def _encode_state(mask_data, axis, slice_idx, force_3d):
    """Capture a compressed 2D slice delta or full 3D mask snapshot."""
    if not force_3d:
        slice_arr = _slice_view(mask_data, axis, slice_idx).copy()
        compressed = zlib.compress(slice_arr.tobytes(), level=1)
        return ("2D", axis, slice_idx, compressed, slice_arr.shape)
    compressed_mask = zlib.compress(mask_data.tobytes(), level=1)
    return ("3D", compressed_mask, mask_data.shape)

def _restore_entry(mask_data, entry):
    """Decompress and apply an undo/redo snapshot entry, returning updated mask_data."""
    m_dtype = mask_data.dtype if mask_data is not None else np.uint8

    if entry[0] == "2D":
        _, axis, s_idx, compressed, shape = entry
        decompressed = zlib.decompress(compressed)
        restored_slice = np.frombuffer(decompressed, dtype=m_dtype).reshape(shape)
        if mask_data is not None and axis < mask_data.ndim and s_idx < mask_data.shape[axis]:
            _slice_view(mask_data, axis, s_idx)[...] = restored_slice
        return mask_data

    if entry[0] == "3D":
        _, compressed, shape = entry
    else:
        # Backward compatibility with legacy tuple (compressed, shape)
        compressed, shape = entry
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=m_dtype).reshape(shape).copy()

class HistoryMixin:
    """Mixin providing compressed 2D/3D Undo & Redo operations for NiftiMaskEditor."""

    MAX_HISTORY_DEPTH = 50

    def push_undo(self, force_3d=False):
        """Record current mask state on undo stack and clear redo stack."""
        if self.mask_data is None:
            return

        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self.redo_stack.clear()

        is_2d = not force_3d and self.scope_var.get() == "2D"
        entry = _encode_state(self.mask_data, self.axis_index, self.current_slice, force_3d=not is_2d)
        self.undo_stack.append(entry)

        if len(self.undo_stack) > self.MAX_HISTORY_DEPTH:
            self.undo_stack.pop(0)

        self.set_unsaved_status()

    def _transfer_history_step(self, source_stack, target_stack):
        """Move one state transition between undo and redo stacks."""
        entry = source_stack.pop()

        if self.mask_data is not None:
            if entry[0] == "2D":
                _, axis, s_idx, _, _ = entry
                if axis < self.mask_data.ndim and s_idx < self.mask_data.shape[axis]:
                    cur_entry = _encode_state(self.mask_data, axis, s_idx, force_3d=False)
                    target_stack.append(cur_entry)
            else:
                target_stack.append(_encode_state(self.mask_data, self.axis_index, self.current_slice, force_3d=True))

            if len(target_stack) > self.MAX_HISTORY_DEPTH:
                target_stack.pop(0)

        self.mask_data = _restore_entry(self.mask_data, entry)
        self.update_views()
        self.update_volume_stats()
        self.set_unsaved_status()

    def undo_action(self):
        """Revert the most recent segmentation or interpolation action."""
        if not self.undo_stack:
            return
        if not hasattr(self, "redo_stack"):
            self.redo_stack = []
        self._transfer_history_step(self.undo_stack, self.redo_stack)

    def redo_action(self):
        """Re-apply the most recently undone action."""
        if not hasattr(self, "redo_stack") or not self.redo_stack:
            return
        self._transfer_history_step(self.redo_stack, self.undo_stack)
