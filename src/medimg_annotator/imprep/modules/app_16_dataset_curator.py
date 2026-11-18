# imprep/modules/app_16_dataset_curator.py
from medimg_annotator.imports import (
    csv,
    ctk,
    DateTime as datetime,
    difflib,
    filedialog,
    messagebox,
    Path,
    re,
    shutil,
    threading,
    tk,
)

from medimg_annotator.imprep.base import BaseNiftiTool
from medimg_annotator.imprep.config import COLORS, FONTS

class DatasetCuratorFrame(BaseNiftiTool):
    def __init__(self, parent, title, controller):
        super().__init__(
            parent,
            title,
            controller,
            description="Smart fuzzy one-to-one pairing of images and masks, standardized continuous indexing, and Train/Test dataset split curation."
        )

        # Dataset curator focuses on batch curation rather than single 2D slice visualization
        self.slider_frame.pack_forget()
        self.vis_frame.grid_forget()
        self.grid_columnconfigure(0, weight=1)

    def add_custom_widgets(self, parent):
        info_card = ctk.CTkFrame(parent, fg_color="transparent")
        info_card.pack(fill="x", padx=8, pady=4)

        ctk.CTkLabel(
            info_card,
            text="• Multi-tiered string token & fuzzy Jaccard similarity algorithms.\n• Enforces strict 1-to-1 matching (no duplicate image/mask pairings).\n• Standardizes naming (img_0001 / msk_0001) into train or test subfolders.\n• Generates full audit log CSV in output directory.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        # Paths
        grp_paths = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        grp_paths.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(grp_paths, text="Source & Target Directory Configuration", font=FONTS["sub_header"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.src_img = tk.StringVar()
        self.src_msk = tk.StringVar()
        self.dst_root = tk.StringVar()

        self._make_custom_row(grp_paths, "Input Images:", self.src_img)
        self._make_custom_row(grp_paths, "Input Masks:", self.src_msk)
        self._make_custom_row(grp_paths, "Dest Root:", self.dst_root)

        # Target Split
        grp_conf = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        grp_conf.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(grp_conf, text="Dataset Split Target:", font=FONTS["sub_header"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.split_var = tk.StringVar(value="train")
        row_split = ctk.CTkFrame(grp_conf, fg_color="transparent")
        row_split.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkRadioButton(row_split, text="Train Set (trainimg / trainmsk)", variable=self.split_var, value="train", font=FONTS["body"]).pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(row_split, text="Test Set (testimg / testmsk)", variable=self.split_var, value="test", font=FONTS["body"]).pack(side="left")

        # Progress
        grp_prog = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=6)
        grp_prog.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(grp_prog, text="Curation Execution Status", font=FONTS["sub_header"]).pack(anchor="w", padx=10, pady=(6, 2))

        self.status_var = tk.StringVar(value="Ready")
        self.lbl_status = ctk.CTkLabel(grp_prog, textvariable=self.status_var, font=FONTS["small"], text_color=COLORS["accent"], anchor="w")
        self.lbl_status.pack(fill="x", padx=10, pady=2)

        self.pbar = ctk.CTkProgressBar(grp_prog, progress_color=COLORS["accent"])
        self.pbar.pack(fill="x", padx=10, pady=(2, 10))
        self.pbar.set(0)

        # Re-configure main button text
        self.btn_run.configure(text="START SMART CURATION", command=self.run_curation)

    def _make_custom_row(self, parent, label, var):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row, text=label, width=100, anchor="w", font=FONTS["small"]).pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=28).pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row, text="...", width=32, height=28, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=lambda: var.set(filedialog.askdirectory())).pack(side="right")

    def get_numbers(self, filename):
        name = filename.lower()
        name = re.sub(r'_mask_from_rater\d+', '', name)
        return tuple(int(n) for n in re.findall(r'\d+', name))

    def clean_filename(self, filename):
        name = filename.lower()
        name = re.sub(r'_mask_from_rater\d+', '', name)
        while True:
            if name.endswith('.nii.gz'):
                name = name[:-7]
            elif name.endswith('.nii'):
                name = name[:-4]
            elif name.endswith('.gz'):
                name = name[:-3]
            else:
                break
        name = name.replace('.nii.gz', '').replace('.nii', '')
        name_spaced = re.sub(r'[^a-z0-9]', ' ', name)
        keywords = ["volume", "vol", "image", "img", "mask", "segmentation", "seg", "label", "gt", "ground", "truth"]
        for k in keywords:
            name_spaced = re.sub(rf'\b{k}\b', '', name_spaced)
        return re.sub(r'\s+', '', name_spaced)

    def get_tokens(self, filename):
        name = filename.lower()
        name = re.sub(r'_mask_from_rater\d+', '', name)
        name = name.replace('.nii.gz', '').replace('.nii', '')
        keywords = {"volume", "vol", "image", "img", "mask", "segmentation", "seg", "label", "gt", "ground", "truth"}
        tokens = set(re.findall(r'[a-z0-9]+', name))
        return tokens - keywords

    def calculate_similarity(self, str1, str2):
        c1 = self.clean_filename(str1)
        c2 = self.clean_filename(str2)

        if c1 == c2 and c1 != "":
            return 1.0

        t1 = self.get_tokens(str1)
        t2 = self.get_tokens(str2)
        jaccard = 0.0
        if t1 and t2:
            intersection = len(t1.intersection(t2))
            union = len(t1.union(t2))
            jaccard = intersection / union if union > 0 else 0

        nums1 = self.get_numbers(str1)
        nums2 = self.get_numbers(str2)
        num_penalty = 0.0
        if nums1 and nums2:
            if not set(nums1).intersection(set(nums2)):
                num_penalty = 0.5

        fuzzy = difflib.SequenceMatcher(None, c1, c2).ratio()
        final_score = max(jaccard, fuzzy * 0.8) - num_penalty
        return max(0.0, min(final_score, 1.0))

    def zero_name(self, prefix, idx, ext, digits=4):
        return f"{prefix}_{idx:0{digits}d}{ext}"

    def get_next_index(self, folder: Path, prefix: str):
        pattern = re.compile(rf"^{prefix}_(\d+)", re.IGNORECASE)
        max_num = 0
        if folder.exists():
            for f in folder.iterdir():
                match = pattern.match(f.stem)
                if match:
                    try:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
                    except:
                        pass
        return max_num + 1

    def run_curation(self):
        img_dir = self.src_img.get()
        msk_dir = self.src_msk.get()
        out_dir = self.dst_root.get()

        if not (img_dir and msk_dir and out_dir):
            messagebox.showerror("Missing Paths", "Please select Input Images, Input Masks, and Destination Root folders.")
            return

        self.btn_run.configure(state="disabled", text="CURATING DATASET...")
        threading.Thread(target=self._worker_curation, args=(img_dir, msk_dir, out_dir), daemon=True).start()

    def _worker_curation(self, img_dir_str, msk_dir_str, out_dir_str):
        try:
            img_dir = Path(img_dir_str)
            msk_dir = Path(msk_dir_str)
            out_root = Path(out_dir_str)
            split_mode = self.split_var.get()

            self.status_var.set("Scanning folders...")
            images = sorted([p for p in img_dir.iterdir() if p.is_file() and (p.suffix == ".nii" or p.name.lower().endswith(".nii.gz"))])
            masks = sorted([p for p in msk_dir.iterdir() if p.is_file() and (p.suffix == ".nii" or p.name.lower().endswith(".nii.gz"))])

            if not images or not masks:
                messagebox.showerror("Empty Folders", "Input folder(s) contain no valid NIfTI files.")
                return

            self.status_var.set("Calculating similarity matrix...")
            possible_matches = []
            for i, img in enumerate(images):
                for j, msk in enumerate(masks):
                    score = self.calculate_similarity(img.name, msk.name)
                    if score > 0.45:
                        possible_matches.append((score, i, j))
                self.pbar.set(i / len(images))

            possible_matches.sort(key=lambda x: x[0], reverse=True)

            final_pairs = []
            used_images = set()
            used_masks = set()
            for score, i, j in possible_matches:
                if i not in used_images and j not in used_masks:
                    final_pairs.append((images[i], masks[j]))
                    used_images.add(i)
                    used_masks.add(j)

            if not final_pairs:
                self.status_var.set("No matching pairs found.")
                messagebox.showwarning("No Matches", "No image-mask pairs met the similarity threshold.")
                return

            sub_img = "trainimg" if split_mode == "train" else "testimg"
            sub_msk = "trainmsk" if split_mode == "train" else "testmsk"
            dst_img_dir = out_root / sub_img
            dst_msk_dir = out_root / sub_msk
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_msk_dir.mkdir(parents=True, exist_ok=True)

            current_idx = self.get_next_index(dst_img_dir, "img")
            success_count = 0
            log_rows = []

            for k, (img_path, mask_path) in enumerate(final_pairs):
                self.status_var.set(f"Pairing ({k+1}/{len(final_pairs)}): {img_path.name}")
                self.pbar.set((k + 1) / len(final_pairs))

                ext_img = ".nii.gz" if img_path.name.lower().endswith(".nii.gz") else ".nii"
                ext_msk = ".nii.gz" if mask_path.name.lower().endswith(".nii.gz") else ".nii"

                new_img_name = self.zero_name("img", current_idx, ext_img)
                new_msk_name = self.zero_name("msk", current_idx, ext_msk)

                try:
                    shutil.copy2(img_path, dst_img_dir / new_img_name)
                    shutil.copy2(mask_path, dst_msk_dir / new_msk_name)
                    log_rows.append([img_path.name, mask_path.name, new_img_name, new_msk_name])
                    success_count += 1
                    current_idx += 1
                except Exception as e:
                    print(f"Error copying pair: {e}")
                    log_rows.append([img_path.name, mask_path.name, "ERROR", str(e)])

            log_file = out_root / f"curation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Original_Image", "Matched_Mask", "New_Image_Name", "New_Mask_Name"])
                writer.writerows(log_rows)

            skipped_img = len(images) - success_count
            self.status_var.set("Curation complete.")
            messagebox.showinfo(
                "Curation Complete",
                f"Successfully matched and curated: {success_count} pairs.\nUnmatched Images: {skipped_img}\n\nLog saved to:\n{log_file}"
            )
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Curation Error", str(e))
        finally:
            self.btn_run.configure(state="normal", text="START SMART CURATION")
