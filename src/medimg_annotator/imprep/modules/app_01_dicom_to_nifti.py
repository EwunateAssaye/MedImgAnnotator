# imprep/modules/app_01_dicom_to_nifti.py
from medimg_annotator.imports import (
    csv,
    ctk,
    dicom2nifti,
    filedialog,
    messagebox,
    os,
    pydicom,
    shutil,
    tempfile,
    threading,
    tk,
)

from medimg_annotator.imprep.config import COLORS, FONTS

class DicomConverterFrame(ctk.CTkFrame):
    def __init__(self, parent, title, controller):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.controller = controller

        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.csv_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.is_running = False
        self.metadata_map = {}

        container = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_panel"], corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Header
        ctk.CTkLabel(
            container,
            text=title,
            font=FONTS["title"],
            text_color=COLORS["text_primary"],
            anchor="w"
        ).pack(fill="x", pady=(5, 2))

        ctk.CTkLabel(
            container,
            text="Advanced DICOM to NIfTI Converter: Handles In/Out Phase Echo-Time splitting, metadata CSV renaming, and anti-overwrite safety.",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
            wraplength=700,
            justify="left"
        ).pack(fill="x", pady=(0, 15))

        # I/O Card
        io_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        io_card.pack(fill="x", pady=8, padx=2)

        ctk.CTkLabel(io_card, text="Directory & Metadata Configuration", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 6))

        self._make_row(io_card, "DICOM Folder:", self.input_path_var, self.browse_input)
        self._make_row(io_card, "Output Folder:", self.output_path_var, self.browse_output)
        self._make_row(io_card, "Metadata CSV (Opt):", self.csv_path_var, self.browse_csv)

        # Status Card
        status_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        status_card.pack(fill="x", pady=8, padx=2)

        ctk.CTkLabel(status_card, text="Conversion Progress", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 4))

        self.pbar = ctk.CTkProgressBar(status_card, variable=self.progress_var, progress_color=COLORS["accent"])
        self.pbar.pack(fill="x", padx=12, pady=6)
        self.pbar.set(0)

        self.lbl_status = ctk.CTkLabel(
            status_card,
            textvariable=self.status_var,
            font=FONTS["body"],
            text_color=COLORS["accent"],
            anchor="w",
            justify="left",
            wraplength=650
        )
        self.lbl_status.pack(fill="x", padx=12, pady=(2, 10))

        # Action Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(fill="x", pady=15, padx=2)

        self.btn_run = ctk.CTkButton(
            btn_frame,
            text="START CONVERSION",
            font=FONTS["button"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=40,
            command=self.start_thread
        )
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_reset = ctk.CTkButton(
            btn_frame,
            text="Reset",
            font=FONTS["button"],
            fg_color=COLORS["warning"],
            hover_color="#d35400",
            width=100,
            height=40,
            command=self.reset_form
        )
        self.btn_reset.pack(side="right")

    def _make_row(self, parent, label, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label, width=130, anchor="w", font=FONTS["body"]).pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=30).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(row, text="Browse", width=70, height=30, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=cmd).pack(side="right")

    def browse_input(self):
        f = filedialog.askdirectory(title="Select Root DICOM Directory")
        if f:
            self.input_path_var.set(f)

    def browse_output(self):
        f = filedialog.askdirectory(title="Select Output NIfTI Directory")
        if f:
            self.output_path_var.set(f)

    def browse_csv(self):
        f = filedialog.askopenfilename(title="Select Metadata CSV", filetypes=[("CSV Files", "*.csv")])
        if f:
            self.csv_path_var.set(f)

    def reset_form(self):
        if self.is_running:
            return
        self.input_path_var.set("")
        self.output_path_var.set("")
        self.csv_path_var.set("")
        self.status_var.set("Ready")
        self.progress_var.set(0)
        self.pbar.set(0)

    def sanitize_filename(self, name):
        return str(name).replace(" ", "_").replace("/", "-").replace("\\", "-").replace(":", "").replace("*", "")

    def is_dicom_file(self, filepath):
        if os.path.basename(filepath).upper() == "DICOMDIR":
            return False
        if not os.path.isfile(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                f.seek(128)
                if f.read(4) == b'DICM':
                    return True
        except:
            pass
        try:
            pydicom.dcmread(filepath, stop_before_pixels=True)
            return True
        except:
            return False

    def load_csv_data(self, csv_path):
        data_map = {}
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                key_col = "SeriesInstanceUID" if "SeriesInstanceUID" in headers else "PatientID"
                if key_col not in headers:
                    return {}
                for row in reader:
                    val = row[key_col].strip()
                    if val:
                        data_map[val] = row
            return data_map
        except:
            return {}

    def start_thread(self):
        input_dir = self.input_path_var.get()
        output_dir = self.output_path_var.get()
        csv_path = self.csv_path_var.get()

        if not input_dir or not output_dir:
            messagebox.showerror("Missing Paths", "Please select both input and output folders.")
            return

        self.btn_run.configure(state="disabled", text="CONVERTING...")
        self.is_running = True
        threading.Thread(target=self.run_echo_split_conversion, args=(input_dir, output_dir, csv_path), daemon=True).start()

    def run_echo_split_conversion(self, root_dir, output_dir, csv_path):
        try:
            os.makedirs(output_dir, exist_ok=True)

            if csv_path and os.path.isfile(csv_path):
                self.status_var.set("Loading CSV Data...")
                self.metadata_map = self.load_csv_data(csv_path)
            else:
                self.metadata_map = {}

            self.status_var.set("Deep Scan: indexing files by UID and Echo Time...")
            series_registry = {}

            file_count = 0
            for current_root, _, files in os.walk(root_dir):
                for f in files:
                    full_path = os.path.join(current_root, f)
                    if file_count % 100 == 0:
                        self.status_var.set(f"Scanning file #{file_count}...")
                    file_count += 1

                    if self.is_dicom_file(full_path):
                        try:
                            ds = pydicom.dcmread(full_path, stop_before_pixels=True)
                            uid = ds.get("SeriesInstanceUID")
                            te = ds.get("EchoTime", 0)
                            try:
                                te_key = round(float(te), 1)
                            except:
                                te_key = 0

                            if uid:
                                unique_key = (uid, te_key)
                                if unique_key not in series_registry:
                                    series_registry[unique_key] = []
                                series_registry[unique_key].append(full_path)
                        except:
                            continue

            total_groups = len(series_registry)
            if total_groups == 0:
                self.status_var.set("No DICOM series found.")
                self.finish_process()
                return

            # Convert Series
            for i, (key, file_list) in enumerate(series_registry.items()):
                uid, echo_time = key
                try:
                    first_ds = pydicom.dcmread(file_list[0], stop_before_pixels=True)
                    patient_id = str(first_ds.get("PatientID", "UnknownID"))
                    series_desc = str(first_ds.get("SeriesDescription", first_ds.get("ProtocolName", "UnknownSeq")))

                    final_base = ""
                    if uid in self.metadata_map:
                        final_base = self.metadata_map[uid].get("TargetName", "")
                    if not final_base and patient_id in self.metadata_map:
                        final_base = f"{self.metadata_map[patient_id].get('TargetName', '')}_{series_desc}"

                    if not final_base:
                        clean_pid = self.sanitize_filename(patient_id)
                        clean_seq = self.sanitize_filename(series_desc)
                        final_base = f"{clean_pid}_{clean_seq}"

                    if echo_time > 0:
                        final_base = f"{final_base}_TE{int(echo_time)}"

                    final_base = self.sanitize_filename(final_base)

                    self.status_var.set(f"Converting ({i+1}/{total_groups}): {final_base}")
                    progress = (i + 1) / total_groups
                    self.progress_var.set(progress)
                    self.pbar.set(progress)

                    with tempfile.TemporaryDirectory() as temp_dir:
                        for src_file in file_list:
                            base_name = os.path.basename(src_file)
                            dst_path = os.path.join(temp_dir, base_name)
                            if os.path.exists(dst_path):
                                dst_path = os.path.join(temp_dir, f"dup_{os.urandom(2).hex()}_{base_name}")
                            try:
                                os.link(src_file, dst_path)
                            except Exception:
                                shutil.copy(src_file, dst_path)

                        dicom2nifti.convert_directory(temp_dir, temp_dir, compression=True, reorient=True)

                        generated = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz')]
                        for nifti in generated:
                            src = os.path.join(temp_dir, nifti)
                            target_name = f"{final_base}.nii.gz"
                            if len(generated) > 1:
                                target_name = f"{final_base}_{nifti}"

                            dest = os.path.join(output_dir, target_name)
                            c = 1
                            orig = dest
                            while os.path.exists(dest):
                                dest = orig.replace(".nii.gz", f"_{c}.nii.gz")
                                c += 1
                            shutil.move(src, dest)

                except Exception as e:
                    print(f"Error on {uid}: {e}")

            self.status_var.set("All conversions complete!")
            messagebox.showinfo("Success", f"Processed {total_groups} unique volumes (split by TE).")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.finish_process()

    def finish_process(self):
        self.is_running = False
        self.btn_run.configure(state="normal", text="START CONVERSION")
