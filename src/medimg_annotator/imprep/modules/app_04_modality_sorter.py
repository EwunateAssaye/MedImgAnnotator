# imprep/modules/app_04_modality_sorter.py
from medimg_annotator.imports import ctk, filedialog, messagebox, os, pd, re, shutil, threading, tk

from medimg_annotator.imprep.config import COLORS, FONTS

class ModalitySorterFrame(ctk.CTkFrame):
    def __init__(self, parent, title, controller):
        super().__init__(parent, fg_color=COLORS["bg_dark"])
        self.controller = controller

        self.input_folder_var = tk.StringVar()
        self.csv_file_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        container = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_panel"], corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Header
        ctk.CTkLabel(container, text=title, font=FONTS["title"], text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 2))
        ctk.CTkLabel(
            container,
            text="Automatically sorts and organizes medical image volumes into modality-specific folders (MR, CT, PET) using metadata tables (CSV/XLSX).",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            anchor="w",
            wraplength=700,
            justify="left"
        ).pack(fill="x", pady=(0, 15))

        # I/O Card
        io_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        io_card.pack(fill="x", pady=8, padx=2)
        ctk.CTkLabel(io_card, text="Source Folder & Metadata Configuration", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 6))

        self._make_row(io_card, "NIfTI / DICOM Folder:", self.input_folder_var, self.browse_folder)
        self._make_row(io_card, "Metadata Table (CSV/XLSX):", self.csv_file_var, self.browse_csv)

        # Action Buttons
        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(fill="x", pady=15, padx=2)

        self.btn_sort = ctk.CTkButton(
            btn_frame,
            text="START MODALITY SORTING",
            font=FONTS["button"],
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=40,
            command=self.start_sorting_thread
        )
        self.btn_sort.pack(fill="x")

        # Log Card
        log_card = ctk.CTkFrame(container, fg_color=COLORS["bg_card"], corner_radius=6)
        log_card.pack(fill="both", expand=True, pady=8, padx=2)
        ctk.CTkLabel(log_card, text="Sorting Log", font=FONTS["sub_header"], text_color=COLORS["text_secondary"]).pack(anchor="w", padx=12, pady=(10, 4))

        self.log_box = ctk.CTkTextbox(log_card, height=200, font=("Consolas", 10), text_color=COLORS["text_secondary"], fg_color=COLORS["bg_dark"])
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def _make_row(self, parent, label, var, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label, width=170, anchor="w", font=FONTS["body"]).pack(side="left")
        ctk.CTkEntry(row, textvariable=var, font=FONTS["small"], height=30).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(row, text="Browse", width=70, height=30, fg_color=COLORS["bg_panel"], hover_color=COLORS["accent"], command=cmd).pack(side="right")

    def browse_folder(self):
        f = filedialog.askdirectory(title="Select Folder to Sort")
        if f:
            self.input_folder_var.set(f)

    def browse_csv(self):
        f = filedialog.askopenfilename(
            title="Select Metadata Table",
            filetypes=[("Metadata Tables", "*.csv *.xlsx"), ("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if f:
            self.csv_file_var.set(f)

    def log(self, message):
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def sanitize_string(self, text):
        if pd.isna(text):
            return ""
        return re.sub(r'[^a-z0-9]', '', str(text).lower())

    def start_sorting_thread(self):
        if not self.input_folder_var.get() or not self.csv_file_var.get():
            messagebox.showwarning("Missing Input", "Please select both the folder to sort and the metadata file.")
            return

        self.btn_sort.configure(state="disabled", text="SORTING IN PROGRESS...")
        threading.Thread(target=self.run_sorting, daemon=True).start()

    def run_sorting(self):
        folder_path = self.input_folder_var.get()
        csv_path = self.csv_file_var.get()

        try:
            self.log(f"Loading metadata from {os.path.basename(csv_path)}...")
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(csv_path)

            cols_lower = {str(c).lower().strip(): c for c in df.columns}
            sub_col = cols_lower.get('subject id') or cols_lower.get('patientid') or cols_lower.get('id')
            desc_col = cols_lower.get('series description') or cols_lower.get('description') or cols_lower.get('protocol')
            mod_col = cols_lower.get('modality') or cols_lower.get('series modality')

            if not sub_col or not desc_col or not mod_col:
                self.log(f"Warning: Expected columns ('Subject ID', 'Series Description', 'Modality') not all identified. Available: {list(df.columns)}")
                messagebox.showerror("Column Error", "Metadata table must contain Subject ID, Series Description, and Modality columns.")
                return

            files_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(('.nii', '.nii.gz'))]
            self.log(f"Found {len(files_in_dir)} NIfTI files in folder.")

            files_moved = 0
            for _, row in df.iterrows():
                subject_id = str(row[sub_col]).strip()
                series_desc = str(row[desc_col]).strip()
                modality = str(row[mod_col]).strip().upper()

                clean_id = self.sanitize_string(subject_id)
                clean_desc = self.sanitize_string(series_desc)

                if not clean_id or not clean_desc or not modality or modality == "NAN":
                    continue

                target_folder = os.path.join(folder_path, modality)
                os.makedirs(target_folder, exist_ok=True)

                for fname in files_in_dir:
                    clean_fname = self.sanitize_string(fname)
                    if clean_id in clean_fname and clean_desc in clean_fname:
                        src = os.path.join(folder_path, fname)
                        dst = os.path.join(target_folder, fname)
                        if os.path.exists(src):
                            shutil.move(src, dst)
                            self.log(f"Moved: {fname} -> {modality}/")
                            files_moved += 1

            self.log(f"Done. Successfully sorted {files_moved} files.")
            messagebox.showinfo("Sorting Complete", f"Successfully sorted {files_moved} files into modality subfolders.")
        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_sort.configure(state="normal", text="START MODALITY SORTING")
