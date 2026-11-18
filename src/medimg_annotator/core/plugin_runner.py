"""
Backend helpers for external plugin discovery, interpreter resolution,
archive extraction, and isolated subprocess launching.
"""

from medimg_annotator.imports import json, os, shutil, subprocess, sys, tarfile, zipfile

SKIP_DIRS = {
    "_internal",
    "venv",
    ".env",
    "env",
    "site-packages",
    "__pycache__",
    ".git",
    ".vscode",
    "lib",
    "include",
    "tcl",
    "tk",
    "weights",
    "models",
    "checkpoints",
    "data",
    "dataset",
    "logs",
    "assets",
    "__macosx",
}

SKIP_FILES = {"__init__.py", "setup.py"}

def get_plugin_dir():
    """Return path to the workstation plugin directory and migrate legacy plugins if present."""
    plugin_dir = os.path.join(os.path.expanduser("~"), "MedImgAnnotator_Plugins")
    try:
        os.makedirs(plugin_dir, exist_ok=True)
    except Exception:
        pass

    old_dir = os.path.join(os.path.expanduser("~"), "MaskBoard_Plugins")
    if os.path.exists(old_dir) and os.path.isdir(old_dir):
        try:
            for item in os.listdir(old_dir):
                src_p = os.path.join(old_dir, item)
                dst_p = os.path.join(plugin_dir, item)
                if not os.path.exists(dst_p):
                    if os.path.isdir(src_p):
                        shutil.copytree(src_p, dst_p)
                    else:
                        shutil.copy2(src_p, dst_p)
        except Exception as mig_err:
            print(f"Migration note: {mig_err}")

    return plugin_dir

def is_valid_python(exe_path):
    """Validate that exe_path points to a standard Python interpreter executable."""
    if not exe_path or not isinstance(exe_path, str):
        return False
    if not os.path.isfile(exe_path):
        return False
    name = os.path.basename(exe_path).lower()
    if "medimg" in name:
        return False
    return name in ("python.exe", "pythonw.exe", "python", "python3.exe", "python3")

def find_python_interpreter(cached_exec=None):
    """Discover a suitable host Python interpreter (preferring Anaconda/Miniconda or active env)."""
    if cached_exec and is_valid_python(cached_exec):
        return cached_exec

    home = os.path.expanduser("~")
    candidates = [
        r"C:\Users\assay\anaconda3\python.exe",
        os.path.join(home, "anaconda3", "python.exe"),
        os.path.join(home, "miniconda3", "python.exe"),
        os.path.join(home, "AppData", "Local", "anaconda3", "python.exe"),
        os.path.join(home, "AppData", "Local", "miniconda3", "python.exe"),
        r"C:\ProgramData\anaconda3\python.exe",
        r"C:\ProgramData\miniconda3\python.exe",
        shutil.which("python.exe"),
        shutil.which("python"),
    ]

    if not getattr(sys, "frozen", False) and is_valid_python(sys.executable):
        candidates.append(sys.executable)

    appdata_python = os.path.join(home, "AppData", "Local", "Programs", "Python")
    if os.path.exists(appdata_python):
        try:
            for folder in sorted(os.listdir(appdata_python), reverse=True):
                candidates.append(os.path.join(appdata_python, folder, "python.exe"))
        except Exception:
            pass

    for path in candidates:
        if path and is_valid_python(path):
            return os.path.abspath(path)

    return "python"

def get_plugin_interpreter(target_path, default_interpreter=None):
    """Prefer a plugin-local virtual environment interpreter when present."""
    plugin_dir_loc = os.path.dirname(target_path)
    for venv_name in ("venv", ".venv", "env"):
        venv_py = os.path.join(plugin_dir_loc, venv_name, "Scripts", "python.exe")
        if is_valid_python(venv_py):
            return os.path.abspath(venv_py)
    return default_interpreter or find_python_interpreter()

def build_clean_plugin_env(context_vars=None):
    """Build a sanitized environment dict with optional MIA_* workstation context variables."""
    clean_env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS", "_MEIPASS2", "TCL_LIBRARY", "TK_LIBRARY"):
        clean_env.pop(var, None)

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        try:
            meipass = os.path.normcase(os.path.abspath(getattr(sys, "_MEIPASS")))
            cleaned_paths = [
                p for p in clean_env.get("PATH", "").split(os.pathsep)
                if p and os.path.normcase(os.path.abspath(p)) != meipass
            ]
            clean_env["PATH"] = os.pathsep.join(cleaned_paths)
        except Exception:
            pass

    if context_vars:
        clean_env.update(context_vars)
    return clean_env

def launch_pip_install(pkg_or_req, py_exec, cwd):
    """Spawn pip install in an isolated terminal/process for a package or requirements file."""
    clean_env = build_clean_plugin_env()
    target_str = pkg_or_req.strip()
    is_file = os.path.isfile(target_str) or target_str.lower().endswith(".txt")

    if is_file:
        banner = f"Installing dependencies from {os.path.basename(target_str)}"
        inner_cmd = f'echo {banner}... & "{py_exec}" -m pip install -r "{target_str}"'
    else:
        banner = f"Installing Python library: {target_str}"
        inner_cmd = f'echo {banner}... & "{py_exec}" -m pip install {target_str}'

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        cmd_line = (
            f'cmd.exe /k "{inner_cmd} & echo. & '
            f'echo ==================================================== & '
            f'echo [Completed] Package installation routine finished. & '
            f'echo You can close this console window now. & '
            f'echo ===================================================="'
        )
        subprocess.Popen(cmd_line, cwd=cwd, env=clean_env, creationflags=creationflags, close_fds=True)
    else:
        if is_file:
            subprocess.Popen([py_exec, "-m", "pip", "install", "-r", target_str], cwd=cwd, env=clean_env, close_fds=True)
        else:
            subprocess.Popen([py_exec, "-m", "pip", "install"] + target_str.split(), cwd=cwd, env=clean_env, close_fds=True)

def launch_plugin_process(target_path, py_exec, context_vars=None):
    """Execute a runnable plugin script or binary with isolated workstation environment variables."""
    cwd = os.path.dirname(target_path)
    clean_env = build_clean_plugin_env(context_vars)

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        if target_path.endswith(".py"):
            cmd_line = f'cmd.exe /k ""{py_exec}" "{target_path}""'
            subprocess.Popen(cmd_line, cwd=cwd, env=clean_env, creationflags=creationflags, close_fds=True)
        elif target_path.endswith(".bat"):
            cmd_line = f'cmd.exe /k ""{target_path}""'
            subprocess.Popen(cmd_line, cwd=cwd, env=clean_env, creationflags=creationflags, close_fds=True)
        else:
            subprocess.Popen([target_path], cwd=cwd, env=clean_env, creationflags=creationflags, close_fds=True)
    else:
        if target_path.endswith(".py"):
            subprocess.Popen([py_exec, target_path], cwd=cwd, env=clean_env, close_fds=True)
        else:
            subprocess.Popen([target_path], cwd=cwd, env=clean_env, close_fds=True)

def scan_plugins(plugin_dir):
    """Walk plugin_dir up to 3 levels deep and return list of (name, full_path, rel_path)."""
    runnables = []
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        rel_parts = os.path.relpath(root, plugin_dir).split(os.sep)
        if len(rel_parts) > 3:
            dirs.clear()
            continue

        manifest_entry = None
        for mf in ("plugin.json", "metadata.json", "manifest.json"):
            mf_path = os.path.join(root, mf)
            if os.path.isfile(mf_path):
                try:
                    with open(mf_path, "r", encoding="utf-8") as f:
                        mdata = json.load(f)
                        manifest_entry = mdata.get("entry") or mdata.get("main")
                except Exception:
                    pass
                break

        if manifest_entry and os.path.isfile(os.path.join(root, manifest_entry)):
            full_path = os.path.join(root, manifest_entry)
            runnables.append((manifest_entry, full_path, os.path.relpath(full_path, plugin_dir)))
            continue

        for file in files:
            if file.lower() in SKIP_FILES or file.startswith("test_") or file.endswith("_test.py"):
                continue
            if file.endswith((".exe", ".py", ".bat")):
                full_path = os.path.join(root, file)
                runnables.append((file, full_path, os.path.relpath(full_path, plugin_dir)))
    return runnables

def find_requirements_in_tree(search_dir):
    """Return path to the first requirements.txt inside search_dir, or None."""
    if search_dir and os.path.isdir(search_dir):
        for root_c, _, files_c in os.walk(search_dir):
            if "requirements.txt" in files_c:
                return os.path.join(root_c, "requirements.txt")
    return None

def install_plugin_archive_or_file(selected_file, plugin_dir):
    """Install a single plugin file or extract a .zip/.tar.gz archive into plugin_dir."""
    base_name = os.path.basename(selected_file)
    target_dest = os.path.normpath(os.path.abspath(os.path.join(plugin_dir, base_name)))
    lower_name = selected_file.lower()
    extract_target = None

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(selected_file, "r") as zip_ref:
            namelist = [n for n in zip_ref.namelist() if not n.startswith("__MACOSX")]
            top_levels = {n.split("/")[0] for n in namelist if "/" in n}
            if len(top_levels) == 1 and all(
                n.startswith(list(top_levels)[0] + "/") or n == list(top_levels)[0] for n in namelist
            ):
                extract_target = plugin_dir
            else:
                extract_target = os.path.join(plugin_dir, os.path.splitext(base_name)[0])
            os.makedirs(extract_target, exist_ok=True)
            zip_ref.extractall(extract_target)
    elif lower_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(selected_file, "r:*") as tar_ref:
            folder_name = base_name.replace(".tar.gz", "").replace(".tgz", "")
            extract_target = os.path.join(plugin_dir, folder_name)
            os.makedirs(extract_target, exist_ok=True)
            tar_ref.extractall(extract_target)
    else:
        os.makedirs(os.path.dirname(target_dest), exist_ok=True)
        shutil.copy2(selected_file, target_dest)

    return find_requirements_in_tree(extract_target) if extract_target else None
