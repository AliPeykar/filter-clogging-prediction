"""
gui.py  -  Filter Analysis App  v2.1
=====================================
Hormozgan Gas Corporation  |  Developed by Ali Peykar

Main window  : file selection, filter parameters, run button, live log
Results viewer (opens when run completes):
    - Left panel  : scrollable summary report + clickable plot list
    - Right panel : full-size plot image with Prev / Next navigation
    - Keyboard    : Left/Right arrow keys to navigate plots
"""

import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import io
import glob
import base64
import shutil
import threading
import subprocess

import FreeSimpleGUI as sg
from PIL import Image
from datakiller import FilterAnalysis


# ============================================================
#  Constants
# ============================================================
APP_TITLE = "Filter Analysis App  v2.1  |  Hormozgan Gas Corporation"
ACCENT    = "#1565C0"
IMG_MAX   = (980, 620)

PLOT_CATALOGUE = [
    ("01_raw_data",                 "1  |  Raw Data Overview"),
    ("02_data_quality",             "2  |  Data Quality Check"),
    ("03_permeability",             "3  |  Darcy Permeability Tracking"),
    ("04_xgboost_predictions",      "4  |  XGBoost Predictions"),
    ("05_residuals",                "5  |  Residual Diagnostics"),
    ("06_shap_summary",             "6  |  SHAP Feature Importance"),
    ("07_shap_bar",                 "7  |  Feature Importance Bar Chart"),
    ("08_clogging_prediction_rul",  "8  |  Clogging Prediction & RUL"),
]


# ============================================================
#  Helpers
# ============================================================
def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def image_to_b64(path, size):
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue())
    except Exception:
        return None


def plot_to_b64(path, max_size=IMG_MAX):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue())
    except Exception:
        return None


def open_folder(path):
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


# ============================================================
#  About window
# ============================================================
def show_about():
    h_b64 = image_to_b64(resource_path("hormozgan.jpg"),   (120, 120))
    d_b64 = image_to_b64(resource_path("Removal-569.png"), (100, 100))
    logo = []
    if h_b64:
        logo.append(sg.Image(data=h_b64))
    if d_b64:
        logo.append(sg.Image(data=d_b64))
    layout = [
        logo or [sg.Text("")],
        [sg.Text("Filter Analysis App", font=("Helvetica", 16, "bold"),
                 justification="center", expand_x=True)],
        [sg.HSeparator()],
        [sg.Text("Developed by:",  font=("Helvetica", 10, "bold")),
         sg.Text("Ali Peykar",     font=("Helvetica", 10))],
        [sg.Text("Organization:",  font=("Helvetica", 10, "bold")),
         sg.Text("Hormozgan Gas Corporation", font=("Helvetica", 10))],
        [sg.Text("Purpose:",       font=("Helvetica", 10, "bold")),
         sg.Text("Filter Lifecycle & Permeability Analysis", font=("Helvetica", 10))],
        [sg.Text("Version:",       font=("Helvetica", 10, "bold")),
         sg.Text("2.1", font=("Helvetica", 10))],
        [sg.HSeparator()],
        [sg.Button("Close", size=(10, 1))],
    ]
    win = sg.Window("About", layout, modal=True,
                    element_justification="center", finalize=True)
    while True:
        ev, _ = win.read()
        if ev in (sg.WIN_CLOSED, "Close"):
            break
    win.close()


# ============================================================
#  Results viewer window
# ============================================================
def show_results_window(results_dir, source_file=""):
    available = []
    for stem, title in PLOT_CATALOGUE:
        p = os.path.join(results_dir, stem + ".png")
        if os.path.exists(p):
            available.append((stem, title, p))
    for p in sorted(glob.glob(os.path.join(results_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if not any(s == stem for s, _, _ in available):
            available.append((stem, stem.replace("_", " ").title(), p))

    if not available:
        sg.popup_error("No result plots found.", title="Results Viewer")
        return

    report_path = os.path.join(results_dir, "report.txt")
    report_text = "No report.txt found."
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report_text = fh.read()

    idx    = [0]
    titles = [t for _, t, _ in available]

    def load(i):
        _, title, path = available[i]
        b64     = plot_to_b64(path)
        counter = str(i + 1) + "  /  " + str(len(available))
        return b64, title, counter

    b64_0, title_0, counter_0 = load(0)

    left_col = sg.Column([
        [sg.Text("Summary Report", font=("Helvetica", 10, "bold"))],
        [sg.Multiline(report_text, size=(44, 28), key="-REPORT-",
                      font=("Courier New", 8), disabled=True,
                      autoscroll=False, horizontal_scroll=True)],
        [sg.HSeparator()],
        [sg.Text("Plots", font=("Helvetica", 10, "bold"))],
        [sg.Listbox(titles, size=(44, 12), key="-PLOTLIST-",
                    enable_events=True,
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                    default_values=[titles[0]],
                    font=("Helvetica", 9))],
    ], vertical_alignment="top", pad=((5, 8), 5))

    right_col = sg.Column([
        [sg.Text(title_0, key="-PLOT-TITLE-",
                 font=("Helvetica", 11, "bold"),
                 justification="center", expand_x=True, size=(80, 1))],
        [sg.Image(data=b64_0, key="-IMAGE-", pad=(0, 4))],
        [sg.Button("< Prev", key="-PREV-", size=(10, 1)),
         sg.Push(),
         sg.Text(counter_0, key="-COUNTER-",
                 font=("Helvetica", 10, "bold"), justification="center"),
         sg.Push(),
         sg.Button("Next >", key="-NEXT-", size=(10, 1))],
        [sg.Text("Tip: use Left / Right arrow keys to navigate",
                 font=("Helvetica", 8), text_color="grey",
                 justification="center", expand_x=True)],
    ], vertical_alignment="top", pad=((8, 5), 5))

    win_layout = [
        [sg.Text("Analysis Results  --  " + os.path.basename(source_file),
                 font=("Helvetica", 13, "bold")),
         sg.Push(),
         sg.Button("Export This Plot", key="-EXPORT-"),
         sg.Button("Open Results Folder", key="-OPENFOLDER-"),
         sg.Button("Close", key="-CLOSE-")],
        [sg.HSeparator()],
        [left_col, sg.VSeparator(), right_col],
    ]

    win = sg.Window("Filter Analysis  --  Results Viewer",
                    win_layout, finalize=True, resizable=True,
                    return_keyboard_events=True, size=(1500, 860))
    win.bring_to_front()

    def navigate(new_idx):
        idx[0] = new_idx % len(available)
        b64, title, counter = load(idx[0])
        win["-IMAGE-"].update(data=b64)
        win["-PLOT-TITLE-"].update(title)
        win["-COUNTER-"].update(counter)
        win["-PLOTLIST-"].update(set_to_index=[idx[0]],
                                  scroll_to_index=idx[0])

    while True:
        ev, vals = win.read(timeout=100)
        if ev in (sg.WIN_CLOSED, "-CLOSE-"):
            break
        elif ev in ("-NEXT-", "Right:39", "Right"):
            navigate(idx[0] + 1)
        elif ev in ("-PREV-", "Left:37", "Left"):
            navigate(idx[0] - 1)
        elif ev == "-PLOTLIST-":
            sel = vals.get("-PLOTLIST-")
            if sel:
                try:
                    navigate(titles.index(sel[0]))
                except ValueError:
                    pass
        elif ev == "-EXPORT-":
            _, title, src = available[idx[0]]
            safe = title.replace(" ", "_").replace("|", "").replace("/", "-")
            dest = sg.popup_get_file(
                "Export plot as PNG", save_as=True,
                default_extension=".png",
                file_types=(("PNG Image", "*.png"),),
                default_path=safe + ".png")
            if dest:
                shutil.copy2(src, dest)
                sg.popup_quick_message("Saved to:\n" + dest,
                                       auto_close_duration=2)
        elif ev == "-OPENFOLDER-":
            open_folder(results_dir)

    win.close()


# ============================================================
#  Background analysis thread
# ============================================================
class _GuiLogger:
    def __init__(self, window):
        self._win = window
    def write(self, msg):
        self._win["-OUTPUT-"].print(msg, end="")
    def flush(self):
        pass


def run_analysis(file_path, area, depth, window):
    old_stdout = sys.stdout
    sys.stdout  = _GuiLogger(window)
    try:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(file_path)), "results")
        analyzer = FilterAnalysis(
            file_path=file_path,
            filter_area=area,
            filter_length=depth,
            output_dir=output_dir,
            show_plots=False,
        )
        success = analyzer.run_full_analysis(optimize_xgb=True)
        if success:
            window.write_event_value("-DONE-", output_dir)
        else:
            window.write_event_value("-ERROR-",
                                     "Analysis failed -- check console output.")
    except Exception:
        import traceback
        window.write_event_value("-ERROR-", traceback.format_exc())
    finally:
        sys.stdout = old_stdout


# ============================================================
#  Main window
# ============================================================
sg.theme("LightGrey1")

main_layout = [
    [sg.Text("Filter Analysis App", font=("Helvetica", 18, "bold")),
     sg.Push(),
     sg.Text("Hormozgan Gas Corporation", font=("Helvetica", 10, "italic"))],
    [sg.HSeparator()],
    [sg.Text("Data file (Excel):"),
     sg.Input(key="-FILE-", enable_events=True, expand_x=True),
     sg.FileBrowse(file_types=(("Excel Files", "*.xlsx;*.xls"),))],
    [sg.Text("Filter face area (m2):"),
     sg.Input("2.0",  key="-AREA-",  size=(8, 1)),
     sg.Text("   Filter depth (m):"),
     sg.Input("0.02", key="-DEPTH-", size=(8, 1)),
     sg.Push(),
     sg.Button("Run Analysis", key="-RUN-", disabled=True,
               button_color=("white", ACCENT), size=(14, 1)),
     sg.Button("About"),
     sg.Button("Exit")],
    [sg.HSeparator()],
    [sg.Text("Ready.", key="-STATUS-", font=("Helvetica", 9),
             text_color="grey")],
    [sg.Multiline(size=(90, 22), key="-OUTPUT-", autoscroll=True,
                  disabled=True, font=("Courier New", 9),
                  background_color="#F5F5F5",
                  expand_x=True, expand_y=True)],
    [sg.Button("View Results", key="-VIEW-", disabled=True,
               button_color=("white", "#2E7D32"), size=(14, 1)),
     sg.Button("Open Results Folder", key="-OPENFOLDER-", disabled=True),
     sg.Push(),
     sg.Text("Developed by Ali Peykar  |  Hormozgan Gas Corporation",
             font=("Helvetica", 8), text_color="grey")],
]

main_win = sg.Window(APP_TITLE, main_layout,
                     finalize=True, resizable=True, size=(900, 580))

_results_dir = None
_source_file = None

while True:
    event, values = main_win.read()

    if event in (sg.WIN_CLOSED, "Exit"):
        break

    elif event == "About":
        show_about()

    elif event == "-FILE-":
        ok = bool(values["-FILE-"]) and os.path.isfile(values["-FILE-"])
        main_win["-RUN-"].update(disabled=not ok)
        if ok:
            main_win["-STATUS-"].update(
                "File: " + os.path.basename(values["-FILE-"]))

    elif event == "-RUN-":
        try:
            area  = float(values["-AREA-"])
            depth = float(values["-DEPTH-"])
        except ValueError:
            sg.popup_error(
                "Filter area and depth must be numbers (e.g. 2.0 and 0.02).",
                title="Invalid Input")
            continue
        _source_file = values["-FILE-"]
        main_win["-OUTPUT-"].update("")
        main_win["-RUN-"].update(disabled=True)
        main_win["-VIEW-"].update(disabled=True)
        main_win["-OPENFOLDER-"].update(disabled=True)
        main_win["-STATUS-"].update("Running analysis ...")
        threading.Thread(
            target=run_analysis,
            args=(_source_file, area, depth, main_win),
            daemon=True).start()

    elif event == "-DONE-":
        _results_dir = values[event]
        main_win["-OUTPUT-"].print(
            "\n  Analysis complete.\n  Results saved to: " + _results_dir + "\n")
        main_win["-STATUS-"].update("Done. Results in: " + _results_dir)
        main_win["-RUN-"].update(disabled=False)
        main_win["-VIEW-"].update(disabled=False)
        main_win["-OPENFOLDER-"].update(disabled=False)
        show_results_window(_results_dir, _source_file)

    elif event == "-ERROR-":
        main_win["-OUTPUT-"].print("\n  ERROR:\n" + values[event] + "\n")
        main_win["-STATUS-"].update("Analysis failed -- see console output.")
        main_win["-RUN-"].update(disabled=False)

    elif event == "-VIEW-":
        if _results_dir and os.path.isdir(_results_dir):
            show_results_window(_results_dir, _source_file or "")

    elif event == "-OPENFOLDER-":
        if _results_dir and os.path.isdir(_results_dir):
            open_folder(_results_dir)

main_win.close()
