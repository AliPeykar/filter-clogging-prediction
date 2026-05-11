# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
import os, site

datas, binaries, hiddenimports = [], [], []

# Bundle logo files
datas += [('hormozgan.jpg', '.'), ('Removal-569.png', '.')]

# --- xgboost: manual collection to avoid xgboost.testing (needs pytest/hypothesis) ---
xgb_site = next(p for p in site.getsitepackages()
                if os.path.isdir(os.path.join(p, 'xgboost')))
xgb_dir = os.path.join(xgb_site, 'xgboost')
binaries += [(os.path.join(xgb_dir, 'lib', 'xgboost.dll'), 'xgboost/lib')]
datas   += [(os.path.join(xgb_dir, 'VERSION'), 'xgboost')]
hiddenimports += [
    'xgboost', 'xgboost.core', 'xgboost.sklearn', 'xgboost.training',
    'xgboost.callback', 'xgboost.data', 'xgboost.tracker',
    'xgboost.compat', 'xgboost.config', 'xgboost.plotting',
    'xgboost.dask', 'xgboost.spark',
]

# --- FreeSimpleGUI: fully collect (small package, no test bloat) ---
d, b, h = collect_all('FreeSimpleGUI')
datas += d; binaries += b; hiddenimports += h

# --- Pillow: needed for logo image loading in About dialog ---
d, b, h = collect_all('PIL')
datas += d; binaries += b; hiddenimports += h

# --- matplotlib: needs mpl-data (fonts, styles, colormaps) ---
datas += collect_data_files('matplotlib')
hiddenimports += [
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
]

# --- seaborn: data files only ---
datas += collect_data_files('seaborn')
hiddenimports += ['seaborn', 'seaborn.cm']

# --- shap: fully collect (small) ---
d, b, h = collect_all('shap')
datas += d; binaries += b; hiddenimports += h

# --- sklearn: data files + key submodules (skip tests) ---
datas += collect_data_files('sklearn')
hiddenimports += [
    'sklearn', 'sklearn.utils._cython_blas', 'sklearn.neighbors._partition_nodes',
    'sklearn.tree._utils', 'sklearn.utils._weight_vector',
    'sklearn.model_selection', 'sklearn.model_selection._search',
    'sklearn.preprocessing', 'sklearn.metrics',
]

# --- statsmodels: data files + key submodules ---
datas += collect_data_files('statsmodels')
hiddenimports += [
    'statsmodels', 'statsmodels.graphics.tsaplots',
    'statsmodels.tsa', 'statsmodels.tsa.stattools',
]

# --- numpy / scipy / pandas: PyInstaller hooks cover these; just ensure key imports ---
hiddenimports += [
    'numpy', 'scipy', 'scipy.signal', 'scipy.interpolate', 'scipy.optimize',
    'pandas', 'pandas.plotting',
    'multiprocessing', 'multiprocessing.freeze_support',
]

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['xgboost.testing', 'pytest', 'hypothesis', 'IPython', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='FilterAnalysisApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FilterAnalysisApp',
)
