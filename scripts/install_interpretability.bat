@echo off
REM Installation script for model interpretability packages
REM Run this to install SHAP, LIME, and other visualization dependencies

echo ============================================
echo Installing Model Interpretability Packages
echo ============================================
echo.

echo [1/3] Installing SHAP...
pip install shap>=0.41.0
echo.

echo [2/3] Installing LIME...
pip install lime>=0.2.0.1
echo.

echo [3/3] Installing Plotly (for interactive visualizations)...
pip install plotly>=5.0.0
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo You can now use the enhanced visualization features:
echo   - SHAP explanations
echo   - LIME explanations
echo   - Partial Dependence Plots
echo   - Permutation Importance
echo.
echo Run your pipeline: python main.py your_data.csv
echo Check plots/interpretability/ for visualizations
echo.
echo For help, see INTERPRETABILITY_GUIDE.md
echo ============================================

pause
