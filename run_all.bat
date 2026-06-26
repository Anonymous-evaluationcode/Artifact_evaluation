@echo off
echo ==========================================
echo LiquidLens Artifact Evaluation
echo ==========================================

echo.
echo [1/3] Running spectral reconstruction test...
python LiquidLens_VIS-NIR_reconstruction_test.py
if errorlevel 1 (
    echo [ERROR] Spectral reconstruction test failed!
    exit /b 1
)
echo Done.

echo.
echo [2/3] Running adulteration detection...
python LiquidLens_adulteration_detection.py
if errorlevel 1 (
    echo [ERROR] Adulteration detection failed!
    exit /b 1
)
echo Done.

echo.
echo [3/3] Running transfer learning on orange juice...
python LiquidLens_transferlearning.py
if errorlevel 1 (
    echo [ERROR] Transfer learning failed!
    exit /b 1
)
echo Done.

echo.
echo ==========================================
echo All evaluations completed successfully!
echo ==========================================
pause