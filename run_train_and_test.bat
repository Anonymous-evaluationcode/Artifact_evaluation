@echo off
echo ==========================================
echo LiquidLens: Train & Test Spectral Reconstruction
echo ==========================================

echo.
echo ==========================================
echo Starting Training + Testing...
echo ==========================================
echo Estimated time: ~65 minutes on GPU
echo (CPU training will take significantly longer)
echo.

python LiquidLens_VIS-NIR_reconstruction_train_test.py

if errorlevel 1 (
    echo.
    echo ==========================================
    echo [ERROR] Training + Testing failed!
    echo Please check the error messages above.
    echo ==========================================
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [SUCCESS] Training + Testing completed!
echo Trained models saved to ./model/
echo ==========================================
pause