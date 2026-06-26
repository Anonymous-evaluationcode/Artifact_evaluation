#!/bin/bash
# LiquidLens - Train and Test Spectral Reconstruction

echo "=========================================="
echo "LiquidLens: Train & Test Spectral Reconstruction"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Run: source venv/bin/activate"
    echo "   Then: pip install -r requirements.txt"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Starting Training + Testing..."
echo "=========================================="
echo "⏱️  Estimated time: ~65 minutes"
echo ""

# Run the training + test script
python LiquidLens_VIS-NIR_reconstruction_train_test.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training + Testing completed successfully!"
    echo "📁 Trained models saved to ./model/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Training + Testing failed!"
    echo "Please check the error messages above."
    echo "=========================================="
    exit 1
fi