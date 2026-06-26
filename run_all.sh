#!/bin/bash
# LiquidLens - Run all evaluations

echo "=========================================="
echo "LiquidLens Artifact Evaluation"
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
echo "[1/3] Running spectral reconstruction test..."
python LiquidLens_VIS-NIR_reconstruction_test.py
if [ $? -ne 0 ]; then
    echo "❌ Spectral reconstruction test failed!"
    exit 1
fi
echo "✅ Done."

echo ""
echo "[2/3] Running adulteration detection..."
python LiquidLens_adulteration_detection.py
if [ $? -ne 0 ]; then
    echo "❌ Adulteration detection failed!"
    exit 1
fi
echo "✅ Done."

echo ""
echo "[3/3] Running transfer learning on orange juice..."
python LiquidLens_transferlearning.py
if [ $? -ne 0 ]; then
    echo "❌ Transfer learning failed!"
    exit 1
fi
echo "✅ Done."

echo ""
echo "=========================================="
echo "🎉 All evaluations completed successfully!"
echo "Results saved to ./output/"
echo "=========================================="