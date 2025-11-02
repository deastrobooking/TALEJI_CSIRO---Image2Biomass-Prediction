#!/bin/bash
# test_install.sh - Test the pip installation commands from README

echo "Testing core package installation..."
echo "Command: pip install -q numpy>=1.26 pandas>=2.1 scikit-learn>=1.3 scikit-image>=0.22 opencv-python-headless>=4.8.1 pyarrow>=12"

# Test the core installation command (dry run)
pip install --dry-run numpy>=1.26 pandas>=2.1 scikit-learn>=1.3 scikit-image>=0.22 opencv-python-headless>=4.8.1 pyarrow>=12 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Core packages: Installation command syntax is valid"
else
    echo "❌ Core packages: Installation command has issues"
fi

echo ""
echo "Testing optional packages..."
echo "Command: pip install -q optuna>=3.0 scikit-optimize>=0.9 fastparquet>=2023.10.1"

# Test optional packages (dry run)
pip install --dry-run optuna>=3.0 scikit-optimize>=0.9 fastparquet>=2023.10.1 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Optional packages: Installation command syntax is valid"
else
    echo "⚠️ Optional packages: May not be available in all environments"
fi

echo ""
echo "Testing requirements file..."
if [ -f "requirements-min.txt" ]; then
    echo "✅ requirements-min.txt exists"
    pip install --dry-run -r requirements-min.txt 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ requirements-min.txt: Valid syntax"
    else
        echo "⚠️ requirements-min.txt: Some packages may not be available"
    fi
else
    echo "❌ requirements-min.txt not found"
fi

echo ""
echo "Test complete!"