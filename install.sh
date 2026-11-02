#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.14}"

# Locate the ED_MRI repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installing EDMRI..."
echo "Installation directory: $INSTALL_DIR"

# Clone TADRED alongside ED_MRI
if [ ! -d "$INSTALL_DIR/tadred" ]; then
    git clone https://github.com/PaddySlator/tadred.git \
        "$INSTALL_DIR/tadred"
fi

# Find Python (prefer pyenv if available)
if command -v pyenv >/dev/null 2>&1; then
    if ! pyenv prefix "$PYTHON_VERSION" >/dev/null 2>&1; then
        echo "Python $PYTHON_VERSION is not installed."
        echo "Install it using: pyenv install $PYTHON_VERSION"
        exit 1
    fi
    PYTHON="$(pyenv prefix "$PYTHON_VERSION")/bin/python"
else
    PYTHON="python3"
    echo "pyenv not found; using $PYTHON"
fi

# Create virtual environment inside ED_MRI
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi

# Install packages in editable mode
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -e "$INSTALL_DIR/tadred"
"$VENV/bin/python" -m pip install -e "$SCRIPT_DIR"

echo ""
echo "EDMRI installation complete."
echo "Activate the environment using:"
echo "  source \"$VENV/bin/activate\""
