#!/bin/bash

VENV_DIR="simGF"

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    read -p "Virtual environment '$VENV_DIR' already exists. Recreate it? (y/n): " choice
    if [ "$choice" == "y" ] || [ "$choice" == "Y" ]; then
        echo "Deleting old environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing environment."
    fi
fi

# If environment does not exist, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# Activate environment (Linux/macOS)
echo "Activating environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping installation."
fi

echo "✅ Setup complete!"
echo "To activate later:"
echo "   source $VENV_DIR/bin/activate   (Linux/macOS)"
echo "   .\\$VENV_DIR\\Scripts\\Activate  (Windows PowerShell)"
