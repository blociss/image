#!/bin/bash

# Setup script for Image Classification Project
# This script helps configure your GitHub username for GHCR deployment

echo "Image Classification Project Setup"
echo "================================="
echo ""
echo "This script will configure your GitHub username for deployment."
echo ""

# Get GitHub username
read -p "Enter your GitHub username: " USERNAME

if [ -z "$USERNAME" ]; then
    echo "Error: Username cannot be empty"
    exit 1
fi

echo ""
echo "Setting GITHUB_USERNAME to: $USERNAME"

# Export for current session
export GITHUB_USERNAME=$USERNAME

# Add to shell profile for persistence
SHELL_RC=""
if [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.profile"
fi

# Add to shell profile if not already there
if ! grep -q "GITHUB_USERNAME" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Image Classification Project" >> "$SHELL_RC"
    echo "export GITHUB_USERNAME=$USERNAME" >> "$SHELL_RC"
    echo ""
    echo "✅ Added GITHUB_USERNAME to $SHELL_RC"
else
    echo "ℹ️  GITHUB_USERNAME already exists in $SHELL_RC"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Login to GitHub Container Registry:"
echo "   echo \"YOUR_GITHUB_TOKEN\" | docker login ghcr.io -u $USERNAME --password-stdin"
echo ""
echo "2. Deploy the application:"
echo "   ./deploy.sh run"
echo ""
echo "3. Or build and push (if you're a developer):"
echo "   ./deploy.sh build"
echo ""
echo "Note: Restart your terminal or run 'source $SHELL_RC' to load the environment variable."
