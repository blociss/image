#!/bin/bash

# Setup script for Image Classification Project
# This script helps configure your GitHub username for GHCR deployment

echo "Image Classification Project Setup"
echo "================================="
echo ""
echo "This script will configure your GitHub username for deployment."
echo ""

# Get GitHub username
read -p "Enter your GitHub username: "  USERNAME

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

# Interactive guide for next steps
echo "🚀 Next Steps - Interactive Guide"
echo "================================="
echo ""
echo "This guide will walk you through each step. Press Enter to continue."
echo ""

# Step 1: Login to GHCR
read -p "Step 1: Login to GitHub Container Registry (Press Enter to continue)..."
echo ""
echo "Run this command with your actual GitHub token:"
echo "  echo \"YOUR_GITHUB_TOKEN\" | docker login ghcr.io -u $USERNAME --password-stdin"
echo ""
echo "Replace YOUR_GITHUB_TOKEN with your GitHub personal access token."
echo "Token permissions needed: 'write:packages' and 'read:packages'"
echo ""
read -p "Press Enter after you've logged in to GHCR..."

# Step 2: User vs Developer
echo ""
echo "Step 2: Are you a user or developer?"
echo "==================================="
echo ""
echo "👤 USER - Just run the application (recommended for most users)"
echo "   - Pull pre-built images"
echo "   - Start the application"
echo "   - No building required"
echo ""
echo "👨‍💻 DEVELOPER - Build and push images"
echo "   - Build from source code"
echo "   - Push to GitHub Container Registry"
echo "   - For developers only"
echo ""
while true; do
    read -p "Choose 'user' or 'developer': " choice
    case $choice in
        user|USER|User|u|U)
            echo ""
            echo "✅ You chose: USER MODE"
            echo ""
            echo "Perfect! Just run this command to start the application:"
            echo "  ./deploy.sh run"
            echo ""
            echo "This will automatically:"
            echo "  - Pull pre-built images from GitHub Container Registry"
            echo "  - Start all containers"
            echo "  - Make the app available at http://localhost:8501"
            echo ""
            echo "No building required - just run and enjoy! 🚀"
            break
            ;;
        developer|DEVELOPER|Developer|dev|DEV|d|D)
            echo ""
            echo "✅ You chose: DEVELOPER MODE"
            echo ""
            echo "Run these commands to build and push:"
            echo "  export GITHUB_USERNAME=$USERNAME"
            echo "  ./deploy.sh build"
            echo ""
            echo "This will:"
            echo "  - Build Docker images from source code"
            echo "  - Tag them for GitHub Container Registry"
            echo "  - Push images to GHCR"
            echo ""
            echo "After building, test with:"
            echo "  ./deploy.sh run"
            echo ""
            echo "📝 Note: This is only needed if you're modifying the code"
            break
            ;;
        *)
            echo "Please choose 'user' or 'developer'"
            ;;
    esac
done

echo ""
echo "📚 Additional Resources"
echo "======================="
echo ""
echo "Documentation:"
echo "  - DEPLOY.md     - Complete deployment guide"
echo "  - RUN.md        - Detailed run instructions"
echo "  - TROUBLESHOOTING.md - Common issues"
echo ""
echo "Useful commands:"
echo "  ./deploy.sh status  - Check deployment status"
echo "  ./deploy.sh train   - Train models after deployment"
echo "  ./deploy.sh stop    - Stop all containers"
echo ""
echo "Access the application:"
echo "  Streamlit Dashboard: http://localhost:8501"
echo "  API Documentation:   http://localhost:8000/docs"
echo ""
echo "✅ You're all set! Happy image classifying! 🎉"
