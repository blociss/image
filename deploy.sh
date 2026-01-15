#!/bin/bash

# Image Classification Deployment Script
# 
# This script handles the complete deployment workflow:
# 1. BUILD: Build images from source and push to GHCR (developers)
# 2. RUN: Pull pre-built images from GHCR and start containers (users)
#
# Usage:
#   ./deploy.sh build    - Build and push to GHCR (developers)
#   ./deploy.sh run      - Pull and run from GHCR (users)
#   ./deploy.sh train    - Train models after deployment
#   ./deploy.sh stop     - Stop all containers
#   ./deploy.sh status   - Check deployment status

set -e

# Configuration
GITHUB_USERNAME="blociss"
API_IMAGE="ghcr.io/${GITHUB_USERNAME}/image-classification-api:latest"
STREAMLIT_IMAGE="ghcr.io/${GITHUB_USERNAME}/image-classification-streamlit:latest"
COMPOSE_FILE="docker-compose.ghcr.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if logged in to GHCR
check_ghcr_login() {
    print_step "Checking GitHub Container Registry login..."
    if ! docker info | grep -q "Username.*${GITHUB_USERNAME}"; then
        print_warning "Not logged in to GitHub Container Registry"
        print_info "Please login first:"
        echo "echo \"YOUR_GITHUB_TOKEN\" | docker login ghcr.io -u ${GITHUB_USERNAME} --password-stdin"
        exit 1
    fi
    print_info "✓ Logged in to GHCR"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "Compose file $COMPOSE_FILE not found"
        exit 1
    fi
    
    print_info "✓ Prerequisites satisfied"
}

# Build and push images (for developers)
build_and_push() {
    print_header "BUILD & PUSH TO GITHUB CONTAINER REGISTRY"
    
    check_ghcr_login
    check_prerequisites
    
    print_step "Building Docker images from source..."
    docker compose build
    
    print_step "Tagging images for GHCR..."
    docker tag image-api:latest ${API_IMAGE}
    docker tag image-streamlit:latest ${STREAMLIT_IMAGE}
    
    print_step "Pushing images to GitHub Container Registry..."
    docker push ${API_IMAGE}
    docker push ${STREAMLIT_IMAGE}
    
    print_info "✅ Images successfully pushed to GHCR!"
    echo ""
    echo "Images pushed:"
    echo "  API:      ${API_IMAGE}"
    echo "  Streamlit: ${STREAMLIT_IMAGE}"
    echo ""
    print_info "Now users can deploy with: ./deploy.sh run"
}

# Pull and run images (for users)
pull_and_run() {
    print_header "DEPLOY FROM GITHUB CONTAINER REGISTRY"
    
    check_prerequisites
    
    print_step "Pulling images from GitHub Container Registry..."
    docker pull ${API_IMAGE}
    docker pull ${STREAMLIT_IMAGE}
    
    print_step "Creating necessary directories..."
    mkdir -p outputs/models outputs/figures outputs/logs data/train data/test
    
    print_step "Starting containers..."
    docker compose -f ${COMPOSE_FILE} up -d
    
    # Wait for containers to be healthy
    print_step "Waiting for services to be ready..."
    sleep 10
    
    print_info "✅ Deployment completed successfully!"
    echo ""
    echo "🚀 Access the application:"
    echo "  Streamlit Dashboard: http://localhost:8501"
    echo "  API Documentation:   http://localhost:8000/docs"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Upload your dataset to data/train and data/test folders"
    echo "  2. Train models: ./deploy.sh train"
    echo "  3. Start using the dashboard!"
    echo ""
    echo "🛠 Management commands:"
    echo "  Stop:    ./deploy.sh stop"
    echo "  Status:  ./deploy.sh status"
    echo "  Logs:    docker compose -f ${COMPOSE_FILE} logs -f"
}

# Train models
train_models() {
    print_header "TRAIN MODELS"
    
    print_step "Checking if containers are running..."
    if ! docker compose -f ${COMPOSE_FILE} ps | grep -q "Up"; then
        print_error "Containers are not running. Please deploy first with: ./deploy.sh run"
        exit 1
    fi
    
    print_step "Training models in speed mode (quick test)..."
    docker compose -f ${COMPOSE_FILE} exec api python scripts/train_pipeline.py --speed-mode
    
    print_info "✅ Model training completed!"
    echo ""
    print_info "Models are now available in the Streamlit dashboard"
    print_info "Check outputs/models/ for trained model files"
}

# Stop containers
stop_containers() {
    print_header "STOP DEPLOYMENT"
    
    print_step "Stopping all containers..."
    docker compose -f ${COMPOSE_FILE} down
    
    print_info "✅ All containers stopped"
}

# Check status
check_status() {
    print_header "DEPLOYMENT STATUS"
    
    print_step "Checking container status..."
    docker compose -f ${COMPOSE_FILE} ps
    
    echo ""
    print_step "Checking available models..."
    if [ -d "outputs/models" ]; then
        model_count=$(ls outputs/models/*.keras 2>/dev/null | wc -l)
        if [ $model_count -gt 0 ]; then
            print_info "✅ Found $model_count trained model(s)"
            ls outputs/models/
        else
            print_warning "No trained models found. Run './deploy.sh train' to train models."
        fi
    else
        print_warning "Models directory not found"
    fi
    
    echo ""
    print_step "Service health check..."
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        print_info "✅ API is healthy (http://localhost:8000)"
    else
        print_warning "API is not responding"
    fi
    
    if curl -s http://localhost:8501/ > /dev/null 2>&1; then
        print_info "✅ Streamlit is healthy (http://localhost:8501)"
    else
        print_warning "Streamlit is not responding"
    fi
}

# Show comprehensive help
show_help() {
    print_header "IMAGE CLASSIFICATION DEPLOYMENT SCRIPT"
    
    echo "This script manages the complete deployment workflow."
    echo ""
    echo "📦 DEPLOYMENT MODES:"
    echo "  1. GitHub Container Registry (GHCR) - Pre-built images"
    echo "  2. Local Development - Run from source"
    echo "  3. Docker Build - Build locally (developers)"
    echo ""
    echo "🚀 COMMANDS:"
    echo "  build    Build images from source and push to GHCR (developers)"
    echo "  run      Pull pre-built images from GHCR and deploy (users)"
    echo "  train    Train models after deployment"
    echo "  stop     Stop all running containers"
    echo "  status   Check deployment status and health"
    echo "  help     Show this help message"
    echo ""
    echo "📋 EXAMPLES:"
    echo "  # For developers - build and push to GHCR"
    echo "  ./deploy.sh build"
    echo ""
    echo "  # For users - deploy from GHCR"
    echo "  ./deploy.sh run"
    echo ""
    echo "  # After deployment - train models"
    echo "  ./deploy.sh train"
    echo ""
    echo "  # Check everything is working"
    echo "  ./deploy.sh status"
    echo ""
    echo "🔧 REQUIREMENTS:"
    echo "  - Docker and Docker Compose installed"
    echo "  - GitHub token with packages permissions (for build)"
    echo "  - Dataset in data/train and data/test folders"
    echo ""
    echo "📚 DOCUMENTATION:"
    echo "  - DEPLOY.md - Complete deployment guide"
    echo "  - RUN.md - Detailed run instructions"
    echo "  - DATASET_SETUP.md - Dataset configuration"
}

# Main script logic
case "${1:-help}" in
    build)
        build_and_push
        ;;
    run)
        pull_and_run
        ;;
    train)
        train_models
        ;;
    stop)
        stop_containers
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
