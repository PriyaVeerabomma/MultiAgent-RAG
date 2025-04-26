#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log function
log() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1"
  exit 1
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  error "Please run as root or with sudo"
fi

log "Starting Docker installation and setup..."

# Install Docker
log "Installing Docker..."
apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common gnupg

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker service
systemctl start docker
systemctl enable docker

log "Docker installed successfully"

# Run docker compose commands
log "Running docker compose build..."
cd "${APP_DIR:-.}" # Use current directory by default
docker compose build

log "Running docker compose up..."
docker compose up -d

log "Docker setup complete! Containers are now running."