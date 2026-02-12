#!/usr/bin/env bash
set -e

#######################################
# Config
#######################################
ENV_NAME="segmesh3d"
PYTHON_VERSION="3.12"
NODE_VERSION="lts/*"
SAM3_REPO="https://github.com/facebookresearch/sam3.git"
SAM3_DIR="./sam3"

#######################################
# Conda environment
#######################################
echo ">>> Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo ">>> Activating conda environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

#######################################
# Python / PyTorch
#######################################
echo ">>> Upgrading pip"
pip install --upgrade pip

echo ">>> Installing PyTorch (CUDA 13.0)"
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.9.1+cu130 \
  torchvision==0.24.1+cu130

echo ">>> Installing PyTorch3D"
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  pytorch3d==0.7.9+pt2.9.1cu130

echo ">>> Pin setuptools"
pip install "setuptools<80"

echo ">>> Installing Python project dependencies"
pip install -e .

#######################################
# Clone SAM3
#######################################
if [ ! -d "${SAM3_DIR}" ]; then
  echo ">>> Cloning SAM3 repository into ${SAM3_DIR}"
  git clone ${SAM3_REPO} ${SAM3_DIR}
else
  echo ">>> SAM3 repository already exists, skipping clone"
fi

#######################################
# Node.js / npm (via nvm)
#######################################
echo ">>> Installing nvm (Node Version Manager)"
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

source "$NVM_DIR/nvm.sh"

echo ">>> Installing Node.js (${NODE_VERSION})"
nvm install ${NODE_VERSION}
nvm use ${NODE_VERSION}
nvm alias default ${NODE_VERSION}

echo ">>> Node version:"
node -v
echo ">>> npm version:"
npm -v

#######################################
# Frontend dependencies
#######################################
echo ">>> Installing frontend npm dependencies"
npm install

#######################################
# Sanity check
#######################################
echo ">>> Verifying Python + CUDA"
python - << 'EOF'
import torch, pytorch3d
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch3D:", pytorch3d.__version__)
EOF

#######################################
# Check SAM3 checkpoint
#######################################
SAM3_CKPT_PATH="./sam3/sam3/models/sam3.pt"

if [ ! -f "${SAM3_CKPT_PATH}" ]; then
  echo ""
  echo "⚠️  SAM3 checkpoint not found:"
  echo "    ${SAM3_CKPT_PATH}"
  echo ""
  echo "👉 You are one step away from running SAM3."
  echo "   Please download the checkpoint manually and place it at:"
  echo "   ${SAM3_CKPT_PATH}"
  echo ""
  echo ""
else
    echo ">>> SAM3 checkpoint found."
    echo ">>> Setup complete."
    echo ">>> Activate env later with: conda activate ${ENV_NAME}"
    echo ">>> Start dev server with: npm run dev"
fi