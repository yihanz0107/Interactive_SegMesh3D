#!/usr/bin/env bash
set -e

#######################################
# Config
#######################################
ENV_NAME="interactive_segmesh3d"
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
# Auto-detect CUDA wheel tag from NVIDIA driver
#######################################
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. NVIDIA driver not installed."
  exit 1
fi

# Example parsed value: 13.1
CUDA_MAX="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"

if [ -z "${CUDA_MAX}" ]; then
  echo "ERROR: Failed to parse CUDA version from nvidia-smi."
  exit 1
fi

echo ">>> Detected driver-supported CUDA version: ${CUDA_MAX}"

# Split major.minor
CUDA_MAJOR="${CUDA_MAX%%.*}"
CUDA_MINOR="${CUDA_MAX#*.}"

CUDA_VERSION=""

if [ "${CUDA_MAJOR}" -ge 13 ]; then
  CUDA_VERSION="130"
elif [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -ge 8 ]; then
  CUDA_VERSION="128"
elif [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -ge 6 ]; then
  CUDA_VERSION="126"
else
  echo "ERROR: Driver CUDA version ${CUDA_MAX} is below 12.6."
  echo "This setup supports cu126 / cu128 / cu130 only, please update your Nvidia Driver."
  exit 1
fi

echo ">>> Selecting PyTorch wheels: cu${CUDA_VERSION}"

#######################################
# Python / PyTorch/ PyTorch3d
#######################################
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
PYTORCH3D_VERSION="0.7.9"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
PYTORCH3D_INDEX_URL="https://miropsota.github.io/torch_packages_builder"

echo ">>> Installing PyTorch ${TORCH_VERSION} (cu${CUDA_VERSION})"
pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}+cu${CUDA_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION}"

echo ">>> Installing PyTorch3D ${PYTORCH3D_VERSION} (+pt${TORCH_VERSION}cu${CUDA_VERSION})"
pip install --extra-index-url "${PYTORCH3D_INDEX_URL}" \
  "pytorch3d==${PYTORCH3D_VERSION}+pt${TORCH_VERSION}cu${CUDA_VERSION}"

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
# Flatten SAM3 directory structure
#######################################
if [ -d "./sam3/sam3" ]; then
  echo ">>> Flattening SAM3 directory structure (sam3/sam3 -> sam3/)"

  # Copy all contents (including hidden files) from sam3/sam3 to sam3
  cp -r ./sam3/sam3/. ./sam3/

  echo ">>> SAM3 directory flattened."
else
  echo ">>> No nested sam3/sam3 directory found, skipping flatten."
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
SAM3_CKPT_PATH="./sam3/models/sam3.pt"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

SAM3_BPE_PATH="./sam3/assets/bpe_simple_vocab_16e6.txt.gz"

if [ ! -f "${SAM3_BPE_PATH}" ]; then
  echo ""
  echo "⚠️  SAM3 BPE vocab not found:"
  echo -e "    ${YELLOW}${BOLD}${SAM3_BPE_PATH}${RESET}"
  echo ""
  echo "👉 This file is required for SAM3 text/token processing."
  echo "   Please make sure the SAM3 repository is fully initialized"
  echo "   and that this file exists at the path above."
  echo ""
else
  echo -e "${GREEN}${BOLD}✔ SAM3 BPE vocab found.${RESET}"
fi

if [ ! -f "${SAM3_CKPT_PATH}" ]; then
  echo ""
  echo "⚠️  SAM3 checkpoint not found"
  echo "👉 You are one step away from running SAM3."
  echo "   Please download the checkpoint manually and place it at:"
  echo -e "   ${YELLOW}${BOLD}${SAM3_CKPT_PATH}${RESET}"
  echo ""
  echo ""
else
    echo -e "${GREEN}${BOLD}✔ SAM3 checkpoint found.${RESET}"
    echo -e "${GREEN}${BOLD}✔ Setup complete.${RESET}"
    echo -e "➜ Please activate the environment manually${RESET}"
    echo -e "➜ Then start the dev server with${RESET}"
        echo -e "    ${BOLD}conda activate segmesh3d${RESET}"
    echo -e "    ${BOLD}npm run dev${RESET}"

fi
find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} + >/dev/null 2>&1
