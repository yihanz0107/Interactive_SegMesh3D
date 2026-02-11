// popup.js

export function initializePopup({ 
  BACKEND, 
  canvasId, 
  popupId, 
  btnPosId, 
  btnNegId, 
  btnRemoveId, 
  btnConfirmId, 
  coordinatesDisplayId,
  onConfirm,
  onImageParamsChange
}) {
  const popup = document.getElementById(popupId);
  const segCanvas = document.getElementById(canvasId);
  const ctx = segCanvas.getContext("2d");
  const baseImgElement = document.getElementById("segBaseImage"); 
  const chkUseNormal = document.getElementById("chkUseNormal");

  let currentImageName = "";
  let isImageLoaded = false;

  const btnPos = document.getElementById(btnPosId);
  const btnNeg = document.getElementById(btnNegId);
  const btnRemove = document.getElementById(btnRemoveId);
  const btnConfirm = document.getElementById(btnConfirmId);
  const coordinatesDisplay = document.getElementById(coordinatesDisplayId);
  const closePopupButton = document.getElementById("closePopup");

  let currentLabel = 'pos';
  let clickLocations = [];
  let maskImage = null; 
  let currentImageUrl = ""; 

  let currentRenderContext = null;

  const RENDER_SIZE = 1024;
  const CANVAS_SIZE = 512;
  const SCALE_FACTOR = RENDER_SIZE / CANVAS_SIZE;

  // --------------------------------------------------
  function redraw() {
    ctx.clearRect(0, 0, segCanvas.width, segCanvas.height);

    if (!isImageLoaded) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "20px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Loading Image...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      return; 
    }

    if (maskImage) {
      ctx.drawImage(maskImage, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }

    const fontSize = 18;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    clickLocations.forEach(({ x, y, label }) => {
      ctx.fillStyle = label === 'pos' ? '#00ff00' : '#ff0000';
      ctx.fillText(label === 'pos' ? "+" : "-", x, y);
    });

    updateCoordinatesDisplay();
  }

// --------------------------------------------------
  function showPopup(imageUrl, restoreState = null, explicitImageName = null, context = null) {
    currentImageUrl = imageUrl;
    isImageLoaded = false;
    currentRenderContext = context; 
    if (explicitImageName) {
        currentImageName = explicitImageName;
    } else {
        const urlParts = imageUrl.split('/');
        currentImageName = urlParts[urlParts.length - 1]; 
    }

    currentLabel = 'pos';
    updateButtonStyles();
    segCanvas.width = CANVAS_SIZE;
    segCanvas.height = CANVAS_SIZE;
    
    if (chkUseNormal) {
        chkUseNormal.checked = false;
    }

    redraw(); 
    popup.style.display = "flex";

    if (baseImgElement) {
        baseImgElement.onload = null; 
        baseImgElement.onload = () => {
            console.log("Base image loaded:", currentImageName);
            isImageLoaded = true;
            redraw(); 
        };
        baseImgElement.src = imageUrl;
    }

    if (restoreState) {
      clickLocations = [...restoreState.points]; 
      maskImage = restoreState.maskImage;        
    } else {
      clickLocations = [];
      maskImage = null;
    }
  }

  // --------------------------------------------------
  if (chkUseNormal) {
      chkUseNormal.addEventListener('change', async () => {
          if (!currentRenderContext || !onImageParamsChange) return;

          const useNormal = chkUseNormal.checked;
          console.log("Switching mode. Use Normal:", useNormal);

          isImageLoaded = false;
          redraw();

          const result = await onImageParamsChange(currentRenderContext, useNormal);

          if (result && result.imageUrl) {
              currentImageUrl = result.imageUrl;
              currentImageName = result.imageName;
              baseImgElement.src = currentImageUrl;
              maskImage = null; 
          }
      });
  }

  async function fetchSegmentation() {
    if (!currentImageName || clickLocations.length === 0) {
      maskImage = null;
      redraw();
      return;
    }

    if (!isImageLoaded) return;

    const points = clickLocations.map(p => [p.x * SCALE_FACTOR, p.y * SCALE_FACTOR]);
    const labels = clickLocations.map(p => (p.label === 'pos' ? 1 : 0));

    try {
      const response = await fetch(`${BACKEND}/api/interactive_seg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points, labels, image_name: currentImageName }),
      });
      if (!response.ok) throw new Error("Segmentation failed");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        maskImage = img; 
        redraw(); 

      };
      img.src = url;
    } catch (err) {
      console.error("Error fetching mask:", err);
    }
  }

  segCanvas.addEventListener("click", (e) => {
    if (!isImageLoaded) return;
    const rect = segCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (segCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (segCanvas.height / rect.height);
    if (!currentLabel) return;
    clickLocations.push({ x, y, label: currentLabel });
    redraw();
    fetchSegmentation();
  });
  
  btnPos.addEventListener("click", () => { currentLabel = "pos"; updateButtonStyles(); });
  btnNeg.addEventListener("click", () => { currentLabel = "neg"; updateButtonStyles(); });
  
  btnRemove.addEventListener("click", () => {
    if (!isImageLoaded) return;
    if (clickLocations.length > 0) {
      clickLocations.pop();
      redraw();
      fetchSegmentation();
    }
  });
  
  function updateButtonStyles() {
    btnPos.style.backgroundColor = currentLabel === 'pos' ? '#4CAF50' : '';
    btnNeg.style.backgroundColor = currentLabel === 'neg' ? '#f44336' : '';
  }

  function updateCoordinatesDisplay() {
      if(coordinatesDisplay) coordinatesDisplay.textContent = `Points: ${clickLocations.length}`;
  }

  closePopupButton.addEventListener("click", () => {
      popup.style.display = "none";
  });
  
  btnConfirm.addEventListener("click", () => {
     popup.style.display = "none";

     if (onConfirm) {
         onConfirm({
             imageUrl: currentImageUrl,
             points: [...clickLocations], 
             maskImage: maskImage 
         });
     }
  });

  return { showPopup };
}