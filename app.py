import os
import time
import secrets
import uuid
import json
import shutil
from pathlib import Path
import base64
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

import io
import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import List

from fastapi.middleware.cors import CORSMiddleware
from backend.get_camscene import getcamScene
from backend.seg_single import seg_single_scene
from backend.multiview_glb import multiview_voting_to_glb
from backend.holopart_parts import run_holopart, extract_original_parts

APP_DIR = Path(__file__).resolve().parent 
UPLOAD_DIR = APP_DIR / "uploads"  
TEMP_DIR = APP_DIR / "temp"  
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="SAMesh Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SegRequest(BaseModel):
    points: List[List[float]] # [[x1, y1], [x2, y2]]
    labels: List[int]         # [1, 0]
    image_name: str

class ViewData(BaseModel):
    matrix: List[float]
    mask_base64: str


class MultiViewMergeRequest(BaseModel):
    file_id: str
    views: List[ViewData]
    use_holopart: bool = False

class GetImageRequest(BaseModel):
    meshFile: str
    selectedView: List[float] # 16 floats (4x4 matrix flattened)
    useNormal: bool = False   # New field, default False

class ViewItem(BaseModel):
    matrix: List[float] 
    mask_base64: str         

class MergeRequest(BaseModel):
    file_id: str
    views: List[ViewItem]
    use_holopart: bool = False

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/upload_mesh")
async def upload_mesh(file: UploadFile = File(...), object_name: str = Form(...)):
    try:
        # Ensure the file name is unique
        file_id = str(uuid.uuid4())  # Generate a unique ID for the file
        file_location = UPLOAD_DIR / f"{file_id}.glb"  # Store as a .glb file

        # Save the uploaded file to disk
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Return the file ID to the front-end
        return {"ok": True, "file_id": file_id, "message": "File uploaded successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/get_image")
async def get_image(req: GetImageRequest):
    try:
        initial_path = UPLOAD_DIR / req.meshFile
        glb_path = None
        if initial_path.exists():
            glb_path = initial_path
        else:
            print(f"File not found at {initial_path}, searching for candidates...")
            candidates = list(UPLOAD_DIR.glob(f"{req.meshFile}.*"))
            if candidates:
                glb_path = candidates[0]
            else:
                pass

        if glb_path is None or not glb_path.exists():
             raise HTTPException(status_code=404, detail=f"Mesh file not found: {req.meshFile}")

        print(f"Loading mesh from: {glb_path}")
        img = getcamScene(str(glb_path), req.selectedView, use_normal=req.useNormal)

        suffix = "_normal" if req.useNormal else "_gray"
        image_filename = f"{glb_path.stem}_{uuid.uuid4().hex[:6]}{suffix}.png"
        image_path = TEMP_DIR / image_filename
        
        img.save(image_path)
        
        return JSONResponse({
            "imageUrl": f"/temp/{image_filename}",
            "imageName": image_filename
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error rendering image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/temp/{image_name}")
async def get_temp_image(image_name: str):
    image_path = TEMP_DIR / image_name
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    
    return FileResponse(image_path)


@app.post("/api/interactive_seg")
async def interactive_seg(req: SegRequest):
    try:
        if not req.image_name:
             raise HTTPException(status_code=400, detail="Image name is required")

        image_path = TEMP_DIR / req.image_name

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Base image not found")
        
        image_pil = Image.open(image_path).convert("RGB")
        input_point = np.array(req.points, dtype=np.int32)
        input_label = np.array(req.labels, dtype=np.int32)

        print(f"Interactive Seg Request - Points: {len(input_point)}, Labels: {input_label}")

        mask = seg_single_scene(image_pil, input_point, input_label)

        h, w = mask.shape
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Use Mask index (mask is now numpy bool
        rgba_img[mask] = [255, 255, 255, 140] # white semi-transparent

        mask_pil = Image.fromarray(rgba_img, mode="RGBA")

        img_byte_arr = io.BytesIO()
        mask_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc() 
        print(f"Segmentation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/multiview_merge")
async def multiview_merge(req: MergeRequest):
    original_glb_path = UPLOAD_DIR / f"{req.file_id}.glb"
    if not original_glb_path.exists():
        raise HTTPException(status_code=404, detail="Original GLB not found")

    print(f"Processing Merge Request: {len(req.views)} views for {req.file_id}")
    parsed_views = []
    
    for v in req.views:
        try:
            img_data = base64.b64decode(v.mask_base64)
            img = Image.open(io.BytesIO(img_data)).convert("RGBA")
            img_arr = np.array(img)
            
            alpha = img_arr[..., 3]
            binary_mask = (alpha > 0).astype(np.uint8)
            
            parsed_views.append({
                "matrix": np.array(v.matrix),
                "mask": binary_mask
            })
        except Exception as e:
            print(f"Error decoding mask: {e}")
            continue

    if not parsed_views:
        raise HTTPException(status_code=400, detail="No valid views parsed")

    merged_filename = f"merged_temp_{uuid.uuid4()}.glb"
    merged_path = TEMP_DIR / merged_filename
    
    try:
        multiview_voting_to_glb(
            glb_path=str(original_glb_path),
            views_data=parsed_views,
            output_path=str(merged_path),
            vote_threshold=0.5,
            device="cuda"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge algorithm failed: {str(e)}")


    try:
        print(f"Running HoloPart on {merged_path}...")
        
        if req.use_holopart:
            # Checkbox is Checked: Use Neural Segmentation
            meshes = run_holopart(str(merged_path)) 
        else:
            # Checkbox is Unchecked: Use Simple Extraction
            # Note: We apply this to merged_path (the result of voting)
            meshes = extract_original_parts(str(merged_path))
        
        part_urls = []
        for i, mesh in enumerate(meshes):
            part_filename = f"part_{uuid.uuid4()}_{i}.glb"
            part_path = TEMP_DIR / part_filename
            mesh.export(str(part_path))
            part_urls.append(f"/temp/{part_filename}")
            
        if merged_path.exists():
            os.remove(merged_path)
            
        return {"parts": part_urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HoloPart execution failed: {str(e)}")