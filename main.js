import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { initializePopup } from './popup.js'; 

import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

const BACKEND = "http://127.0.0.1:8000";

const savedScenes = []; 
let currentEditingSceneId = null; 
const sceneListEl = document.getElementById("sceneList");
let tempSceneContext = null;

// Initialize the popup functionality
const { showPopup } = initializePopup({
  BACKEND,
  canvasId: "segCanvas",
  popupId: "popup",
  btnPosId: "btnPos",
  btnNegId: "btnNeg",
  btnRemoveId: "btnRemove",
  btnConfirmId: "btnConfirm",
  coordinatesDisplayId: "coordinatesDisplay",
  onImageParamsChange: async (context, useNormal) => {
      const data = await fetchRenderedImage(context.fileId, context.matrix, useNormal);
      if (data) {
          return { imageUrl: `${BACKEND}${data.imageUrl}`, imageName: data.imageName };
      }
      return null;
  },
  onConfirm: (state) => { 
    
    if (currentEditingSceneId !== null) {
        const scene = savedScenes.find(s => s.id === currentEditingSceneId);
        if (scene) {
            scene.points = state.points;
            scene.maskImage = state.maskImage;
        }
    } else {
        const newId = Date.now(); 
        const newScene = {
            id: newId,
            name: `Scene ${savedScenes.length + 1}`, 
            imageUrl: state.imageUrl,
            points: state.points,
            maskImage: state.maskImage,
            fileId: tempSceneContext ? tempSceneContext.fileId : null,
            matrix: tempSceneContext ? tempSceneContext.matrix : null
        };
        savedScenes.push(newScene);
    }

    renderSceneList();
    currentEditingSceneId = null; 
    tempSceneContext = null; 
  }
  
});


function renderSceneList() {
  if (!sceneListEl) return;
  sceneListEl.innerHTML = "";

  savedScenes.forEach((scene) => {
    const div = document.createElement("div");
    div.className = "item"; 
    
    const left = document.createElement("div");
    left.className = "name";
    left.textContent = scene.name;
    
    const delBtn = document.createElement("button");
    delBtn.textContent = "×";
    delBtn.style.background = "transparent";
    delBtn.style.border = "none";
    delBtn.style.color = "#888";
    delBtn.style.fontSize = "16px";
    delBtn.style.cursor = "pointer";
    delBtn.style.padding = "0 5px";
    
    delBtn.addEventListener("click", (e) => {
        e.stopPropagation(); 
        const idx = savedScenes.findIndex(s => s.id === scene.id);
        if (idx !== -1) {
            savedScenes.splice(idx, 1);
            renderSceneList();
        }
    });

    div.appendChild(left);
    div.appendChild(delBtn);

    div.addEventListener("click", () => {
        currentEditingSceneId = scene.id; 
        showPopup(scene.imageUrl, {
            points: scene.points,
            maskImage: scene.maskImage
        });
    });

    sceneListEl.appendChild(div);
  });
}


const btnUseViewpoint = document.getElementById("btnUseViewpoint");

btnUseViewpoint.addEventListener("click", async () => {
  let selectedFileId = null;
  if (selected && selected.userData && selected.userData.fileId) {
    selectedFileId = selected.userData.fileId;
  }
  
  let selectedViewMatrix = null;
  if (camera) {
    selectedViewMatrix = Array.from(camera.matrixWorld.elements); 
  }

  if (!selectedFileId || !selectedViewMatrix) {
    alert("Please select an object and ensure the viewpoint is valid."); 
    return;
  }
  tempSceneContext = {
      fileId: selectedFileId,
      matrix: selectedViewMatrix
  };

  const data = await fetchRenderedImage(selectedFileId, selectedViewMatrix, false);

  if (data) {
    const fullImageUrl = `${BACKEND}${data.imageUrl}`;
    currentEditingSceneId = null; 
    showPopup(fullImageUrl, null, data.imageName, tempSceneContext);
  }
});

const btnMultiViewMerge = document.getElementById("btnMultiViewMerge");
const chkUseHolopart = document.getElementById("chkUseHolopart");

btnMultiViewMerge.addEventListener("click", async () => {
    if (!selected) {
        alert("Please select an object (the object must have associated segmentation scenes).");
        return;
    }

    const targetFileId = selected.userData.fileId;
    if (!targetFileId) {
        alert("The selected object does not have a fileId, cannot merge.");
        return;
    }

    const relatedScenes = savedScenes.filter(s => s.fileId === targetFileId);
    if (relatedScenes.length === 0) {
        alert("No associated segmentation scenes found for this object. Please use 'Seg current view' to perform segmentation first.");
        return;
    }

    logBackend(`Merging results from ${relatedScenes.length} viewpoints...`);

    try {
        const viewsData = await Promise.all(relatedScenes.map(async (scene) => {
            const base64Mask = await imageToBase64(scene.maskImage);
            return {
                matrix: scene.matrix,
                mask_base64: base64Mask
            };
        }));

        const response = await fetch(`${BACKEND}/api/multiview_merge`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_id: targetFileId,
                views: viewsData,
                use_holopart: chkUseHolopart.checked,
            })
        });

        if (!response.ok) {
            const errJson = await response.json();
            throw new Error(errJson.detail || "Merge failed.");
        }

        const data = await response.json(); 

        if (data.parts && Array.isArray(data.parts)) {
            logBackend(`HoloPart completed, loading ${data.parts.length} parts...`);
          
            data.parts.forEach((partUrl, index) => {
                const fullUrl = `${BACKEND}${partUrl}`;
                const partName = `Part_${index}_${selected.name}`;
                loadGlbFromUrl(fullUrl, partName);
            });
            
            logBackend("All parts loaded.");
        } else {
            throw new Error("Backend returned incorrect data format.");
        }

    } catch (e) {
        console.error(e);
        alert("Segmentation merge failed: " + e.message);
        logBackend("Error: " + e.message);
    }
});

function imageToBase64(imgElement) {
    return new Promise((resolve) => {
        if (typeof imgElement === 'string') {
            resolve(imgElement); 
            return;
        }
        
        const canvas = document.createElement("canvas");
        canvas.width = imgElement.naturalWidth || imgElement.width;
        canvas.height = imgElement.naturalHeight || imgElement.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imgElement, 0, 0);
        const dataURL = canvas.toDataURL("image/png");
        const base64 = dataURL.split(",")[1];
        resolve(base64);
    });
}

function loadGlbFromUrl(url, name) {
    gltfLoader.load(url, (gltf) => {
        const mesh = gltf.scene;
        mesh.name = name;
        scene.add(mesh);
        
        mesh.traverse((child) => {
            if (child.isMesh) {
                selectableMeshes.push(child);
                child.userData = { ...child.userData, isGenerated: true };
            }
        });
    });
}

const canvas = document.getElementById("c");
const fileInput = document.getElementById("fileInput");

const selectedInfo = document.getElementById("selectedInfo");
const meshList = document.getElementById("meshList");

const btnTranslate = document.getElementById("btnTranslate");
const btnRotate = document.getElementById("btnRotate");
const btnScale = document.getElementById("btnScale");

const btnDuplicate = document.getElementById("btnDuplicate");
const btnDelete = document.getElementById("btnDelete");
const btnFrame = document.getElementById("btnFrame");
const btnReset = document.getElementById("btnReset");
const backendLog = document.getElementById("backendLog");

const chkGrid = document.getElementById("chkGrid");
const chkWire = document.getElementById("chkWire");
chkGrid.addEventListener("change", (e) => {
  grid.visible = e.target.checked;
});
chkWire.addEventListener("change", (e) => {
  applyWireframeToSelected(e.target.checked);
});

function logBackend(msg) {
  if (!backendLog) return;
  backendLog.textContent = String(msg ?? "");
}

async function uploadMeshToBackend(file, objectName) {
  const api = `${BACKEND}/api/upload_mesh`;
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("object_name", objectName || file.name);

  const res = await fetch(api, { method: "POST", body: form });
  const j = await res.json();
  if (!res.ok || !j.ok) {
    throw new Error(j.detail || j.error || "upload failed");
  }
  return j.file_id;
}

const gltfLoader = new GLTFLoader();
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);

const camera = new THREE.PerspectiveCamera(60, 2, 0.01, 5000);
camera.position.set(1.2, 1.0, 1.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const transform = new TransformControls(camera, renderer.domElement);
transform.setSpace("world");
scene.add(transform);

transform.addEventListener("dragging-changed", (e) => {
  controls.enabled = !e.value;
});


const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
keyLight.position.set(2, 3, 2);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
fillLight.position.set(-2, 3, -2);
scene.add(fillLight);
const topLight = new THREE.DirectionalLight(0xffffff, 0.5);
topLight.position.set(0, 4, 0);
scene.add(topLight);
const botLight = new THREE.DirectionalLight(0xffffff, 0.5);
botLight.position.set(0, -4, 0);
scene.add(botLight);
const ttLight = new THREE.DirectionalLight(0xffffff, 0.5);
ttLight.position.set(2, -3, 2);
scene.add(ttLight);
const yyLight = new THREE.DirectionalLight(0xffffff, 0.5);
yyLight.position.set(-2, 3, 2);
scene.add(yyLight);
const zzLight = new THREE.DirectionalLight(0xffffff, 0.5);
zzLight.position.set(2, 3, -2);
scene.add(zzLight);
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
hemiLight.position.set(0, 1, 0);
scene.add(hemiLight);

const grid = new THREE.GridHelper(10, 100, 0x334155, 0x1f2937);
grid.position.y = 0;
scene.add(grid);

const axes = new THREE.AxesHelper(0.5);
scene.add(axes);

const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();

const selectableMeshes = [];
let selected = null;
let selectedWireframeBackup = null;

//-------------------------------------------------
function formatMatrix(matrix) {
  let formatted = "";
  for (let i = 0; i < 4; i++) {
    // Round each value to 3 decimal places and handle small values close to 0
    const row = matrix.elements.slice(i * 4, (i + 1) * 4)
      .map(val => {
        // If the value is small (close to zero), convert it to 0
        if (Math.abs(val) < 0.001) {
          return "0.000";  // Replace small values with 0.000
        } else {
          return val.toFixed(3);  // Round to 3 decimal places
        }
      })
      .join(", ");
    formatted += row + ",\n";
  }
  return formatted;
}

// Function to update the camera matrices in the UI
function updateCameraMatrices() {
  // Extrinsic matrix (camera world matrix)
  const extrinsicMatrix = new THREE.Matrix4().copy(camera.matrixWorld);
  document.getElementById("extrinsicMatrix").textContent = formatMatrix(extrinsicMatrix);

  // Intrinsic matrix (you can retrieve it from camera parameters, assuming it's set correctly)
  // For simplicity, using a basic camera intrinsic matrix with assumed focal lengths
  const fov = camera.fov; // Field of view
  const aspect = camera.aspect; // Aspect ratio
  const near = camera.near;
  const far = camera.far;
  const focalLength = 1 / Math.tan(THREE.MathUtils.degToRad(fov) / 2); // Approximate focal length

  const intrinsicMatrix = new THREE.Matrix3();
  intrinsicMatrix.set(
    focalLength, 0, 0,
    0, focalLength, 0,
    0, 0, 1
  );

  // Adjust for aspect ratio
  intrinsicMatrix.elements[0] *= aspect;
  
  document.getElementById("intrinsicMatrix").textContent = formatMatrix(intrinsicMatrix);
  const projectionMatrix = camera.projectionMatrix.clone();
  
  // P[0] (elements[0])   1 / (tan(fov/2) * aspect)
  // P[5] (elements[5])   1 / tan(fov/2)
  document.getElementById("intrinsicMatrix").textContent = formatMatrix(projectionMatrix);
}

function getPixelIntrinsics(camera, width, height) {
  const fov = camera.fov; 
  const aspect = width / height; 
  
  //  fx = (H / 2) / tan(fov / 2)
  const fy = (height / 2) / Math.tan(THREE.MathUtils.degToRad(fov) / 2);
  const fx = fy; 
  
  const cx = width / 2;
  const cy = height / 2;

  return [
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1
  ];
}

// -------------------- tools --------------------
function resizeRendererToDisplaySize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const needResize = renderer.domElement.width !== Math.floor(w * renderer.getPixelRatio()) ||
                     renderer.domElement.height !== Math.floor(h * renderer.getPixelRatio());
  if (needResize) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  return needResize;
}

function fmtV3(v) {
  return `${v.x.toFixed(3)}, ${v.y.toFixed(3)}, ${v.z.toFixed(3)}`;
}
function fmtEulerDeg(e) {
  const dx = THREE.MathUtils.radToDeg(e.x);
  const dy = THREE.MathUtils.radToDeg(e.y);
  const dz = THREE.MathUtils.radToDeg(e.z);
  return `${dx.toFixed(1)}°, ${dy.toFixed(1)}°, ${dz.toFixed(1)}°`;
}

let geometricCenter = new THREE.Vector3();  // Stores the geometric center

// Calculate geometric center of an object
function updateGeometricCenter(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  box.getCenter(geometricCenter);
}

const controlCenters = new Map();
function setSelected(obj) {
  if (selected && selectedWireframeBackup) {
    selected.traverse((n) => {
      if (n.isMesh && selectedWireframeBackup.has(n.uuid)) {
        n.material = selectedWireframeBackup.get(n.uuid);
      }
    });
  }
  selectedWireframeBackup = null;

  selected = obj || null;

  if (!selected) {
    transform.detach();
    selectedInfo.textContent = "未选中";
    renderMeshList();
    return;
  }

  if (controlCenters.has(selected)) {
    const controlCenter = controlCenters.get(selected);
    transform.position.copy(controlCenter);
  } else {
    updateGeometricCenter(selected);
    controlCenters.set(selected, geometricCenter.clone()); 
    transform.position.copy(geometricCenter); 
  }


  transform.attach(selected);

  selectedInfo.textContent =
    `name: ${selected.name || "(unnamed)"}\n` +
    `pos: ${fmtV3(selected.position)}\n` +
    `rot: ${fmtEulerDeg(selected.rotation)}\n` +
    `scl: ${fmtV3(selected.scale)}\n` +
    `fileId: ${selected.userData?.fileId || "(none)"}`;

  if (chkWire.checked) applyWireframeToSelected(true);
  renderMeshList();
}

function computeObjectBounds(object3D) {
  const box = new THREE.Box3();
  box.setFromObject(object3D);
  return box;
}

function frameObject(object3D) {
  const box = computeObjectBounds(object3D);
  if (!box.isEmpty()) {
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = (maxDim * 0.5) / Math.tan(fov * 0.5);

    const dir = new THREE.Vector3(1, 0.8, 1).normalize();
    camera.position.copy(center.clone().add(dir.multiplyScalar(dist * 1.6)));
    camera.near = Math.max(0.001, dist / 100);
    camera.far = dist * 200;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.update();
  }
}

function applyWireframeToSelected(enabled) {
  if (!selected) return;

  if (!enabled) {
    if (selectedWireframeBackup) {
      selected.traverse((n) => {
        if (n.isMesh && selectedWireframeBackup.has(n.uuid)) {
          n.material = selectedWireframeBackup.get(n.uuid);
        }
      });
    }
    selectedWireframeBackup = null;
    return;
  }

  selectedWireframeBackup = new Map();
  selected.traverse((n) => {
    if (!n.isMesh) return;
    selectedWireframeBackup.set(n.uuid, n.material);

    const toWire = (mat) => {
      const m = mat.clone();
      m.wireframe = true;
      m.transparent = false;
      m.depthWrite = true;
      return m;
    };

    if (Array.isArray(n.material)) n.material = n.material.map(toWire);
    else n.material = toWire(n.material);
  });
}

function addSelectableRoot(root, nameHint = "mesh") {
  root.name = root.name || nameHint;

  root.traverse((n) => {
    if (n.isMesh) {
      n.castShadow = false;
      n.receiveShadow = true;
      selectableMeshes.push(n);
    }
  });

  scene.add(root);
  renderMeshList();
}

function pick(event) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  mouseNDC.set(x, y);

  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(selectableMeshes, true);
  if (!hits.length) {
    setSelected(null);
    return;
  }

  let hitObj = hits[0].object;
  while (hitObj.parent && hitObj.parent !== scene) hitObj = hitObj.parent;
  setSelected(hitObj);
}

function duplicateSelected() {
  if (!selected) return;

  const copy = selected.clone(true);
  copy.name = (selected.name || "mesh") + "_copy";
  copy.position.add(new THREE.Vector3(0.05, 0, 0.05));

  copy.traverse((n) => {
    if (n.isMesh) selectableMeshes.push(n);
  });

  scene.add(copy);
  setSelected(copy);
}

function deleteSelected() {
  if (!selected) return;

  const uuids = new Set();
  selected.traverse((n) => {
    if (n.isMesh) uuids.add(n.uuid);
  });
  for (let i = selectableMeshes.length - 1; i >= 0; i--) {
    if (uuids.has(selectableMeshes[i].uuid)) selectableMeshes.splice(i, 1);
  }

  transform.detach();
  scene.remove(selected);
  setSelected(null);
}

function resetSelectedTransform() {
  if (!selected) return;
  selected.position.set(0, 0, 0);
  selected.rotation.set(0, 0, 0);
  selected.scale.set(1, 1, 1);
  setSelected(selected);
}

function setTransformMode(mode) {
  transform.setMode(mode);
}

async function fetchRenderedImage(fileId, viewMatrix, useNormal) {
    try {
        const response = await fetch(`${BACKEND}/api/get_image`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                meshFile: fileId,
                selectedView: viewMatrix,
                useNormal: useNormal 
            }),
        });
        
        if (!response.ok) {
             const err = await response.json();
             throw new Error(err.detail || "Render failed");
        }
        return await response.json(); 
    } catch (e) {
        console.error("Failed to fetch image", e);
        alert("render failed: " + e.message);
        return null;
    }
}


function renderMeshList() {
  meshList.innerHTML = "";
  const roots = scene.children.filter((n) => n !== grid && n !== axes && n !== transform);
  const list = roots.filter((n) => !(n.isLight || n.isCamera));

  for (const obj of list) {
    const div = document.createElement("div");
    div.className = "item" + (selected === obj ? " active" : "");
    div.style.display = "flex";
    div.style.justifyContent = "space-between";
    div.style.alignItems = "center";
    div.style.paddingRight = "5px"; 
    const nameDiv = document.createElement("div");
    nameDiv.className = "name";
    nameDiv.textContent = obj.name || "(unnamed)";
    nameDiv.style.flex = "1"; 
    nameDiv.style.overflow = "hidden";
    nameDiv.style.textOverflow = "ellipsis";
    nameDiv.style.whiteSpace = "nowrap";
    const btnGroup = document.createElement("div");
    btnGroup.style.display = "flex";
    btnGroup.style.alignItems = "center";
    btnGroup.style.gap = "10px"; 
    const eyeBtn = document.createElement("span");
    eyeBtn.textContent = "👁‍🗨";
    eyeBtn.style.cursor = "pointer";
    eyeBtn.style.fontSize = "16px";
    eyeBtn.style.opacity = obj.visible ? "1.0" : "0.3";
    eyeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      obj.visible = !obj.visible;
      eyeBtn.style.opacity = obj.visible ? "1.0" : "0.3";
    });
    const delBtn = document.createElement("span");
    delBtn.textContent = "×"; 
    delBtn.style.cursor = "pointer";
    delBtn.style.fontSize = "20px";
    delBtn.style.fontWeight = "bold";
    delBtn.style.lineHeight = "1";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation(); 
      if (selected === obj) {
          transform.detach();
          selected = null;
          selectedInfo.textContent = "未选中";
      }
      const uuidsToRemove = new Set();
      obj.traverse((n) => { if (n.isMesh) uuidsToRemove.add(n.uuid); });
      
      for (let i = selectableMeshes.length - 1; i >= 0; i--) {
        if (uuidsToRemove.has(selectableMeshes[i].uuid)) {
            selectableMeshes.splice(i, 1);
        }
      }
      scene.remove(obj); 
      renderMeshList();  
    });
    btnGroup.appendChild(eyeBtn);
    btnGroup.appendChild(delBtn);
    div.appendChild(nameDiv);
    div.appendChild(btnGroup);
    div.addEventListener("click", () => setSelected(obj));
    meshList.appendChild(div);
  }
}

const objLoader = new OBJLoader();
const fbxLoader = new FBXLoader();
const stlLoader = new STLLoader();

async function loadFile(file) {
  const name = file.name || "mesh";
  const ext = name.split(".").pop()?.toLowerCase() || "";
  const url = URL.createObjectURL(file);

  logBackend(`upload to backend：${name} ...`);
  let fileId = null;
  try {
    fileId = await uploadMeshToBackend(file, name);
    logBackend(`upload finished：fileId=${fileId}`);
  } catch (e) {
    console.error(e);
    logBackend(`upload failed：${String(e?.message || e)}`);
    URL.revokeObjectURL(url);
    alert(`upload failed: ${name}\n${String(e?.message || e)}`);
    return;
  }

  const finalize = (root) => {
    root.name = root.name || name;
    root.userData = root.userData || {};
    root.userData.fileId = fileId; 
    addSelectableRoot(root, name);
    frameObject(root);
    URL.revokeObjectURL(url);
  };

  const fail = (err) => {
    console.error("load failed:", name, err);
    URL.revokeObjectURL(url);
    alert(`load failed: ${name}\n${String(err?.message || err)}`);
  };

  if (ext === "glb" || ext === "gltf") {
    gltfLoader.load(url, (gltf) => finalize(gltf.scene || gltf.scenes?.[0]), undefined, fail);
    return;
  }

  if (ext === "obj") {
    objLoader.load(url, (obj) => finalize(obj), undefined, fail);
    return;
  }

  if (ext === "fbx") {
    fbxLoader.load(url, (obj) => finalize(obj), undefined, fail);
    return;
  }

  if (ext === "stl") {
    stlLoader.load(
      url,
      (geo) => {
        geo.computeVertexNormals();
        const mat = new THREE.MeshStandardMaterial({ metalness: 0.0, roughness: 0.9 });
        const mesh = new THREE.Mesh(geo, mat);
        finalize(mesh);
      },
      undefined,
      fail
    );
    return;
  }

  URL.revokeObjectURL(url);
  alert(`format unsupported: ${name}\n support: obj / glb / gltf / fbx / stl`);
}

fileInput.addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;

  for (const f of files) {
    await loadFile(f);
  }
  fileInput.value = "";
});

// -------------------- ui --------------------
// Modify the pointerdown event to ignore clicks outside the selected object
canvas.addEventListener("pointerdown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  mouseNDC.set(x, y);

  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(selectableMeshes, true);
  if (!hits.length) {
    return; // Ignore click if not hitting the selected object
  }

  let hitObj = hits[0].object;
  while (hitObj.parent && hitObj.parent !== scene) hitObj = hitObj.parent;

  if (hitObj !== selected) {  // Only change object when clicked on another object
    setSelected(hitObj);
  }
});

btnTranslate.addEventListener("click", () => setTransformMode("translate"));
btnRotate.addEventListener("click", () => setTransformMode("rotate"));
btnScale.addEventListener("click", () => setTransformMode("scale"));

btnDuplicate.addEventListener("click", duplicateSelected);
btnDelete.addEventListener("click", deleteSelected);
btnFrame.addEventListener("click", () => { if (selected) frameObject(selected); });
btnReset.addEventListener("click", resetSelectedTransform);


window.addEventListener("keydown", (e) => {
  const tag = (document.activeElement?.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;

  if (e.key === "w" || e.key === "W") setTransformMode("translate");
  if (e.key === "e" || e.key === "E") setTransformMode("rotate");
  if (e.key === "r" || e.key === "R") setTransformMode("scale");

  if ((e.ctrlKey || e.metaKey) && (e.key === "d" || e.key === "D")) {
    e.preventDefault();
    duplicateSelected();
  }

  if (e.key === "Delete" || e.key === "Backspace") {
    deleteSelected();
  }

  if (e.key === "Escape") setSelected(null);
});

transform.addEventListener("objectChange", () => {
  if (!selected) return;
  selectedInfo.textContent =
    `name: ${selected.name || "(unnamed)"}\n` +
    `pos: ${fmtV3(selected.position)}\n` +
    `rot: ${fmtEulerDeg(selected.rotation)}\n` +
    `scl: ${fmtV3(selected.scale)}\n` +
    `fileId: ${selected.userData?.fileId || "(none)"}`;
});

setTransformMode("translate");
renderMeshList();
logBackend("Ready!");

function animate() {
  requestAnimationFrame(animate);
  resizeRendererToDisplaySize();
  controls.update();
  renderer.render(scene, camera);
  updateCameraMatrices();
}
animate();
