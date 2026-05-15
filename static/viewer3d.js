// 3D Tumor Viewer using Three.js with real reconstruction payloads only.
function resolveOrbitControls() {
  if (typeof window !== 'undefined') {
    if (typeof window.OrbitControls === 'function') return window.OrbitControls;
    if (window.THREE && typeof window.THREE.OrbitControls === 'function') return window.THREE.OrbitControls;
  }
  if (typeof OrbitControls === 'function') return OrbitControls;
  return null;
}

class TumorViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.tumorMesh = null;
    this.uncertaintyShell = null;
    this.tumorWire = null;
    this.pointCloud = null;
    this.studyFrame = null;
    this.centroidMarker = null;
    this.axisGroup = null;
    this.measureGroup = null;
    this.overlayGroup = null;
    this.volumeSliceGroup = null;
    this.lights = [];
    this.autoRotate = false;
    this.stats = { volume: 0, surface: 0, density: 0 };
    this.init();
  }

  init() {
    if (!this.container) return;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080d18);
    this.scene.fog = new THREE.Fog(0x080d18, 240, 760);

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 4000);
    this.camera.position.set(0, -180, 120);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    const OrbitControlsClass = resolveOrbitControls();
    if (OrbitControlsClass) {
      this.controls = new OrbitControlsClass(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.06;
      this.controls.autoRotate = false;
      this.controls.autoRotateSpeed = 1.6;
      this.controls.minDistance = 10;
      this.controls.maxDistance = 3000;
    } else {
      console.warn('OrbitControls is unavailable; viewer will run in static-camera mode.');
      this.controls = {
        target: new THREE.Vector3(0, 0, 0),
        update() {},
        autoRotate: false,
      };
    }

    this.setupLights();
    this.animate();
    window.addEventListener('resize', () => this.onWindowResize());
  }

  setupLights() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambientLight);
    this.lights.push(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
    keyLight.position.set(140, -100, 180);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    this.scene.add(keyLight);
    this.lights.push(keyLight);

    const fillLight = new THREE.DirectionalLight(0xb9d9ff, 0.55);
    fillLight.position.set(-180, 120, 80);
    this.scene.add(fillLight);
    this.lights.push(fillLight);

    const rimLight = new THREE.PointLight(0xffd7ca, 0.35);
    rimLight.position.set(0, 0, 220);
    this.scene.add(rimLight);
    this.lights.push(rimLight);
  }

  clearObjects() {
    [
      this.tumorMesh,
      this.uncertaintyShell,
      this.tumorWire,
      this.pointCloud,
      this.studyFrame,
      this.centroidMarker,
      this.axisGroup,
      this.measureGroup,
      this.overlayGroup,
      this.volumeSliceGroup,
    ].forEach((item) => {
      if (!item) return;
      this.scene.remove(item);
      item.traverse?.((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((material) => {
              material.map?.dispose?.();
              material.dispose();
            });
          } else {
            child.material.map?.dispose?.();
            child.material.dispose();
          }
        }
      });
      if (item.geometry) item.geometry.dispose();
      if (item.material) {
        item.material.map?.dispose?.();
        item.material.dispose();
      }
    });

    this.tumorMesh = null;
    this.uncertaintyShell = null;
    this.tumorWire = null;
    this.pointCloud = null;
    this.studyFrame = null;
    this.centroidMarker = null;
    this.axisGroup = null;
    this.measureGroup = null;
    this.overlayGroup = null;
    this.volumeSliceGroup = null;
    this.stats = { volume: 0, surface: 0, density: 0 };
  }

  getTumorMaterial() {
    return new THREE.MeshPhysicalMaterial({
      color: 0xdc5f49,
      emissive: 0x6f1b10,
      emissiveIntensity: 0.18,
      roughness: 0.38,
      metalness: 0.08,
      transmission: 0.04,
      thickness: 0.6,
      transparent: true,
      opacity: 0.94,
      clearcoat: 0.32,
      clearcoatRoughness: 0.28,
      side: THREE.DoubleSide,
    });
  }

  getUncertaintyMaterial() {
    return new THREE.MeshPhysicalMaterial({
      color: 0x4eced1,
      emissive: 0x0d5356,
      emissiveIntensity: 0.08,
      roughness: 0.5,
      metalness: 0.04,
      transparent: true,
      opacity: 0.16,
      clearcoat: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
  }

  createStudyFrame(payload = {}) {
    const min = payload.bounds_min_mm || [];
    const max = payload.bounds_max_mm || [];
    if (min.length !== 3 || max.length !== 3) return;

    const minVector = new THREE.Vector3(min[2], min[1], min[0]);
    const maxVector = new THREE.Vector3(max[2], max[1], max[0]);
    const box = new THREE.Box3(minVector, maxVector);
    const size = box.getSize(new THREE.Vector3());
    if (size.lengthSq() === 0) return;

    const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({
      color: 0x38d9c8,
      transparent: true,
      opacity: 0.35,
    });

    this.studyFrame = new THREE.LineSegments(edges, material);
    this.studyFrame.position.copy(box.getCenter(new THREE.Vector3()));
    this.scene.add(this.studyFrame);

    const grid = new THREE.GridHelper(Math.max(size.x, size.y, 20), 8, 0x1a3040, 0x0f2030);
    grid.rotation.x = Math.PI / 2;
    grid.position.copy(this.studyFrame.position);
    grid.position.z = minVector.z;
    this.scene.add(grid);
    this.measureGroup = grid;
  }

  createCentroidMarker(payload = {}) {
    const centroid = payload.centroid_mm || [];
    if (centroid.length !== 3) return;

    const geometry = new THREE.SphereGeometry(1.8, 20, 20);
    const material = new THREE.MeshStandardMaterial({
      color: 0x083b4c,
      emissive: 0x0f7a76,
      emissiveIntensity: 0.35,
      roughness: 0.35,
      metalness: 0.15,
    });

    this.centroidMarker = new THREE.Mesh(geometry, material);
    this.centroidMarker.position.set(centroid[2], centroid[1], centroid[0]);
    this.scene.add(this.centroidMarker);
  }

  createPrincipalAxes(payload = {}) {
    const axes = Array.isArray(payload.principal_axes) ? payload.principal_axes : [];
    const centroid = payload.centroid_mm || [];
    if (!axes.length || centroid.length !== 3) return;

    const origin = new THREE.Vector3(centroid[2], centroid[1], centroid[0]);
    const colors = [0x0f7a76, 0x2d8ccf, 0xf06f5a];
    this.axisGroup = new THREE.Group();

    axes.forEach((axis, index) => {
      if (!Array.isArray(axis) || axis.length < 4) return;
      const direction = new THREE.Vector3(axis[2], axis[1], axis[0]).normalize();
      const halfLength = axis[3] * 0.5;
      const start = origin.clone().addScaledVector(direction, -halfLength);
      const end = origin.clone().addScaledVector(direction, halfLength);

      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      const material = new THREE.LineBasicMaterial({ color: colors[index] || 0xffffff, transparent: true, opacity: 0.85 });
      const line = new THREE.Line(geometry, material);
      this.axisGroup.add(line);
    });

    this.scene.add(this.axisGroup);
  }

  createMesh(payload = {}) {
    if (!Array.isArray(payload.vertices) || !Array.isArray(payload.faces) || !payload.vertices.length || !payload.faces.length) {
      return;
    }

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(payload.vertices.flatMap((vertex) => [vertex[2], vertex[1], vertex[0]]));
    const indices = payload.faces.flat();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    this.tumorMesh = new THREE.Mesh(geometry, this.getTumorMaterial());
    this.tumorMesh.castShadow = true;
    this.tumorMesh.receiveShadow = true;
    this.scene.add(this.tumorMesh);

    const edgeGeometry = new THREE.EdgesGeometry(geometry, 22);
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x742314, transparent: true, opacity: 0.3 });
    this.tumorWire = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    this.scene.add(this.tumorWire);

    if (Array.isArray(payload.shell_vertices) && Array.isArray(payload.shell_faces) && payload.shell_vertices.length && payload.shell_faces.length) {
      const shellGeometry = new THREE.BufferGeometry();
      const shellPositions = new Float32Array(payload.shell_vertices.flatMap((vertex) => [vertex[2], vertex[1], vertex[0]]));
      const shellIndices = payload.shell_faces.flat();
      shellGeometry.setAttribute('position', new THREE.BufferAttribute(shellPositions, 3));
      shellGeometry.setIndex(shellIndices);
      shellGeometry.computeVertexNormals();

      this.uncertaintyShell = new THREE.Mesh(shellGeometry, this.getUncertaintyMaterial());
      this.uncertaintyShell.renderOrder = 2;
      this.scene.add(this.uncertaintyShell);
    }
  }

  createPointCloud(payload = {}) {
    if (!Array.isArray(payload.points) || !payload.points.length) return;

    const positions = [];
    payload.points.forEach((point) => {
      positions.push(Number(point.z || 0), Number(point.y || 0), Number(point.x || 0));
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xd65f49,
      size: 1.2,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.86,
    });

    this.pointCloud = new THREE.Points(geometry, material);
    this.scene.add(this.pointCloud);
  }

  composeOverlayTexture(overlay = {}) {
    const width = Number(overlay.width || 0);
    const height = Number(overlay.height || 0);
    if (!width || !height) return null;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(width, height);
    const imagePixels = Array.isArray(overlay.image_pixels) ? overlay.image_pixels : [];
    const maskPixels = Array.isArray(overlay.mask_pixels) ? overlay.mask_pixels : [];
    const probabilityPixels = Array.isArray(overlay.probability_pixels) ? overlay.probability_pixels : [];

    for (let index = 0; index < width * height; index += 1) {
      const offset = index * 4;
      const base = Number(imagePixels[index] || 0);
      const maskValue = Number(maskPixels[index] || 0);
      const probabilityValue = Number(probabilityPixels[index] || 0);

      let red = base;
      let green = base;
      let blue = base;
      let alpha = 210;

      if (probabilityValue > 0) {
        red = Math.min(255, red + Math.round(probabilityValue * 0.38));
        green = Math.min(255, green + Math.round(probabilityValue * 0.08));
        blue = Math.max(0, blue - Math.round(probabilityValue * 0.12));
        alpha = 222;
      }

      if (maskValue > 0) {
        red = 28;
        green = 208;
        blue = 188;
        alpha = 242;
      }

      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = alpha;
    }

    ctx.putImageData(image, 0, 0);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  addOverlayPlane(overlay = {}) {
    const size = Array.isArray(overlay.size_mm) ? overlay.size_mm : [];
    const position = Array.isArray(overlay.position_mm) ? overlay.position_mm : [];
    if (size.length !== 2 || position.length !== 3) return;

    const texture = this.composeOverlayTexture(overlay);
    if (!texture) return;

    const geometry = new THREE.PlaneGeometry(Math.max(Number(size[0] || 0), 1), Math.max(Number(size[1] || 0), 1));
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.62,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(Number(position[0] || 0), Number(position[1] || 0), Number(position[2] || 0));
    mesh.renderOrder = 3;

    if (overlay.plane === 'coronal') {
      mesh.rotation.x = Math.PI / 2;
    } else if (overlay.plane === 'sagittal') {
      mesh.rotation.y = Math.PI / 2;
    }

    const edges = new THREE.EdgesGeometry(geometry);
    const border = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x0f7a76, transparent: true, opacity: 0.4 }),
    );
    border.position.copy(mesh.position);
    border.rotation.copy(mesh.rotation);

    this.overlayGroup.add(mesh);
    this.overlayGroup.add(border);
  }

  createOrthogonalOverlays(overlays = {}) {
    const planes = ['axial', 'coronal', 'sagittal']
      .map((plane) => overlays?.[plane])
      .filter((overlay) => overlay && overlay.width && overlay.height);
    if (!planes.length) return;

    this.overlayGroup = new THREE.Group();
    planes.forEach((overlay) => this.addOverlayPlane(overlay));
    this.scene.add(this.overlayGroup);
  }

  composeVolumeSliceTexture(width, height, zIndex, mprPayload) {
    const shape = Array.isArray(mprPayload?.shape) ? mprPayload.shape : [0, 0, 0];
    const volumeBytes = mprPayload?.volumeBytes;
    const maskBytes = mprPayload?.maskBytes;
    const probabilityBytes = mprPayload?.probabilityBytes;
    if (!volumeBytes || !maskBytes || !probabilityBytes) return null;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(width, height);

    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width; col += 1) {
        const voxelIndex = (zIndex * shape[1] * shape[2]) + (row * shape[2]) + col;
        const offset = (row * width + col) * 4;
        const base = Number(volumeBytes[voxelIndex] || 0);
        const maskValue = Number(maskBytes[voxelIndex] || 0);
        const probabilityValue = Number(probabilityBytes[voxelIndex] || 0);

        let red = base;
        let green = base;
        let blue = base;
        let alpha = 88;

        if (probabilityValue > 0) {
          red = Math.min(255, red + Math.round(probabilityValue * 0.32));
          green = Math.min(255, green + Math.round(probabilityValue * 0.08));
          blue = Math.max(0, blue - Math.round(probabilityValue * 0.16));
          alpha = 102;
        }
        if (maskValue > 0) {
          red = 36;
          green = 214;
          blue = 188;
          alpha = 168;
        }

        image.data[offset] = red;
        image.data[offset + 1] = green;
        image.data[offset + 2] = blue;
        image.data[offset + 3] = alpha;
      }
    }

    ctx.putImageData(image, 0, 0);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  createVolumeSliceStack(mprPayload = null) {
    const shape = Array.isArray(mprPayload?.shape) ? mprPayload.shape.map((value) => Number(value || 0)) : [0, 0, 0];
    const spacing = Array.isArray(mprPayload?.voxel_spacing_mm)
      ? mprPayload.voxel_spacing_mm.map((value) => Number(value || 1))
      : [1, 1, 1];
    if (!shape[0] || !shape[1] || !shape[2]) return;
    if (!mprPayload?.volumeBytes || !mprPayload?.maskBytes || !mprPayload?.probabilityBytes) return;

    const bboxMin = Array.isArray(mprPayload?.lesion_bbox_min) ? mprPayload.lesion_bbox_min.map((value) => Number(value || 0)) : [0, 0, 0];
    const bboxMax = Array.isArray(mprPayload?.lesion_bbox_max)
      ? mprPayload.lesion_bbox_max.map((value) => Number(value || 0))
      : [shape[0] - 1, shape[1] - 1, shape[2] - 1];

    const zMin = Math.max(0, Math.min(shape[0] - 1, bboxMin[0]));
    const zMax = Math.max(0, Math.min(shape[0] - 1, bboxMax[0]));
    const zRange = Math.max(1, zMax - zMin + 1);
    const maxSlices = 20;
    const step = Math.max(1, Math.floor(zRange / maxSlices));

    this.volumeSliceGroup = new THREE.Group();
    const planeWidth = Math.max(1, shape[2] * spacing[2]);
    const planeHeight = Math.max(1, shape[1] * spacing[1]);

    for (let z = zMin; z <= zMax; z += step) {
      const texture = this.composeVolumeSliceTexture(shape[2], shape[1], z, mprPayload);
      if (!texture) continue;

      const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.24,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const plane = new THREE.Mesh(geometry, material);
      plane.position.set((shape[2] * spacing[2]) * 0.5, (shape[1] * spacing[1]) * 0.5, z * spacing[0]);
      plane.renderOrder = 1;
      this.volumeSliceGroup.add(plane);
    }

    this.scene.add(this.volumeSliceGroup);
  }

  loadPayload(payload = {}, stats = {}, overlays = null, mprPayload = null) {
    this.clearObjects();
    this.createVolumeSliceStack(mprPayload);
    this.createStudyFrame(payload);
    this.createCentroidMarker(payload);
    this.createPrincipalAxes(payload);
    this.createOrthogonalOverlays(overlays);

    if (payload.mode === 'mesh') {
      this.createMesh(payload);
    } else if (payload.mode === 'point-cloud') {
      this.createPointCloud(payload);
    }

    this.calculateStats(stats);
    this.focusOnTarget();
    this.updateStats();
  }

  calculateStats(data = {}) {
    this.stats = {
      volume: Number(data.volume || 0),
      surface: Number(data.surface || 0),
      density: data.density ?? 0,
    };
  }

  focusOnTarget(target = this.tumorMesh || this.pointCloud || this.studyFrame || this.volumeSliceGroup) {
    if (!target) return;

    const box = new THREE.Box3().setFromObject(target);
    if (!box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) * 0.9;

      this.controls.target.copy(center);
      this.camera.position.copy(center).add(new THREE.Vector3(radius * 1.2, -radius * 1.8, radius * 1.1));
      this.camera.near = Math.max(radius / 200, 0.1);
      this.camera.far = Math.max(radius * 20, 1000);
      this.camera.updateProjectionMatrix();
      this.controls.update();
    }
  }

  toggleAutoRotate() {
    this.autoRotate = !this.autoRotate;
    this.controls.autoRotate = this.autoRotate;
  }

  zoomIn() {
    this.camera.position.sub(this.controls.target).multiplyScalar(0.82).add(this.controls.target);
    this.controls.update();
  }

  zoomOut() {
    this.camera.position.sub(this.controls.target).multiplyScalar(1.2).add(this.controls.target);
    this.controls.update();
  }

  reset() {
    this.autoRotate = false;
    this.controls.autoRotate = false;
    this.focusOnTarget();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
  }

  onWindowResize() {
    if (!this.container || !this.camera || !this.renderer) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  updateStats() {
    const volumeEl = document.getElementById('tumor-volume');
    const surfaceEl = document.getElementById('tumor-surface');
    const densityEl = document.getElementById('tumor-density');

    if (volumeEl) volumeEl.textContent = Number(this.stats.volume || 0).toLocaleString('vi-VN', { maximumFractionDigits: 2 });
    if (surfaceEl) surfaceEl.textContent = Number(this.stats.surface || 0).toLocaleString('vi-VN', { maximumFractionDigits: 2 });
    if (densityEl) densityEl.textContent = typeof this.stats.density === 'number'
      ? this.stats.density.toLocaleString('vi-VN', { maximumFractionDigits: 4 })
      : String(this.stats.density ?? '--');
  }

  dispose() {
    this.clearObjects();
    if (this.renderer) this.renderer.dispose();
    if (this.scene) {
      this.scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((material) => material.dispose());
          } else {
            obj.material.dispose();
          }
        }
      });
    }
  }
}

window.TumorViewer = TumorViewer;
