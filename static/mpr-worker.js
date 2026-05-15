self.onmessage = (event) => {
  const payload = event.data || {};
  const jobId = payload.jobId;
  try {
    const shape = Array.isArray(payload.shape) ? payload.shape : [0, 0, 0];
    const plane = payload.plane || 'axial';
    const cursor = Array.isArray(payload.cursor) ? payload.cursor : [0, 0, 0];
    const volumeBytes = payload.volumeBytes ? new Uint8Array(payload.volumeBytes) : new Uint8Array(0);
    const maskBytes = payload.maskBytes ? new Uint8Array(payload.maskBytes) : new Uint8Array(0);
    const probabilityBytes = payload.probabilityBytes ? new Uint8Array(payload.probabilityBytes) : new Uint8Array(0);

    let width = 0;
    let height = 0;
    if (plane === 'axial') {
      width = Number(shape[2] || 0);
      height = Number(shape[1] || 0);
    } else if (plane === 'coronal') {
      width = Number(shape[2] || 0);
      height = Number(shape[0] || 0);
    } else {
      width = Number(shape[1] || 0);
      height = Number(shape[0] || 0);
    }

    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width; col += 1) {
        let z = Number(cursor[0] || 0);
        let y = Number(cursor[1] || 0);
        let x = Number(cursor[2] || 0);
        if (plane === 'axial') {
          y = row;
          x = col;
        } else if (plane === 'coronal') {
          z = row;
          x = col;
        } else {
          z = row;
          y = col;
        }
        const voxelIndex = (z * shape[1] * shape[2]) + (y * shape[2]) + x;
        const offset = (row * width + col) * 4;
        const base = Number(volumeBytes[voxelIndex] || 0);
        const maskValue = Number(maskBytes[voxelIndex] || 0);
        const probabilityValue = Number(probabilityBytes[voxelIndex] || 0);

        let red = base;
        let green = base;
        let blue = base;
        if (probabilityValue > 0) {
          red = Math.min(255, red + Math.round(probabilityValue * 0.45));
          green = Math.min(255, green + Math.round(probabilityValue * 0.08));
          blue = Math.max(0, blue - Math.round(probabilityValue * 0.18));
        }
        if (maskValue > 0) {
          red = 36;
          green = 214;
          blue = 188;
        }
        pixels[offset] = red;
        pixels[offset + 1] = green;
        pixels[offset + 2] = blue;
        pixels[offset + 3] = 255;
      }
    }

    self.postMessage({
      jobId,
      width,
      height,
      pixels: pixels.buffer,
    }, [pixels.buffer]);
  } catch (error) {
    self.postMessage({
      jobId,
      error: String(error && error.message ? error.message : error),
    });
  }
};
