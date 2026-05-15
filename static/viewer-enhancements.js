// Enhanced 3D Viewer Interactions & Controls
class TumorViewerEnhancements {
  constructor(viewer) {
    this.viewer = viewer;
    this.init();
  }

  init() {
    this.setupKeyboardControls();
    this.setupGestureControls();
    this.setupAnimations();
  }

  setupKeyboardControls() {
    document.addEventListener('keydown', (event) => {
      if (!this.viewer.controls) return;

      switch (event.key.toLowerCase()) {
        case 'r':
          this.viewer.reset();
          break;
        case '+':
        case '=':
          this.viewer.zoomIn();
          break;
        case '-':
          this.viewer.zoomOut();
          break;
        case ' ':
          event.preventDefault();
          this.viewer.toggleAutoRotate();
          break;
      }
    });
  }

  setupGestureControls() {
    if (!this.viewer.container) return;

    let lastDistance = 0;

    this.viewer.container.addEventListener('touchmove', (event) => {
      if (event.touches.length === 2) {
        event.preventDefault();

        const touch1 = event.touches[0];
        const touch2 = event.touches[1];
        const distance = Math.hypot(
          touch2.clientX - touch1.clientX,
          touch2.clientY - touch1.clientY
        );

        if (lastDistance > 0) {
          if (distance > lastDistance) {
            this.viewer.zoomOut();
          } else {
            this.viewer.zoomIn();
          }
        }

        lastDistance = distance;
      }
    });

    this.viewer.container.addEventListener('touchend', () => {
      lastDistance = 0;
    });
  }

  setupAnimations() {
    // Animate stats when they appear
    const statsObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          this.animateStats(entry.target);
          statsObserver.unobserve(entry.target);
        }
      });
    });

    const statElements = document.querySelectorAll('.stat-value');
    statElements.forEach((el) => statsObserver.observe(el));
  }

  animateStats(element) {
    const text = element.textContent;
    const isNumeric = !isNaN(parseFloat(text));

    if (isNumeric) {
      const start = 0;
      const end = parseFloat(text);
      const duration = 800;
      const startTime = Date.now();

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const current = Math.floor(start + (end - start) * this.easeOutQuad(progress));
        element.textContent = current.toLocaleString();

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      animate();
    }
  }

  easeOutQuad(t) {
    return t * (2 - t);
  }

  // Create color map for tumor density visualization
  static createDensityColorMap() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 1;

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 256, 0);

    gradient.addColorStop(0, '#0f7a76'); // Dark teal (low)
    gradient.addColorStop(0.25, '#16a3a0'); // Teal
    gradient.addColorStop(0.5, '#2db8b5'); // Light teal
    gradient.addColorStop(0.75, '#f06f5a'); // Coral
    gradient.addColorStop(1, '#c41e3a'); // Dark red (high)

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 1);

    return canvas;
  }

  // Export viewer state as JSON
  exportState() {
    if (!this.viewer) return null;

    return {
      cameraPosition: {
        x: this.viewer.camera.position.x,
        y: this.viewer.camera.position.y,
        z: this.viewer.camera.position.z
      },
      controlsTarget: {
        x: this.viewer.controls.target.x,
        y: this.viewer.controls.target.y,
        z: this.viewer.controls.target.z
      },
      autoRotate: this.viewer.autoRotate,
      zoom: this.viewer.camera.zoom,
      stats: this.viewer.stats
    };
  }

  // Import viewer state from JSON
  importState(state) {
    if (!this.viewer || !state) return;

    if (state.cameraPosition) {
      const { x, y, z } = state.cameraPosition;
      this.viewer.camera.position.set(x, y, z);
    }

    if (state.controlsTarget) {
      const { x, y, z } = state.controlsTarget;
      this.viewer.controls.target.set(x, y, z);
    }

    if (state.autoRotate !== undefined) {
      this.viewer.autoRotate = state.autoRotate;
      this.viewer.controls.autoRotate = state.autoRotate;
    }

    this.viewer.controls.update();
  }
}

// Export for use in other scripts
window.TumorViewerEnhancements = TumorViewerEnhancements;
