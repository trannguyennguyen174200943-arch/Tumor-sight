const TEXT = {
  initError: 'Không thể khởi tạo giao diện',
  platformLoadError: 'Không tải được thông tin nền tảng',
  casesLoadError: 'Không tải được danh sách ca',
  noDataSelected: 'Chưa có dữ liệu được chọn.',
  previewUnavailable: 'Preview chưa sẵn sàng',
  previewLoading: 'Đang tải preview...',
  processing: 'Đang xử lý...',
  analyzeRunning: 'Đang phân tích study đã tải lên...',
  analyzeSuccess: 'Phân tích hoàn tất',
  analyzeFailed: 'Phân tích thất bại',
  exportMissing: 'Chưa có dữ liệu khối u để xuất STL',
  exportSuccess: 'Đã xuất mô hình STL',
};

const LABEL_MAPS = {
  classification: {
    malignant: 'Nghi ngờ ác tính',
    benign: 'Nghi ngờ lành tính',
    indeterminate: 'Chưa xác định',
    uncertain: 'Chưa xác định',
  },
  reviewStatus: {
    'Ready for board': 'Sẵn sàng hội chẩn',
    'Ready for review': 'Sẵn sàng rà soát',
    'Monitoring ready': 'Sẵn sàng theo dõi',
  },
  triageLevel: {
    Critical: 'Khẩn cấp',
    High: 'Ưu tiên cao',
    Moderate: 'Theo dõi chuẩn',
  },
  growthBand: {
    accelerating: 'Tăng nhanh',
    growing: 'Đang tăng',
    stable: 'Ổn định',
    regressing: 'Đang giảm',
  },
  sourceType: {
    uploaded_dicom: 'DICOM tải lên',
    uploaded_volume: 'Volume tải lên',
  },
  healthStatus: {
    healthy: 'Ổn định',
    degraded: 'Cần chú ý',
  },
};

const app = {
  state: {
    platform: null,
    cases: [],
    selectedCase: null,
    submitting: false,
    viewer: null,
    viewerEnhancements: null,
    previewByCase: {},
    previewLoadingByCase: {},
    mprByCase: {},
    mprLoadingByCase: {},
    mprCursorByCase: {},
    viewerOverlaysByCase: {},
    viewerOverlayLoadingByCase: {},
    mprWorker: null,
    mprWorkerJobs: {},
    mprWorkerSeq: 0,
  },

  async init() {
    this.normalizePageUrl();
    this.setupEvents();
    try {
      if (document.getElementById('viewer-3d') && window.TumorViewer) {
        try {
          this.state.viewer = new TumorViewer('viewer-3d');
          this.state.viewerEnhancements = new TumorViewerEnhancements(this.state.viewer);
        } catch (viewerError) {
          console.error(viewerError);
          this.state.viewer = null;
          this.state.viewerEnhancements = null;
          this.showMessage('Viewer 3D chưa khởi tạo được, nhưng upload và phân tích vẫn hoạt động.', 'error');
        }
      }
      this.initMprWorker();

      await this.loadPlatform();
      this.render();

      if (this.state.selectedCase?.id) {
        this.loadSegmentationPreview(this.state.selectedCase.id);
        this.loadMprPayload(this.state.selectedCase.id);
        this.loadViewerOverlays(this.state.selectedCase.id);
        this.initializeTumorVisualization(this.state.selectedCase);
        this.focusSelectedCase();
      }
    } catch (err) {
      console.error(err);
      this.render();
      this.showMessage(err.message || TEXT.initError, 'error');
    }
  },

  initMprWorker() {
    if (typeof Worker === 'undefined') return;
    if (this.state.mprWorker) return;
    try {
      const worker = new Worker('/static/mpr-worker.js?v=20260424-opt');
      worker.onmessage = (event) => {
        const { jobId, width, height, pixels, error } = event.data || {};
        const resolver = this.state.mprWorkerJobs[jobId];
        if (!resolver) return;
        delete this.state.mprWorkerJobs[jobId];
        if (error) {
          resolver.reject(new Error(error));
          return;
        }
        resolver.resolve({
          width: Number(width || 0),
          height: Number(height || 0),
          pixels: new Uint8ClampedArray(pixels),
        });
      };
      this.state.mprWorker = worker;
    } catch (_) {
      this.state.mprWorker = null;
    }
  },

  normalizePageUrl() {
    if (typeof window === 'undefined' || !window.location?.search) return;
    const search = window.location.search.toLowerCase();
    const knownFormKeys = ['patient_name=', 'modality=', 'timepoint_count=', 'dicom_files='];
    if (!knownFormKeys.some((key) => search.includes(key))) return;
    if (window.history?.replaceState) {
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  },

  async loadPlatform() {
    const res = await fetch('/api/platform');
    if (!res.ok) throw new Error(TEXT.platformLoadError);
    this.state.platform = await res.json();
    await this.loadCases();
  },

  async loadCases() {
    const res = await fetch('/api/cases');
    if (!res.ok) throw new Error(TEXT.casesLoadError);
    const cases = await res.json();
    const selectedId = this.state.selectedCase?.id;
    this.state.cases = Array.isArray(cases) ? cases : [];
    this.state.selectedCase =
      this.state.cases.find((item) => item.id === selectedId) ||
      this.state.cases[0] ||
      null;
  },

  setupEvents() {
    const bindings = [
      ['analysis-form', 'submit', (event) => {
        event.preventDefault();
        this.submitAnalysis();
      }],
      ['submit-analysis-btn', 'click', (event) => {
        event.preventDefault();
        this.submitAnalysis();
      }],
      ['load-demo', 'click', () => this.loadDemo()],
      ['jump-upload', 'click', () => {
        document.getElementById('stage-input')?.scrollIntoView({ behavior: 'smooth' });
      }],
      ['dicom-files', 'change', () => this.updateUploadMode()],
      ['viewer-reset', 'click', () => this.state.viewer?.reset()],
      ['viewer-zoom-in', 'click', () => this.state.viewer?.zoomIn()],
      ['viewer-zoom-out', 'click', () => this.state.viewer?.zoomOut()],
      ['viewer-rotate', 'click', () => this.state.viewer?.toggleAutoRotate()],
      ['export-stl-btn', 'click', () => this.exportTumorAsSTL()],
    ];

    bindings.forEach(([id, eventName, handler]) => {
      const element = document.getElementById(id);
      if (!element || element.dataset.bound) return;
      element.dataset.bound = '1';
      element.addEventListener(eventName, handler);
    });
  },

  bindCaseListEvents() {
    document.querySelectorAll('.case-item').forEach((item) => {
      if (item.dataset.bound) return;
      item.dataset.bound = '1';
      item.addEventListener('click', () => this.selectCase(item.dataset.id));
    });
  },

  updateUploadMode() {
    const fileInput = document.getElementById('dicom-files');
    const timeInput = document.querySelector('input[name="timepoint_count"]');
    const badge = document.getElementById('upload-mode-badge');
    if (!badge || !fileInput || !timeInput) return;

    const fileCount = fileInput.files?.length || 0;
    const timeCount = Number(timeInput.value || 1);

    if (!fileCount) {
      badge.textContent = TEXT.noDataSelected;
      return;
    }

    let mode = 'Nhanh';
    if (fileCount >= 40 || timeCount >= 3) mode = 'Cân bằng';
    if (fileCount >= 120 || timeCount >= 4) mode = 'Biên tối ưu';
    badge.textContent = `Chế độ suy luận: ${mode}`;
  },

  async submitAnalysis() {
    const form = document.getElementById('analysis-form');
    if (!form || this.state.submitting) return;

    try {
      this.state.submitting = true;
      const button = document.getElementById('submit-analysis-btn');
      if (button) button.textContent = TEXT.processing;

      this.renderAnalysisStatus(TEXT.analyzeRunning, 'info');

      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: new FormData(form),
      });

      if (!res.ok) {
        let message = TEXT.analyzeFailed;
        try {
          const payload = await res.json();
          message = payload.detail || payload.error || message;
        } catch (_) {
          // Ignore JSON parse errors.
        }
        throw new Error(message);
      }

      const data = await res.json();
      await this.loadCases();
      this.state.selectedCase =
        this.state.cases.find((item) => item.id === data.case_id) ||
        data.case ||
        this.state.selectedCase;

      this.render();

      if (this.state.selectedCase?.id) {
        this.loadSegmentationPreview(this.state.selectedCase.id);
        this.loadMprPayload(this.state.selectedCase.id);
        this.loadViewerOverlays(this.state.selectedCase.id);
        this.initializeTumorVisualization(this.state.selectedCase);
        this.focusSelectedCase();
      }

      this.renderAnalysisStatus(
        `Hoàn tất: ${this.state.selectedCase?.patient_name || data.case_id || 'ca mới'}`,
        'success',
      );
      this.showMessage(TEXT.analyzeSuccess, 'success');
    } catch (err) {
      console.error(err);
      this.renderAnalysisStatus(err.message || TEXT.analyzeFailed, 'error');
      this.showMessage(err.message || TEXT.analyzeFailed, 'error');
    } finally {
      this.state.submitting = false;
      const button = document.getElementById('submit-analysis-btn');
      if (button) button.textContent = 'Phân tích';
    }
  },

  loadDemo() {
    if (!this.state.cases.length) {
      this.showMessage('Chưa có ca bệnh thực. Hãy tải study để phân tích.', 'info');
      return;
    }
    this.selectCase(this.state.cases[0].id);
  },

  selectCase(id) {
    const selected = this.state.cases.find((item) => item.id === id);
    if (!selected) return;
    this.state.selectedCase = selected;
    this.render();
    this.loadSegmentationPreview(id);
    this.loadMprPayload(id);
    this.loadViewerOverlays(id);
    this.initializeTumorVisualization(selected);
  },

  focusSelectedCase() {
    document.getElementById('stage-viewing')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  initializeTumorVisualization(caseData) {
    if (!this.state.viewer || !caseData) return;
    this.renderViewer(caseData);
  },

  exportTumorAsSTL() {
    if (!this.state.viewer?.tumorMesh) {
      this.showMessage(TEXT.exportMissing, 'error');
      return;
    }

    const geometry = this.state.viewer.tumorMesh.geometry;
    const name = this.state.selectedCase?.patient_name || 'tumor';
    const stlData = this.geometryToSTL(geometry, name);
    const blob = new Blob([stlData], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${name}_tumor.stl`;
    link.click();
    URL.revokeObjectURL(url);
    this.showMessage(TEXT.exportSuccess, 'success');
  },

  geometryToSTL(geometry, name = 'tumor') {
    const positions = geometry.attributes.position.array;
    const indices = geometry.index ? geometry.index.array : null;
    const triangles = indices ? indices.length / 3 : positions.length / 9;
    const buffer = new ArrayBuffer(84 + triangles * 50);
    const view = new DataView(buffer);
    const header = new TextEncoder().encode(name.padEnd(80));

    for (let i = 0; i < 80; i += 1) view.setUint8(i, header[i] || 0);
    view.setUint32(80, triangles, true);

    let offset = 84;
    for (let i = 0; i < triangles; i += 1) {
      const idx = i * 3;
      const pi = indices ? indices[idx] * 3 : idx * 9;
      const pj = indices ? indices[idx + 1] * 3 : (idx * 9) + 3;
      const pk = indices ? indices[idx + 2] * 3 : (idx * 9) + 6;
      const v1 = [positions[pi], positions[pi + 1], positions[pi + 2]];
      const v2 = [positions[pj], positions[pj + 1], positions[pj + 2]];
      const v3 = [positions[pk], positions[pk + 1], positions[pk + 2]];
      const normal = this.computeNormal(v1, v2, v3);

      [...normal, ...v1, ...v2, ...v3].forEach((value) => {
        view.setFloat32(offset, value, true);
        offset += 4;
      });
      view.setUint16(offset, 0, true);
      offset += 2;
    }

    return buffer;
  },

  computeNormal(v1, v2, v3) {
    const a = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
    const b = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
    const normal = [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const len = Math.hypot(normal[0], normal[1], normal[2]);
    return len > 0 ? normal.map((value) => value / len) : [0, 0, 1];
  },

  async loadSegmentationPreview(caseId) {
    if (!caseId || this.state.previewByCase[caseId] || this.state.previewLoadingByCase[caseId]) return;

    this.state.previewLoadingByCase[caseId] = true;
    this.renderSegmentationPreview(this.state.selectedCase);

    try {
      const res = await fetch(`/api/cases/${caseId}/segmentation/preview`);
      if (!res.ok) throw new Error(TEXT.previewUnavailable);
      this.state.previewByCase[caseId] = await res.json();
    } catch (_) {
      this.state.previewByCase[caseId] = null;
    } finally {
      delete this.state.previewLoadingByCase[caseId];
      if (this.state.selectedCase?.id === caseId) {
        this.renderSegmentationPreview(this.state.selectedCase);
        this.renderViewerReference(this.state.selectedCase);
      }
    }
  },

  decodeBase64Bytes(base64) {
    const binary = window.atob(base64 || '');
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  },

  async loadMprPayload(caseId) {
    if (!caseId || this.state.mprByCase[caseId] || this.state.mprLoadingByCase[caseId]) return;

    this.state.mprLoadingByCase[caseId] = true;
    this.renderSegmentationPreview(this.state.selectedCase);

    try {
      const res = await fetch(`/api/cases/${caseId}/mpr`);
      if (!res.ok) throw new Error('Không tải được dữ liệu MPR');
      const payload = await res.json();
      this.state.mprByCase[caseId] = {
        ...payload,
        volumeBytes: this.decodeBase64Bytes(payload.volume_base64),
        maskBytes: this.decodeBase64Bytes(payload.mask_base64),
        probabilityBytes: this.decodeBase64Bytes(payload.probability_base64),
      };
      if (!this.state.mprCursorByCase[caseId] && Array.isArray(payload.default_index)) {
        this.state.mprCursorByCase[caseId] = payload.default_index.map((value) => Number(value || 0));
      }
    } catch (_) {
      this.state.mprByCase[caseId] = null;
    } finally {
      delete this.state.mprLoadingByCase[caseId];
      if (this.state.selectedCase?.id === caseId) {
        this.renderSegmentationPreview(this.state.selectedCase);
        this.renderViewerReference(this.state.selectedCase);
        this.renderViewer(this.state.selectedCase);
      }
    }
  },

  async loadViewerOverlays(caseId) {
    if (!caseId || this.state.viewerOverlaysByCase[caseId] || this.state.viewerOverlayLoadingByCase[caseId]) return;

    this.state.viewerOverlayLoadingByCase[caseId] = true;
    try {
      const res = await fetch(`/api/cases/${caseId}/viewer/orthogonal-overlays`);
      if (!res.ok) throw new Error('Không tải được overlay trực giao');
      this.state.viewerOverlaysByCase[caseId] = await res.json();
    } catch (_) {
      this.state.viewerOverlaysByCase[caseId] = null;
    } finally {
      delete this.state.viewerOverlayLoadingByCase[caseId];
      if (this.state.selectedCase?.id === caseId) {
        this.renderViewer(this.state.selectedCase);
      }
    }
  },

  render() {
    this.renderPlatform();
    this.renderCases();
    this.renderPortfolio();
    if (this.state.selectedCase) {
      this.renderCase();
    } else {
      this.renderEmptyState();
    }
  },

  renderPlatform() {
    const platform = this.state.platform;
    if (!platform) return;

    const metricsHost = document.getElementById('platform-metrics');
    if (metricsHost) {
      metricsHost.innerHTML = [
        this.metric('Nền tảng', platform.availability ? 'Sẵn sàng' : 'Gián đoạn'),
        this.metric('Số ca', platform.case_count ?? '--'),
        this.metric('Bộ nhớ', platform.memory || '--'),
        this.metric('Thiết bị', platform.gpu ? 'GPU sẵn sàng' : 'CPU mode'),
      ].join('');
    }

    const statusHost = document.getElementById('model-status-strip');
    if (statusHost) {
      const runtimeModels = Array.isArray(platform.runtime_models) ? platform.runtime_models : [];
      const segmentationModel = runtimeModels.find((item) => item.task === 'segmentation');
      const classificationModel = runtimeModels.find((item) => item.task === 'classification');
      statusHost.innerHTML = `
        <div class="status-badge">v${this.escapeHtml(platform.version || '--')}</div>
        <div class="status-badge">${platform.gpu ? 'GPU' : 'CPU'}</div>
        <div class="status-badge">${this.escapeHtml(this.mapLabel('healthStatus', platform.health?.status) || '--')}</div>
        <div class="status-badge">${this.escapeHtml(segmentationModel?.name || 'Segmentation --')}</div>
        <div class="status-badge">${this.escapeHtml(classificationModel?.name || 'Classification --')}</div>
      `;
    }

    const evidenceHost = document.getElementById('evidence-card');
    if (evidenceHost) {
      const health = platform.health || {};
      const checkpoints = health.checks?.checkpoints || {};
      const training = platform.evidence?.training || {};
      const evaluation = platform.evidence?.evaluation || {};
      const availableModels = Array.isArray(checkpoints.available_models) ? checkpoints.available_models : [];
      const runtimeModels = Array.isArray(platform.runtime_models) ? platform.runtime_models : [];
      const runtimeSummary = runtimeModels
        .map((item) => `${item.task}: ${item.name}${item.ready ? '' : ' (chua san sang)'}`)
        .join(' | ');
      evidenceHost.innerHTML = `
        <div class="card">
          <h3>Dấu vết kiểm chứng</h3>
          <div class="benchmark-grid">
            <div class="benchmark-card-item">
              <div class="benchmark-card-top"><span>Checkpoint ưu tiên</span><small>${this.escapeHtml(this.mapLabel('healthStatus', health.status) || '--')}</small></div>
              <strong class="benchmark-card-value">${this.escapeHtml(checkpoints.preferred_model_name || '--')}</strong>
              <small>${checkpoints.training_checkpoint_ready ? 'Sẵn sàng suy luận' : 'Thiếu checkpoint'}</small>
            </div>
            <div class="benchmark-card-item">
              <div class="benchmark-card-top"><span>Dice validation</span><small>Segmentation</small></div>
              <strong class="benchmark-card-value">${this.formatNumber(training.best_val_dice, 4)}</strong>
              <small>Số epoch: ${this.escapeHtml(String(training.epochs ?? '--'))}</small>
            </div>
            <div class="benchmark-card-item">
              <div class="benchmark-card-top"><span>MAE clean volume</span><small>Evaluation</small></div>
              <strong class="benchmark-card-value">${this.formatNumber(evaluation.clean_mae_cm3, 2)}</strong>
              <small>Validation cases: ${this.escapeHtml(String(evaluation.validation_cases ?? '--'))}</small>
            </div>
          </div>
          <div class="confidence-chip-row" style="margin-top: 12px;">
            <div class="confidence-chip">
              <span>Thiết bị</span>
              <strong>${platform.gpu ? 'GPU Runtime' : 'CPU Runtime'}</strong>
            </div>
            <div class="confidence-chip">
              <span>Model khả dụng</span>
              <strong>${this.escapeHtml(String(availableModels.length || 0))}</strong>
            </div>
            <div class="confidence-chip">
              <span>Bộ nhớ / cache</span>
              <strong>${this.escapeHtml(platform.memory || '--')}</strong>
            </div>
          </div>
          <p style="margin-top: 14px; color: var(--muted);">${this.escapeHtml(runtimeSummary || 'Chua co thong tin runtime model')}</p>
        </div>
      `;
    }

    const governanceHost = document.getElementById('governance-card');
    if (governanceHost) {
      const health = platform.health || {};
      const checkpoints = health.checks?.checkpoints || {};
      const training = platform.evidence?.training || {};
      const evaluation = platform.evidence?.evaluation || {};
      const runtimeModels = Array.isArray(platform.runtime_models) ? platform.runtime_models : [];
      const readyCount = runtimeModels.filter((item) => item.ready).length;
      const runtimeModes = runtimeModels
        .map((item) => item.runtime_mode || item.mode || item.name)
        .filter(Boolean)
        .slice(0, 4);

      governanceHost.innerHTML = `
        <div class="governance-card">
          <div class="governance-head">
            <strong>Runtime, benchmark va checkpoint duoc dat trong mot lop governance</strong>
            <p>Khung nay tom tat model dang chay, checkpoint uu tien va muc san sang cua pipeline de giao dien co tinh hoc thuat hon thay vi chi hien thi card don le.</p>
          </div>
          <div class="governance-grid">
            ${this.metric('Model san sang', `${readyCount}/${runtimeModels.length || 0}`)}
            ${this.metric('Checkpoint uu tien', checkpoints.preferred_model_name || '--')}
            ${this.metric('Dice validation', this.formatNumber(training.best_val_dice, 4))}
            ${this.metric('Validation cases', evaluation.validation_cases ?? '--')}
          </div>
          <div class="evidence-pill-row">
            ${runtimeModes.length
              ? runtimeModes.map((item) => `<span class="evidence-pill">${this.escapeHtml(String(item))}</span>`).join('')
              : '<span class="evidence-pill">Runtime dang duoc khoi tao</span>'}
          </div>
          <p class="governance-note">Trang thai he thong: ${this.escapeHtml(this.mapLabel('healthStatus', health.status) || '--')} | Thiet bi suy luan: ${this.escapeHtml(platform.gpu ? 'GPU runtime' : 'CPU runtime')} | So model kha dung: ${this.escapeHtml(String((checkpoints.available_models || []).length || 0))}</p>
        </div>
      `;
    }
  },

  renderCases() {
    const host = document.getElementById('case-list');
    if (!host) return;

    if (!this.state.cases.length) {
      host.innerHTML = '<p style="color: var(--muted); text-align: center; padding: 26px 12px;">Chưa có ca bệnh nào.</p>';
      return;
    }

    host.innerHTML = this.state.cases.map((item) => {
      const active = this.state.selectedCase?.id === item.id ? 'active' : '';
      return `
        <div class="case-item ${active}" data-id="${this.escapeHtml(item.id || '')}">
          <div class="case-item-header">
            <strong>${this.escapeHtml(item.patient_name || item.patient?.name || 'Chưa rõ')}</strong>
            <span>${this.escapeHtml(item.modality || item.study?.modality || '--')}</span>
          </div>
          <div class="case-item-meta">
            <span>${this.escapeHtml(item.date || '--')}</span>
            <span>${this.escapeHtml(this.formatReviewStatus(item.status || '--'))}</span>
          </div>
        </div>
      `;
    }).join('');

    this.bindCaseListEvents();
  },

  renderCase() {
    const current = this.state.selectedCase;
    if (!current) return;

    const titleHost = document.getElementById('case-title');
    if (titleHost) titleHost.textContent = current.patient_name || current.patient?.name || 'Ca bệnh';

    this.renderSummary(current);
    this.renderCaseKpis(current);
    this.renderCaseDetails(current);
    this.renderViewer(current);
    this.renderViewerReference(current);
    this.renderInterpretability(current);
    this.renderIngestion(current);
    this.renderPreprocessing(current);
    this.renderWorkflow(current);
    this.renderSegmentationHero(current);
    this.renderSegmentation(current);
    this.renderSegmentationDownloads(current);
    this.renderSegmentationPreview(current);
    this.renderMorphology(current);
    this.renderClassification(current);
    this.renderTreatment(current);
    this.renderFollowup(current);
    this.renderTeleconsultationBoard(current);
    this.renderTimeline(current);
  },

  renderEmptyState() {
    [
      'case-kpi-grid',
      'viewer-reference-panel',
      'interpretability-card',
      'case-details',
      'ingestion-card',
      'preprocessing-card',
      'workflow-steps',
      'segmentation-hero-card',
      'segmentation-card',
      'segmentation-downloads',
      'segmentation-preview',
      'morphology-card',
      'classification-card',
      'treatment-card',
      'followup-card',
      'teleconsult-board',
      'timeline-card',
    ].forEach((id) => {
      const host = document.getElementById(id);
      if (host) host.innerHTML = '';
    });

    const exportButton = document.getElementById('export-stl-btn');
    if (exportButton) exportButton.disabled = true;

    if (this.state.viewer?.clearObjects) {
      this.state.viewer.clearObjects();
      this.state.viewer.updateStats();
    }

    this.renderInterpretability(null);
  },

  renderSummary(current) {
    const summaryRisk = document.getElementById('summary-risk');
    const summaryForecast = document.getElementById('summary-forecast');
    const summaryReady = document.getElementById('summary-ready');
    const summaryPriority = document.getElementById('summary-priority');

    if (summaryRisk) {
      summaryRisk.textContent = `${this.formatClassificationLabel(current.classification?.label)} | ${current.insight?.portfolio_tag || '--'}`;
    }
    if (summaryForecast) summaryForecast.textContent = current.progression?.forecast_summary || '--';
    if (summaryReady) summaryReady.textContent = this.formatReviewStatus(current.ai_summary?.review_status || '--');
    if (summaryPriority) summaryPriority.textContent = this.formatTriageLevel(current.ai_summary?.triage_level || '--');
  },

  renderCaseKpis(current) {
    const host = document.getElementById('case-kpi-grid');
    if (!host) return;

    host.innerHTML = [
      this.kpiCard('Thể tích', `${this.formatNumber(current.tumor_volume, 1)} cm³`, current.segmentation?.model_name || '--'),
      this.kpiCard('Nguy cơ', `${this.formatNumber(current.classification?.risk_score, 1)}/100`, this.formatClassificationLabel(current.classification?.label)),
      this.kpiCard('Tăng 6 tháng', `${this.formatNumber(current.progression?.volume_change_6m_percent, 1)}%`, this.formatGrowthBand(current.progression?.growth_velocity_band)),
      this.kpiCard('Tái khám', current.follow_up?.next_recommended_date || '--', current.follow_up?.channel || '--'),
    ].join('');
  },

  renderCaseDetails(current) {
    const patient = current.patient || {};
    const study = current.study || {};
    const host = document.getElementById('case-details');
    if (!host) return;
    const reconstruction = current.reconstruction || {};
    const primaryHospital = patient.hospital || current.referral_site || 'Chua khai bao';
    const studyLabel = [study.modality || current.modality || '--', study.timepoints ? `${study.timepoints} moc thoi gian` : null]
      .filter(Boolean)
      .join(' | ');
    const volumeLabel = `${this.formatNumber(reconstruction.volume_cm3 || current.tumor_volume, 2)} cm3`;
    const surfaceLabel = `${this.formatNumber(reconstruction.surface_area_cm2, 2)} cm2`;

    host.innerHTML = `
      <div class="clinical-brief-card">
        <span class="clinical-brief-label">Benh nhan</span>
        <strong class="clinical-brief-title">${this.escapeHtml(patient.code || patient.name || current.patient_name || '--')}</strong>
        <span class="clinical-brief-meta">${this.escapeHtml([patient.age ? `${patient.age} tuoi` : null, patient.sex || null].filter(Boolean).join(' | ') || 'Ho so lam sang')}</span>
      </div>
      <div class="clinical-brief-card">
        <span class="clinical-brief-label">Noi gui</span>
        <strong class="clinical-brief-title">${this.escapeHtml(primaryHospital)}</strong>
        <span class="clinical-brief-meta">${this.escapeHtml(study.region || 'Don vi chuyen ho so')}</span>
      </div>
      <div class="clinical-brief-card">
        <span class="clinical-brief-label">Ca chup</span>
        <strong class="clinical-brief-title">${this.escapeHtml(studyLabel || '--')}</strong>
        <span class="clinical-brief-meta">${this.escapeHtml(`Tai len ${study.uploaded_at || current.created_at || '--'}`)}</span>
      </div>
      <div class="clinical-brief-card accent-coral">
        <span class="clinical-brief-label">The tich 3D</span>
        <strong class="clinical-brief-title">${this.escapeHtml(volumeLabel)}</strong>
        <span class="clinical-brief-meta">${this.escapeHtml(`${surfaceLabel} dien tich be mat`)}</span>
      </div>
    `;
    return;

    host.innerHTML = `
      <div class="card">
        <h3>Hồ sơ ca bệnh</h3>
        <div class="card-item"><span>Bệnh nhân</span><strong>${this.escapeHtml(patient.name || current.patient_name || '--')}</strong></div>
        <div class="card-item"><span>Mã ca</span><strong>${this.escapeHtml(patient.code || '--')}</strong></div>
        <div class="card-item"><span>Bệnh viện</span><strong>${this.escapeHtml(patient.hospital || '--')}</strong></div>
        <div class="card-item"><span>Phương thức</span><strong>${this.escapeHtml(study.modality || current.modality || '--')}</strong></div>
        <div class="card-item"><span>Vùng khảo sát</span><strong>${this.escapeHtml(study.region || '--')}</strong></div>
        <div class="card-item"><span>Số timepoint</span><strong>${this.escapeHtml(String(study.timepoints ?? '--'))}</strong></div>
      </div>
    `;
  },

  renderViewerReference(current) {
    const host = document.getElementById('viewer-reference-panel');
    if (!host || !current?.id) return;

    const mpr = this.state.mprByCase[current.id];
    const preview = this.state.previewByCase[current.id];
    const mprLoading = !!this.state.mprLoadingByCase[current.id];
    const previewLoading = !!this.state.previewLoadingByCase[current.id];

    if (!mpr && !preview) {
      host.innerHTML = `
        <div class="reference-card reference-card-empty">
          <div class="reference-card-header">
            <div>
              <span class="reference-kicker">Lat cat tham chieu</span>
              <h3>Chua co MPR sau phan tich</h3>
            </div>
          </div>
          <p class="reference-empty-copy">${this.escapeHtml(mprLoading || previewLoading ? TEXT.previewLoading : TEXT.previewUnavailable)}</p>
        </div>
      `;
      return;
    }

    const planes = ['axial', 'coronal', 'sagittal'];
    const dominantPlane = preview?.dominant_plane || 'axial';
    const cursor = (this.state.mprCursorByCase[current.id] || mpr?.default_index || preview?.lesion_center_index || [0, 0, 0]).map((value) => Number(value || 0));
    const dominantSlice = preview?.[dominantPlane] || null;
    const lesionRatio = dominantSlice?.lesion_area_ratio ?? preview?.axial?.lesion_area_ratio ?? null;
    const canvasSize = dominantSlice ? `${dominantSlice.width || '--'} x ${dominantSlice.height || '--'}` : (mpr?.shape ? `${mpr.shape[2] || '--'} x ${mpr.shape[1] || '--'}` : '--');
    const sliceIndex = dominantSlice?.slice_index ?? (dominantPlane === 'coronal' ? cursor[1] : dominantPlane === 'sagittal' ? cursor[2] : cursor[0]);
    const peakProbability = Array.isArray(dominantSlice?.probability_pixels) && dominantSlice.probability_pixels.length
      ? Math.max(...dominantSlice.probability_pixels) / 255
      : null;

    host.innerHTML = `
      <div class="reference-card">
        <div class="reference-card-header">
          <div>
            <span class="reference-kicker">Lat cat tham chieu</span>
            <h3>${this.escapeHtml(current.patient_name || current.patient?.name || 'Ca benh moi')}</h3>
            <p>Lat ${this.escapeHtml(String(sliceIndex ?? '--'))} | Ti le ton thuong ${this.escapeHtml(this.formatPercent(lesionRatio, 2))}</p>
          </div>
          <div class="reference-legend">
            <span class="legend-pill legend-probability">Ban do xac suat</span>
            <span class="legend-pill legend-mask">Vung phan doan</span>
          </div>
        </div>
        <div class="reference-stat-grid">
          <div class="reference-stat">
            <span>Slice</span>
            <strong>${this.escapeHtml(String(sliceIndex ?? '--'))}</strong>
          </div>
          <div class="reference-stat">
            <span>Lesion ratio</span>
            <strong>${this.escapeHtml(this.formatPercent(lesionRatio, 2))}</strong>
          </div>
          <div class="reference-stat">
            <span>Canvas</span>
            <strong>${this.escapeHtml(canvasSize)}</strong>
          </div>
        </div>
        <div class="reference-plane-grid">
          ${planes.map((plane) => {
            const planeSlice = preview?.[plane] || null;
            const canvasId = `viewer-reference-${plane}-${current.id}`;
            const fallbackIndex = plane === 'axial' ? cursor[0] : plane === 'coronal' ? cursor[1] : cursor[2];
            const footer = planeSlice
              ? `${this.formatNumber(planeSlice.position_mm?.[0] ?? 0, 0)} mm -> ${this.formatNumber(planeSlice.size_mm?.[0] ?? 0, 0)} mm`
              : `Crosshair ${fallbackIndex}`;
            return `
              <div class="reference-plane">
                <div class="reference-plane-head">
                  <span>${plane}</span>
                  <strong>${this.escapeHtml(String(planeSlice?.slice_index ?? fallbackIndex))}</strong>
                </div>
                <canvas id="${canvasId}" class="reference-canvas"></canvas>
                <span class="reference-plane-foot">${this.escapeHtml(footer)}</span>
              </div>
            `;
          }).join('')}
        </div>
        <div class="reference-meta-grid">
          <div class="reference-meta-card">
            <span>Peak probability</span>
            <strong>${this.escapeHtml(this.formatPercent(peakProbability, 0))}</strong>
          </div>
          <div class="reference-meta-card">
            <span>Lesion area</span>
            <strong>${this.escapeHtml(this.formatPercent(lesionRatio, 2))}</strong>
          </div>
          <div class="reference-meta-card">
            <span>Workstation mode</span>
            <strong>Axial / Coronal / Sagittal</strong>
          </div>
        </div>
      </div>
    `;

    requestAnimationFrame(() => {
      planes.forEach((plane) => {
        const canvasId = `viewer-reference-${plane}-${current.id}`;
        if (mpr) {
          this.drawMprCanvas(canvasId, mpr, plane, cursor);
        } else if (preview?.[plane]) {
          this.drawPreviewCanvas(canvasId, preview[plane]);
        }
      });
    });
  },

  renderInterpretability(current) {
    const host = document.getElementById('interpretability-card');
    if (!host) return;

    if (!current) {
      host.innerHTML = `
        <div class="interpretability-card">
          <div class="interpretability-head">
            <strong>Interpretability se hien sau khi chon ca</strong>
            <p>Khung nay tong hop confidence, uncertainty va cac co can ra soat de bac si doc nhanh muc do tin cay cua ket qua.</p>
          </div>
        </div>
      `;
      return;
    }

    const segmentation = current.segmentation || {};
    const classification = current.classification || {};
    const ai = current.ai_summary || {};
    const notes = (Array.isArray(segmentation.diagnostic_notes) ? segmentation.diagnostic_notes : []).filter(Boolean).slice(0, 3);
    const decisionFactors = (Array.isArray(classification.decision_factors) ? classification.decision_factors : []).filter(Boolean).slice(0, 2);
    const reviewState = segmentation.review_recommended
      ? 'Cần rà soát'
      : this.formatReviewStatus(ai.review_status || 'ready');

    host.innerHTML = `
      <div class="interpretability-card">
        <div class="interpretability-head">
          <strong>Lop giai thich cua tung ca benh</strong>
          <p>Du lieu duoi day di truc tiep tu segmentation, morphology va classification de doi chieu ket qua voi chat luong study dau vao.</p>
        </div>
        <div class="interpretability-grid">
          ${this.metric('Muc tin cay', segmentation.confidence_band || '--')}
          ${this.metric('Do bat dinh', segmentation.uncertainty_score !== undefined && segmentation.uncertainty_score !== null ? this.formatNumber(segmentation.uncertainty_score, 6) : '--')}
          ${this.metric('Tin cay du doan', segmentation.prediction_confidence_score !== undefined ? `${this.formatNumber(segmentation.prediction_confidence_score, 1)}%` : '--')}
          ${this.metric('Tin cay phan loai', this.formatPercent(classification.confidence))}
          ${this.metric('Ra soat lam sang', reviewState)}
          ${this.metric('Consensus AI', this.formatPercent(ai.consensus_score))}
        </div>
        <div class="evidence-pill-row">
          <span class="evidence-pill">${this.escapeHtml(segmentation.inference_mode || 'Inference mode --')}</span>
          <span class="evidence-pill">${this.escapeHtml(segmentation.model_name || segmentation.model || 'Segmentation model --')}</span>
          <span class="evidence-pill">${this.escapeHtml(this.formatClassificationLabel(classification.label || '--'))}</span>
        </div>
        <div class="interpretability-note">
          ${(notes.length || decisionFactors.length)
            ? `<ul>${[...notes, ...decisionFactors].slice(0, 4).map((item) => `<li>${this.escapeHtml(String(item))}</li>`).join('')}</ul>`
            : 'Chua co ghi chu giai thich cho ca nay.'}
        </div>
      </div>
    `;
  },

  renderIngestion(current) {
    const ingestion = current.ingestion || {};
    const host = document.getElementById('ingestion-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Nguồn dữ liệu', this.mapLabel('sourceType', ingestion.source_type) || '--')}
        ${this.metric('Số file', ingestion.file_count ?? '--')}
        ${this.metric('Số series', ingestion.series_count ?? '--')}
        ${this.metric('Chuỗi MRI', (ingestion.series_labels || []).join(' | ') || '--')}
        ${this.metric('Độ sẵn sàng', ingestion.readiness_score ?? '--')}
        ${this.metric('Định dạng', (ingestion.accepted_formats || []).join(', ') || '--')}
      </div>
    `;
  },

  renderPreprocessing(current) {
    const preprocessing = current.preprocessing || {};
    const host = document.getElementById('preprocessing-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Voxel spacing', (preprocessing.resample_spacing_mm || []).join(' × ') || '--')}
        ${this.metric('Chuẩn hóa', preprocessing.intensity_normalization || '--')}
        ${this.metric('Chiến lược ROI', preprocessing.roi_strategy || '--')}
        ${this.metric('Chất lượng dữ liệu', preprocessing.data_quality_score ?? '--')}
      </div>
    `;
  },

  renderWorkflow(current) {
    const host = document.getElementById('workflow-steps');
    if (!host) return;
    const pipeline = Array.isArray(current.pipeline) ? current.pipeline : [];
    host.innerHTML = pipeline.map((step, index) => `
      <div class="card">
        <div class="card-item">
          <span>${index + 1}. ${this.escapeHtml(step.step || '--')}</span>
          <strong>${this.escapeHtml(this.formatPipelineStatus(step.status || '--'))}</strong>
        </div>
      </div>
    `).join('');
  },

  renderSegmentationHero(current) {
    const seg = current.segmentation || {};
    const host = document.getElementById('segmentation-hero-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        <h3>Điểm nổi bật của segmentation</h3>
        <div class="metric-grid">
          ${this.metric('Thể tích', `${this.formatNumber(seg.tumor_volume_cm3, 2)} cm³`)}
          ${this.metric('Dice nội bộ', this.formatNumber(seg.dice_score, 4))}
          ${this.metric('IoU nội bộ', this.formatNumber(seg.iou, 4))}
          ${this.metric('Chất lượng ca', this.formatNumber(seg.case_quality_score, 4))}
        </div>
      </div>
    `;
  },

  renderSegmentation(current) {
    const seg = current.segmentation || {};
    const payload = current.reconstruction?.viewer_payload || {};
    const dimensions = Array.isArray(payload.dimensions_mm) ? payload.dimensions_mm : [];
    const centroid = Array.isArray(payload.centroid_mm) ? payload.centroid_mm : [];
    const host = document.getElementById('segmentation-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        <h3>Thông số segmentation</h3>
        ${this.metric('Mô hình', seg.model_name || seg.model || '--')}
        ${this.metric('Chế độ suy luận', seg.inference_mode || '--')}
        ${this.metric('Ngưỡng', this.formatNumber(seg.threshold, 4))}
        ${this.metric('Sai lệch biên (mm)', this.formatNumber(seg.hausdorff_mm, 3))}
        ${this.metric('Tin cậy dự đoán', `${this.formatNumber(seg.prediction_confidence_score, 1)}%`)}
        ${this.metric('Tin cậy đo đạc', `${this.formatNumber(seg.measurement_confidence_score, 1)}%`)}
        ${this.metric('Mức tin cậy', seg.confidence_band || '--')}
        ${this.metric('Độ bất định', seg.uncertainty_score !== undefined && seg.uncertainty_score !== null ? this.formatNumber(seg.uncertainty_score, 6) : '--')}
        ${this.metric('Độ mở rộng tổn thương (mm)', (seg.lesion_extent_mm || []).join(' × ') || '--')}
        ${this.metric('Kích thước đầu vào', (seg.inference_shape || []).join(' × ') || '--')}
        ${this.metric('Kích thước 3D (mm)', dimensions.length ? dimensions.map((value) => this.formatNumber(value, 1)).join(' × ') : '--')}
        ${this.metric('Tâm khối (mm)', centroid.length ? centroid.map((value) => this.formatNumber(value, 1)).join(' / ') : '--')}
        ${this.renderBulletBlock('Ghi chú chất lượng', seg.diagnostic_notes, 4)}
      </div>
    `;
  },

  renderSegmentationDownloads(current) {
    const host = document.getElementById('segmentation-downloads');
    if (!host) return;
    const caseId = current.case_id || current.id;

    host.innerHTML = `
      <a class="ghost" href="/api/cases/${this.escapeHtml(caseId)}/segmentation/mask" target="_blank" rel="noopener">Tải mask</a>
      <a class="ghost" href="/api/cases/${this.escapeHtml(caseId)}/segmentation/probability-map" target="_blank" rel="noopener">Bản đồ xác suất</a>
      <a class="ghost" href="/api/cases/${this.escapeHtml(caseId)}/report.md" target="_blank" rel="noopener">Báo cáo Markdown</a>
      <a class="ghost" href="/api/cases/${this.escapeHtml(caseId)}/bundle.json" target="_blank" rel="noopener">Bundle JSON</a>
    `;
  },

  renderSegmentationPreview(current) {
    const host = document.getElementById('segmentation-preview');
    if (!host || !current?.id) return;

    const mpr = this.state.mprByCase[current.id];
    const mprLoading = !!this.state.mprLoadingByCase[current.id];
    const preview = this.state.previewByCase[current.id];
    const isLoading = !!this.state.previewLoadingByCase[current.id];
    if (!mpr && !preview) {
      host.innerHTML = `
        <div class="card">
          <h3>MPR 3 mặt phẳng</h3>
          <div class="card-item">
            <span>Trạng thái</span>
            <strong>${mprLoading || isLoading ? TEXT.previewLoading : TEXT.previewUnavailable}</strong>
          </div>
        </div>
      `;
      return;
    }

    if (mpr) {
      const shape = Array.isArray(mpr.shape) ? mpr.shape : [0, 0, 0];
      const cursor = (this.state.mprCursorByCase[current.id] || mpr.default_index || [0, 0, 0]).map((value, index) => {
        const limit = Math.max((shape[index] || 1) - 1, 0);
        return Math.max(0, Math.min(limit, Number(value || 0)));
      });
      this.state.mprCursorByCase[current.id] = cursor;

      const planes = ['axial', 'coronal', 'sagittal'];
      host.innerHTML = `
        <div class="card">
          <h3>MPR 3 mặt phẳng có crosshair đồng bộ</h3>
          <div class="card-item"><span>Tâm hiện tại</span><strong>${this.escapeHtml(cursor.join(' / '))}</strong></div>
          <div class="card-item"><span>Hộp bao</span><strong>${this.escapeHtml(`${(mpr.lesion_bbox_min || []).join(' / ')} -> ${(mpr.lesion_bbox_max || []).join(' / ')}`)}</strong></div>
          <div class="card-item"><span>Điều khiển</span><strong>Click để đặt crosshair, cuộn chuột để đổi lát cắt</strong></div>
          <div class="mpr-control-grid">
            ${['Z / Axial', 'Y / Coronal', 'X / Sagittal'].map((label, axis) => `
              <label class="mpr-slider">
                <span>${label}: ${this.escapeHtml(String(cursor[axis] ?? '--'))}</span>
                <input type="range" min="0" max="${Math.max((shape[axis] || 1) - 1, 0)}" value="${cursor[axis]}" data-case-id="${this.escapeHtml(current.id)}" data-axis="${axis}">
              </label>
            `).join('')}
          </div>
          <div class="mpr-grid">
            ${planes.map((plane) => `
              <div class="mpr-panel">
                <span class="mpr-panel-label">${plane}</span>
                <canvas id="mpr-${plane}-${current.id}" class="mpr-canvas"></canvas>
              </div>
            `).join('')}
          </div>
        </div>
      `;

      requestAnimationFrame(() => {
        this.bindMprControls(current.id);
        planes.forEach((plane) => this.drawMprCanvas(`mpr-${plane}-${current.id}`, mpr, plane, cursor));
      });
      return;
    }

    const planes = ['axial', 'coronal', 'sagittal'];
    host.innerHTML = `
      <div class="card">
        <h3>Preview đa mặt phẳng</h3>
        <div class="card-item"><span>Mặt phẳng nổi bật</span><strong>${this.escapeHtml(preview.dominant_plane || '--')}</strong></div>
        <div class="card-item"><span>Tâm tổn thương</span><strong>${this.escapeHtml((preview.lesion_center_index || []).join(' / ') || '--')}</strong></div>
        <div class="card-item"><span>Hộp bao</span><strong>${this.escapeHtml(`${(preview.lesion_bbox_min || []).join(' / ')} -> ${(preview.lesion_bbox_max || []).join(' / ')}`)}</strong></div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px;">
          ${planes.map((plane) => {
            const slice = preview[plane] || {};
            const canvasId = `preview-${plane}-${current.id}`;
            return `
              <div>
                <span style="display:block; margin-bottom: 8px; color: var(--muted); font-size: 12px; text-transform: capitalize;">${plane} | lát ${this.escapeHtml(String(slice.slice_index ?? '--'))}</span>
                <canvas id="${canvasId}" style="width:100%; border-radius: 16px; background:#081018;"></canvas>
                <span style="display:block; margin-top: 6px; color: var(--muted); font-size: 12px;">Tỉ lệ tổn thương: ${this.formatNumber(slice.lesion_area_ratio, 4)}</span>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    requestAnimationFrame(() => {
      planes.forEach((plane) => {
        this.drawPreviewCanvas(`preview-${plane}-${current.id}`, preview[plane]);
      });
    });
  },

  bindMprControls(caseId) {
    const host = document.getElementById('segmentation-preview');
    if (!host) return;

    host.querySelectorAll('.mpr-slider input[type="range"]').forEach((input) => {
      if (input.dataset.bound) return;
      input.dataset.bound = '1';
      input.addEventListener('input', (event) => {
        const axis = Number(event.target.dataset.axis || 0);
        const cursor = [...(this.state.mprCursorByCase[caseId] || [0, 0, 0])];
        cursor[axis] = Number(event.target.value || 0);
        this.state.mprCursorByCase[caseId] = cursor;
        if (this.state.selectedCase?.id === caseId) {
          this.renderSegmentationPreview(this.state.selectedCase);
        }
      });
    });
  },

  mprIndex(shape, z, y, x) {
    return (z * shape[1] * shape[2]) + (y * shape[2]) + x;
  },

  mprPlaneDimensions(shape, plane) {
    if (plane === 'axial') return { width: shape[2], height: shape[1] };
    if (plane === 'coronal') return { width: shape[2], height: shape[0] };
    return { width: shape[1], height: shape[0] };
  },

  mprCrosshairForPlane(plane, cursor) {
    if (plane === 'axial') return { x: cursor[2], y: cursor[1] };
    if (plane === 'coronal') return { x: cursor[2], y: cursor[0] };
    return { x: cursor[1], y: cursor[0] };
  },

  updateMprCursor(caseId, plane, canvasX, canvasY) {
    const mpr = this.state.mprByCase[caseId];
    if (!mpr) return;
    const shape = Array.isArray(mpr.shape) ? mpr.shape : [0, 0, 0];
    const cursor = [...(this.state.mprCursorByCase[caseId] || mpr.default_index || [0, 0, 0])];

    if (plane === 'axial') {
      cursor[1] = Math.max(0, Math.min(shape[1] - 1, canvasY));
      cursor[2] = Math.max(0, Math.min(shape[2] - 1, canvasX));
    } else if (plane === 'coronal') {
      cursor[0] = Math.max(0, Math.min(shape[0] - 1, canvasY));
      cursor[2] = Math.max(0, Math.min(shape[2] - 1, canvasX));
    } else {
      cursor[0] = Math.max(0, Math.min(shape[0] - 1, canvasY));
      cursor[1] = Math.max(0, Math.min(shape[1] - 1, canvasX));
    }

    this.state.mprCursorByCase[caseId] = cursor;
    if (this.state.selectedCase?.id === caseId) {
      this.renderSegmentationPreview(this.state.selectedCase);
    }
  },

  renderMprPixelsSync(mpr, plane, cursor) {
    const shape = Array.isArray(mpr.shape) ? mpr.shape : [0, 0, 0];
    const { width, height } = this.mprPlaneDimensions(shape, plane);
    const volumeBytes = mpr.volumeBytes;
    const maskBytes = mpr.maskBytes;
    const probabilityBytes = mpr.probabilityBytes;
    const pixels = new Uint8ClampedArray(width * height * 4);

    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width; col += 1) {
        let z = cursor[0];
        let y = cursor[1];
        let x = cursor[2];
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
        const voxelIndex = this.mprIndex(shape, z, y, x);
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
    return { width, height, pixels };
  },

  async renderMprPixels(mpr, plane, cursor) {
    const shape = Array.isArray(mpr.shape) ? mpr.shape : [0, 0, 0];
    const { width, height } = this.mprPlaneDimensions(shape, plane);
    if (!width || !height) return { width: 0, height: 0, pixels: new Uint8ClampedArray(0) };

    if (!this.state.mprWorker) {
      return this.renderMprPixelsSync(mpr, plane, cursor);
    }

    const jobId = `mpr_${++this.state.mprWorkerSeq}`;
    const payload = {
      jobId,
      shape,
      plane,
      cursor: [...cursor],
      volumeBytes: mpr.volumeBytes,
      maskBytes: mpr.maskBytes,
      probabilityBytes: mpr.probabilityBytes,
    };

    return new Promise((resolve, reject) => {
      this.state.mprWorkerJobs[jobId] = { resolve, reject };
      try {
        this.state.mprWorker.postMessage(payload);
      } catch (error) {
        delete this.state.mprWorkerJobs[jobId];
        reject(error);
      }
    }).catch(() => this.renderMprPixelsSync(mpr, plane, cursor));
  },

  async drawMprCanvas(canvasId, mpr, plane, cursor) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !mpr) return;
    const shape = Array.isArray(mpr.shape) ? mpr.shape : [0, 0, 0];
    const { width, height, pixels } = await this.renderMprPixels(mpr, plane, cursor);
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(width, height);
    image.data.set(pixels);
    ctx.putImageData(image, 0, 0);
    const crosshair = this.mprCrosshairForPlane(plane, cursor);
    ctx.save();
    ctx.strokeStyle = '#ff9b54';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(crosshair.x + 0.5, 0);
    ctx.lineTo(crosshair.x + 0.5, height);
    ctx.moveTo(0, crosshair.y + 0.5);
    ctx.lineTo(width, crosshair.y + 0.5);
    ctx.stroke();
    ctx.restore();

    if (!canvas.dataset.bound) {
      canvas.dataset.bound = '1';
      canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.max(0, Math.min(width - 1, Math.round(((event.clientX - rect.left) / rect.width) * (width - 1))));
        const y = Math.max(0, Math.min(height - 1, Math.round(((event.clientY - rect.top) / rect.height) * (height - 1))));
        this.updateMprCursor(this.state.selectedCase?.id, plane, x, y);
      });
      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const currentCaseId = this.state.selectedCase?.id;
        if (!currentCaseId) return;
        const currentCursor = [...(this.state.mprCursorByCase[currentCaseId] || cursor)];
        const axis = plane === 'axial' ? 0 : plane === 'coronal' ? 1 : 2;
        const limit = Math.max((shape[axis] || 1) - 1, 0);
        currentCursor[axis] = Math.max(0, Math.min(limit, currentCursor[axis] + (event.deltaY > 0 ? 1 : -1)));
        this.state.mprCursorByCase[currentCaseId] = currentCursor;
        if (this.state.selectedCase?.id === currentCaseId) {
          this.renderSegmentationPreview(this.state.selectedCase);
        }
      }, { passive: false });
    }
  },

  drawPreviewCanvas(canvasId, preview) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !preview) return;
    const width = Number(preview.width || 0);
    const height = Number(preview.height || 0);
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(width, height);
    const imagePixels = Array.isArray(preview.image_pixels) ? preview.image_pixels : [];
    const maskPixels = Array.isArray(preview.mask_pixels) ? preview.mask_pixels : [];
    const probabilityPixels = Array.isArray(preview.probability_pixels) ? preview.probability_pixels : [];

    for (let i = 0; i < width * height; i += 1) {
      const offset = i * 4;
      const base = Number(imagePixels[i] || 0);
      const maskValue = Number(maskPixels[i] || 0);
      const probabilityValue = Number(probabilityPixels[i] || 0);

      let red = base;
      let green = base;
      let blue = base;

      if (probabilityValue > 0) {
        red = Math.min(255, red + Math.round(probabilityValue * 0.45));
        green = Math.min(255, green + Math.round(probabilityValue * 0.12));
        blue = Math.max(0, blue - Math.round(probabilityValue * 0.18));
      }

      if (maskValue > 0) {
        red = 36;
        green = 214;
        blue = 188;
      }

      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }

    ctx.putImageData(image, 0, 0);
  },

  renderMorphology(current) {
    const morph = current.morphology || {};
    const host = document.getElementById('morphology-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Độ bất thường bờ', this.formatNumber(morph.surface_irregularity_index, 4))}
        ${this.metric('Fractal dimension', this.formatNumber(morph.fractal_dimension, 4))}
        ${this.metric('Phân tán gradient', this.formatNumber(morph.gradient_distribution, 4))}
        ${this.metric('Texture signature', morph.texture_signature || '--')}
        ${this.metric('Độ bất đối xứng', morph.asymmetry || '--')}
        ${this.metric('Độ phức tạp bờ', this.formatNumber(morph.margin_complexity_index, 4))}
        ${this.metric('Heterogeneity hướng tâm', this.formatNumber(morph.radial_heterogeneity_index, 4))}
        ${this.metric('Heterogeneity cường độ', this.formatNumber(morph.intensity_heterogeneity_index, 4))}
      </div>
    `;
  },

  renderClassification(current) {
    const classification = current.classification || {};
    const progression = current.progression || {};
    const ai = current.ai_summary || {};
    const host = document.getElementById('classification-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        <h3>Nguy cơ và diễn tiến</h3>
        ${this.metric('Phân loại', this.formatClassificationLabel(classification.label || '--'))}
        ${this.metric('Điểm nguy cơ', `${this.formatNumber(classification.risk_score, 1)}/100`)}
        ${this.metric('Độ tin cậy', this.formatPercent(classification.confidence))}
        ${this.metric('Độ ổn định', this.formatPercent(classification.reliability_score))}
        ${this.metric('Tăng thể tích 6 tháng', this.formatNumber(progression.volume_change_6m_percent, 2) + '%')}
        ${this.metric('Nguy cơ chuyển ác tính', this.formatPercent(progression.malignant_transition_risk))}
        ${this.metric('Consensus', this.formatPercent(ai.consensus_score))}
        ${this.metric('Trạng thái rà soát', this.formatReviewStatus(ai.review_status || '--'))}
        ${this.renderBulletBlock('Cờ lâm sàng', classification.clinical_flags, 4)}
        ${this.renderBulletBlock('Yếu tố quyết định', classification.decision_factors, 4)}
      </div>
    `;
  },

  renderTreatment(current) {
    const treatment = current.treatment_simulation || {};
    const options = Array.isArray(treatment.options) ? treatment.options : [];
    const host = document.getElementById('treatment-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Khuyến nghị chính', treatment.recommended_strategy || '--')}
        ${this.metric('Độ tin cậy mô phỏng', this.formatPercent(treatment.simulation_confidence))}
        ${this.metric('Lý do', treatment.rationale || '--')}
        ${this.renderBulletBlock(
          'Phương án thay thế',
          options.map((item) => `${item.name}: thay đổi ${item.expected_volume_shift_percent}% | giảm rủi ro ${item.risk_reduction_score}`),
          4,
        )}
      </div>
    `;
  },

  renderFollowup(current) {
    const followup = current.follow_up || {};
    const reminder = current.follow_up_reminder || {};
    const teleconsultation = current.teleconsultation || {};
    const ai = current.ai_summary || {};
    const host = document.getElementById('followup-card');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Ngày tái khám', followup.next_recommended_date || '--')}
        ${this.metric('Kênh xử trí', followup.channel || '--')}
        ${this.metric('Lý do', followup.reason || '--')}
        ${this.metric('Nhắc lịch', reminder.message || '--')}
        ${this.metric('Mức nhắc', reminder.severity || '--')}
        ${this.metric('Teleconsultation', teleconsultation.status || '--')}
        ${this.metric('Dải tăng trưởng', this.formatGrowthBand(current.progression?.growth_velocity_band || '--'))}
        ${this.renderBulletBlock('Hành động đề xuất', ai.recommended_actions, 4)}
        ${this.renderBulletBlock('Cảnh báo', ai.alerts, 4)}
      </div>
    `;
  },

  renderTeleconsultationBoard(current) {
    const board = current.teleconsultation_board || {};
    const notes = Array.isArray(board.notes) ? board.notes : [];
    const host = document.getElementById('teleconsult-board');
    if (!host) return;

    host.innerHTML = `
      <div class="card">
        ${this.metric('Trạng thái gói', board.packet_ready ? 'Sẵn sàng' : 'Chưa đủ dữ liệu')}
        ${this.metric('Ưu tiên', board.board_priority || '--')}
        ${this.metric('Tóm tắt', board.summary || '--')}
        ${this.renderBulletBlock('Chuyên khoa khuyến nghị', board.recommended_specialists, 5)}
        ${this.renderBulletBlock(
          'Ghi chú hội chẩn',
          notes.map((note) => {
            const sign = note.sign_off ? ' | đã ký duyệt' : '';
            return `${note.author || '--'} (${note.role || '--'}) | ${note.created_at || '--'}${sign}: ${note.content || '--'}`;
          }),
          5,
        )}
      </div>
    `;
  },

  renderTimeline(current) {
    const timeline = Array.isArray(current.patient_timeline) ? current.patient_timeline : [];
    const host = document.getElementById('timeline-card');
    if (!host) return;

    if (!timeline.length) {
      host.innerHTML = `
        <div class="card">
          <div class="card-item">
            <span>Trạng thái</span>
            <strong>Chưa có mốc theo dõi trước đó</strong>
          </div>
        </div>
      `;
      return;
    }

    host.innerHTML = `
      <div class="card">
        ${timeline.slice(-6).map((item, index) => `
          <div class="card" style="margin-top: ${index === 0 ? '0' : '12px'};">
            <div class="card-item"><span>Ngày</span><strong>${this.escapeHtml(item.date || '--')}</strong></div>
            <div class="card-item"><span>Ca</span><strong>${this.escapeHtml(item.case_id || '--')}</strong></div>
            <div class="card-item"><span>Phương thức</span><strong>${this.escapeHtml(item.modality || '--')}</strong></div>
            <div class="card-item"><span>Thể tích</span><strong>${this.formatNumber(item.tumor_volume_cm3, 2)} cm³</strong></div>
            <div class="card-item"><span>Điểm nguy cơ</span><strong>${this.formatNumber(item.risk_score, 1)}/100</strong></div>
            <div class="card-item"><span>Trạng thái</span><strong>${this.escapeHtml(this.formatReviewStatus(item.review_status || '--'))}</strong></div>
          </div>
        `).join('')}
      </div>
    `;
  },

  renderPortfolio() {
    const host = document.getElementById('portfolio-card');
    if (!host || !this.state.platform) return;
    const portfolio = this.state.platform.portfolio || {};

    host.innerHTML = `
      <div class="card">
        <div class="benchmark-grid">
          <div class="benchmark-card-item">
            <div class="benchmark-card-top"><span>Số ca</span><small>Registry</small></div>
            <strong class="benchmark-card-value">${this.escapeHtml(String(portfolio.case_count ?? '--'))}</strong>
            <small>Danh mục đang hoạt động</small>
          </div>
          <div class="benchmark-card-item">
            <div class="benchmark-card-top"><span>Hoàn tất trung bình</span><small>Pipeline</small></div>
            <strong class="benchmark-card-value">${this.formatNumber(portfolio.avg_completion_score, 1)}</strong>
            <small>Điểm chất lượng tổng hợp</small>
          </div>
          <div class="benchmark-card-item">
            <div class="benchmark-card-top"><span>Ưu tiên trung bình</span><small>Triage</small></div>
            <strong class="benchmark-card-value">${this.formatNumber(portfolio.avg_priority_score, 1)}</strong>
            <small>Mức ưu tiên danh mục</small>
          </div>
        </div>
        <div class="confidence-chip-row" style="margin-top: 12px;">
          <div class="confidence-chip">
            <span>Sẵn sàng hội chẩn</span>
            <strong>${this.escapeHtml(String(portfolio.ready_for_board ?? '--'))}</strong>
          </div>
          <div class="confidence-chip">
            <span>Cần chú ý</span>
            <strong>${this.escapeHtml(String(portfolio.needs_attention ?? '--'))}</strong>
          </div>
        </div>
        ${this.renderBulletBlock('Ca ưu tiên', portfolio.top_case_ids, 5)}
      </div>
    `;
  },

  renderAnalysisStatus(message, type = 'info') {
    const host = document.getElementById('analysis-job-status');
    if (!host) return;
    host.innerHTML = `
      <div class="card">
        <div class="card-item">
          <span>Trạng thái</span>
          <strong>${this.escapeHtml(this.formatStatusType(type))}</strong>
        </div>
        <div class="card-item">
          <span>Thông điệp</span>
          <strong>${this.escapeHtml(message || '--')}</strong>
        </div>
      </div>
    `;
  },

  renderViewer(current) {
    const host = document.getElementById('viewer-3d');
    if (!host || !window.THREE || !window.TumorViewer) return;

    if (!this.state.viewer || !this.state.viewer.container) {
      host.innerHTML = '';
      this.state.viewer = new TumorViewer('viewer-3d');
      this.state.viewerEnhancements = new TumorViewerEnhancements(this.state.viewer);
    }

    const payload = current.reconstruction?.viewer_payload;
    const overlays = current.id ? this.state.viewerOverlaysByCase[current.id] : null;
    const mprPayload = current.id ? this.state.mprByCase[current.id] : null;
    const stats = {
      volume: current.tumor_volume || current.segmentation?.tumor_volume_cm3 || 0,
      surface: current.tumor_surface || current.reconstruction?.surface_area_cm2 || 0,
      density: current.tumor_density || 0,
    };

    if (payload) {
      this.state.viewer.loadPayload(payload, stats, overlays, mprPayload);
    } else if (mprPayload?.shape && mprPayload?.voxel_spacing_mm) {
      const shape = mprPayload.shape.map((value) => Number(value || 0));
      const spacing = mprPayload.voxel_spacing_mm.map((value) => Number(value || 1));
      const fallbackPayload = {
        mode: 'point-cloud',
        points: [],
        bounds_min_mm: [0, 0, 0],
        bounds_max_mm: [
          Math.max((shape[2] - 1) * spacing[2], 1),
          Math.max((shape[1] - 1) * spacing[1], 1),
          Math.max((shape[0] - 1) * spacing[0], 1),
        ],
        centroid_mm: [
          Math.max((shape[2] - 1) * spacing[2] * 0.5, 0),
          Math.max((shape[1] - 1) * spacing[1] * 0.5, 0),
          Math.max((shape[0] - 1) * spacing[0] * 0.5, 0),
        ],
        dimensions_mm: [
          Math.max(shape[2] * spacing[2], 1),
          Math.max(shape[1] * spacing[1], 1),
          Math.max(shape[0] * spacing[0], 1),
        ],
        voxel_spacing_mm: spacing,
        principal_axes: [],
        mesh_quality_score: 0,
        vertex_count: 0,
        face_count: 0,
      };
      this.state.viewer.loadPayload(fallbackPayload, stats, overlays, mprPayload);
    } else {
      this.state.viewer.clearObjects();
      this.state.viewer.updateStats();
    }

    const exportButton = document.getElementById('export-stl-btn');
    if (exportButton) exportButton.disabled = !this.state.viewer.tumorMesh;
  },

  destroyViewer() {
    if (this.state.viewer?.dispose) this.state.viewer.dispose();
    const host = document.getElementById('viewer-3d');
    if (host) host.innerHTML = '';
    this.state.viewer = null;
    this.state.viewerEnhancements = null;
  },

  kpiCard(label, value, subvalue) {
    return `
      <div class="kpi-card">
        <span class="kpi-label">${this.escapeHtml(label)}</span>
        <strong class="kpi-value">${this.escapeHtml(this.valueOrDash(value))}</strong>
        <span class="kpi-subvalue">${this.escapeHtml(this.valueOrDash(subvalue))}</span>
      </div>
    `;
  },

  metric(label, value) {
    return `
      <div class="metric">
        <span>${this.escapeHtml(label)}</span>
        <strong>${this.escapeHtml(this.valueOrDash(value))}</strong>
      </div>
    `;
  },

  renderBulletBlock(title, items, limit = 4) {
    const safeItems = (Array.isArray(items) ? items : []).filter(Boolean).slice(0, limit);
    if (!safeItems.length) return '';
    return `
      <div class="card" style="margin-top: 12px;">
        <h3>${this.escapeHtml(title)}</h3>
        <ul>
          ${safeItems.map((item) => `<li>${this.escapeHtml(String(item))}</li>`).join('')}
        </ul>
      </div>
    `;
  },

  formatNumber(value, digits = 2) {
    if (value === null || value === undefined || value === '') return '--';
    const number = Number(value);
    if (Number.isNaN(number)) return String(value);
    return number.toFixed(digits);
  },

  formatPercent(value, digits = 0) {
    if (value === null || value === undefined || value === '') return '--';
    const number = Number(value);
    if (Number.isNaN(number)) return String(value);
    const normalized = Math.abs(number) <= 1 ? number * 100 : number;
    return `${normalized.toFixed(digits)}%`;
  },

  formatClassificationLabel(value) {
    return this.mapLabel('classification', value) || value || '--';
  },

  formatReviewStatus(value) {
    return this.mapLabel('reviewStatus', value) || value || '--';
  },

  formatTriageLevel(value) {
    return this.mapLabel('triageLevel', value) || value || '--';
  },

  formatGrowthBand(value) {
    return this.mapLabel('growthBand', value) || value || '--';
  },

  formatPipelineStatus(value) {
    if (value === 'completed') return 'Hoàn tất';
    if (value === 'running') return 'Đang chạy';
    if (value === 'queued') return 'Chờ xử lý';
    return value || '--';
  },

  formatStatusType(value) {
    if (value === 'info') return 'Đang xử lý';
    if (value === 'success') return 'Hoàn tất';
    if (value === 'error') return 'Lỗi';
    return value || '--';
  },

  mapLabel(group, value) {
    return LABEL_MAPS[group]?.[value];
  },

  valueOrDash(value) {
    if (value === null || value === undefined || value === '') return '--';
    return String(value);
  },

  escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  },

  showMessage(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast visible ${type}`;
    clearTimeout(this.messageTimeout);
    this.messageTimeout = setTimeout(() => {
      toast.classList.remove('visible');
    }, 3200);
  },
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.init());
} else {
  app.init();
}
