function formatDecisionStatus(value) {
  const mapping = {
    escalate_now: "Cần hội chẩn ngay",
    expedited_review: "Rà soát ưu tiên",
    monitor_with_plan: "Theo dõi có kế hoạch",
  };
  return mapping[value] || humanizeKey(value);
}

function formatConfidenceLabel(value) {
  const mapping = {
    segmentation: "Phân đoạn",
    classification: "Phân loại",
    temporal: "Diễn tiến",
    data_readiness: "Độ sẵn sàng dữ liệu",
  };
  return mapping[value] || humanizeKey(value);
}

function formatFactorImpact(value) {
  const mapping = {
    pressure: "Tăng áp lực",
    support: "Tăng hỗ trợ",
    review: "Cần rà soát",
  };
  return mapping[value] || humanizeKey(value);
}

function formatBenchmarkMetric(metric) {
  if (!metric) return "--";
  const unit = metric.unit && metric.unit !== "index" ? ` ${metric.unit}` : "";
  const digits = metric.unit === "index" ? 3 : metric.unit === "/100" ? 1 : 2;
  return `${formatMaybeNumber(metric.value, digits)}${unit}`;
}

function decisionTone(status) {
  if (status === "escalate_now") return "critical";
  if (status === "expedited_review") return "high";
  return "low";
}

const baseSetDashboardSkeletonClinicalDepth = setDashboardSkeleton;
setDashboardSkeleton = function setDashboardSkeletonClinicalDepth() {
  baseSetDashboardSkeletonClinicalDepth();
  setPanelSkeleton("benchmark-card", 3);
  setPanelSkeleton("decision-support-card", 3);
  setPanelSkeleton("longitudinal-card", 3);
};

const baseRenderCaseDetailsClinicalDepth = renderCaseDetails;
renderCaseDetails = function renderCaseDetailsWithClinicalDepth() {
  baseRenderCaseDetailsClinicalDepth();

  const benchmarkHost = document.getElementById("benchmark-card");
  const decisionHost = document.getElementById("decision-support-card");
  const longitudinalHost = document.getElementById("longitudinal-card");
  const item = state.cases.find((entry) => entry.case_id === state.selectedCaseId);

  if (!benchmarkHost || !decisionHost || !longitudinalHost) return;

  if (!item) {
    const emptyMarkup = `
      <div class="empty-state empty-state-soft">
        <strong>Chọn hồ sơ để xem chi tiết</strong>
        <span>Đối chiếu, ưu tiên và diễn tiến sẽ hiển thị tại đây.</span>
      </div>
    `;
    benchmarkHost.innerHTML = emptyMarkup;
    decisionHost.innerHTML = emptyMarkup;
    longitudinalHost.innerHTML = emptyMarkup;
    return;
  }

  const benchmark = item.benchmark || {};
  const decisionSupport = item.decision_support || {};
  const longitudinalStory = item.longitudinal_story || {};
  const viewerCaption = document.querySelector(".viewer-caption");
  const tone = priorityTone(item.insight?.priority_band);
  const decisionStatusTone = decisionTone(decisionSupport.overall_status);
  const benchmarkMetrics = benchmark.benchmark_metrics || [];
  const confidenceBreakdown = Object.entries(decisionSupport.confidence_breakdown || {});
  const timelinePoints = longitudinalStory.timeline_points || [];
  const maxVolume = Math.max(1, ...timelinePoints.map((point) => Number(point.volume_cm3) || 0));

  if (viewerCaption && item.reconstruction) {
    viewerCaption.textContent = "Dựng hình 3D";
  }

  benchmarkHost.innerHTML = `
    <div class="diagnosis-banner diagnosis-banner--${tone}">
      <div>
        <span>Vị thế ca bệnh</span>
        <strong>${formatValue(benchmark.case_mix_position)}</strong>
      </div>
      <div>
        <span>Ca tương đồng</span>
        <strong>${benchmark.similar_case_ids?.length || 0}</strong>
      </div>
      <div>
        <span>Nhóm tham chiếu</span>
        <strong>${formatValue(benchmark.cohort_label)}</strong>
      </div>
    </div>
    <div class="inference-note">${benchmark.summary || "Chưa có benchmark nội bộ."}</div>
    <div class="benchmark-grid">
      ${benchmarkMetrics.map((metric) => `
        <div class="benchmark-card-item">
          <div class="benchmark-card-top">
            <span>${metric.label}</span>
            <strong>Bách phân vị ${metric.percentile}</strong>
          </div>
          <div class="benchmark-card-value">${formatBenchmarkMetric(metric)}</div>
          <div class="signal-track"><div class="signal-fill" style="width:${Math.max(6, metric.percentile || 0)}%"></div></div>
          <small>Trung vị ${formatMaybeNumber(metric.cohort_median, metric.unit === "index" ? 3 : 2)}${metric.unit === "index" ? "" : ` ${metric.unit}`}</small>
        </div>
      `).join("")}
    </div>
    ${(benchmark.alerts || []).map((note) => `<div class="inference-note">Tín hiệu nội bộ: ${note}</div>`).join("")}
    ${benchmark.similar_case_ids?.length ? `<div class="inference-note">Ca tương đồng: ${benchmark.similar_case_ids.join(", ")}</div>` : ""}
  `;

  decisionHost.innerHTML = `
    <div class="diagnosis-banner diagnosis-banner--${decisionStatusTone}">
      <div>
        <span>Trạng thái</span>
        <strong>${formatDecisionStatus(decisionSupport.overall_status)}</strong>
      </div>
      <div>
        <span>Áp lực phân luồng</span>
        <strong>${formatScore(decisionSupport.triage_pressure_score)}</strong>
      </div>
      <div>
        <span>Khuyến nghị</span>
        <strong>${formatValue(decisionSupport.recommendation)}</strong>
      </div>
    </div>
    <div class="confidence-chip-row">
      ${confidenceBreakdown.map(([label, value]) => `
        <div class="confidence-chip">
          <span>${formatConfidenceLabel(label)}</span>
          <strong>${formatScore(value)}</strong>
        </div>
      `).join("")}
    </div>
    ${(decisionSupport.decision_factors || []).map((factor) => `
      <div class="signal-card">
        <strong>${factor.factor}</strong>
        <span>Tác động: ${formatFactorImpact(factor.impact)}</span>
        <div>Điểm: ${formatMaybeNumber(factor.score, 1)}</div>
        <div>${factor.evidence}</div>
      </div>
    `).join("")}
    ${(decisionSupport.escalation_reasons || []).map((reason) => `<div class="check-row done"><strong>Cần hội chẩn</strong><span>${reason}</span></div>`).join("")}
    ${(decisionSupport.blockers || []).map((reason) => `<div class="check-row"><strong>Điểm nghẽn</strong><span>${reason}</span></div>`).join("")}
  `;

  longitudinalHost.innerHTML = `
    <div class="diagnosis-banner diagnosis-banner--${decisionStatusTone}">
      <div>
        <span>Kiểu diễn tiến</span>
        <strong>${formatValue(longitudinalStory.growth_pattern)}</strong>
      </div>
      <div>
        <span>Cửa sổ xử trí</span>
        <strong>${formatValue(longitudinalStory.treatment_window)}</strong>
      </div>
      <div>
        <span>Mốc kế tiếp</span>
        <strong>${formatValue(longitudinalStory.next_milestone)}</strong>
      </div>
    </div>
    <div class="inference-note">${longitudinalStory.trajectory_summary || "Chưa có tóm tắt diễn tiến."}</div>
    <div class="trend-strip">
      ${timelinePoints.map((point) => `
        <div class="trend-point trend-point--${point.status}">
          <span>${point.label}</span>
          <div class="trend-bar-shell">
            <div class="trend-bar" style="height:${Math.max(10, ((Number(point.volume_cm3) || 0) / maxVolume) * 100)}%"></div>
          </div>
          <strong>${formatMaybeNumber(point.volume_cm3, 2)}</strong>
          <small>${point.date}</small>
        </div>
      `).join("")}
    </div>
    ${(timelinePoints || []).map((point) => `
      <div class="signal-row">
        <span>${point.label}</span>
        <div class="signal-track"><div class="signal-fill" style="width:${Math.max(6, Number(point.risk_score) || 0)}%"></div></div>
        <strong>${formatMaybeNumber(point.risk_score, 1)}</strong>
      </div>
    `).join("")}
    ${(longitudinalStory.watch_items || []).map((itemText) => `<div class="inference-note">Điểm cần theo dõi: ${itemText}</div>`).join("")}
  `;

  if (typeof pruneOperationalChrome === "function") {
    pruneOperationalChrome();
  }
  if (typeof pruneOperationalNotes === "function") {
    pruneOperationalNotes();
  }
};
