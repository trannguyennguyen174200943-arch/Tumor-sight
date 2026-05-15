// TumorSight Notification Center
class NotificationCenter {
  constructor() {
    this.notifications = [];
    this.maxItems = 50;
    this.unreadCount = 0;
    this.container = null;
    this.bellBadge = null;
    this.panelOpen = false;
    this.init();
  }

  init() {
    this.injectPanel();
    this.injectBell();
  }

  injectBell() {
    const brandBar = document.querySelector('.brand-bar');
    if (!brandBar) return;
    const bellWrap = document.createElement('div');
    bellWrap.className = 'notif-bell-wrap';
    bellWrap.id = 'notif-bell';
    bellWrap.innerHTML = `
      <button class="notif-bell-btn" title="Thông báo" aria-label="Trung tâm thông báo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
          <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
        </svg>
        <span class="notif-badge" id="notif-badge" style="display:none;">0</span>
      </button>
    `;
    brandBar.appendChild(bellWrap);
    this.bellBadge = document.getElementById('notif-badge');
    bellWrap.querySelector('.notif-bell-btn').addEventListener('click', () => this.toggle());
  }

  injectPanel() {
    const panel = document.createElement('div');
    panel.className = 'notif-panel';
    panel.id = 'notif-panel';
    panel.innerHTML = `
      <div class="notif-panel-header">
        <h3>Thông báo</h3>
        <button class="notif-clear-btn" id="notif-clear" title="Xoá tất cả">Xoá tất cả</button>
      </div>
      <div class="notif-list" id="notif-list">
        <p class="notif-empty">Chưa có thông báo.</p>
      </div>
    `;
    document.body.appendChild(panel);
    this.container = document.getElementById('notif-list');
    document.getElementById('notif-clear')?.addEventListener('click', () => this.clearAll());
    document.addEventListener('click', (e) => {
      if (this.panelOpen && !panel.contains(e.target) && !e.target.closest('#notif-bell')) {
        this.close();
      }
    });
  }

  toggle() {
    this.panelOpen ? this.close() : this.open();
  }

  open() {
    const panel = document.getElementById('notif-panel');
    if (panel) panel.classList.add('open');
    this.panelOpen = true;
    this.markAllRead();
  }

  close() {
    const panel = document.getElementById('notif-panel');
    if (panel) panel.classList.remove('open');
    this.panelOpen = false;
  }

  add(type, title, message, severity = 'info') {
    const item = {
      id: `notif_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      type,
      title,
      message,
      severity,
      read: false,
      timestamp: new Date(),
    };
    this.notifications.unshift(item);
    if (this.notifications.length > this.maxItems) {
      this.notifications = this.notifications.slice(0, this.maxItems);
    }
    this.unreadCount++;
    this.updateBadge();
    this.renderList();

    if (severity === 'critical' && 'Notification' in window && Notification.permission === 'granted') {
      new Notification(`TumorSight: ${title}`, { body: message, icon: '/static/icon.png' });
    }

    return item;
  }

  addCaseAlert(caseName, triageLevel) {
    const severityMap = { Critical: 'critical', High: 'warning', Moderate: 'info' };
    const sev = severityMap[triageLevel] || 'info';
    const titleMap = { Critical: 'Ca khẩn cấp', High: 'Ca ưu tiên cao', Moderate: 'Ca mới' };
    this.add('case', titleMap[triageLevel] || 'Ca mới', `${caseName} — Mức ưu tiên: ${triageLevel}`, sev);
  }

  addFollowUpReminder(caseName, dueInDays, overdue) {
    const sev = overdue ? 'warning' : 'info';
    const msg = overdue
      ? `${caseName} đã quá hạn tái khám ${Math.abs(dueInDays)} ngày`
      : `${caseName} cần tái khám trong ${dueInDays} ngày`;
    this.add('followup', overdue ? 'Quá hạn tái khám' : 'Nhắc lịch tái khám', msg, sev);
  }

  addSystemNotification(message) {
    this.add('system', 'Hệ thống', message, 'info');
  }

  markAllRead() {
    this.notifications.forEach((n) => (n.read = true));
    this.unreadCount = 0;
    this.updateBadge();
    this.renderList();
  }

  clearAll() {
    this.notifications = [];
    this.unreadCount = 0;
    this.updateBadge();
    this.renderList();
  }

  updateBadge() {
    if (!this.bellBadge) return;
    if (this.unreadCount > 0) {
      this.bellBadge.textContent = this.unreadCount > 9 ? '9+' : String(this.unreadCount);
      this.bellBadge.style.display = '';
    } else {
      this.bellBadge.style.display = 'none';
    }
  }

  severityIcon(severity) {
    const icons = {
      critical: '🔴',
      warning: '🟡',
      info: '🟢',
      success: '✅',
    };
    return icons[severity] || '🔵';
  }

  formatTime(date) {
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);
    if (diff < 60) return 'Vừa xong';
    if (diff < 3600) return `${Math.floor(diff / 60)} phút trước`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} giờ trước`;
    return date.toLocaleDateString('vi-VN');
  }

  renderList() {
    if (!this.container) return;
    if (!this.notifications.length) {
      this.container.innerHTML = '<p class="notif-empty">Chưa có thông báo.</p>';
      return;
    }
    this.container.innerHTML = this.notifications.map((n) => `
      <div class="notif-item ${n.read ? '' : 'unread'} notif-${n.severity}" data-id="${n.id}">
        <span class="notif-icon">${this.severityIcon(n.severity)}</span>
        <div class="notif-body">
          <strong class="notif-title">${this.escapeHtml(n.title)}</strong>
          <span class="notif-msg">${this.escapeHtml(n.message)}</span>
          <span class="notif-time">${this.formatTime(n.timestamp)}</span>
        </div>
      </div>
    `).join('');
  }

  escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  }
}

window.NotificationCenter = NotificationCenter;
