const API_BASE =
  window.location.protocol.startsWith("http") && window.location.port === "8000"
    ? window.location.origin
    : "http://127.0.0.1:8000";

const SCAN_INTERVAL_MS = 350;
const VIEW_STORAGE_KEY = "ai_attendance_active_view";
const DEPARTMENT_ORDER = [
  "B.tech IT",
  "B.tech AI&DS",
  "B.E CSE",
  "B.E CSE (CS)",
  "B.E ECE",
  "B.E R&A",
];

const state = {
  staffName: localStorage.getItem("staffName") || "",
  currentView: sessionStorage.getItem(VIEW_STORAGE_KEY) || "dashboard",
  summary: null,
  students: [],
  attendance: [],
  stream: null,
  selectedCameraId: "",
  cameraLabel: "",
  stopRequested: true,
  scanBusy: false,
  scanTimer: null,
  recentEvents: {},
  editingStudentId: null,
  liveness: {
    candidateName: "",
    count: 0,
    lastEyeState: "open",
    motionScore: 0,
    lastCenter: null,
    verified: false,
    blinkComplete: false,
  },
};

const els = {
  apiHealth: document.getElementById("apiHealth"),
  pageTitle: document.getElementById("pageTitle"),
  staffLabel: document.getElementById("staffLabel"),
  logoutBtn: document.getElementById("logoutBtn"),
  exportPdfBtn: document.getElementById("exportPdfBtn"),
  exportModal: document.getElementById("exportModal"),
  closeExportModalBtn: document.getElementById("closeExportModalBtn"),
  exportDeptSelect: document.getElementById("exportDeptSelect"),
  confirmExportPdfBtn: document.getElementById("confirmExportPdfBtn"),
  openStudentModalBtn: document.getElementById("openStudentModalBtn"),
  closeStudentModalBtn: document.getElementById("closeStudentModalBtn"),
  studentModal: document.getElementById("studentModal"),
  studentModalTitle: document.getElementById("studentModalTitle"),
  studentForm: document.getElementById("studentForm"),
  studentRecordId: document.getElementById("studentRecordId"),
  studentIdInput: document.getElementById("studentIdInput"),
  studentNameInput: document.getElementById("studentNameInput"),
  studentClassInput: document.getElementById("studentClassInput"),
  studentDeptInput: document.getElementById("studentDeptInput"),
  studentBatchInput: document.getElementById("studentBatchInput"),
  studentImageInput: document.getElementById("studentImageInput"),
  studentImageHint: document.getElementById("studentImageHint"),
  navButtons: [...document.querySelectorAll(".nav-btn")],
  views: [...document.querySelectorAll(".view")],
  totalStudents: document.getElementById("totalStudents"),
  presentStudents: document.getElementById("presentStudents"),
  absentStudents: document.getElementById("absentStudents"),
  onDutyStudents: document.getElementById("onDutyStudents"),
  refreshDashboardBtn: document.getElementById("refreshDashboardBtn"),
  todayAttendanceTable: document.getElementById("todayAttendanceTable"),
  cameraStatus: document.getElementById("cameraStatus"),
  cameraSelect: document.getElementById("cameraSelect"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  stopCameraBtn: document.getElementById("stopCameraBtn"),
  cameraFeed: document.getElementById("cameraFeed"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  captureCanvas: document.getElementById("captureCanvas"),
  livenessState: document.getElementById("livenessState"),
  recognitionState: document.getElementById("recognitionState"),
  lastMatchState: document.getElementById("lastMatchState"),
  eventFeed: document.getElementById("eventFeed"),
  studentSearchInput: document.getElementById("studentSearchInput"),
  autoArrangeBtn: document.getElementById("autoArrangeBtn"),
  studentTable: document.getElementById("studentTable"),
  refreshAttendanceBtn: document.getElementById("refreshAttendanceBtn"),
  attendanceHistoryTable: document.getElementById("attendanceHistoryTable"),
};

function apiUrl(path) {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function studentImageSrc(student) {
  if (student?.image_url) {
    return escapeHtml(apiUrl(student.image_url));
  }
  return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='96' viewBox='0 0 96 96'%3E%3Crect width='96' height='96' rx='18' fill='%23e8f0ff'/%3E%3Ccircle cx='48' cy='36' r='16' fill='%23759fd4'/%3E%3Cpath d='M24 76c4-14 16-22 24-22s20 8 24 22' fill='%23759fd4'/%3E%3C/svg%3E";
}

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function ensureAuth() {
  if (!state.staffName) {
    window.location.replace("/frontend/login.html");
    return false;
  }
  return true;
}

async function requestJson(path, options) {
  const response = await fetch(apiUrl(path), options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function pushEvent(message) {
  const item = document.createElement("div");
  item.className = "event-item";
  item.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong><span>${escapeHtml(message)}</span>`;
  els.eventFeed.prepend(item);
  while (els.eventFeed.children.length > 30) {
    els.eventFeed.removeChild(els.eventFeed.lastChild);
  }
}

function shouldLogRecognition(name, status) {
  const key = `${String(name || "").trim().toLowerCase()}::${status}`;
  const now = Date.now();
  const last = state.recentEvents[key];
  if (last && now - last < 12000) {
    return false;
  }
  state.recentEvents[key] = now;
  return true;
}

function setView(viewName) {
  state.currentView = viewName;
  sessionStorage.setItem(VIEW_STORAGE_KEY, viewName);
  els.navButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewName);
  });
  els.views.forEach((view) => {
    view.classList.toggle("active", view.id === `view-${viewName}`);
  });

  const titles = {
    dashboard: "Attendance Dashboard",
    monitor: "Live Monitor",
    students: "Student Directory",
    attendance: "Attendance Records",
  };
  setText(els.pageTitle, titles[viewName] || "AI Attendance");
}

function openStudentModal(student = null) {
  state.editingStudentId = student?.id ?? null;
  setText(els.studentModalTitle, student ? "Edit Student" : "Add Student");
  els.studentRecordId.value = student?.id ?? "";
  els.studentIdInput.value = student?.student_id ?? "";
  els.studentNameInput.value = student?.name ?? "";
  els.studentClassInput.value = student?.class_name ?? "";
  els.studentDeptInput.value = student?.dept ?? "";
  els.studentBatchInput.value = student?.batch ?? "";
  els.studentImageInput.value = "";
  els.studentImageHint.textContent = student
    ? "Upload a new image only if you want to update the stored face profile."
    : "Student image is required for new students.";
  els.studentModal.classList.add("open");
}

function closeStudentModal() {
  state.editingStudentId = null;
  els.studentForm.reset();
  els.studentRecordId.value = "";
  els.studentModal.classList.remove("open");
}

function openExportModal() {
  els.exportDeptSelect.value = "all";
  els.exportModal.classList.add("open");
}

function closeExportModal() {
  els.exportModal.classList.remove("open");
}

function renderEmptyRow(colspan, message) {
  return `<tr><td colspan="${colspan}" class="empty-state">${escapeHtml(message)}</td></tr>`;
}

function renderSummary() {
  const summary = state.summary;
  if (!summary) {
    return;
  }

  setText(els.totalStudents, String(summary.total_students || 0));
  setText(els.presentStudents, String(summary.present_students || 0));
  setText(els.absentStudents, String(summary.absent_students || 0));
  setText(els.onDutyStudents, String(summary.on_duty_students || 0));
  els.todayAttendanceTable.innerHTML = summary.attendance?.length
    ? summary.attendance.map((row) => `
        <tr>
          <td>${escapeHtml(row.name)}</td>
          <td>${escapeHtml(row.student_id || "-")}</td>
          <td>${escapeHtml(row.class_name || "-")}</td>
          <td>${escapeHtml(row.dept || "-")}</td>
          <td>${escapeHtml(row.time || "-")}</td>
        </tr>
      `).join("")
    : renderEmptyRow(5, "No attendance marked today yet.");
}

function renderStudents() {
  const query = els.studentSearchInput.value.trim().toLowerCase();
  const rows = state.students.filter((student) => {
    if (!query) {
      return true;
    }
      return [
      student.student_id,
      student.name,
      student.class_name,
      student.dept,
      student.batch,
    ].some((value) => String(value || "").toLowerCase().includes(query));
  });

  if (!rows.length) {
    els.studentTable.innerHTML = renderEmptyRow(8, "No students found.");
    return;
  }

  const groups = new Map();
  DEPARTMENT_ORDER.forEach((dept) => groups.set(dept, []));
  rows.forEach((student) => {
    const dept = student.dept || "No Department";
    if (!groups.has(dept)) {
      groups.set(dept, []);
    }
    groups.get(dept).push(student);
  });

  const sections = Array.from(groups.entries()).filter(([, students]) => students.length);
  els.studentTable.innerHTML = sections.map(([dept, students]) => `
      <tr class="group-row">
        <td colspan="8">
          <div class="group-row-content">
            <span>${escapeHtml(dept)}</span>
            <button type="button" class="mini-btn" data-action="auto-arrange-dept" data-dept="${escapeHtml(dept)}">
              Auto Arrange ${escapeHtml(dept)}
            </button>
          </div>
        </td>
      </tr>
      ${students.map((student) => `
          <tr>
            <td>
              <img class="student-photo" src="${studentImageSrc(student)}" alt="${escapeHtml(student.name)}">
            </td>
          <td>${escapeHtml(student.student_id)}</td>
          <td>${escapeHtml(student.name)}</td>
          <td>${escapeHtml(student.class_name || "-")}</td>
          <td>${escapeHtml(student.dept || "-")}</td>
          <td>${escapeHtml(student.batch || "-")}</td>
          <td>
            <span class="inline-badge ${student.on_duty ? "success" : "muted"}">
              ${student.on_duty ? "Yes" : "No"}
            </span>
          </td>
          <td>
            <div class="inline-actions">
              <button type="button" class="mini-btn" data-action="edit-student" data-id="${student.id}">Edit</button>
              <button type="button" class="mini-btn danger" data-action="delete-student" data-id="${student.id}">Delete</button>
            </div>
          </td>
        </tr>
      `).join("")}
    `).join("");
}

function renderAttendance() {
  const rows = state.attendance;

  els.attendanceHistoryTable.innerHTML = rows.length
    ? rows.map((row) => `
        <tr>
          <td>${escapeHtml(row.name)}</td>
          <td>${escapeHtml(row.student_id || "-")}</td>
          <td>${escapeHtml(row.date || "-")}</td>
          <td>${escapeHtml(row.slot_label || "-")}</td>
          <td>${escapeHtml(row.time || "-")}</td>
          <td>${escapeHtml(row.class_name || "-")}</td>
          <td>${escapeHtml(row.dept || "-")}</td>
          <td>
            <div class="inline-actions">
              <button type="button" class="mini-btn" data-action="toggle-od" data-id="${row.id}">
                ${row.on_duty ? "Remove OD" : "Mark OD"}
              </button>
              <button type="button" class="mini-btn danger" data-action="delete-attendance" data-id="${row.id}">Delete</button>
            </div>
          </td>
        </tr>
      `).join("")
    : renderEmptyRow(8, "No attendance records found.");
}

function resetBlinkState() {
  state.liveness.candidateName = "";
  state.liveness.count = 0;
  state.liveness.lastEyeState = "open";
  state.liveness.motionScore = 0;
  state.liveness.lastCenter = null;
  state.liveness.verified = false;
  state.liveness.blinkComplete = false;
  setText(els.livenessState, "Waiting for 2 eye blinks");
}

function updateBlinkState(candidate) {
  if (!candidate?.name) {
    resetBlinkState();
    return 0;
  }

  const currentName = String(candidate.name).trim();
  if (state.liveness.candidateName !== currentName) {
    state.liveness.candidateName = currentName;
    state.liveness.count = 0;
    state.liveness.lastEyeState = "open";
    state.liveness.motionScore = 0;
    state.liveness.lastCenter = null;
    state.liveness.verified = false;
    state.liveness.blinkComplete = false;
  }

  const eyeState = candidate.eye_state === "closed" ? "closed" : "open";
  if (!state.liveness.blinkComplete && state.liveness.lastEyeState === "open" && eyeState === "closed") {
    state.liveness.count += 1;
  }

  state.liveness.lastEyeState = eyeState;

  if (candidate.bbox?.length === 4) {
    const [x1, y1, x2, y2] = candidate.bbox;
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;
    const faceSpan = Math.max(x2 - x1, y2 - y1, 1);
    if (state.liveness.lastCenter) {
      const dx = centerX - state.liveness.lastCenter.x;
      const dy = centerY - state.liveness.lastCenter.y;
      const normalizedMove = Math.sqrt(dx * dx + dy * dy) / faceSpan;
      if (normalizedMove > 0.015) {
        state.liveness.motionScore = Math.min(1, state.liveness.motionScore + normalizedMove);
      }
    }
    state.liveness.lastCenter = { x: centerX, y: centerY };
  }

  if (state.liveness.count >= 2) {
    state.liveness.blinkComplete = true;
    state.liveness.count = 2;
  }

  state.liveness.verified = state.liveness.count >= 2 && state.liveness.motionScore >= 0.7;

  if (state.liveness.verified) {
    setText(els.livenessState, `${currentName}: live blink verified`);
  } else if (state.liveness.blinkComplete) {
    setText(
      els.livenessState,
      `${currentName}: 2 / 2 blinks locked, waiting for motion ${Math.round(state.liveness.motionScore * 100)} / 70%`
    );
  } else {
    setText(
      els.livenessState,
      `${currentName}: blink ${Math.min(state.liveness.count, 2)} / 2, motion ${Math.round(state.liveness.motionScore * 100)} / 70%`
    );
  }

  return state.liveness.count;
}

function clearOverlay() {
  const ctx = els.overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
}

function drawDetections(detections = []) {
  const video = els.cameraFeed;
  const canvas = els.overlayCanvas;
  const ctx = canvas.getContext("2d");

  if (!video.videoWidth || !video.videoHeight) {
    clearOverlay();
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = '700 14px "Segoe UI", sans-serif';

  for (const item of detections) {
    if (!item.bbox || item.bbox.length !== 4) {
      continue;
    }
    const [x1, y1, x2, y2] = item.bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    const isMobile = item.label === "Mobile";

    const known = !isMobile && item.name && item.name !== "Unknown";
    ctx.strokeStyle = isMobile
    ? "#ff0000"
    : known
        ? "#24a35a"
        : "#f47d20";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, width, height);
    const label = isMobile
    ? `Mobile ${item.confidence}%`
    : known
        ? `${item.name} ${item.confidence}%`
        : "Unknown";
    ctx.fillStyle = isMobile
    ? "rgba(255, 0, 0, 0.94)"
    : known
        ? "rgba(36, 163, 90, 0.94)"
        : "rgba(244, 125, 32, 0.94)";
    ctx.fillRect(x1, Math.max(10, y1 - 34), 220, 28);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x1 + 10, Math.max(28, y1 - 15));
  }
}

async function checkHealth() {
  try {
    const result = await requestJson("/health");
    setText(els.apiHealth, `Backend ready - ${result.students} profiles`);
  } catch (error) {
    setText(els.apiHealth, "Backend offline");
    pushEvent(error.message);
  }
}

async function loadSummary() {
  state.summary = await requestJson("/dashboard/summary");
  renderSummary();
}

async function loadStudents() {
  state.students = await requestJson("/students");
  renderStudents();
}

async function loadAttendance() {
  state.attendance = await requestJson("/attendance");
  renderAttendance();
}

async function refreshAll() {
  await Promise.all([loadSummary(), loadStudents(), loadAttendance()]);
}

async function saveStudent(event) {
  event.preventDefault();

  const imageFile = els.studentImageInput.files?.[0];
  const isEditing = Boolean(state.editingStudentId);
  const numericStudentId = els.studentIdInput.value.replace(/\D+/g, "");
  if (!isEditing && !imageFile) {
    window.alert("Student image is required.");
    return;
  }
  if (!numericStudentId) {
    window.alert("Student ID must contain numbers only.");
    return;
  }

  const formData = new FormData();
  formData.append("student_id", numericStudentId);
  formData.append("name", els.studentNameInput.value.trim());
  formData.append("class_name", els.studentClassInput.value.trim());
  formData.append("dept", els.studentDeptInput.value.trim());
  formData.append("batch", els.studentBatchInput.value.trim());
  if (imageFile) {
    formData.append("image", imageFile);
  }

  try {
    const result = await requestJson(
      isEditing ? `/students/${state.editingStudentId}` : "/students",
      {
        method: isEditing ? "PUT" : "POST",
        body: formData,
      }
    );
    pushEvent(result.message || (isEditing ? "Student updated." : "Student registered."));
    closeStudentModal();
    await refreshAll();
  } catch (error) {
    window.alert(error.message);
  }
}

async function deleteStudent(studentId) {
  const student = state.students.find((item) => String(item.id) === String(studentId));
  if (!student) {
    return;
  }
  if (!window.confirm(`Delete ${student.name}?`)) {
    return;
  }
  await requestJson(`/students/${studentId}`, { method: "DELETE" });
  pushEvent(`Deleted student: ${student.name}`);
  await refreshAll();
}

async function deleteAttendance(recordId) {
  if (!window.confirm("Delete this attendance record?")) {
    return;
  }
  await requestJson(`/attendance/${recordId}`, { method: "DELETE" });
  pushEvent("Attendance record deleted.");
  await refreshAll();
}

async function toggleOd(recordId) {
  const result = await requestJson(`/attendance/${recordId}/toggle-od`, { method: "POST" });
  pushEvent(`${result.student_name}: ${result.on_duty ? "OD enabled" : "OD removed"}.`);
  await refreshAll();
}

async function autoArrangeStudentIds() {
  const result = await requestJson("/students/reassign-ids", { method: "POST" });
  pushEvent(result.message || "Student IDs rearranged.");
  await refreshAll();
}

async function autoArrangeDepartmentStudentIds(dept) {
  const result = await requestJson(`/students/reassign-ids?dept=${encodeURIComponent(dept)}`, { method: "POST" });
  pushEvent(result.message || `Student IDs rearranged for ${dept}.`);
  await refreshAll();
}

function exportPdfByDepartment() {
  const dept = els.exportDeptSelect.value || "all";
  window.open(apiUrl(`/export-pdf?dept=${encodeURIComponent(dept)}`), "_blank", "noopener");
  closeExportModal();
}

async function refreshCameraList(preferredDeviceId = state.selectedCameraId) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    els.cameraSelect.innerHTML = `<option value="">Camera access unavailable</option>`;
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");

  els.cameraSelect.innerHTML = cameras.length
    ? cameras.map((camera, index) => `
        <option value="${escapeHtml(camera.deviceId)}">${escapeHtml(camera.label || `Camera ${index + 1}`)}</option>
      `).join("")
    : `<option value="">No cameras found</option>`;

  if (preferredDeviceId && cameras.some((camera) => camera.deviceId === preferredDeviceId)) {
    els.cameraSelect.value = preferredDeviceId;
  } else if (cameras.length) {
    els.cameraSelect.value = cameras[0].deviceId;
  }

  state.selectedCameraId = els.cameraSelect.value || "";
}

function releaseCurrentStream() {
  if (!state.stream) {
    els.cameraFeed.srcObject = null;
    clearOverlay();
    return;
  }

  state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  els.cameraFeed.srcObject = null;
  clearOverlay();
}

function scheduleNextScan(delay = SCAN_INTERVAL_MS) {
  window.clearTimeout(state.scanTimer);
  if (!state.stopRequested && state.stream) {
    state.scanTimer = window.setTimeout(() => {
      runScan().catch((error) => {
        pushEvent(error.message || "Scanning failed");
      });
    }, delay);
  }
}

async function startCamera() {
  state.stopRequested = false;
  state.scanBusy = false;
  window.clearTimeout(state.scanTimer);
  releaseCurrentStream();
  resetBlinkState();

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      deviceId: state.selectedCameraId ? { exact: state.selectedCameraId } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 24, max: 30 },
    },
  });

  state.stream = stream;
  const videoTrack = stream.getVideoTracks()[0];
  state.cameraLabel = videoTrack?.label || "Connected camera";
  state.selectedCameraId = videoTrack?.getSettings?.().deviceId || state.selectedCameraId;
  els.cameraFeed.srcObject = stream;
  await els.cameraFeed.play().catch(() => {});
  setText(els.cameraStatus, "Camera running");
  setText(els.recognitionState, "Searching face...");
  setText(els.lastMatchState, "Waiting for live face");
  pushEvent(`Camera started: ${state.cameraLabel}`);
  await refreshCameraList(state.selectedCameraId);
  scheduleNextScan(220);
}

function stopCamera() {
  state.stopRequested = true;
  state.scanBusy = false;
  window.clearTimeout(state.scanTimer);
  releaseCurrentStream();
  resetBlinkState();
  setText(els.cameraStatus, "Camera stopped");
  setText(els.recognitionState, "No attendance marked yet");
  setText(els.lastMatchState, "None");
}

function captureFrameBlob() {
  return new Promise((resolve, reject) => {
    const video = els.cameraFeed;
    if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      reject(new Error("Camera frame is not ready"));
      return;
    }

    const canvas = els.captureCanvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Unable to capture frame"));
        return;
      }
      resolve(blob);
    }, "image/jpeg", 0.88);
  });
}

function handleRecognitionResponse(result) {
  const detections = [
    ...(result.results || []),
    ...(result.low_confidence || []),
    ...(result.mobile_detections || [])
];
  drawDetections(detections);

  const primaryMatch = (result.results || [])[0];
  function handleRecognitionResponse(result) {

  // =========================
  // MOBILE BLOCKED
  // =========================

  if (result.status === "blocked") {

    clearOverlay();

    drawDetections(result.mobile_detections || []);

    setText(
      els.recognitionState,
      result.reason || "Mobile phone detected"
    );

    setText(
      els.livenessState,
      "Spoof attack blocked"
    );

    setText(
      els.lastMatchState,
      "Mobile detected"
    );

    pushEvent("Mobile phone detected. Attendance blocked.");

    return;
  }

  // =========================
  // NORMAL DETECTIONS
  // =========================

  const detections = [
    ...(result.results || []),
    ...(result.low_confidence || [])
  ];

  drawDetections(detections);

  const primaryMatch = (result.results || [])[0];
}
  if (result.status === "searching" || result.status === "no_match") {
    setText(els.recognitionState, result.reason || "Searching face...");
    setText(els.lastMatchState, "No proper match yet");
    if (!state.liveness.blinkComplete) {
      resetBlinkState();
    } else {
      setText(
        els.livenessState,
        `${state.liveness.candidateName || "Candidate"}: 2 / 2 blinks locked, waiting for motion ${Math.round(state.liveness.motionScore * 100)} / 70%`
      );
    }
    return;
  }

  if (result.status === "awaiting_blink") {
    const blinkCount = updateBlinkState(primaryMatch);
    setText(els.recognitionState, result.reason || "Complete 2 blinks and 70% motion");
    setText(els.lastMatchState, primaryMatch ? `${primaryMatch.name} ${primaryMatch.confidence}%` : "Face matched");
    if (state.liveness.verified || state.liveness.blinkComplete || blinkCount >= 2) {
      setText(
        els.livenessState,
        state.liveness.verified
          ? "Live check passed. Marking attendance..."
          : `${state.liveness.candidateName}: 2 / 2 blinks locked, waiting for motion ${Math.round(state.liveness.motionScore * 100)} / 70%`
      );
    }
    return;
  }

  if (result.status === "marked" || result.status === "already_marked") {
    if (primaryMatch) {
      setText(els.lastMatchState, `${primaryMatch.name} ${primaryMatch.confidence}%`);
    }
    setText(els.recognitionState, result.reason || "Attendance processed");
    setText(els.livenessState, "Live blink verified");

    if (primaryMatch && shouldLogRecognition(primaryMatch.name, result.status)) {
      pushEvent(
        result.status === "marked"
          ? `Attendance marked for ${primaryMatch.name}.`
          : `${primaryMatch.name} already marked today.`
      );
    }

    refreshAll().catch(() => {});
    window.setTimeout(() => {
      if (!state.stopRequested) {
        resetBlinkState();
      }
    }, 600);
  }
}

async function runScan() {
  if (!state.stream || state.stopRequested) {
    return;
  }

  if (state.scanBusy) {
    scheduleNextScan(120);
    return;
  }

  state.scanBusy = true;
  try {
    const blob = await captureFrameBlob();
    const data = new FormData();
    data.append("file", blob, "frame.jpg");
    data.append("blink_verified", String(state.liveness.count >= 2));
    data.append("motion_verified", String(state.liveness.motionScore >= 0.7));
    data.append("camera_label", state.cameraLabel || "");

    const result = await requestJson("/recognize", {
      method: "POST",
      body: data,
    });

    handleRecognitionResponse(result);
  } catch (error) {
    pushEvent(error.message || "Scanning failed");
  } finally {
    state.scanBusy = false;
    if (state.stream && !state.stopRequested) {
      scheduleNextScan(220);
    }
  }
}

function handleTableActions(event) {
  const actionElement = event.target.closest("[data-action]");
  if (!actionElement) {
    return;
  }

  const action = actionElement.dataset.action;
  const id = actionElement.dataset.id;

  if (action === "edit-student") {
    const student = state.students.find((item) => String(item.id) === String(id));
    if (student) {
      openStudentModal(student);
    }
    return;
  }

  if (action === "delete-student") {
    deleteStudent(id).catch((error) => window.alert(error.message));
    return;
  }

  if (action === "delete-attendance") {
    deleteAttendance(id).catch((error) => window.alert(error.message));
    return;
  }

  if (action === "auto-arrange-dept") {
    autoArrangeDepartmentStudentIds(actionElement.dataset.dept).catch((error) => window.alert(error.message));
    return;
  }

  if (action === "toggle-od") {
    toggleOd(id).catch((error) => window.alert(error.message));
  }
}

function bindEvents() {
  els.navButtons.forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  els.openStudentModalBtn.addEventListener("click", () => openStudentModal());
  els.closeStudentModalBtn.addEventListener("click", closeStudentModal);
  els.studentModal.addEventListener("click", (event) => {
    if (event.target === els.studentModal) {
      closeStudentModal();
    }
  });
  els.exportPdfBtn.addEventListener("click", openExportModal);
  els.closeExportModalBtn.addEventListener("click", closeExportModal);
  els.confirmExportPdfBtn.addEventListener("click", exportPdfByDepartment);
  els.exportModal.addEventListener("click", (event) => {
    if (event.target === els.exportModal) {
      closeExportModal();
    }
  });

  els.studentForm.addEventListener("submit", saveStudent);
  els.logoutBtn.addEventListener("click", () => {
    stopCamera();
    localStorage.removeItem("staffName");
    window.location.replace("/frontend/login.html");
  });

  els.refreshDashboardBtn.addEventListener("click", () => refreshAll().catch((error) => pushEvent(error.message)));
  els.refreshAttendanceBtn.addEventListener("click", () => loadAttendance().catch((error) => pushEvent(error.message)));
  els.autoArrangeBtn.addEventListener("click", () => {
    autoArrangeStudentIds().catch((error) => window.alert(error.message));
  });
  els.studentSearchInput.addEventListener("input", renderStudents);
  els.studentTable.addEventListener("click", handleTableActions);
  els.attendanceHistoryTable.addEventListener("click", handleTableActions);

  els.cameraSelect.addEventListener("change", () => {
    state.selectedCameraId = els.cameraSelect.value;
  });
  els.startCameraBtn.addEventListener("click", () => {
    startCamera().catch((error) => {
      pushEvent(error.message);
      setText(els.cameraStatus, "Unable to start");
    });
  });
  els.stopCameraBtn.addEventListener("click", stopCamera);
  window.addEventListener("beforeunload", stopCamera);
}

async function initialize() {
  if (!ensureAuth()) {
    return;
  }

  setText(els.staffLabel, state.staffName);
  bindEvents();
  setView(state.currentView || "dashboard");
  await Promise.all([
    checkHealth(),
    refreshAll(),
    refreshCameraList().catch((error) => {
      setText(els.cameraStatus, "Camera access unavailable");
      pushEvent(error.message || "Unable to enumerate cameras");
    }),
  ]);
  pushEvent(`Staff login successful for ${state.staffName}.`);
}

initialize().catch((error) => {
  pushEvent(error.message || "Initialization failed");
});
