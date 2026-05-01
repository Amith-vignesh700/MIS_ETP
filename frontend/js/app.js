/**
 * app.js — Ultrasound Elastography Clinical Dashboard
 * =====================================================
 * Handles: .mat file upload, /api/predict/ API call,
 * 3-class result rendering (Benign / Precancerous / Malignant),
 * strain heatmap display, confidence bars, protocol card
 * highlighting, parallax, scroll-reveal, mobile menu, toasts.
 *
 * API Contract (POST /api/predict/):
 *   Request : FormData { mat_file: <File> }
 *   Response: {
 *     status: "success" | "error",
 *     diagnosis: "Benign" | "Precancerous" | "Malignant",
 *     class_index: 0 | 1 | 2,
 *     confidence_scores: { Benign: f, Precancerous: f, Malignant: f },
 *     strain_heatmap_b64: string,   // base64-encoded PNG
 *     dmd_rank: number,
 *     rf_shape: [D, W, F],
 *     filename: string,
 *     warning: string | null,
 *     message: string               // present on error
 *   }
 */

'use strict';

/* ══════════════════════════════════════════════════════════════════════════════
   CONFIGURATION
   ══════════════════════════════════════════════════════════════════════════════ */

const CONFIG = {
  API_ENDPOINT: '/api/predict/',
  REQUEST_TIMEOUT_MS: 180_000,       // 3 min — DMD + CNN-LSTM can be slow on CPU
  MAX_FILE_SIZE_MB: 200,
  ALLOWED_EXTENSIONS: ['.mat'],
  TOAST_DURATION_MS: 3500,
  DIET_SCROLL_DELAY_MS: 2500,        // delay before scrolling to protocol section
  CONFIDENCE_ANIM_DELAY_MS: 150,     // stagger delay between confidence bar animations

  // Per-class display config
  CLASSES: {
    Benign: {
      barClass: 'bar-benign',
      badgeClass: 'bg-blue-100 text-blue-800',
      barColor: '#3b82f6',
      protocolId: 'protocol-benign',
      ringColor: 'ring-blue-400',
      recommendation:
        `<strong class='block mb-2 text-blue-300 text-base flex items-center gap-2'>
           <svg class='w-5 h-5 flex-shrink-0' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
             <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
               d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'/>
           </svg>
           Routine Surveillance
         </strong>
         No signs of malignancy detected. Tissue mechanics indicate uniform stiffness 
         consistent with benign tissue. Schedule a follow-up elastography scan in 12 months 
         and maintain regular clinical monitoring.`,
    },
    Precancerous: {
      barClass: 'bar-precancerous',
      badgeClass: 'bg-yellow-100 text-yellow-800',
      barColor: '#f59e0b',
      protocolId: 'protocol-precancerous',
      ringColor: 'ring-yellow-400',
      recommendation:
        `<strong class='block mb-2 text-yellow-300 text-base flex items-center gap-2'>
           <svg class='w-5 h-5 flex-shrink-0' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
             <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
               d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'/>
           </svg>
           Close Monitoring Required
         </strong>
         Heterogeneous stiffness profile detected — early-stage dysplasia pattern. 
         Recommend confirmatory ultrasound biopsy and multidisciplinary team review. 
         Repeat elastography in 3–6 months to assess progression or regression.`,
    },
    Malignant: {
      barClass: 'bar-malignant',
      badgeClass: 'bg-red-100 text-red-800',
      barColor: '#8b2635',
      protocolId: 'protocol-malignant',
      ringColor: 'ring-dr-accent',
      recommendation:
        `<strong class='block mb-2 text-red-300 text-base flex items-center gap-2'>
           <svg class='w-5 h-5 flex-shrink-0' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
             <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
               d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'/>
           </svg>
           Urgent Clinical Action Required
         </strong>
         Stiff, irregular lesion with high elastographic contrast ratio — malignant pattern detected. 
         Immediate oncology referral and confirmatory core-needle biopsy are required. 
         Staging MRI/CT imaging recommended without delay.`,
    },
  },

  // Simulated loading status messages — timed to match real pipeline steps
  LOADING_STAGES: [
    { text: 'Loading .mat RF Signal…',           subtext: 'scipy.io / h5py parsing',             delayMs: 0     },
    { text: 'Applying 2D-DFT Bandpass Filter…',  subtext: 'Removing DC drift & HF artefacts',    delayMs: 2500  },
    { text: 'Building Hankel Matrix…',           subtext: 'Time-delay embedding (Unit 1)',        delayMs: 5500  },
    { text: 'Running Dynamic Mode Decomposition…', subtext: 'SVD → eigenmodes → artefact rejection', delayMs: 9000  },
    { text: 'CNN-LSTM Forward Pass…',            subtext: 'Spatial CNN + Bi-LSTM temporal track', delayMs: 14000 },
    { text: 'Generating Strain Heatmap…',        subtext: 'Displacement field → Base64 PNG',     delayMs: 20000 },
  ],
};

/* ══════════════════════════════════════════════════════════════════════════════
   APPLICATION STATE
   ══════════════════════════════════════════════════════════════════════════════ */

const state = {
  fileSelected: false,
  currentFile: null,
  isProcessing: false,
  mobileMenuOpen: false,
  loadingTimers: [],             // track setTimeout IDs so we can clear on cancel
};

/* ══════════════════════════════════════════════════════════════════════════════
   DOM REFERENCES (resolved once on DOMContentLoaded)
   ══════════════════════════════════════════════════════════════════════════════ */

let DOM = {};

function resolveDom() {
  DOM = {
    // Toast
    toastContainer:    document.getElementById('toast-container'),

    // Mobile menu
    mobileMenu:        document.getElementById('mobile-menu'),
    menuBtn:           document.getElementById('mobile-menu-btn'),
    closeBtn:          document.getElementById('close-menu-btn'),
    mobileLinks:       document.querySelectorAll('.mobile-link'),

    // Navbar
    navbar:            document.getElementById('navbar'),

    // Upload zone
    dropZone:          document.getElementById('drop-zone'),
    fileInput:         document.getElementById('file-input'),
    uploadUi:          document.getElementById('upload-ui'),
    fileInfoCard:      document.getElementById('file-info-card'),
    fileNameDisplay:   document.getElementById('file-name-display'),
    fileSizeDisplay:   document.getElementById('file-size-display'),
    scanLine:          document.getElementById('scan-line'),
    loadingOverlay:    document.getElementById('loading-overlay'),
    loadingStatus:     document.getElementById('loading-status'),
    loadingSubstatus:  document.getElementById('loading-substatus'),
    scanBtn:           document.getElementById('scan-btn'),

    // Result panel
    resultPanel:       document.getElementById('result-panel'),
    gradeBadge:        document.getElementById('grade-badge'),
    gradeText:         document.getElementById('grade-text'),

    // Confidence bars
    barBenign:         document.getElementById('bar-benign'),
    barPrecancerous:   document.getElementById('bar-precancerous'),
    barMalignant:      document.getElementById('bar-malignant'),
    pctBenign:         document.getElementById('pct-benign'),
    pctPrecancerous:   document.getElementById('pct-precancerous'),
    pctMalignant:      document.getElementById('pct-malignant'),

    // Stats
    processTime:       document.getElementById('process-time'),
    dmdRank:           document.getElementById('dmd-rank'),
    rfShape:           document.getElementById('rf-shape'),

    // Heatmap
    heatmapContainer:  document.getElementById('heatmap-container'),
    strainHeatmap:     document.getElementById('strain-heatmap'),
    heatmapDownloadBtn: document.getElementById('heatmap-download-btn'),

    // Recommendation
    recommendationBox: document.getElementById('recommendation-box'),

    // Protocol cards
    protocolBenign:      document.getElementById('protocol-benign'),
    protocolPrecancerous: document.getElementById('protocol-precancerous'),
    protocolMalignant:   document.getElementById('protocol-malignant'),

    // Parallax elements
    parallaxEls:       document.querySelectorAll('.parallax'),

    // Reveal elements
    revealEls:         document.querySelectorAll('.reveal'),
  };
}

/* ══════════════════════════════════════════════════════════════════════════════
   UTILITY FUNCTIONS
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Format bytes into a human-readable string (KB / MB).
 */
function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

/**
 * Format elapsed milliseconds into a readable string.
 */
function formatElapsed(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Format an RF shape array [D, W, F] into a readable dimension string.
 */
function formatRfShape(shape) {
  if (!Array.isArray(shape) || shape.length < 3) return '—';
  return `${shape[0]} × ${shape[1]} × ${shape[2]}`;
}

/**
 * Validate that the selected file is a .mat file within the size limit.
 * Returns an error string or null if valid.
 */
function validateMatFile(file) {
  if (!file) return 'No file selected.';

  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
  if (!CONFIG.ALLOWED_EXTENSIONS.includes(ext)) {
    return `Invalid file type "${ext}". Only .mat files are accepted.`;
  }

  const sizeMB = file.size / 1_048_576;
  if (sizeMB > CONFIG.MAX_FILE_SIZE_MB) {
    return `File too large (${sizeMB.toFixed(1)} MB). Maximum allowed size is ${CONFIG.MAX_FILE_SIZE_MB} MB.`;
  }

  return null; // valid
}

/**
 * Clear all pending loading-stage setTimeout timers.
 */
function clearLoadingTimers() {
  state.loadingTimers.forEach(clearTimeout);
  state.loadingTimers = [];
}

/* ══════════════════════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
   ══════════════════════════════════════════════════════════════════════════════ */

function showToast(message, type = 'info') {
  if (!DOM.toastContainer) return;

  const icons = {
    success: `<svg class="w-5 h-5 text-green-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
              </svg>`,
    error:   `<svg class="w-5 h-5 text-red-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>`,
    info:    `<svg class="w-5 h-5 text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>`,
    warning: `<svg class="w-5 h-5 text-yellow-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
              </svg>`,
  };

  const toast = document.createElement('div');
  toast.className = [
    'flex items-start w-full max-w-xs p-4 space-x-3',
    'bg-dr-cream text-dr-dark rounded-sm shadow-2xl',
    'border border-l-4 border-dr-accent',
    'animate-slide-in pointer-events-auto',
  ].join(' ');

  toast.innerHTML = `
    ${icons[type] ?? icons.info}
    <div class="text-sm font-sans font-medium leading-snug">${message}</div>
  `;

  DOM.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('animate-fade-out');
    setTimeout(() => toast.remove(), 420);
  }, CONFIG.TOAST_DURATION_MS);
}

/* ══════════════════════════════════════════════════════════════════════════════
   MOBILE MENU
   ══════════════════════════════════════════════════════════════════════════════ */

function initMobileMenu() {
  const { mobileMenu, menuBtn, closeBtn, mobileLinks } = DOM;
  if (!mobileMenu || !menuBtn) return;

  function openMenu() {
    state.mobileMenuOpen = true;
    mobileMenu.classList.remove('translate-x-full', 'opacity-0', 'pointer-events-none');
    document.body.style.overflow = 'hidden';
  }

  function closeMenu() {
    state.mobileMenuOpen = false;
    mobileMenu.classList.add('translate-x-full', 'opacity-0', 'pointer-events-none');
    document.body.style.overflow = '';
  }

  menuBtn.addEventListener('click', openMenu);
  if (closeBtn) closeBtn.addEventListener('click', closeMenu);

  mobileLinks.forEach((link) =>
    link.addEventListener('click', () => { if (state.mobileMenuOpen) closeMenu(); })
  );

  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.mobileMenuOpen) closeMenu();
  });
}

/* ══════════════════════════════════════════════════════════════════════════════
   NAVBAR SCROLL EFFECT
   ══════════════════════════════════════════════════════════════════════════════ */

function initNavbarScroll() {
  const { navbar } = DOM;
  if (!navbar) return;

  let ticking = false;

  function onScroll() {
    if (!ticking) {
      requestAnimationFrame(() => {
        if (window.scrollY > 50) {
          navbar.classList.add('bg-white/95', 'backdrop-blur-md', 'shadow-sm', 'p-3');
          navbar.classList.remove('p-6');
        } else {
          navbar.classList.remove('bg-white/95', 'backdrop-blur-md', 'shadow-sm', 'p-3');
          navbar.classList.add('p-6');
        }
        ticking = false;
      });
      ticking = true;
    }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
}

/* ══════════════════════════════════════════════════════════════════════════════
   PARALLAX ENGINE
   ══════════════════════════════════════════════════════════════════════════════ */

function initParallax() {
  const { parallaxEls } = DOM;
  if (!parallaxEls.length) return;

  // Disable parallax on reduced-motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) return;

  let ticking = false;

  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        const scrolled = window.pageYOffset;
        parallaxEls.forEach((el) => {
          const speed = parseFloat(el.dataset.speed) || 0;
          el.style.transform = `translateY(${scrolled * speed}px)`;
        });
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

/* ══════════════════════════════════════════════════════════════════════════════
   SCROLL REVEAL (IntersectionObserver)
   ══════════════════════════════════════════════════════════════════════════════ */

function initScrollReveal() {
  const { revealEls } = DOM;
  if (!revealEls.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
        } else {
          entry.target.classList.remove('active');
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -50px 0px' }
  );

  revealEls.forEach((el) => observer.observe(el));
}

/* ══════════════════════════════════════════════════════════════════════════════
   FILE UPLOAD — HANDLER
   ══════════════════════════════════════════════════════════════════════════════ */

function initFileUpload() {
  const { dropZone, fileInput } = DOM;
  if (!dropZone || !fileInput) return;

  // Click to open file dialog
  dropZone.addEventListener('click', (e) => {
    // Don't re-trigger if clicking the file info card's inner elements
    if (state.isProcessing) return;
    fileInput.click();
  });

  // Keyboard accessibility
  dropZone.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !state.isProcessing) {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Drag & Drop
  dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); });
  dropZone.addEventListener('dragover',  (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', (e) => {
    if (!dropZone.contains(e.relatedTarget)) {
      dropZone.classList.remove('drag-over');
    }
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (state.isProcessing) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // File input change
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });
}

/**
 * Validate and display the selected .mat file in the upload zone.
 */
function handleFile(file) {
  const error = validateMatFile(file);
  if (error) {
    showToast(error, 'error');
    shakeButton(DOM.scanBtn);
    return;
  }

  state.fileSelected = true;
  state.currentFile  = file;

  showFileInfo(file);
  resetResultPanel(false); // reset results but keep upload zone intact
  showToast('File loaded — ready for analysis.', 'success');
}

/**
 * Show the file info card inside the drop zone, replacing the upload icon.
 */
function showFileInfo(file) {
  const { uploadUi, fileInfoCard, fileNameDisplay, fileSizeDisplay } = DOM;
  if (!fileInfoCard) return;

  if (fileNameDisplay)  fileNameDisplay.textContent  = file.name;
  if (fileSizeDisplay)  fileSizeDisplay.textContent  =
    `${formatFileSize(file.size)}  •  .mat format`;

  if (uploadUi) uploadUi.style.opacity = '0';
  fileInfoCard.style.display = 'flex';
}

/**
 * Reset the drop zone back to the initial "upload" state.
 */
function clearFileInfo() {
  const { uploadUi, fileInfoCard, fileInput } = DOM;
  if (fileInfoCard) fileInfoCard.style.display = 'none';
  if (uploadUi)     uploadUi.style.opacity    = '1';
  if (fileInput)    fileInput.value            = '';
  state.fileSelected = false;
  state.currentFile  = null;
}

/* ══════════════════════════════════════════════════════════════════════════════
   SCANNER — MAIN ENTRY POINT
   ══════════════════════════════════════════════════════════════════════════════ */

function initScanner() {
  const { scanBtn } = DOM;
  if (!scanBtn) return;

  scanBtn.addEventListener('click', () => {
    if (state.isProcessing) return;

    if (!state.fileSelected || !state.currentFile) {
      showToast('Please upload a .mat RF file to begin analysis.', 'error');
      shakeButton(scanBtn);
      return;
    }

    runInference();
  });
}

/**
 * Run the full inference pipeline:
 * 1. Show loading UI with progressive status messages
 * 2. POST to /api/predict/
 * 3. Parse response and render results
 */
async function runInference() {
  const { scanBtn, scanLine, loadingOverlay } = DOM;

  state.isProcessing = true;
  if (scanBtn) {
    scanBtn.disabled     = true;
    scanBtn.textContent  = 'Processing…';
  }

  // Show scan line briefly, then switch to overlay
  if (scanLine) scanLine.style.display = 'block';
  await new Promise((r) => setTimeout(r, 400));
  if (scanLine) scanLine.style.display = 'none';

  showLoadingOverlay();
  startLoadingStageMessages();

  const startTime = performance.now();

  try {
    const formData = new FormData();
    formData.append('mat_file', state.currentFile);

    const controller = new AbortController();
    const timeoutId  = setTimeout(
      () => controller.abort(),
      CONFIG.REQUEST_TIMEOUT_MS
    );

    let response;
    try {
      response = await fetch(CONFIG.API_ENDPOINT, {
        method: 'POST',
        body:   formData,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }

    const elapsedMs = Math.round(performance.now() - startTime);

    // HTTP-level error
    if (!response.ok) {
      let errMsg = `Server error (HTTP ${response.status})`;
      try {
        const errData = await response.json();
        if (errData.message) errMsg = errData.message;
      } catch (_) { /* ignore parse error */ }
      throw new Error(errMsg);
    }

    const data = await response.json();

    // Application-level error
    if (data.status === 'error') {
      throw new Error(data.message || 'Inference pipeline failed.');
    }

    hideLoadingOverlay();
    renderResults(data, elapsedMs);

    if (data.warning) {
      showToast(`Warning: ${data.warning}`, 'warning');
    }

  } catch (err) {
    hideLoadingOverlay();
    console.error('[ElastoGrade] Inference error:', err);

    if (err.name === 'AbortError') {
      showToast('Request timed out. The server may still be processing — please retry.', 'error');
    } else {
      showToast(`Analysis failed: ${err.message}`, 'error');
    }

    if (scanBtn) {
      scanBtn.disabled    = false;
      scanBtn.textContent = 'Initialize Grading';
    }
    state.isProcessing = false;
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   LOADING OVERLAY
   ══════════════════════════════════════════════════════════════════════════════ */

function showLoadingOverlay() {
  const { loadingOverlay } = DOM;
  if (!loadingOverlay) return;
  loadingOverlay.classList.remove('hidden');
  loadingOverlay.classList.add('flex');
}

function hideLoadingOverlay() {
  const { loadingOverlay } = DOM;
  if (!loadingOverlay) return;
  clearLoadingTimers();
  loadingOverlay.classList.add('hidden');
  loadingOverlay.classList.remove('flex');
}

/**
 * Cycle through processing stage messages while the API call is in flight.
 * These are time-based approximations of the real pipeline steps.
 */
function startLoadingStageMessages() {
  const { loadingStatus, loadingSubstatus } = DOM;
  clearLoadingTimers();

  CONFIG.LOADING_STAGES.forEach(({ text, subtext, delayMs }) => {
    const id = setTimeout(() => {
      if (loadingStatus) {
        // Fade out, update, fade in
        loadingStatus.style.opacity = '0';
        setTimeout(() => {
          if (loadingStatus)  loadingStatus.textContent  = text;
          if (loadingSubstatus) loadingSubstatus.textContent = subtext;
          loadingStatus.style.opacity = '1';
        }, 200);
      }
    }, delayMs);
    state.loadingTimers.push(id);
  });
}

/* ══════════════════════════════════════════════════════════════════════════════
   RESULT RENDERING
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Main result renderer — called after a successful API response.
 */
function renderResults(data, elapsedMs) {
  const {
    resultPanel, gradeBadge, gradeText,
    processTime, dmdRank, rfShape,
    recommendationBox, scanBtn,
  } = DOM;

  const diagnosis = data.diagnosis || 'Benign';
  const cfg       = CONFIG.CLASSES[diagnosis] || CONFIG.CLASSES.Benign;

  // 1 ── Unlock result panel
  if (resultPanel) {
    resultPanel.classList.remove('opacity-30', 'blur-[4px]', 'pointer-events-none');
    resultPanel.classList.add('select-auto');
  }

  // 2 ── Primary diagnosis
  if (gradeText)  gradeText.textContent   = diagnosis;
  if (gradeBadge) {
    gradeBadge.textContent = diagnosis;
    gradeBadge.className   =
      `px-3 py-1 text-xs font-bold uppercase tracking-wide rounded-sm ${cfg.badgeClass}`;
  }

  // 3 ── Confidence bars
  const scores = data.confidence_scores || {};
  renderConfidenceBars(scores);

  // 4 ── Stats
  if (processTime) processTime.textContent = formatElapsed(elapsedMs);
  if (dmdRank)     dmdRank.textContent     = data.dmd_rank != null ? data.dmd_rank : '—';
  if (rfShape)     rfShape.textContent     = formatRfShape(data.rf_shape);

  // 5 ── Strain heatmap
  if (data.strain_heatmap_b64) {
    renderHeatmap(data.strain_heatmap_b64, data.filename || 'strain_map');
  }

  // 6 ── Clinical recommendation
  if (recommendationBox) {
    recommendationBox.innerHTML = cfg.recommendation;
  }

  // 7 ── Highlight protocol card
  highlightProtocolCard(cfg.protocolId);

  // 8 ── Reset scan button
  if (scanBtn) {
    scanBtn.disabled    = false;
    scanBtn.textContent = 'Analyse New File';
  }
  state.isProcessing = false;

  showToast(`Analysis complete — ${diagnosis} tissue detected.`, 'success');
}

/**
 * Animate the three confidence bars to their respective percentages.
 */
function renderConfidenceBars(scores) {
  const barMap = {
    Benign:       { bar: DOM.barBenign,       pct: DOM.pctBenign       },
    Precancerous: { bar: DOM.barPrecancerous,  pct: DOM.pctPrecancerous  },
    Malignant:    { bar: DOM.barMalignant,     pct: DOM.pctMalignant     },
  };

  Object.entries(barMap).forEach(([className, { bar, pct }], idx) => {
    const value   = scores[className] ?? 0;
    const percent = Math.round(value * 100);

    // Stagger each bar's animation
    setTimeout(() => {
      if (bar) bar.style.width = `${percent}%`;
      if (pct) pct.textContent = `${percent}%`;
    }, idx * CONFIG.CONFIDENCE_ANIM_DELAY_MS + 100);
  });
}

/**
 * Decode and display the base64-encoded strain heatmap PNG.
 */
function renderHeatmap(b64String, filename) {
  const { heatmapContainer, strainHeatmap, heatmapDownloadBtn } = DOM;
  if (!heatmapContainer || !strainHeatmap || !b64String) return;

  const src = `data:image/png;base64,${b64String}`;
  strainHeatmap.src = src;

  strainHeatmap.onload = () => {
    heatmapContainer.classList.add('visible');
  };

  // Wire up download button
  if (heatmapDownloadBtn) {
    heatmapDownloadBtn.onclick = () => {
      const link    = document.createElement('a');
      link.href     = src;
      link.download = `strain_heatmap_${filename.replace('.mat', '')}.png`;
      link.click();
    };
  }
}

/**
 * Highlight the relevant clinical protocol card and dim the others.
 */
function highlightProtocolCard(targetProtocolId) {
  const allIds = ['protocol-benign', 'protocol-precancerous', 'protocol-malignant'];

  // First scroll to the section
  setTimeout(() => {
    const section = document.getElementById('care');
    if (section) section.scrollIntoView({ behavior: 'smooth', block: 'center' });

    allIds.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;

      // Remove all state classes first
      el.classList.remove('dimmed', 'highlighted', 'ring-4',
        'ring-blue-400', 'ring-yellow-400', 'ring-dr-accent',
        'scale-105', 'z-10', 'shadow-2xl', '-translate-y-2');

      if (id === targetProtocolId) {
        el.classList.add('highlighted', 'ring-4', 'scale-105', 'z-10', 'shadow-2xl');
        // Add the specific ring color per class
        const diagnosisKey = Object.keys(CONFIG.CLASSES).find(
          (k) => CONFIG.CLASSES[k].protocolId === id
        );
        if (diagnosisKey) {
          el.classList.add(CONFIG.CLASSES[diagnosisKey].ringColor);
        }
      } else {
        el.classList.add('dimmed');
      }
    });
  }, CONFIG.DIET_SCROLL_DELAY_MS);
}

/**
 * Reset the result panel to its initial blurred/locked state.
 * Also resets confidence bars and hides the heatmap.
 * @param {boolean} [clearFile=true] - whether to also clear the file info
 */
function resetResultPanel(clearFile = true) {
  const {
    resultPanel, gradeBadge, gradeText,
    barBenign, barPrecancerous, barMalignant,
    pctBenign, pctPrecancerous, pctMalignant,
    processTime, dmdRank, rfShape,
    heatmapContainer, strainHeatmap,
    recommendationBox, scanBtn,
  } = DOM;

  if (resultPanel) {
    resultPanel.classList.add('opacity-30', 'blur-[4px]', 'pointer-events-none');
    resultPanel.classList.remove('select-auto');
  }

  if (gradeText)  gradeText.textContent  = '—';
  if (gradeBadge) {
    gradeBadge.textContent = 'Pending';
    gradeBadge.className   =
      'px-3 py-1 bg-gray-200 text-xs font-bold uppercase tracking-wide text-gray-500 rounded-sm';
  }

  // Reset bars
  [barBenign, barPrecancerous, barMalignant].forEach((b) => {
    if (b) b.style.width = '0%';
  });
  [pctBenign, pctPrecancerous, pctMalignant].forEach((p) => {
    if (p) p.textContent = '0%';
  });

  // Reset stats
  if (processTime) processTime.textContent = '—';
  if (dmdRank)     dmdRank.textContent     = '—';
  if (rfShape)     rfShape.textContent     = '—';

  // Hide heatmap
  if (heatmapContainer) heatmapContainer.classList.remove('visible');
  if (strainHeatmap)    strainHeatmap.src = '';

  // Reset recommendation
  if (recommendationBox) {
    recommendationBox.innerHTML =
      '<p class="opacity-80">Upload and scan a .mat file to receive a detailed analysis and clinical protocol recommendation.</p>';
  }

  // Reset protocol cards
  ['protocol-benign', 'protocol-precancerous', 'protocol-malignant'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove('dimmed', 'highlighted', 'ring-4',
        'ring-blue-400', 'ring-yellow-400', 'ring-dr-accent',
        'scale-105', 'z-10', 'shadow-2xl');
    }
  });

  // Reset button
  if (scanBtn) {
    scanBtn.disabled    = false;
    scanBtn.textContent = 'Initialize Grading';
  }

  if (clearFile) clearFileInfo();
}

/* ══════════════════════════════════════════════════════════════════════════════
   UTILITY — BUTTON SHAKE
   ══════════════════════════════════════════════════════════════════════════════ */

function shakeButton(btn) {
  if (!btn) return;
  btn.classList.add('animate-shake');
  setTimeout(() => btn.classList.remove('animate-shake'), 550);
}

/* ══════════════════════════════════════════════════════════════════════════════
   ENTRY POINT
   ══════════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  resolveDom();
  initMobileMenu();
  initNavbarScroll();
  initParallax();
  initScrollReveal();
  initFileUpload();
  initScanner();

  // Ensure the result panel starts in the locked state on page load
  resetResultPanel(true);
});
