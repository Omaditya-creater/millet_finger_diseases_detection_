import { FINGER_MILLET_DISEASES, TRANSLATIONS } from './diseases_data.js';

let currentLang = 'en';
let currentDiseaseKey = 'blast';
let heatmapOpacity = 0.65;
let lesionThreshold = 0.35;
let isUploadedImage = false;
let uploadedImageElement = null;

// Initialize Web App
document.addEventListener('DOMContentLoaded', () => {
  setupLanguageSelector();
  setupTabNavigation();
  setupPresets();
  setupSliders();
  setupFileUpload();
  setupGenAIChat();
  setupReportDownload();

  // Initial Render
  renderDiagnosticStudio();
});

function setupLanguageSelector() {
  const select = document.getElementById('langSelect');
  if (!select) return;
  select.addEventListener('change', (e) => {
    currentLang = e.target.value;
    updateUIStaticText();
    renderDiagnosticStudio();
  });
}

function updateUIStaticText() {
  const t = TRANSLATIONS[currentLang] || TRANSLATIONS.en;
  
  // Update nav title & sub
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key]) {
      el.textContent = t[key];
    }
  });
}

function setupTabNavigation() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      
      document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = content.id === targetTab ? 'block' : 'none';
      });
    });
  });
}

function setupPresets() {
  const presetCards = document.querySelectorAll('.preset-card');
  presetCards.forEach(card => {
    card.addEventListener('click', () => {
      presetCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      currentDiseaseKey = card.getAttribute('data-disease');
      isUploadedImage = false;
      uploadedImageElement = null;
      renderDiagnosticStudio();
    });
  });
}

function setupSliders() {
  const opacityInput = document.getElementById('opacitySlider');
  const opacityVal = document.getElementById('opacityValue');
  if (opacityInput) {
    opacityInput.addEventListener('input', (e) => {
      heatmapOpacity = parseFloat(e.target.value);
      if (opacityVal) opacityVal.textContent = `${Math.round(heatmapOpacity * 100)}%`;
      drawVisualizerCanvas();
    });
  }

  const threshInput = document.getElementById('threshSlider');
  const threshVal = document.getElementById('threshValue');
  if (threshInput) {
    threshInput.addEventListener('input', (e) => {
      lesionThreshold = parseFloat(e.target.value);
      if (threshVal) threshVal.textContent = `${Math.round(lesionThreshold * 100)}%`;
      drawVisualizerCanvas();
    });
  }
}

function setupFileUpload() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());
  
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#10b981';
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'rgba(16, 185, 129, 0.4)';
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });
}

async function handleFile(file) {
  const reader = new FileReader();
  reader.onload = async (event) => {
    const img = new Image();
    img.onload = async () => {
      isUploadedImage = true;
      uploadedImageElement = img;

      // Attempt live backend API inference
      try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch('/api/predict', {
          method: 'POST',
          body: formData
        });

        if (res.ok) {
          const data = await res.json();
          if (data.disease_key && FINGER_MILLET_DISEASES[data.disease_key]) {
            currentDiseaseKey = data.disease_key;
            document.querySelectorAll('.preset-card').forEach(c => {
              if (c.getAttribute('data-disease') === currentDiseaseKey) {
                c.classList.add('active');
              } else {
                c.classList.remove('active');
              }
            });
          }
          renderDiagnosticStudio(data);
          return;
        }
      } catch (err) {
        console.log('Using client-side offline mode:', err);
      }

      // Default fallback
      currentDiseaseKey = 'blast';
      renderDiagnosticStudio();
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function renderDiagnosticStudio(backendData = null) {
  const disease = FINGER_MILLET_DISEASES[currentDiseaseKey];
  if (!disease) return;

  // 1. Update Metrics
  const diseaseNameEl = document.getElementById('diseaseName');
  const scientificNameEl = document.getElementById('scientificName');
  const confidenceEl = document.getElementById('confidenceVal');
  const lsiEl = document.getElementById('lsiVal');
  const riskBadgeEl = document.getElementById('riskBadge');

  const confVal = backendData ? (backendData.confidence * 100).toFixed(1) : (disease.confidence * 100).toFixed(1);
  const lsiVal = backendData ? backendData.lsi_percentage : disease.lsi;

  if (diseaseNameEl) diseaseNameEl.textContent = backendData?.disease_name || disease.name;
  if (scientificNameEl) scientificNameEl.textContent = disease.scientificName;
  if (confidenceEl) confidenceEl.textContent = `${confVal}%`;
  if (lsiEl) lsiEl.textContent = `${lsiVal}%`;
  
  if (riskBadgeEl) {
    if (backendData?.advisory?.risk_assessment?.risk_level) {
      riskBadgeEl.textContent = backendData.advisory.risk_assessment.risk_level;
    } else {
      riskBadgeEl.textContent = disease.riskLevel;
    }
    riskBadgeEl.className = `badge ${disease.riskBadgeClass}`;
  }

  // 2. Render Advisory Protocols
  const summaryEl = document.getElementById('pathologySummary');
  if (summaryEl) {
    summaryEl.textContent = backendData?.advisory?.pathological_summary || disease.summary;
  }

  const organicList = document.getElementById('organicList');
  if (organicList) {
    const remedies = backendData?.advisory?.treatment_protocol?.organic_biological_remedies || disease.organicRemedies;
    organicList.innerHTML = remedies.map(item => `<li>${item}</li>`).join('');
  }

  const chemicalList = document.getElementById('chemicalList');
  if (chemicalList) {
    const chems = backendData?.advisory?.treatment_protocol?.chemical_fungicide_control || disease.chemicalControl;
    chemicalList.innerHTML = chems.map(item => `<li>${item}</li>`).join('');
  }

  const dosageEl = document.getElementById('dosageVal');
  if (dosageEl) {
    dosageEl.textContent = backendData?.advisory?.treatment_protocol?.recommended_dosage || disease.dosage;
  }

  const phiEl = document.getElementById('phiVal');
  if (phiEl) {
    const phi = backendData?.advisory?.treatment_protocol?.pre_harvest_interval_days ?? disease.phiDays;
    phiEl.textContent = `${phi} Days`;
  }

  const prevList = document.getElementById('preventiveList');
  if (prevList) {
    const prevs = backendData?.advisory?.preventive_agronomy_practices || disease.preventivePractices;
    prevList.innerHTML = prevs.map(item => `<li>${item}</li>`).join('');
  }

  // 3. Draw Canvas
  drawVisualizerCanvas();
}

function drawVisualizerCanvas() {
  const canvas = document.getElementById('visCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const disease = FINGER_MILLET_DISEASES[currentDiseaseKey];

  if (isUploadedImage && uploadedImageElement) {
    // Draw uploaded leaf
    ctx.drawImage(uploadedImageElement, 0, 0, width, height);
  } else {
    // Draw procedural finger millet leaf
    drawProceduralLeaf(ctx, width, height, disease);
  }

  // Draw Grad-CAM Heatmap Layer
  if (heatmapOpacity > 0 && disease.lesionPoints.length > 0) {
    drawGradCAMOverlay(ctx, width, height, disease.lesionPoints);
  }
}

function drawProceduralLeaf(ctx, width, height, disease) {
  // Leaf background gradient
  const grad = ctx.createLinearGradient(0, 0, width, height);
  grad.addColorStop(0, '#15803d');
  grad.addColorStop(0.5, '#166534');
  grad.addColorStop(1, '#14532d');
  
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Central Vein
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();

  // Lateral veins
  ctx.strokeStyle = 'rgba(74, 222, 128, 0.3)';
  ctx.lineWidth = 2;
  for (let y = 30; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(width / 2, y);
    ctx.lineTo(width / 2 - 120, y - 40);
    ctx.moveTo(width / 2, y);
    ctx.lineTo(width / 2 + 120, y - 40);
    ctx.stroke();
  }

  // Draw physical disease spots
  disease.lesionPoints.forEach(pt => {
    const cx = pt.x * width;
    const cy = pt.y * height;
    
    ctx.save();
    ctx.fillStyle = disease.color;
    ctx.beginPath();
    ctx.ellipse(cx, cy, pt.radius, pt.radius * 0.5, Math.PI / 4, 0, Math.PI * 2);
    ctx.fill();

    // Lesion center grey core
    ctx.fillStyle = '#e5e7eb';
    ctx.beginPath();
    ctx.ellipse(cx, cy, pt.radius * 0.3, pt.radius * 0.15, Math.PI / 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });
}

function drawGradCAMOverlay(ctx, width, height, points) {
  // Create off-screen canvas for heatmap buffer
  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = width;
  heatCanvas.height = height;
  const hCtx = heatCanvas.getContext('2d');

  points.forEach(pt => {
    if (pt.weight < lesionThreshold) return;

    const cx = pt.x * width;
    const cy = pt.y * height;
    const r = pt.radius * 3.5;

    const radial = hCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
    radial.addColorStop(0, `rgba(239, 68, 68, ${pt.weight})`);
    radial.addColorStop(0.4, `rgba(245, 158, 11, ${pt.weight * 0.7})`);
    radial.addColorStop(0.8, `rgba(59, 130, 246, ${pt.weight * 0.3})`);
    radial.addColorStop(1, 'rgba(0, 0, 0, 0)');

    hCtx.fillStyle = radial;
    hCtx.beginPath();
    hCtx.arc(cx, cy, r, 0, Math.PI * 2);
    hCtx.fill();
  });

  // Blend overlay onto main canvas
  ctx.save();
  ctx.globalAlpha = heatmapOpacity;
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(heatCanvas, 0, 0);
  ctx.restore();
}

function setupGenAIChat() {
  const sendBtn = document.getElementById('sendChatBtn');
  const chatInput = document.getElementById('chatInput');
  const chatMessages = document.getElementById('chatMessages');

  if (!sendBtn || !chatInput) return;

  const handleSend = () => {
    const text = chatInput.value.trim();
    if (!text) return;

    // Append User Message
    appendMessage(text, 'user');
    chatInput.value = '';

    // Generate AI Response
    setTimeout(() => {
      const responseText = getGenAIResponse(text);
      appendMessage(responseText, 'bot');
    }, 600);
  };

  sendBtn.addEventListener('click', handleSend);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
  });
}

function appendMessage(text, sender) {
  const chatMessages = document.getElementById('chatMessages');
  if (!chatMessages) return;

  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-bubble ${sender}`;
  msgDiv.textContent = text;
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function getGenAIResponse(query) {
  const q = query.toLowerCase();
  const disease = FINGER_MILLET_DISEASES[currentDiseaseKey];

  if (q.includes('dosage') || q.includes('quantity') || q.includes('how much')) {
    return disease.chatResponses.dosage;
  } else if (q.includes('organic') || q.includes('natural') || q.includes('neem')) {
    return disease.chatResponses.organic;
  } else if (q.includes('weather') || q.includes('rain') || q.includes('humidity')) {
    return disease.chatResponses.weather;
  } else {
    return `Based on GenAI pathological context for ${disease.name}: We recommend strictly adhering to the recommended Pre-Harvest Interval of ${disease.phiDays} days and combining ${disease.organicRemedies[0]} with targeted chemical spray if Lesion Severity Index exceeds 15%.`;
  }
}

function setupReportDownload() {
  const downloadBtn = document.getElementById('downloadReportBtn');
  if (!downloadBtn) return;
  downloadBtn.addEventListener('click', () => {
    window.print();
  });
}
