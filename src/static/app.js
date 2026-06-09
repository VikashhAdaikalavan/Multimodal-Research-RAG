/* ==========================================================================
   Multimodal Research RAG Client Javascript - App Controller & Audio WAV Recorder
   ========================================================================== */

// --- Global States ---
let currentTab = 'chat-view';
let isInitializing = true;
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let audioSamples = [];
let voiceTimerInterval = null;
let voiceDurationSeconds = 0;
let attachedImageFile = null;
let activeUtterance = null;
let pollIngestInterval = null;
let ingestTimerInterval = null;
let ingestSeconds = 0;
let conversationHistory = [];

// --- Initialize App ---
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  checkSystemStatus();
  loadDocumentList();
  
  // Check ingestion status on load in case a build is already running
  checkIngestStatus();
  pollIngestInterval = setInterval(checkIngestStatus, 3000);
});

// --- Tab Switching ---
function setupTabs() {
  const navItems = document.querySelectorAll('.nav-item');
  const panels = document.querySelectorAll('.tab-panel');
  
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabName = item.getAttribute('data-tab');
      if (!tabName) return;
      
      navItems.forEach(nav => nav.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      
      item.classList.add('active');
      document.getElementById(tabName).classList.add('active');
      currentTab = tabName;
    });
  });
}

// --- System Status Check & Polling ---
async function checkSystemStatus(isRetry = false) {
  const errorBox = document.getElementById('init-error-box');
  const errorText = document.getElementById('init-error-text');
  
  if (isRetry) {
    errorBox.style.display = 'none';
  }
  
  try {
    const res = await fetch('/api/status');
    if (!res.ok) {
      throw new Error(`API returned HTTP ${res.status}`);
    }
    const data = await res.json();
    
    // Update step markers
    updateInitStep('step-embeddings', data.steps.embeddings);
    updateInitStep('step-database', data.steps.database);
    updateInitStep('step-llm_ensemble', data.steps.llm_ensemble);
    updateInitStep('step-whisper', data.steps.whisper);
    
    // Set config values in diagnostics tab
    document.getElementById('diag-data-folder').textContent = data.data_folder;
    document.getElementById('diag-db-path').textContent = data.db_path;
    
    if (data.status === 'ready') {
      isInitializing = false;
      document.getElementById('init-overlay').classList.add('hidden');
      document.querySelector('.quick-status .status-dot').className = 'status-dot online';
      document.querySelector('.quick-status .status-desc').textContent = 'All models online';
    } else if (data.status === 'error') {
      throw new Error(data.error_message || "Unknown model loading failure.");
    } else {
      // Still loading, check again in 2 seconds
      setTimeout(checkSystemStatus, 2000);
    }
    
  } catch (err) {
    isInitializing = true;
    errorText.textContent = err.message || "Failed to establish socket connection with backend api.";
    errorBox.style.display = 'block';
    
    document.querySelector('.quick-status .status-dot').className = 'status-dot';
    document.querySelector('.quick-status .status-desc').textContent = 'System offline';
  }
}

function updateInitStep(stepId, status) {
  const row = document.getElementById(stepId);
  if (!row) return;
  
  row.className = `step-row ${status}`;
  const statusLabel = row.querySelector('.step-status');
  
  if (status === 'ready') {
    statusLabel.textContent = '✓ Ready';
  } else if (status === 'loading') {
    statusLabel.textContent = '⚡ Ingesting...';
  } else if (status === 'error') {
    statusLabel.textContent = '✕ Error';
  } else if (status === 'empty') {
    statusLabel.textContent = '⚠ Empty Index';
  } else {
    statusLabel.textContent = 'Pending';
  }
}

// --- Textarea Expand & Key Events ---
function handleTextareaKey(e) {
  // Enter submits, Shift+Enter inserts newline
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
  
  // Autosize
  const textarea = e.target;
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// --- Image Attachment handling ---
function handleImageAttachment(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file (JPG, PNG, WEBP).');
    return;
  }
  
  attachedImageFile = file;
  
  // Show preview
  const reader = new FileReader();
  reader.onload = (event) => {
    document.getElementById('preview-img').src = event.target.result;
    document.getElementById('attachment-filename').textContent = file.name;
    document.getElementById('attachment-preview').style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function removeImageAttachment() {
  attachedImageFile = null;
  document.getElementById('image-query-input').value = '';
  document.getElementById('attachment-preview').style.display = 'none';
  document.getElementById('preview-img').src = '';
}

// --- Chat Form Submission ---
async function handleChatSubmit(e) {
  e.preventDefault();
  
  if (isInitializing) return;
  
  const textInput = document.getElementById('chat-input');
  const queryText = textInput.value.trim();
  
  if (!queryText && !attachedImageFile) {
    return; // Nothing to search
  }
  
  const historyToSend = JSON.stringify(conversationHistory);
  
  // Add User Message Bubble
  appendChatMessage('user', queryText, attachedImageFile);
  
  // Save references and clear inputs
  const queryToSend = queryText;
  const imageToSend = attachedImageFile;
  
  textInput.value = '';
  textInput.style.height = 'auto';
  removeImageAttachment();
  
  // Disable UI inputs
  toggleInputState(false);
  
  // Show Typing Indicator
  document.getElementById('typing-indicator').style.display = 'flex';
  scrollChatToBottom();
  
  try {
    const formData = new FormData();
    formData.append('query', queryToSend || 'Describe this image.');
    formData.append('history', historyToSend);
    if (imageToSend) {
      formData.append('image', imageToSend);
    }
    
    const res = await fetch('/api/query', {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Failed to query RAG model ensemble.');
    }
    
    const data = await res.json();
    
    // Append Bot response
    appendChatMessage('bot', data.answer, null, data.sources, data.image_description);
    
  } catch (err) {
    console.error(err);
    appendChatMessage('bot', `⚠ **Error querying research assistant**: ${err.message}`, null);
  } finally {
    // Hide Typing Indicator and Enable UI
    document.getElementById('typing-indicator').style.display = 'none';
    toggleInputState(true);
    scrollChatToBottom();
    textInput.focus();
  }
}

function toggleInputState(enabled) {
  document.getElementById('chat-input').disabled = !enabled;
  document.getElementById('send-btn').disabled = !enabled;
  document.getElementById('voice-record-btn').disabled = !enabled;
}

// --- Chat Messages Rendering (Custom Markdown Parser) ---
function appendChatMessage(sender, text, imageFile = null, sources = [], imageDesc = null) {
  // Update conversational memory (only store text for context)
  if (text && !text.includes('(Voice Recording sent...)')) {
    conversationHistory.push({ role: sender, content: text });
    if (conversationHistory.length > 10) {
      conversationHistory.shift();
    }
  }
  
  const chatHistory = document.getElementById('chat-history');
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${sender}`;
  
  // Avatar
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = sender === 'user' ? 'ME' : 'AI';
  messageDiv.appendChild(avatar);
  
  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  
  // Render attached image in bubble if user message
  if (sender === 'user' && imageFile) {
    const imgWrapper = document.createElement('div');
    imgWrapper.className = 'message-attached-image';
    imgWrapper.style.maxWidth = '250px';
    imgWrapper.style.borderRadius = '8px';
    imgWrapper.style.overflow = 'hidden';
    imgWrapper.style.marginBottom = '10px';
    imgWrapper.style.border = '1px solid var(--panel-border)';
    
    const img = document.createElement('img');
    img.style.width = '100%';
    img.style.display = 'block';
    
    const reader = new FileReader();
    reader.onload = (e) => { img.src = e.target.result; };
    reader.readAsDataURL(imageFile);
    
    imgWrapper.appendChild(img);
    bubble.appendChild(imgWrapper);
  }
  
  // Text content
  const textDiv = document.createElement('div');
  textDiv.className = 'message-text';
  textDiv.innerHTML = formatMarkdown(text);
  bubble.appendChild(textDiv);
  
  // If bot responded with image description, render it as collapsible segment
  if (sender === 'bot' && imageDesc) {
    const imgDescDiv = document.createElement('div');
    imgDescDiv.className = 'message-image-description';
    imgDescDiv.style.marginTop = '10px';
    imgDescDiv.style.padding = '8px 12px';
    imgDescDiv.style.background = 'rgba(244, 63, 94, 0.05)';
    imgDescDiv.style.borderLeft = '3px solid var(--accent-pink)';
    imgDescDiv.style.borderRadius = '4px';
    imgDescDiv.style.fontSize = '0.85rem';
    imgDescDiv.innerHTML = `<strong>Vision Description:</strong> ${formatMarkdown(imageDesc)}`;
    bubble.appendChild(imgDescDiv);
  }
  
  // Add sources citations accordion if present
  if (sender === 'bot' && sources && sources.length > 0) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'message-sources';
    
    const sourceBtn = document.createElement('button');
    sourceBtn.className = 'sources-header-btn';
    sourceBtn.innerHTML = `
      <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
      Show Reference Sources (${sources.length})
    `;
    
    const sourceContent = document.createElement('div');
    sourceContent.className = 'sources-content';
    sourceContent.style.display = 'none';
    
    // Draw source cards
    sources.forEach(src => {
      const card = document.createElement('div');
      card.className = 'source-card';
      
      const filename = src.filename || 'Source Document';
      const scorePct = Math.max(0, (1 - src.score) * 100).toFixed(1); // Format distance score to confidence % (min 0%)
      const cleanContent = src.content.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
      
      card.innerHTML = `
        <div class="source-meta">
          <span class="source-filename" title="${src.source}">${filename}</span>
          <span class="source-score" title="Distance score: ${src.score.toFixed(4)}">Match: ${scorePct}%</span>
        </div>
        <div class="source-snippet" title="Click to view full chunk">${cleanContent}</div>
      `;
      
      // Click card to show snippet modal
      card.addEventListener('click', () => {
        alert(`Source file: ${src.source}\nMatch: ${scorePct}%\n\nExcerpt:\n${cleanContent}`);
      });
      
      sourceContent.appendChild(card);
    });
    
    sourceBtn.addEventListener('click', () => {
      const isOpen = sourceBtn.classList.toggle('open');
      sourceContent.style.display = isOpen ? 'grid' : 'none';
      scrollChatToBottom();
    });
    
    sourcesContainer.appendChild(sourceBtn);
    sourcesContainer.appendChild(sourceContent);
    bubble.appendChild(sourcesContainer);
  }
  
  // Speak Answer button (TTS) for bot responses
  if (sender === 'bot') {
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'message-actions';
    
    const speakBtn = document.createElement('button');
    speakBtn.className = 'btn-message-action';
    speakBtn.innerHTML = `
      <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" class="tts-icon"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/></svg>
      Read Aloud
    `;
    
    speakBtn.addEventListener('click', () => {
      toggleSpeech(text, speakBtn);
    });
    
    actionsContainer.appendChild(speakBtn);
    bubble.appendChild(actionsContainer);
  }
  
  messageDiv.appendChild(bubble);
  chatHistory.appendChild(messageDiv);
  scrollChatToBottom();
}

function scrollChatToBottom() {
  const container = document.getElementById('chat-history');
  container.scrollTop = container.scrollHeight;
}

// --- Markdown-Lite Formatting parser ---
function formatMarkdown(text) {
  if (!text) return "";
  
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold (**text**)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Code block (`code`)
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');
  
  // Headers (### title)
  html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');
  
  // Bullet lists (- or *)
  html = html.replace(/^\s*[-*]\s+(.*?)$/gm, '<li>$1</li>');
  html = html.replace(/(?:<li>.*?<\/li>\s*)+/g, '<ul>$&</ul>');
  
  // Number lists (1. title)
  html = html.replace(/^\s*(\d+)\.\s+(.*?)$/gm, '<ol-item>$2</ol-item>');
  html = html.replace(/(?:<ol-item>.*?<\/ol-item>\s*)+/g, '<ol>$&</ol>');
  html = html.replace(/<ol-item>(.*?)<\/ol-item>/g, '<li>$1</li>');
  
  // Multi line spacing
  html = html.split('\n\n').map(p => {
    const trimmed = p.trim();
    if (trimmed.startsWith('<h') || trimmed.startsWith('<ul') || trimmed.startsWith('<ol')) {
      return trimmed;
    }
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).join('');
  
  return html;
}

// --- Text-To-Speech browser API ---
function toggleSpeech(text, button) {
  const synth = window.speechSynthesis;
  
  // If already speaking
  if (synth.speaking && activeUtterance) {
    synth.cancel();
    document.querySelectorAll('.btn-message-action.speaking').forEach(btn => {
      btn.classList.remove('speaking');
      btn.innerHTML = btn.innerHTML.replace('Stop Audio', 'Read Aloud');
    });
    activeUtterance = null;
    return;
  }
  
  // Start speaking
  // Strip Markdown markers for clean TTS output
  const cleanText = text.replace(/\*\*|`/g, '').replace(/###|##|#/g, '');
  
  activeUtterance = new SpeechSynthesisUtterance(cleanText);
  button.classList.add('speaking');
  button.innerHTML = button.innerHTML.replace('Read Aloud', 'Stop Audio');
  
  activeUtterance.onend = () => {
    button.classList.remove('speaking');
    button.innerHTML = button.innerHTML.replace('Stop Audio', 'Read Aloud');
    activeUtterance = null;
  };
  
  activeUtterance.onerror = () => {
    button.classList.remove('speaking');
    button.innerHTML = button.innerHTML.replace('Stop Audio', 'Read Aloud');
    activeUtterance = null;
  };
  
  synth.speak(activeUtterance);
}

// --- Client-Side Audio Recording (WAV Encoder) ---
async function toggleAudioRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    isRecording = true;
    mediaStream = stream;
    audioSamples = [];
    voiceDurationSeconds = 0;
    
    // Set up Audio Context and processing nodes
    // Note: Whisper expects 16000Hz mono.
    // We create AudioContext with standard/native rates, collect buffers, 
    // and manually downsample on stop to ensure it runs cleanly on all browsers.
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create script processor (buffer size 4096)
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    
    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const inputBuffer = e.inputBuffer.getChannelData(0);
      // Clone floats
      audioSamples.push(new Float32Array(inputBuffer));
    };
    
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
    
    // Toggle overlays and triggers
    document.getElementById('voice-overlay').style.display = 'flex';
    document.getElementById('voice-record-btn').classList.add('recording');
    
    // Run Timer
    document.getElementById('voice-timer').textContent = '00:00';
    voiceTimerInterval = setInterval(() => {
      voiceDurationSeconds++;
      const mins = String(Math.floor(voiceDurationSeconds / 60)).padStart(2, '0');
      const secs = String(voiceDurationSeconds % 60).padStart(2, '0');
      document.getElementById('voice-timer').textContent = `${mins}:${secs}`;
      
      // Auto stop after 60 seconds
      if (voiceDurationSeconds >= 60) {
        stopRecording();
      }
    }, 1000);
    
  } catch (err) {
    console.error('Mic initialization failed:', err);
    alert('Could not start recording. Please grant microphone access permission in your browser.');
  }
}

async function stopRecording() {
  if (!isRecording) return;
  
  isRecording = false;
  
  // Clear recording timer
  clearInterval(voiceTimerInterval);
  
  // Disable recording processor links
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  
  // Hide UI overlays
  document.getElementById('voice-overlay').style.display = 'none';
  document.getElementById('voice-record-btn').classList.remove('recording');
  
  // Downsample and encode float samples to standard WAV
  if (audioSamples.length === 0) return;
  
  // Show Typing Indicator
  toggleInputState(false);
  document.getElementById('typing-indicator').style.display = 'flex';
  document.getElementById('typing-indicator').querySelector('.status-text').textContent = 'Transcribing your voice query...';
  
  try {
    // 1. Flatten all float arrays
    let totalLength = 0;
    audioSamples.forEach(arr => totalLength += arr.length);
    const flattenedSamples = new Float32Array(totalLength);
    let offset = 0;
    audioSamples.forEach(arr => {
      flattenedSamples.set(arr, offset);
      offset += arr.length;
    });
    
    // 2. Downsample Float samples to 16000Hz (required by Whisper)
    // Assume input sample rate is browser's context rate, usually 44100Hz or 48000Hz.
    const browserSampleRate = audioContext ? audioContext.sampleRate : 48000;
    const downsampledSamples = downsampleBuffer(flattenedSamples, browserSampleRate, 16000);
    
    // 3. Package as WAV blob
    const wavBlob = encodeWAV(downsampledSamples, 16000);
    
    const historyToSend = JSON.stringify(conversationHistory);
    
    // Add user message with voice label
    appendChatMessage('user', '🎙️ (Voice Recording sent...)');
    
    // Upload WAV to backend for Whisper transcription
    const formData = new FormData();
    formData.append('audio', wavBlob, 'query.wav');
    formData.append('history', historyToSend);
    
    const res = await fetch('/api/voice-query', {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Whisper voice transcription failed.');
    }
    
    const data = await res.json();
    
    // Replace the voice placeholder with the transcribed text
    const chatHistory = document.getElementById('chat-history');
    const userMsgs = chatHistory.querySelectorAll('.chat-message.user');
    if (userMsgs.length > 0) {
      const lastUserMsg = userMsgs[userMsgs.length - 1];
      const bubble = lastUserMsg.querySelector('.message-text');
      bubble.innerHTML = `🎙️ "${data.transcription}"`;
    }
    
    // Update memory with transcription
    conversationHistory.push({ role: 'user', content: data.transcription });
    
    // Show synthesised output
    appendChatMessage('bot', data.answer, null, data.sources);
    
  } catch (err) {
    console.error(err);
    appendChatMessage('bot', `⚠ **Voice transcription failed**: ${err.message}`, null);
  } finally {
    document.getElementById('typing-indicator').style.display = 'none';
    document.getElementById('typing-indicator').querySelector('.status-text').textContent = 'Generating synthesised response...';
    toggleInputState(true);
    scrollChatToBottom();
  }
}

// WAV Downsampling function
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

// WAV Byte Encoder
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);

  return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// --- Knowledge Base - Document List Loader ---
async function loadDocumentList() {
  const documentList = document.getElementById('document-list');
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) throw new Error('Failed to retrieve file repository list.');
    const files = await res.json();
    
    document.getElementById('file-count').textContent = files.length;
    
    if (files.length === 0) {
      documentList.innerHTML = '<div class="empty-docs">No documents uploaded. Drag and drop files above!</div>';
      return;
    }
    
    documentList.innerHTML = '';
    
    files.forEach(file => {
      const row = document.createElement('div');
      row.className = `doc-row ${file.type}`;
      
      const sizeKB = (file.size / 1024).toFixed(1);
      
      // Icon mapping based on extension types
      let svgIcon = '';
      if (file.type === 'pdf') {
        svgIcon = `<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg>`;
      } else if (file.type === 'audio') {
        svgIcon = `<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/></svg>`;
      } else if (file.type === 'image') {
        svgIcon = `<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>`;
      } else {
        svgIcon = `<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>`;
      }
      
      row.innerHTML = `
        <div class="doc-info">
          <span class="doc-icon">${svgIcon}</span>
          <div>
            <div class="doc-name" title="${file.name}">${file.name}</div>
            <span class="doc-size">${sizeKB} KB</span>
          </div>
        </div>
        <button class="btn-delete" title="Delete document">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
        </button>
      `;
      
      // Handle delete click
      row.querySelector('.btn-delete').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Are you sure you want to delete ${file.name}?`)) {
          try {
            const delRes = await fetch(`/api/documents/${file.name}`, { method: 'DELETE' });
            if (!delRes.ok) throw new Error('Failed to delete file.');
            loadDocumentList();
          } catch (err) {
            alert(err.message);
          }
        }
      });
      
      documentList.appendChild(row);
    });
    
  } catch (err) {
    documentList.innerHTML = `<div class="empty-docs">Error loading repository: ${err.message}</div>`;
  }
}

// --- Knowledge Base - Document Upload Dropzone ---
const dropZone = document.getElementById('drop-zone');

['dragenter', 'dragover'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(88, 101, 242, 0.7)';
    dropZone.style.background = 'rgba(88, 101, 242, 0.04)';
  }, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    dropZone.style.background = 'rgba(255, 255, 255, 0.01)';
  }, false);
});

dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  uploadFiles(files);
});

function handleFileUpload(e) {
  const files = e.target.files;
  uploadFiles(files);
}

async function uploadFiles(files) {
  if (files.length === 0) return;
  
  const progressContainer = document.getElementById('upload-progress-container');
  const progressBar = document.getElementById('upload-progress-bar');
  const progressLabel = document.getElementById('upload-progress-label');
  
  progressContainer.style.display = 'block';
  
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    progressLabel.textContent = `Uploading ${file.name} (${i + 1}/${files.length})...`;
    progressBar.style.width = '0%';
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // XHR upload wrapper to track upload percentage
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/documents/upload', true);
        
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            progressBar.style.width = percent + '%';
          }
        };
        
        xhr.onload = () => {
          if (xhr.status === 200) {
            resolve();
          } else {
            reject(new Error(`Server error HTTP ${xhr.status}`));
          }
        };
        
        xhr.onerror = () => reject(new Error('Network upload error.'));
        xhr.send(formData);
      });
      
    } catch (err) {
      alert(`Failed to upload ${file.name}: ${err.message}`);
    }
  }
  
  // Hide progress and reload files
  progressBar.style.width = '100%';
  progressLabel.textContent = 'All uploads complete!';
  setTimeout(() => {
    progressContainer.style.display = 'none';
  }, 1500);
  
  loadDocumentList();
}

// --- Database Ingestion Wizard ---
async function triggerIngestion() {
  if (confirm('Are you sure you want to rebuild the vector index?\nThis will clear the current database cache and rebuild embeddings for all documents in the Data folder.')) {
    try {
      const res = await fetch('/api/ingest', { method: 'POST' });
      if (!res.ok) throw new Error('Could not trigger parser pipeline.');
      
      const data = await res.json();
      console.log('Ingestion triggered:', data);
      
      // Transition UI
      toggleIngestUI(true);
      
    } catch (err) {
      alert(err.message);
    }
  }
}

function toggleIngestUI(isRunning) {
  const statusBox = document.getElementById('ingest-status-box');
  const idleBox = document.getElementById('ingest-idle-box');
  
  if (isRunning) {
    statusBox.style.display = 'block';
    idleBox.style.display = 'none';
    
    // Ingest timer loop
    ingestSeconds = 0;
    document.getElementById('ingest-time').textContent = '00:00';
    clearInterval(ingestTimerInterval);
    ingestTimerInterval = setInterval(() => {
      ingestSeconds++;
      const mins = String(Math.floor(ingestSeconds / 60)).padStart(2, '0');
      const secs = String(ingestSeconds % 60).padStart(2, '0');
      document.getElementById('ingest-time').textContent = `${mins}:${secs}`;
    }, 1000);
    
  } else {
    statusBox.style.display = 'none';
    idleBox.style.display = 'flex';
    clearInterval(ingestTimerInterval);
  }
}

async function checkIngestStatus() {
  try {
    const res = await fetch('/api/ingest/status');
    if (!res.ok) return;
    const data = await res.json();
    
    if (data.status === 'running') {
      toggleIngestUI(true);
      document.getElementById('ingest-phase-title').textContent = data.phase || "Parsing Documents";
      document.getElementById('ingest-phase-desc').textContent = data.message || "Warming up processor...";
      document.getElementById('ingest-status-badge').className = 'ingest-badge badge-running';
      document.getElementById('ingest-status-badge').textContent = 'Running';
      
      // Update sidebar status
      document.querySelector('.quick-status .status-dot').className = 'status-dot';
      document.querySelector('.quick-status .status-dot').style.background = '#3b82f6'; // Blue for loading
      document.querySelector('.quick-status .status-desc').textContent = 'Rebuilding index...';
      
    } else if (data.status === 'success') {
      if (document.getElementById('ingest-status-box').style.display === 'block') {
        // Just finished
        toggleIngestUI(false);
        alert(data.message || 'Vector Database rebuilt successfully!');
        loadDocumentList();
        checkSystemStatus();
      }
      
      // Reset sidebar status
      document.querySelector('.quick-status .status-dot').style.background = '';
      document.querySelector('.quick-status .status-dot').className = 'status-dot online';
      document.querySelector('.quick-status .status-desc').textContent = 'All models online';
      
      const lastRun = data.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString() : 'Unknown';
      document.getElementById('ingest-last-run').textContent = `Last build: ${lastRun}`;
      
    } else if (data.status === 'error') {
      if (document.getElementById('ingest-status-box').style.display === 'block') {
        toggleIngestUI(false);
        alert(`Ingestion Error: ${data.message}`);
        checkSystemStatus();
      }
      document.getElementById('ingest-last-run').textContent = `Index Build failed!`;
    }
  } catch (err) {
    console.error('Ingest status check failed:', err);
  }
}
