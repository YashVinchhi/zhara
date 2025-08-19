(function(){
  async function fetchJSON(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn('Fetch failed for', url, e);
      return null;
    }
  }

  async function populateModels() {
    const sel = document.querySelector('#model-select');
    if (!sel) return;
    sel.innerHTML = '<option value="">Loading models...</option>';
    const data = await fetchJSON('/models');
    let models = Array.isArray(data) ? data : [];
    if (!models || models.length === 0) {
      // Fallback option if Ollama unavailable
      sel.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = 'offline';
      opt.textContent = 'Offline';
      sel.appendChild(opt);
      sel.value = 'offline';
      return;
    }
    sel.innerHTML = '';
    models.forEach(m => {
      if (!m || !m.name) return;
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      sel.appendChild(opt);
    });
    // Select first model by default
    if (sel.options.length > 0) sel.selectedIndex = 0;
  }

  async function populateCoquiVoices() {
    const sel = document.querySelector('#voice-select');
    if (!sel) return;
    const data = await fetchJSON('/tts/voices?provider=coqui');
    const voices = Array.isArray(data) ? data : [];
    sel.innerHTML = '';
    if (!voices.length) {
      const opt = document.createElement('option');
      opt.value = 'default';
      opt.textContent = 'Default';
      sel.appendChild(opt);
      return;
    }
    voices.forEach(v => {
      if (!v || !v.name) return;
      const opt = document.createElement('option');
      opt.value = v.name; // e.g., coqui:tts_models/... or similar
      opt.textContent = v.display_name || v.name;
      sel.appendChild(opt);
    });
  }

  async function init() {
    await Promise.all([
      populateModels(),
      populateCoquiVoices()
    ]);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

