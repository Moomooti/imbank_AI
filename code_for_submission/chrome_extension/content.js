
(function () {
  if (!window.__finhollyState) {
    window.__finhollyState = {
      originalTextMap: new WeakMap(), 
      currentLang: null, 
    };
  }
  const state = window.__finhollyState;

  const KOREAN_RE = /[가-힣ᄀ-ᇿ㄰-㆏]/;
  const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT", "OPTION", "CODE", "PRE"]);
  const BATCH_SIZE = 20; 

  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden" && el.offsetParent !== null;
  }

  function collectTextNodes() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
        if (!node.textContent || !node.textContent.trim()) return NodeFilter.FILTER_SKIP;
        if (!isVisible(parent)) return NodeFilter.FILTER_SKIP;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);
    return nodes;
  }

  function chunk(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
  }

  function sendToBackground(texts, lang) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: "translate_batch", texts, lang }, resolve);
    });
  }

  async function translatePage(lang) {
    if (lang === state.currentLang) {
      return { ok: true, count: 0, note: "이미 이 언어로 번역되어 있습니다." };
    }

    const allNodes = collectTextNodes();
    const targets = [];
    for (const node of allNodes) {
      if (!state.originalTextMap.has(node)) {
        
        if (!KOREAN_RE.test(node.textContent)) continue;
        state.originalTextMap.set(node, node.textContent);
      }
      targets.push(node);
    }

    if (targets.length === 0) {
      return { ok: true, count: 0, note: "번역할 한국어 텍스트를 찾지 못했습니다." };
    }

    const batches = chunk(targets, BATCH_SIZE);
    const t0 = performance.now();
    let done = 0;

    for (const batch of batches) {
      const texts = batch.map((node) => state.originalTextMap.get(node));
      const resp = await sendToBackground(texts, lang);
      if (!resp || !resp.ok) {
        return { ok: false, error: (resp && resp.error) || "번역 서버 응답을 받지 못했습니다.", doneBeforeError: done };
      }
      const translations = resp.data.translations;
      batch.forEach((node, i) => {
        if (translations[i] !== undefined) node.textContent = translations[i];
      });
      done += batch.length;
    }

    state.currentLang = lang;
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    return { ok: true, count: done, elapsed };
  }

  function restoreOriginal() {
    
    const allNodes = collectTextNodes();
    let restored = 0;
    for (const node of allNodes) {
      if (state.originalTextMap.has(node)) {
        node.textContent = state.originalTextMap.get(node);
        restored++;
      }
    }
    state.currentLang = null;
    return { ok: true, count: restored };
  }

  if (!window.__finhollyListenerRegistered) {
    window.__finhollyListenerRegistered = true;
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (msg.action === "translate") {
        translatePage(msg.lang).then(sendResponse);
        return true;
      }
      if (msg.action === "restore") {
        sendResponse(restoreOriginal());
        return false;
      }
    });
  }
})();
