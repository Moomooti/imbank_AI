const LANGS = [
  { code: "vie_Latn", name: "🇻🇳 베트남어" },
  { code: "ind_Latn", name: "🇮🇩 인도네시아어" },
  { code: "tha_Thai", name: "🇹🇭 태국어" },
  { code: "tgl_Latn", name: "🇵🇭 필리핀어" },
  { code: "mya_Mymr", name: "🇲🇲 미얀마어" },
];

const selectEl = document.getElementById("lang");
const goBtn = document.getElementById("go");
const restoreBtn = document.getElementById("restore");
const statusEl = document.getElementById("status");

for (const l of LANGS) {
  const opt = document.createElement("option");
  opt.value = l.code;
  opt.textContent = l.name;
  selectEl.appendChild(opt);
}

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.className = isError ? "error" : "";
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function ensureContentScript(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
}

function sendToTab(tabId, message) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
      } else {
        resolve(response);
      }
    });
  });
}

goBtn.addEventListener("click", async () => {
  const lang = selectEl.value;
  goBtn.disabled = true;
  setStatus("번역 중... (문장이 많으면 시간이 걸릴 수 있습니다)");

  try {
    const tab = await getActiveTab();
    await ensureContentScript(tab.id);
    const resp = await sendToTab(tab.id, { action: "translate", lang });

    if (!resp || !resp.ok) {
      setStatus(`오류: ${(resp && resp.error) || "알 수 없는 오류"}`, true);
    } else if (resp.count === 0) {
      setStatus(resp.note || "번역할 내용이 없습니다.");
    } else {
      setStatus(`완료: ${resp.count}개 문장 번역 (${resp.elapsed}초)`);
    }
  } catch (err) {
    setStatus(`오류: ${err.message || err}`, true);
  } finally {
    goBtn.disabled = false;
  }
});

restoreBtn.addEventListener("click", async () => {
  restoreBtn.disabled = true;
  try {
    const tab = await getActiveTab();
    await ensureContentScript(tab.id);
    const resp = await sendToTab(tab.id, { action: "restore" });
    setStatus(resp && resp.ok ? `원문으로 복원 (${resp.count}개 문장)` : "복원할 내용이 없습니다.");
  } catch (err) {
    setStatus(`오류: ${err.message || err}`, true);
  } finally {
    restoreBtn.disabled = false;
  }
});
