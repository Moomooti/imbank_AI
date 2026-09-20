
const SERVER_BASE = "http://127.0.0.1:5000";

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "translate_batch") {
    fetch(`${SERVER_BASE}/api/translate_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: msg.texts, lang: msg.lang }),
    })
      .then((r) => r.json().then((data) => ({ status: r.status, data })))
      .then(({ status, data }) => {
        if (status !== 200) {
          sendResponse({ ok: false, error: data.error || `서버 오류 (${status})` });
        } else {
          sendResponse({ ok: true, data });
        }
      })
      .catch((err) => {
        sendResponse({
          ok: false,
          error: `번역 서버에 연결할 수 없습니다 (${SERVER_BASE}). 실행.bat으로 서버를 먼저 켜주세요. (${err})`,
        });
      });
    return true; 
  }
});
