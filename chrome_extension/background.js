// FinHOLLY 페이지 번역 — background service worker.
//
// 실제 네트워크 호출은 content script(페이지 컨텍스트, 페이지의 CSP에
// 걸릴 수 있음)가 아니라 여기 background에서 한다. background는 페이지의
// Content-Security-Policy 제약을 받지 않는 확장 전용 컨텍스트라 로컬
// 서버(localhost:5000) 호출이 더 안정적이다.

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
    return true; // 비동기 응답을 위해 채널을 열어둠
  }
});
