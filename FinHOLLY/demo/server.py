"""FinHOLLY 온프레미스 데모 서버.

완전히 로컬에서 동작한다(인터넷 연결 불필요). 실행.bat이 이 파일을 실행하고,
서버가 뜨면 자동으로 기본 브라우저가 열린다. 이 콘솔 창을 닫으면 서버도
같이 종료된다.
"""
from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import config  # noqa: E402
from engine import get_engine  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
URL = "http://127.0.0.1:5000"
MAX_BATCH_TEXTS = 200  # 크롬 확장이 한 번에 보낼 수 있는 최대 문장 수 (페이지당 과호출 방지)

flask_app = Flask(__name__, static_folder=None)


@flask_app.after_request
def add_cors_headers(response):
    # 크롬 확장(별도 origin: chrome-extension://...)에서 로컬 서버를 호출할 수
    # 있게 허용한다. 이 서버는 127.0.0.1에서만 열리는 온프레미스 서버라
    # 와일드카드 허용의 실질적 위험이 낮다(외부 인터넷에 노출 안 됨).
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@flask_app.route("/api/translate", methods=["OPTIONS"])
@flask_app.route("/api/translate_batch", methods=["OPTIONS"])
def cors_preflight():
    return "", 204


@flask_app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@flask_app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "번역할 한국어 문장을 입력해 주세요."}), 400
    if len(text) > 500:
        return jsonify({"error": "문장이 너무 깁니다 (500자 이내)."}), 400

    engine = get_engine()
    t0 = time.perf_counter()
    try:
        results = engine.translate_all(text)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"번역 실패: {e!r}"}), 500
    elapsed = time.perf_counter() - t0

    return jsonify({
        "source": text,
        "translations": [
            {"lang": lang, "name": name, "text": results[lang]}
            for lang, name in config.SUPPORTED_LANGS.items()
        ],
        "elapsed_sec": round(elapsed, 2),
    })


@flask_app.route("/api/translate_batch", methods=["POST"])
def api_translate_batch():
    """크롬 확장 등 여러 문장을 한 번에 번역해야 하는 클라이언트용.
    한 언어로만 번역한다(요청마다 5개 언어를 다 만들지 않음 -- 페이지 번역은
    보통 한 언어만 필요하므로 5배 낭비를 피한다)."""
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    lang = data.get("lang")

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "texts는 비어있지 않은 문자열 배열이어야 합니다."}), 400
    if lang not in config.SUPPORTED_LANGS:
        return jsonify({"error": f"지원하지 않는 언어 코드: {lang!r}. 지원: {list(config.SUPPORTED_LANGS)}"}), 400
    if len(texts) > MAX_BATCH_TEXTS:
        return jsonify({"error": f"한 번에 보낼 수 있는 문장은 최대 {MAX_BATCH_TEXTS}개입니다."}), 400

    texts = [str(t)[:500] for t in texts]  # 문장당 길이 방어

    engine = get_engine()
    t0 = time.perf_counter()
    try:
        translations = engine.translate_batch(texts, lang)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"번역 실패: {e!r}"}), 500
    elapsed = time.perf_counter() - t0

    return jsonify({
        "lang": lang,
        "translations": translations,
        "count": len(translations),
        "elapsed_sec": round(elapsed, 2),
    })


@flask_app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "langs": list(config.SUPPORTED_LANGS.keys())})


if __name__ == "__main__":
    print("=" * 50)
    print(" FinHOLLY 온프레미스 번역 서버")
    print("=" * 50)
    print("모델 로딩 중... (수 초 정도 걸립니다)")
    get_engine()
    print(f"로드 완료. 브라우저를 엽니다: {URL}")
    print("\n※ 이 창을 닫으면 서버가 종료됩니다.\n")
    webbrowser.open(URL)
    flask_app.run(host="127.0.0.1", port=5000, debug=False)
