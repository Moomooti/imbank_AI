"""Phase 7 — FinHOLLY 온프레미스 데모 서버.

완전히 로컬에서 동작한다(인터넷 연결 불필요) -- FinHOLLY 금융 어댑터가
병합된 CTranslate2 INT8 모델로 실시간 번역을 제공하고, 정적 HTML 페이지로
결과를 보여준다. 발표/심사 데모용.

실행: python demo/server.py  (브라우저에서 http://127.0.0.1:5000 접속)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory              

from app.mt.finholly_translate import SUPPORTED_LANGS, translate_all              
from app.mt.ctranslate2_provider import get_finholly_provider              

STATIC_DIR = Path(__file__).resolve().parent / "static"

flask_app = Flask(__name__, static_folder=None)


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

    t0 = time.perf_counter()
    try:
        results = translate_all(text)
    except Exception as e:                
        return jsonify({"error": f"번역 실패: {e!r}"}), 500
    elapsed = time.perf_counter() - t0

    return jsonify({
        "source": text,
        "translations": [
            {"lang": lang, "name": SUPPORTED_LANGS[lang], "text": results[lang]}
            for lang in SUPPORTED_LANGS
        ],
        "elapsed_sec": round(elapsed, 2),
    })


@flask_app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "langs": list(SUPPORTED_LANGS.keys())})


if __name__ == "__main__":
    print("모델 로딩 중 (온프레미스, 인터넷 불필요)...")
    get_finholly_provider()                              
    print("로드 완료. http://127.0.0.1:5000 에서 접속하세요.\n")
    flask_app.run(host="127.0.0.1", port=5000, debug=False)
