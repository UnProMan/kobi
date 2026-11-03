#!/usr/bin/env bash
# 코비 서버 한 번에 켜기
#   1) TTS 서비스(MOSS fast_api.py, kobi-tts 환경)를 뒤에서 띄우고
#   2) 코비 서버(server/app.py, kobi 환경)를 앞에서 띄운다. (TTS가 뜰 때까지 알아서 기다림)
#   Ctrl+C 한 번이면 코비 서버 → TTS 서비스 순서로 모두 꺼진다.
# 사용: ./start.sh      (처음 한 번: chmod +x start.sh)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"          # 이 스크립트가 있는 kobi 폴더
TTS_DIR="$ROOT/tts/MOSS-TTS/moss_tts_realtime"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# conda activate를 스크립트 안에서 쓰기 위한 준비
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 1) TTS 서비스: setsid로 따로 띄워서 Ctrl+C가 바로 전달되지 않게 한다
#      (코비 서버가 먼저 TTS 세션을 닫고 꺼진 뒤에 TTS를 끄기 위함)
conda activate kobi-tts
cd "$TTS_DIR"
setsid python fast_api.py --host 127.0.0.1 --port 8083 > "$LOG_DIR/tts.log" 2>&1 &
TTS_PID=$!
echo "[start] TTS 서비스 시작 (pid $TTS_PID, 로그: logs/tts.log)"

stop_tts() {
    if kill -0 "$TTS_PID" 2>/dev/null; then
        echo "[start] TTS 서비스 종료"
        kill -INT -- "-$TTS_PID" 2>/dev/null || true      # setsid로 만든 프로세스 그룹 전체에 전달
        wait "$TTS_PID" 2>/dev/null || true
    fi
}
trap stop_tts EXIT

# ---- 2) 접속 주소 안내
IP="$(hostname -I | awk '{print $1}')"
PORT="$(grep -A2 '^server:' "$ROOT/server/config.yaml" | awk '/port:/{print $2}')"
echo "[start] 준비되면 브라우저에서 접속:  http://$IP:$PORT/   (이 PC에서는 http://127.0.0.1:$PORT/)"

# ---- 3) 코비 서버 (앞에서 실행, Ctrl+C로 종료)
conda activate kobi
cd "$ROOT"
python server/app.py || true