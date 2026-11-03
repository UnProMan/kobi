"""
코비 대화 서버 (2-2단계: 텍스트 입력 → 음성 답변)

흐름
  클라이언트 --(WebSocket: 텍스트)--> 서버
  서버: Ollama LLM 스트리밍 → 생성되는 글자를 바로 MOSS-TTS 세션에 밀어 넣음
  서버 --(WebSocket: 음성 조각 PCM16)--> 클라이언트

WebSocket 메시지
  클라이언트 → 서버 : {"type": "text", "text": "..."}
  서버 → 클라이언트 : {"type": "state", "value": "thinking" | "speaking"}
                      바이너리 = 음성 조각 (24kHz, 모노, PCM16)
                      {"type": "reply_end", "text": "...", "timing": {...}}
                      {"type": "error", "value": "..."}

실행: python server/app.py   (설정은 server/config.yaml)
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text(encoding="utf-8"))
OLLAMA = CFG["ollama"]
TTS = CFG["tts"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kobi")
logging.getLogger("httpx").setLevel(logging.WARNING)   # 글자 조각마다 찍히는 HTTP 요청 로그 숨기기


def ms_since(t0: float) -> int:
    """t0 이후 경과 시간(ms)."""
    return int((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------- LLM (Ollama)
async def llm_stream(client: httpx.AsyncClient, messages: list[dict]):
    """Ollama /api/chat 스트리밍 호출. 생성되는 글자 조각을 차례로 돌려준다."""
    body = {
        "model": OLLAMA["model"],
        "messages": messages,
        "stream": True,
        "think": False,                      # Qwen3 생각 모드 끄기 (첫 글자 지연 방지)
        "keep_alive": OLLAMA["keep_alive"],
        "options": {"num_ctx": OLLAMA["num_ctx"]},
    }
    async with client.stream("POST", f"{OLLAMA['url']}/api/chat", json=body, timeout=None) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():   # Ollama는 한 줄에 JSON 하나씩 보낸다
            if not line:
                continue
            data = json.loads(line)
            delta = data.get("message", {}).get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                break


async def unload_llm():
    """서버를 끌 때 Ollama에 올라가 있는 모델을 모두 GPU에서 내린다.

    Ollama는 모델마다 llama.cpp 실행 프로세스(llama-server)를 띄우고,
    모델을 내리면 그 프로세스도 같이 끝난다. LLM뿐 아니라 임베딩 모델 등
    다른 경로로 올라간 모델까지 /api/ps로 찾아서 `ollama stop`으로 모두 내린다.
    (이 서버를 같이 쓰는 다른 사람이 올린 Ollama 모델도 내려가는 점에 유의)
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA['url']}/api/ps", timeout=10)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:  # 종료 중이므로 실패해도 넘어간다
        log.warning(f"Ollama 모델 목록 확인 실패: {e}")
        return

    for name in names:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "stop", name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, err = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                log.info(f"Ollama 모델을 GPU에서 내렸습니다: {name}")
            else:
                log.warning(f"Ollama 모델 내리기 실패: {name} ({err.decode().strip()})")
        except Exception as e:
            log.warning(f"Ollama 모델 내리기 실패: {name} ({e})")


# ---------------------------------------------------------------- TTS (MOSS fast_api.py)
# MOSS fast_api.py는 세션을 새로 만들 때마다 모델을 디스크에서 다시 불러온다(약 30초).
# 그래서 세션은 서버 전체에서 하나만 만들어 계속 쓰고, 답변마다 "새 턴"만 시작한다.
# MOSS 서버는 한 번에 하나의 생성만 지원하므로(batch size 1) 잠금으로 순서를 지킨다.
TTS_SID = f"kobi-{uuid.uuid4()}"
TTS_LOCK = asyncio.Lock()


class TTSStream:
    """코비의 답변 하나 = 공용 TTS 세션의 턴 하나.

    start()로 첫 글자와 함께 턴을 시작하고, push()로 이어지는 글자를 넣고,
    finish()로 끝을 알린다. 그동안 _relay_audio()가 생성된 음성을 클라이언트로 넘긴다.
    ws가 None이면(워밍업) 음성을 받기만 하고 버린다.
    """

    def __init__(self, client: httpx.AsyncClient, ws: WebSocket | None, t0: float, timing: dict):
        self.client, self.ws, self.t0, self.timing = client, ws, t0, timing
        self.sid = TTS_SID
        self.audio_task: asyncio.Task | None = None

    async def start(self, first_text: str):
        r = await self.client.post(f"{TTS['url']}/tts/session/start", json={
            "session_id": self.sid,
            "assistant_text": first_text,
            "prompt_audio": TTS["prompt_audio"],
            "new_turn": True,
        }, timeout=60)
        r.raise_for_status()
        self.audio_task = asyncio.create_task(self._relay_audio())

    async def push(self, text: str, is_final: bool = False):
        r = await self.client.post(f"{TTS['url']}/tts/session/push", json={
            "session_id": self.sid, "text": text, "is_final": is_final,
        }, timeout=60)
        r.raise_for_status()

    async def finish(self):
        await self.push("", is_final=True)
        if self.audio_task:
            await self.audio_task            # 음성을 끝까지 다 보낼 때까지 기다림 (세션은 닫지 않음)

    async def _relay_audio(self):
        """TTS 음성 스트림을 받아 클라이언트로 그대로 전달한다."""
        url = f"{TTS['url']}/tts/session/{self.sid}/audio"
        leftover = b""                       # PCM16은 2바이트 단위라 홀수로 끊긴 1바이트를 보관
        async with self.client.stream("GET", url, timeout=None) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                if "tts_first_ms" not in self.timing:
                    self.timing["tts_first_ms"] = ms_since(self.t0)
                    if self.ws:
                        await self.ws.send_json({"type": "state", "value": "speaking"})
                data = leftover + chunk
                cut = len(data) // 2 * 2
                leftover = data[cut:]
                if cut and self.ws:
                    await self.ws.send_bytes(data[:cut])


async def wait_for_tts(timeout_s: float = 120, interval_s: float = 2):
    """TTS 서비스(MOSS fast_api.py)가 응답할 때까지 기다린다."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(f"{TTS['url']}/health", timeout=5)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            waited = time.perf_counter() - t0
            if waited > timeout_s:
                raise SystemExit(f"TTS 서비스({TTS['url']})가 {int(timeout_s)}초 동안 응답하지 않습니다. "
                                 f"kobi-tts 환경에서 fast_api.py를 먼저 실행하세요.")
            log.info(f"TTS 서비스를 기다리는 중... ({int(waited)}/{int(timeout_s)}초)")
            await asyncio.sleep(interval_s)


async def tts_warmup():
    """서버 시작 시 짧은 문장을 한 번 합성해 TTS 모델 로드를 미리 끝내 둔다."""
    log.info("TTS 워밍업 시작 (모델 로드 포함, 처음에는 30초 정도 걸림)")
    t0, timing = time.perf_counter(), {}
    async with httpx.AsyncClient() as client:
        tts = TTSStream(client, None, t0, timing)
        await tts.start("안녕! 나는 코리야.")
        await tts.finish()
    log.info(f"TTS 워밍업 완료: {ms_since(t0)} ms")


async def close_tts():
    """서버를 끌 때 공용 TTS 세션을 닫아 TTS 쪽 GPU 메모리를 풀어 준다."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{TTS['url']}/tts/session/close", json={"session_id": TTS_SID}, timeout=10)
        log.info("TTS 세션을 닫았습니다")
    except Exception as e:
        log.warning(f"TTS 세션 닫기 실패: {e}")


# ---------------------------------------------------------------- 한 번의 질문-답변
async def handle_turn(ws: WebSocket, client: httpx.AsyncClient, history: list[dict], user_text: str):
    t0 = time.perf_counter()
    timing: dict = {}
    await ws.send_json({"type": "state", "value": "thinking"})

    n = CFG["dialog"]["history_turns"] * 2
    messages = ([{"role": "system", "content": CFG["dialog"]["persona"]}]
                + history[-n:]
                + [{"role": "user", "content": user_text}])

    reply, tts = "", None
    try:
        async with TTS_LOCK:                 # 다른 대화가 TTS를 쓰는 중이면 끝날 때까지 기다림
            async for delta in llm_stream(client, messages):
                if "llm_first_ms" not in timing:
                    timing["llm_first_ms"] = ms_since(t0)
                reply += delta
                if tts is None:              # 첫 글자가 나오는 순간 TTS 턴을 시작한다
                    tts = TTSStream(client, ws, t0, timing)
                    await tts.start(delta)
                else:
                    await tts.push(delta)
            timing["llm_done_ms"] = ms_since(t0)
            if tts:
                await tts.finish()
    except Exception as e:
        log.exception("답변 생성 실패")
        await ws.send_json({"type": "error", "value": str(e)})

    timing["total_ms"] = ms_since(t0)
    history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
    await ws.send_json({"type": "reply_end", "text": reply, "timing": timing})
    log.info(f"질문: {user_text} | 답: {reply} | 시간(ms): {timing}")


# ---------------------------------------------------------------- 서버
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await tts_warmup()                       # 첫 방문객이 모델 로드를 기다리지 않게 미리 준비
    yield
    await close_tts()                        # Ctrl+C로 서버를 끄면 TTS 세션과 LLM을 GPU에서 내림
    await unload_llm()


app = FastAPI(lifespan=lifespan)
WEB_PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


@app.get("/")
def index():
    """브라우저용 대화 데모 페이지 (http://<서버IP>:<포트>/)."""
    return FileResponse(WEB_PAGE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info(f"클라이언트 연결: {ws.client}")
    history: list[dict] = []                 # 연결 하나 = 방문객 한 명과의 대화
    async with httpx.AsyncClient() as client:
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "text" and msg.get("text", "").strip():
                    await handle_turn(ws, client, history, msg["text"].strip())
        except WebSocketDisconnect:
            log.info("클라이언트 연결 종료")


if __name__ == "__main__":
    # uvicorn을 띄우기 전에 TTS 서비스부터 확인한다 (실패하면 안내 한 줄만 출력하고 종료)
    asyncio.run(wait_for_tts())
    uvicorn.run(app, host=CFG["server"]["host"], port=CFG["server"]["port"])