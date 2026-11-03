"""
0단계 STT 확인
- faster-whisper large-v3-turbo(fp16)로 한국어 녹음 파일을 변환한다.
- 처리 속도(RTF)와 GPU 메모리 사용량을 출력한다.
사용: python check_stt.py test.wav --runs 5
"""
import argparse
import subprocess
import time

from faster_whisper import WhisperModel
from tqdm import tqdm


def gpu_used_mb() -> int:
    """GPU 0번의 현재 사용 메모리(MB)를 nvidia-smi로 읽는다."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def transcribe(model, wav):
    """음성 파일 전체를 변환해 (텍스트, 오디오 길이)를 돌려준다.
    transcribe()는 제너레이터를 돌려주므로 끝까지 읽어야 실제 변환이 끝난다."""
    segments, info = model.transcribe(wav, language="ko", beam_size=1)
    text = "".join(seg.text for seg in segments).strip()
    return text, info.duration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", help="16kHz 모노 한국어 녹음 파일")
    p.add_argument("--runs", type=int, default=5, help="속도 측정 반복 횟수")
    args = p.parse_args()

    base = gpu_used_mb()

    # 1) 모델 로드 (처음 실행 시 Hugging Face에서 약 1.6GB 다운로드, 진행 막대가 따로 뜸)
    t0 = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    load_s = time.perf_counter() - t0
    after_load = gpu_used_mb()

    # 2) 워밍업: 첫 변환은 CUDA 초기화 때문에 느려서 측정에서 뺀다
    transcribe(model, args.wav)

    # 3) 반복 측정
    times, peak = [], after_load
    for _ in tqdm(range(args.runs), desc="STT 측정", unit="회"):
        t0 = time.perf_counter()
        text, dur = transcribe(model, args.wav)
        times.append(time.perf_counter() - t0)
        peak = max(peak, gpu_used_mb())

    avg = sum(times) / len(times)
    print(f"\n인식 결과 : {text}")
    print(f"오디오 길이: {dur:.2f}초")
    print(f"모델 로드  : {load_s:.1f}초")
    print(f"평균 처리  : {avg:.3f}초 (RTF {avg / dur:.3f}, 기준: 5초 음성을 1초 이내)")
    print(f"GPU 메모리 : 로드 후 +{after_load - base} MB, 측정 중 최대 +{peak - base} MB")


if __name__ == "__main__":
    main()