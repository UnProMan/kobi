"""
0단계 임베딩 확인
- Ollama의 bge-m3로 문장 임베딩을 만들고 코사인 유사도를 비교한다.
- 같은 뜻의 질문이 무관한 문장보다 점수가 높으면 통과.
"""
import time

import numpy as np
import requests
from tqdm import tqdm

URL = "http://localhost:11434/api/embed"

# (기준 질문, 같은 뜻의 다른 표현, 무관한 문장)
CASES = [
    ("컴퓨터공학부 건물은 어디에 있나요?", "컴공 건물 어디야?", "오늘 점심 메뉴 뭐야?"),
    ("이 행사에서 뭘 할 수 있어?", "여기서 체험할 수 있는 프로그램 알려줘", "버스 시간표 알려줘"),
    ("같이 사진 찍을래?", "나랑 사진 한 장 찍자", "도서관은 몇 시에 닫아?"),
]


def embed(texts):
    """문장 목록을 임베딩해 길이 1로 정규화한 행렬을 돌려준다."""
    r = requests.post(URL, json={"model": "bge-m3", "input": texts, "keep_alive": -1}, timeout=60)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    passed, times = 0, []
    for base, para, other in tqdm(CASES, desc="임베딩 비교", unit="쌍"):
        t0 = time.perf_counter()
        v = embed([base, para, other])
        times.append(time.perf_counter() - t0)
        s_para, s_other = float(v[0] @ v[1]), float(v[0] @ v[2])  # 정규화했으므로 내적 = 코사인
        ok = s_para > s_other
        passed += ok
        tqdm.write(f"[{'통과' if ok else '실패'}] {base}\n    같은 뜻 {s_para:.3f} / 무관 {s_other:.3f}")

    # 첫 호출은 모델 로드 시간이 섞이므로 평균에서 뺀다
    avg_ms = np.mean(times[1:]) * 1000
    print(f"\n결과: {passed}/{len(CASES)} 통과, 호출당 평균 {avg_ms:.0f} ms")


if __name__ == "__main__":
    main()