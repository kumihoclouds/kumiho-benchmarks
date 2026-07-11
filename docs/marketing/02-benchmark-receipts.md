# Kumiho Memory — 벤치마크 영수증 (전 수치 + 출처)

> 2026-07-09 기준. 모든 수치는 원본 파일 재계산으로 검증됨 (검증 에이전트 대조 완료).

## 1. LoCoMo 공개 수치 (full-suite, n=1,986, token-F1)

출처: kumiho-benchmarks README.md (0.10.1 온톨로지 스택 = kumiho-memory 0.10.1 + kumiho-server v1.5.0, 2026-07-11, 답변 모델 `gpt-4o-2024-08-06` 핀·`system_fingerprint` 응답별 로깅) · baseline은 arXiv 2504.19413 + Memobase 공개 평가

| 시스템 | single-hop | multi-hop | temporal | open-domain | overall F1 |
|---|---|---|---|---|---|
| Zep | 0.357 | 0.194 | 0.420 | 0.496 | — |
| Mem0 | 0.387 | 0.286 | 0.489 | 0.477 | ~0.40 |
| Mem0-Graph | 0.381 | 0.243 | 0.516 | 0.493 | ~0.40 |
| Memobase | 0.463 | 0.229 | 0.642 | 0.516 | — |
| OpenAI Memory | — | — | — | — | ~0.343 |
| Kumiho (Feb 2026, cosine) <sup>†</sup> | 0.462 | 0.355 | 0.533 | 0.290 | 0.565 |
| Kumiho (v0.9.0, 2026-07-08) <sup>†</sup> | 0.449 | 0.393 | 0.530 | 0.313 | 0.564 |
| **Kumiho (0.10.1 온톨로지, 2026-07-11, 핀)** | **0.424** | **0.361** | **0.457** | **0.248** | **0.531** |

<sup>†</sup> *언핀·이후 드리프트된 `gpt-4o`로 측정된 드리프트-era 최고치 — 현재 스냅샷에서 재현 불가(아래 재현성 주석 참조). 볼드 행이 `gpt-4o-2024-08-06` 핀에서 오늘 재현 가능한 수치.*

- Kumiho overall은 adversarial(0.955, n=446) 포함 — baseline 대부분이 별도 보고하지 않음 (README에 명시적 공개).
  adversarial 제외 4카테고리 = 0.408.
- **재현성 (재정의)**: 예전 "Feb 0.565 ≈ Jul 0.564, 5개월 재현" 주장은 폐기 — 그 두 수치는 *그날의* 언핀 `gpt-4o`로 측정된
  드리프트-era 최고치이며, 동일 회수 컨텍스트를 이후 스냅샷에서 재응답하면 모든 스냅샷에서 −0.056 하락(frozen-context 실험).
  즉 현재 모델에서 재현 불가·타 날짜와 직접 비교 불가. 재현 가능한 것은 **메모리/검색 레이어(방법)와 핀 모델 수치(0.531)** 이지,
  드리프트하는 모델 위의 절대 점수가 아님 — 그래서 이제 `gpt-4o-2024-08-06`로 핀하고 `system_fingerprint`를 응답별 로깅.
- **온톨로지 효과 (동일-era, like-for-like)**: 같은 코퍼스·같은 날·같은 핀에서 ON=0.531 vs OFF 컨트롤=0.521 → **+0.010** (gate-v2 컨트롤).
- **가장 강한 주장**: multi-hop 0.361 > 2월 기록 0.355 — 답변 모델이 약해졌는데도 **검색 레이어는 오히려 개선**.
  회수 정확도는 Kumiho가 통제하는 층이고, end-to-end F1은 앱이 고른 답변 모델에 따라 추가로 스케일(0.564→0.531은 회귀가 아니라 답변 모델 드리프트의 몫).
- summarized 모드 (title+summary 메타데이터만, 원문 대화 미포함) 달성.

## 2. LoCoMo-Plus (인지 벤치마크)

- **93.3%** — 모순 처리·신념 갱신·시점 인지 확장 스위트. README 공개.
- v0.9.0 회복 과정에서 entry-by-entry 무회귀 검증됨.

## 3. fact-recall 레그 페어 측정 (2026-07-09)

**설계**: 같은 코퍼스(locomo-c26-p41facts, conv-26 199문항), answer-only 재평가, 변수는 fact 레그 하나.
컨트롤 = 레그 미발화 런 / 트리트먼트 = 발화 165회 런. 인제스트 노이즈 0.

| 카테고리 | n | 컨트롤 | Kumiho | Δ |
|---|---|---|---|---|
| multi-hop | 32 | 0.213 | 0.299 | **+0.086** |
| temporal | 37 | 0.510 | 0.561 | **+0.051** |
| single-hop | 70 | 0.272 | 0.315 | **+0.043** |
| open-domain | 13 | 0.274 | 0.312 | **+0.038** |
| adversarial | 47 | 0.872 | 0.894 | **+0.021** |
| **4카테고리 가중** | 152 | 0.318 | **0.371** | **+0.054** |
| overall(5) | 199 | 0.449 | 0.495 | +0.046 |

- **문항별: 23승 172무 4패** (부호검정 p≈0.0002). 전 카테고리 상승 = additive 설계의 실증.
- 라벨 주의: **+0.054는 4카테고리(adversarial 제외) 가중**, 5카테고리 overall은 +0.046. 혼용 금지.
- 원본: `results-c26-p41facts/` vs `results-c26-p41nofire/` 의 `_checkpoint.jsonl` (question_id 매칭 재계산).

## 4. 실전 win 사례 (페어에서 추출, 원문)

### conv-26_q3 [multi-hop] Δf1 +1.00
- Q: *What did Caroline research?* / gold: **Adoption agencies**
- 컨트롤: "Counseling and mental health" (F1 0.00 — 유사 주제 오답)
- Kumiho: "Adoption agencies" (F1 1.00)
- 컨텍스트에 추가된 fact 조각: `"Caroline is researching adoption agencies to provide a home for children"`

### conv-26_q126 [single-hop] Δf1 +1.00
- Q: *What activity did Caroline used to do with her dad?* / gold: **Horseback riding**
- 컨트롤: "Camping" (0.00) → Kumiho: "Horseback riding" (1.00)

### conv-26_q179 [adversarial] Δf1 +1.00
- Q: *Where did Oscar hide his bone once?* (본문에 정답 없음 — 함정 문항)
- 컨트롤: 무관 텍스트 출력 (0.00) → Kumiho: "No information available." (1.00)
- 추가된 fact("Melanie has a dog named Luna and a cat named Oliver")가 모델에게 "Oscar 정보는 없음"을 판단할 근거를 제공

## 5. 지표 스탠스 (F1 vs LLM-judge)

- LoCoMo 공식 지표 = **토큰 F1 + Porter 스테밍** (Maharana et al. 2024).
- LLM-as-judge 정확도는 1.5–2× 높게 읽히며 상호 비교 불가 → Kumiho는 F1로만 공개.
- Memobase/Zep 등은 J만 공개하는 경우가 있어 정면 F1 비교가 안 되는 항목은 "—"로 표기.

## 6. 측정 방법론 (신뢰의 근거)

1. **페어 A/B**: 같은 코퍼스 + answer-only 재평가로 읽기 경로만 분리. 단일 런 델타로 판정하지 않음.
2. **분산의 실체**: 단일 대화(conv-26) 4catW는 같은 코드로도 0.307–0.427 밴드 (2일간 n=9 실측) —
   ingest LLM 비결정성이 지배 항. full-suite에서만 분산이 죽음.
3. **프리플라이트**: 유료 런 전 1문항 드라이런으로 측정 대상 기능의 발화 로그 확인.
4. **비용 실측**: conv-26 1런 ≈ $1.25 (gpt-4o 답변 ~425k in / mini 인제스트 $0.03) · full-suite 1런 ≈ $12–13.

## 7. 재현 방법

```bash
pip install kumiho-memory
# 로컬 Kumiho CE + Neo4j + Redis 기동 후:
export KUMIHO_MEMORY_ONTOLOGY=1
python -m kumiho_eval.locomo_eval \
  --data locomo/data/locomo10.json --answer-model gpt-4o-2024-08-06 \
  --recall-limit 3 --context-top-k 5 --recall-candidate-multiplier 3 \
  --recall-mode summarized --no-judge
```
