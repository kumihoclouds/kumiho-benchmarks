# Kumiho Memory — 기술 레퍼런스 (덱 백업 자료)

> 2026-07-09 기준. 덱/쇼케이스의 모든 기술 주장의 상세 근거.
> 코드 기준: kumiho-memory `feat/memory-write-time-ontology` (커밋 6152067), kumiho-server PR#35.
> **원천 논문**: Young Bin Park, *"Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision
> Semantics for Versioned Memory Architectures"*, [arXiv:2603.17244](https://arxiv.org/abs/2603.17244) (2026-03).
> Prospective indexing·event extraction·SUPERSEDES(AGM 형식 의미론)의 최초 제안.

## 1. kref 주소 체계

```
kref://<project>/<space-path>/<item-slug>.<kind>?r=<revision>
```

- **space가 타입을 인코딩**: 대화 `personal/<user>`, 타입 노드는 `facts/ decisions/ events/ entities/ actions/ questions/`
- **item-slug = 내용 기반 아이덴티티**: 제목/주장의 슬러그 + 해시 접미(충돌 방지). 같은 주제 재등장 → get-or-create → 리비전 스택
- **?r=N**: 모든 기억이 버저닝. head = 최신, `get_revision_as_of`로 시점 조회

## 2. 저장 파이프라인

```
대화 턴 → Redis Streams 세션 버퍼 → (임계치) → Consolidation (LLM 1콜, 스키마 강제)
→ conversation revision 저장 (+서버측 임베딩) → 온톨로지 분해 (결정론적, LLM 0콜)
```

### 통합(Consolidation) 구조화 출력 스키마 (요지)
```json
{
  "summary": "...",
  "classification": { "entities": [...] },
  "knowledge": {
    "facts":      [ {"claim": "...", "certainty": "high|medium|low"} ],
    "decisions":  [ {"decision": "...", "reason": "..."} ],
    "actions":    [ {"task": "...", "status": "..."} ],
    "open_questions": ["..."]
  },
  "events": [ {"event": "...", "when": "...", "participants": [...], "consequence": "..."} ]
}
```
- **스키마는 온톨로지 ON/OFF와 무관하게 바이트 동일** (테스트 `test_summary_schema_is_identical_in_both_ontology_modes`로 보증).
  온톨로지 게이트 필드(`based_on`)를 시도했다가 구조화 출력 자체가 변형되는 것을 실측하고 제거 —
  DEPENDS_ON은 사후 토큰 중첩으로 유도.
- 대화 리비전에는 summary 외에 `facts / events / entities / topics / implications / event_date /
  session_id / message_count` 구조화 필드가 함께 저장됨 (실물 예시는 03-live-data-examples.md).

### 온톨로지 분해 (materializer) — 전부 결정론적
| 엣지 | 규칙 | 임계 |
|---|---|---|
| DERIVED_FROM | 타입 노드 → 원본 대화. 항상 생성 | — |
| ABOUT | fact/decision → 엔티티. **토큰 경계 멘션 매칭** (부분문자열 오탐 없음, Hangul-aware: "김"≠"김치") | — |
| INVOLVES | event → participants 엔티티 | — |
| DEPENDS_ON | decision → 같은 통합 내 최고 중첩 fact | Jaccard ≥ 0.4 (top-1만) |
| SUPERSEDES | 새 주장 → 같은 주제의 옛 주장. 후보는 kind-scoped 검색으로 찾고 **판정은 토큰 중첩** (랭킹 점수 아님 — 코퍼스 독립) | Jaccard ≥ 0.6 |

- 임계 미달이면 엣지를 만들지 않음 ("불확실하면 연결하지 않는다").
- 스위치: **기본 ON** (2026-07-10 옵트아웃 전환 — 페어 실측 근거: 읽기 +0.042, fact +0.054). `KUMIHO_MEMORY_ONTOLOGY=0`으로 옵트아웃하면 전 경로가 레거시와 바이트 동일.

## 3. 인덱스 위생 (kumiho-server PR#35)

```rust
pub const FULLTEXT_EXCLUDED_KINDS: &[&str] =
    &["entity", "fact", "decision", "event", "action", "question"];
```
- 파생 kind의 리비전/아이템 `_search_text`를 비움 → 렉시컬(BM25) 코퍼스에서 제외.
  이유: 타입 노드 텍스트는 대화의 복제 → 전역 IDF 왜곡 (실측 4cat −0.031, temporal −0.065).
- 아이템 name/kind 필드는 인덱스 유지 (SUPERSEDES 후보 탐색이 슬러그 매칭으로 생존).
- **임베딩은 유지** → 타입 노드는 벡터 검색과 kref 순회로 완전 도달 가능.
- 레거시 데이터: idempotent 배치 백필이 옛 블롭 정리 (수동 rebuild 필요 — ops 문서화 요청 상태).

## 4. 리콜 파이프라인 (7단계)

| 단계 | 내용 | 주요 설정 (기본값) |
|---|---|---|
| ① 리포뮬레이션 | 질문 → N개 각도 (LLM) | reformulate_queries=True |
| ② 하이브리드 병렬 리콜 | 각도별 BM25+벡터, kref 병합(최고점), 각도 귀속 보존 | recall_limit=3, candidate_multiplier=3 |
| ③ 구조 순회 | 상위 히트의 엣지 (DERIVED_FROM/SUPERSEDES 등, BOTH 방향), score-less | top_k_for_traversal=5 |
| ④ **브리지 조인** | ≥2 각도가 ABOUT으로 도달한 엔티티 = 브리지 → fact/event 노드에 **상속 점수 = 0.9 × 약한 각도** | entity_bridge_score_factor=0.9, max_results=4, hub_degree_max=12 (**defer** not drop) |
| ⑤ **fact 레그** | 원 질문 1회로 `{project}/facts` 시맨틱 검색. **점수 = 0.9 × 최약 base 히트** (축-불변) | fact_recall_limit=3, max_results=2 |
| ⑥ 재랭크 | evidence level · recency · event-proximity(시간 질문만) · MMR. 크로스인코더는 **각도별 서브쿼리 안에서만** | |
| ⑦ 추가-비대체 조립 | base top-K 원형 유지 + 브리지 ≤2 + fact ≤2 온톱 | context_top_k=5 |

핵심 불변식:
- **additive-only**: 구조 증거는 어떤 단계에서도 base 대화 히트를 대체/축출 불가 (캡·트림·조립 3단 예산 미러링).
- **retrieve 툴 우회**: 타입 노드는 published/latest 태그가 없어 retrieve 툴이 버림 → fact 레그는 `kumiho.search` 직접 호출.
- **스코프 정합**: space_paths 스코프 콜은 같은 프로젝트의 facts로 유도(크로스-프로젝트 격리 유지), memory_types 필터 시 레그 스킵.
- fact 레그 실패/타임아웃 → 강등만, 리콜은 죽지 않음 (bounded thread, seen-kref는 await 후에만 클레임).

## 5. 서버 하이브리드 검색

- 레그: item_fulltext(1.0) · revision_fulltext(0.9) · revision **vector**(0.85) · artifact(0.8)
- **레그별 min-max 정규화 후 가중 합산** — BM25 무한 스케일이 코사인을 짓누르던 legacy max-of-raw를 대체, 다중 레그 동의에 가산
- 벡터 레그: `db.index.vector.queryNodes(k=10)` 전역 → 스코프 필터 (개선 예정: 스코프 내장/k 스케일링)
- 퍼지: 편집거리 1 (`getting~1 pet~1`), 한국어 lindera ko-dic 형태소 인덱싱
- 쿼리 임베딩 5초 타임아웃 → 풀텍스트-온리 강등 (경고 로그), 검색 무중단

## 6. 스택 요약
Rust(tokio/tonic gRPC) 서버 · Neo4j(그래프+VECTOR 인덱스) · Redis Streams · text-embedding-3-small(1536d, 서버측) ·
LLM 구조화 출력(provider-agnostic) · Python asyncio SDK · 429+ 유닛테스트 · 페어 A/B 하네스
