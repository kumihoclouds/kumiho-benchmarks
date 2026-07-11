# Kumiho Memory — 실데이터 예시 원장 (Neo4j에서 직접 추출)

> 2026-07-09, 프로젝트 `locomo-c26-p41facts` (LoCoMo conv-26, ONTOLOGY=1 인제스트).
> 덱/쇼케이스의 모든 LIVE DATA 예시의 원본. bolt 쿼리로 재추출 가능.

## 1. 실제 대화 리비전 (전 필드)

```
kref://locomo-c26-p41facts/personal/locomo-conv-26/caroline-shares-her-transgender-journey-at-a-sch-f2c3a633.conversation?r=1

schema        : kumiho.agent_memory.v1
type          : summary          memory_type: summary
title         : Caroline shares her transgender journey at a school event
summary       : On 9 June 2023, Caroline recounted her experience at a school event where she
                spoke about her transgender journey and advocated for increased involvement in th…
facts         : Caroline started transitioning three years ago; Caroline spoke about her journey
                at a school event on 9 June 2023; Caroline has known her friends for 4 years s…
events        : [one week before 9 June 2023] Caroline spoke about her transgender journey at a
                school event; [4 years ago] Caroline moved from her home country and formed a s…
entities      : Caroline, Melanie
topics        : LGBTQ advocacy, support systems, family
event_date    : 2019-06
implications  : Caroline organizes a series of educational workshops at school, inspired by her
                previous event. / Melanie begins to volunteer at a local LGBTQ organization, fee…
session_id    : personal:user-0e90700f10:20260709:221
message_count : 23
embedding     : [1536-dim] (embedding_text = title + "\n" + summary)
tag_history   : summarized(2026-07-09T08:49:53Z) → published
space         : /locomo-c26-p41facts/personal/locomo-conv-26
```

- 이 아이템에는 **리비전 19개가 스택**되어 있음 (conv-26 세션들이 통합될 때마다 get-or-create로 적층).

## 2. 온톨로지 분해 결과 (이 프로젝트 전체, Item 카운트)

| kind | 개수 |
|---|---|
| fact | 103 |
| event | 77 |
| entity | 15 |
| bundle | 11 |
| question | 10 |
| action | 8 |
| decision | 8 |
| conversation | 1 (리비전 19개) |

## 3. 엣지 실측 (Revision→Revision 관계)

| 엣지 | 개수 |
|---|---|
| DERIVED_FROM | 208 |
| ABOUT | 175 |
| INVOLVES | 89 |
| SUPERSEDES | 29 |
| DEPENDS_ON | 0 (임계 Jaccard ≥ 0.4 미달 — 이 대화의 decision 8건은 근거 fact와 중첩 부족) |

## 4. fact 노드 + ABOUT 클러스터 실례

```
kref://locomo-c26-p41facts/facts/becoming-nicole-by-amy-ellis-nutt-inspi-05578a27.fact?r=1
  claim: "'Becoming Nicole' by Amy Ellis Nutt inspired Caroline"
  ──ABOUT──▶ kref://locomo-c26-p41facts/entities/caroline.entity?r=1

kref://locomo-c26-p41facts/facts/caroline-aims-to-create-a-safe-and-lovi-ae391e96.fact?r=1
  claim: "Caroline aims to create a safe and loving home for children in need"
  ──ABOUT──▶ …/entities/caroline.entity?r=1

kref://locomo-c26-p41facts/facts/caroline-and-melanie-attended-a-pride-f-587bdd8b.fact?r=1
  claim: "Caroline and Melanie attended a Pride fest together last year."
  ──ABOUT──▶ …/entities/caroline.entity?r=1
  ──ABOUT──▶ …/entities/melanie.entity?r=1     # 다중 앵커 = 브리지 조인 후보 증거
```

## 5. SUPERSEDES 실례 (신념 갱신)

```
NEW  kref://locomo-c26-p41facts/facts/caroline-attended-a-pride-parade-on-11-d0bc603e.fact?r=1
     "Caroline attended a pride parade on 11 August 2023"
  │ SUPERSEDES (토큰 중첩 Jaccard ≥ 0.6)
  ▼
OLD  kref://locomo-c26-p41facts/facts/caroline-attended-an-lgbtq-pride-parade-5a153231.fact?r=1
     "Caroline attended an LGBTQ+ pride parade on 3 July 2023"
```

## 6. 엔티티 허브 디그리 (브리지 defer의 실증)

| 엔티티 | in-degree (ABOUT+INVOLVES) | 브리지 판정 |
|---|---|---|
| entities/caroline.entity | **149** | 허브 → defer (hub_degree_max=12 초과) |
| entities/melanie.entity | **82** | 허브 → defer |
| entities/family.entity | 4 | 판별 브리지 후보 |
| entities/youth-center.entity | 4 | 판별 브리지 후보 |

## 7. 재추출 쿼리 (bolt, 참고용)

```cypher
// 대화 리비전
MATCH (r:Revision) WHERE r.kref STARTS WITH 'kref://locomo-c26-p41facts/personal' RETURN r LIMIT 1;
// kind별 카운트
MATCH (i:Item) WHERE i.kref STARTS WITH 'kref://locomo-c26-p41facts/' RETURN i.kind, count(*);
// 엣지 카운트
MATCH (a:Revision)-[e]->(b:Revision) WHERE a.kref STARTS WITH 'kref://locomo-c26-p41facts/' RETURN type(e), count(*);
// SUPERSEDES 페어
MATCH (n:Revision)-[:SUPERSEDES]->(o:Revision) WHERE n.kref STARTS WITH 'kref://locomo-c26-p41facts/'
RETURN n.kref, n.summary, o.kref, o.summary LIMIT 5;
// 허브 디그리
MATCH (x:Revision)-[e:ABOUT|INVOLVES]->(ent:Revision) WHERE ent.kref STARTS WITH 'kref://locomo-c26-p41facts/entities'
RETURN ent.kref, count(e) ORDER BY count(e) DESC;
```

> ⚠️ 이 프로젝트는 full-10 게이트 병합에 쓰이므로 **게이트 완료 전 삭제 금지**.
