"""Authored, deterministic corpus for the relation-traversal A/B benchmark.

The corpus is built so that the branch's flag-gated entity->entity relation
traversal (kumiho_memory.graph_augmentation, ``relation_traversal``) can be
measured in isolation against local Community Edition, with NO LLM and NO
embeddings (fulltext search only).

Topology (mirrors what production ``decompose_and_link_agent`` writes):

* Each RELATION query has a *seed* conversation that is ``ABOUT`` entity **A**
  only, and a *gold* conversation that is ``ABOUT`` entity **B** only. A and B
  are joined solely by an entity->entity relation edge (A --predicate--> B).
* The relation edges are declared by a single *relation-source* conversation in
  a SEPARATE space (``relsrc``) that the reader's search never scopes to. Entity
  anchors are project-level (shared across spaces), so the edge lands on the same
  A/B anchors the seed/gold reference — but the seed conversation itself is never
  ``ABOUT`` B. This is what makes the gold UNreachable via plain ABOUT-sibling
  bridging in the OFF condition (verified at runtime), so any ON-only gold-reach
  is attributable to the relation edge and nothing else.
* Seed query text is lexically specific to its seed conversation and disjoint
  from every gold, so the gold is never a direct fulltext hit for a relation
  query.

Predicate mix (exercises the new predicate_registry folding on the write path):
  uses / utilizes / employs        -> USES        (canonical + 2 folded synonyms)
  depends on / relies on / requires -> DEPENDS_ON  (canonical + 2 folded synonyms)
  part of                           -> PART_OF     (canonical passthrough)
  monitors                          -> RELATES_TO  (unregistered -> fallback bucket)
"""

from __future__ import annotations

from typing import Any, Dict, List

CHAT_SPACE = "chat"       # searchable corpus (the reader scopes recall here)
RELSRC_SPACE = "relsrc"   # relation-source conversations (never searched)

# --------------------------------------------------------------------------- #
# 8 relation pairs.
#
# The SEARCHABLE text of every conversation is built around a globally UNIQUE
# codeword (Kestrel, Basalt, ...) plus unique descriptors, and every query is
# composed only of tokens that occur in its own seed conversation and nowhere
# else in the corpus.  Fulltext (BM25) therefore returns *only* the intended
# seed for each query — no noisy tail of other pairs' seeds or the gold itself
# (empirically verified by the setup index probe / scratch diagnostics).  This
# lexical orthogonality is what lets the OFF condition cleanly show the gold is
# unreachable by any non-relation path.
#
# Entity names are independent of this text: ``decompose_and_link_agent`` takes
# entities from the decomposition dict, not by scanning the conversation, so the
# graph uses clean component names while the corpus stays orthogonal.
# --------------------------------------------------------------------------- #
# fields: id, seed_entity, gold_entity, predicate,
#         seed_title/summary/desc, gold_title/summary/desc, query
RELATION_PAIRS: List[Dict[str, str]] = [
    {
        "id": "r1_uses",
        "seed_entity": "Checkout Service",
        "gold_entity": "Stripe SDK",
        "predicate": "uses",  # USES (canonical)
        "seed_title": "Kestrel plateau throughput note",
        "seed_summary": "Kestrel hit a throughput plateau during the concurrency drill.",
        "seed_desc": "The Kestrel plateau throughput drill measured concurrency headroom.",
        "gold_title": "Marlin idempotent callback ledger",
        "gold_summary": "Marlin verified idempotent callbacks and deduplicated signatures.",
        "gold_desc": "Marlin callback deduplication kept signature verification consistent.",
        "query": "Kestrel plateau throughput",
    },
    {
        "id": "r2_utilizes",
        "seed_entity": "Search Indexer",
        "gold_entity": "OpenSearch Cluster",
        "predicate": "utilizes",  # folds -> USES
        "seed_title": "Basalt cadence ingestion memo",
        "seed_summary": "Basalt ingestion cadence tightened so freshness improved.",
        "seed_desc": "The Basalt cadence ingestion memo tracked freshness improvements.",
        "gold_title": "Nimbus shard drift review",
        "gold_summary": "Nimbus rebalanced shard drift across the storage ring.",
        "gold_desc": "Nimbus shard drift review evened out the storage ring load.",
        "query": "Basalt cadence ingestion",
    },
    {
        "id": "r3_employs",
        "seed_entity": "Fraud Engine",
        "gold_entity": "XGBoost Model",
        "predicate": "employs",  # folds -> USES
        "seed_title": "Cinder anomaly scoring memo",
        "seed_summary": "Cinder anomaly scoring flagged an unusual velocity pattern.",
        "seed_desc": "The Cinder anomaly scoring memo captured the velocity pattern.",
        "gold_title": "Onyx booster retrain plan",
        "gold_summary": "Onyx moved the booster retrain schedule to a nightly window.",
        "gold_desc": "Onyx booster retrain plan set a nightly schedule instead of weekly.",
        "query": "Cinder anomaly scoring",
    },
    {
        "id": "r4_depends_on",
        "seed_entity": "Billing Worker",
        "gold_entity": "Ledger Database",
        "predicate": "depends on",  # DEPENDS_ON (canonical)
        "seed_title": "Dovetail reconciliation memo",
        "seed_summary": "Dovetail smoothed reconciliation with a jittered settlement pass.",
        "seed_desc": "The Dovetail reconciliation memo added a jittered settlement pass.",
        "gold_title": "Perch fiscal partition scheme",
        "gold_summary": "Perch repartitioned records by fiscal period to bound scans.",
        "gold_desc": "Perch fiscal partition scheme bounded full-scan record size.",
        "query": "Dovetail reconciliation settlement",
    },
    {
        "id": "r5_relies_on",
        "seed_entity": "Notification Dispatcher",
        "gold_entity": "Twilio Gateway",
        "predicate": "relies on",  # folds -> DEPENDS_ON
        "seed_title": "Ember dispatch quenching memo",
        "seed_summary": "Ember quenched dispatch fan-out per tenant to steady flow.",
        "seed_desc": "The Ember dispatch quenching memo steadied per-tenant flow.",
        "gold_title": "Quartz carrier reroute note",
        "gold_summary": "Quartz added a secondary reroute for carrier-outage coverage.",
        "gold_desc": "Quartz carrier reroute note covered outage failover paths.",
        "query": "Ember dispatch quenching",
    },
    {
        "id": "r6_requires",
        "seed_entity": "Report Builder",
        "gold_entity": "Snowflake Warehouse",
        "predicate": "requires",  # folds -> DEPENDS_ON
        "seed_title": "Fathom aggregation rollup memo",
        "seed_summary": "Fathom memoized aggregation rollups to skip recomputation.",
        "seed_desc": "The Fathom aggregation rollup memo skipped repeated recomputation.",
        "gold_title": "Ridge compute rightsizing note",
        "gold_summary": "Ridge rightsized the analytics compute footprint for spend.",
        "gold_desc": "Ridge compute rightsizing note lowered analytics footprint spend.",
        "query": "Fathom aggregation rollup",
    },
    {
        "id": "r7_part_of",
        "seed_entity": "Auth Module",
        "gold_entity": "Gateway Platform",
        "predicate": "part of",  # PART_OF (canonical)
        "seed_title": "Garnet keyring provenance memo",
        "seed_summary": "Garnet shortened keyring provenance and re-signed anchors.",
        "seed_desc": "The Garnet keyring provenance memo re-signed the trust anchors.",
        "gold_title": "Sable rollout topology note",
        "gold_summary": "Sable documented a blue-green rollout topology for the fleet.",
        "gold_desc": "Sable rollout topology note mapped the blue-green fleet stages.",
        "query": "Garnet keyring provenance",
    },
    {
        "id": "r8_relates_to",
        "seed_entity": "Latency Probe",
        "gold_entity": "Edge CDN",
        "predicate": "monitors",  # unregistered -> RELATES_TO fallback
        "seed_title": "Halcyon telemetry sampling memo",
        "seed_summary": "Halcyon raised telemetry sampling around the tail percentile.",
        "seed_desc": "The Halcyon telemetry sampling memo densified tail-percentile probes.",
        "gold_title": "Talon soft-purge tagging note",
        "gold_summary": "Talon switched to soft-purge tagging to avoid origin stampedes.",
        "gold_desc": "Talon soft-purge tagging note prevented origin stampede storms.",
        "query": "Halcyon telemetry sampling",
    },
]

# --------------------------------------------------------------------------- #
# 3 DIRECT controls: the query hits the gold conversation directly (base recall
# must find it in BOTH conditions).  No relation traversal involved.
# --------------------------------------------------------------------------- #
DIRECT_CONTROLS: List[Dict[str, str]] = [
    {
        "id": "d1_registry",
        "entity": "Container Registry",
        "title": "Umber pullsecret rotation memo",
        "summary": "Umber rotated pullsecret material across the registry mesh.",
        "desc": "The Umber pullsecret rotation memo swept the registry mesh.",
        "query": "Umber pullsecret rotation",
    },
    {
        "id": "d2_flags",
        "entity": "Feature Flag Service",
        "title": "Verdant ramp gating memo",
        "summary": "Verdant ramped a gated cohort in slow percentage steps.",
        "desc": "The Verdant ramp gating memo staged a slow cohort percentage.",
        "query": "Verdant ramp gating",
    },
    {
        "id": "d3_postmortem",
        "entity": "Incident Review",
        "title": "Willow blameless followup memo",
        "summary": "Willow tracked blameless followups through to closure.",
        "desc": "The Willow blameless followup memo closed every tracked item.",
        "query": "Willow blameless followup",
    },
]

# --------------------------------------------------------------------------- #
# 2 NEGATIVE controls: query hits a conversation whose entity has NO relation
# edges.  ON must surface NO relation-path result (no leak).
# --------------------------------------------------------------------------- #
NEGATIVE_CONTROLS: List[Dict[str, str]] = [
    {
        "id": "n1_wiki",
        "entity": "Runbook Wiki",
        "title": "Xenon taxonomy reshuffle memo",
        "summary": "Xenon reshuffled the runbook taxonomy into flatter clusters.",
        "desc": "The Xenon taxonomy reshuffle memo flattened runbook clusters.",
        "query": "Xenon taxonomy reshuffle",
    },
    {
        "id": "n2_survey",
        "entity": "Onboarding Survey",
        "title": "Yarrow questionnaire trim memo",
        "summary": "Yarrow trimmed the newcomer questionnaire to a short core.",
        "desc": "The Yarrow questionnaire trim memo kept only the short core.",
        "query": "Yarrow questionnaire trim",
    },
]


def build_corpus() -> List[Dict[str, Any]]:
    """All conversations to store, each as an authored item + one revision.

    Returns a list of ``{id, space, title, summary, description}`` records.
    """
    convs: List[Dict[str, Any]] = []
    for p in RELATION_PAIRS:
        convs.append({
            "id": f"{p['id']}__seed",
            "space": CHAT_SPACE,
            "title": p["seed_title"], "summary": p["seed_summary"],
            "description": p["seed_desc"],
        })
        convs.append({
            "id": f"{p['id']}__gold",
            "space": CHAT_SPACE,
            "title": p["gold_title"], "summary": p["gold_summary"],
            "description": p["gold_desc"],
        })
    for d in DIRECT_CONTROLS:
        convs.append({
            "id": d["id"], "space": CHAT_SPACE,
            "title": d["title"], "summary": d["summary"], "description": d["desc"],
        })
    for n in NEGATIVE_CONTROLS:
        convs.append({
            "id": n["id"], "space": CHAT_SPACE,
            "title": n["title"], "summary": n["summary"], "description": n["desc"],
        })
    # Single relation-source conversation in the relsrc space that declares all
    # entity->entity relations (ABOUT every relation entity, but never searched).
    convs.append({
        "id": "relsrc_backbone",
        "space": RELSRC_SPACE,
        "title": "Service dependency backbone",
        "summary": "Architecture note enumerating component relationships.",
        "description": "Dependency mapping between infrastructure and product components.",
    })
    return convs


def seed_decomposition(pair: Dict[str, str]) -> Dict[str, Any]:
    """Decomposition for a relation seed conv: ABOUT its seed entity ONLY."""
    return {
        "entities": [{"name": pair["seed_entity"], "type": "component"}],
        "facts": [],
        "relations": [],
    }


def gold_decomposition(pair: Dict[str, str]) -> Dict[str, Any]:
    """Decomposition for a relation gold conv: ABOUT its gold entity ONLY."""
    return {
        "entities": [{"name": pair["gold_entity"], "type": "component"}],
        "facts": [],
        "relations": [],
    }


def single_entity_decomposition(name: str) -> Dict[str, Any]:
    """Decomposition for direct/negative convs: one isolated entity, no edges."""
    return {
        "entities": [{"name": name, "type": "component"}],
        "facts": [],
        "relations": [],
    }


def backbone_decomposition() -> Dict[str, Any]:
    """The relation-source decomposition: all pair entities + all relations.

    All 16 entities are materialized here so ``_resolve`` can link both
    endpoints of every relation onto the shared project-level anchors.
    """
    entities: List[Dict[str, str]] = []
    seen: set = set()
    relations: List[Dict[str, str]] = []
    for p in RELATION_PAIRS:
        for name in (p["seed_entity"], p["gold_entity"]):
            if name not in seen:
                seen.add(name)
                entities.append({"name": name, "type": "component"})
        relations.append({
            "subject": p["seed_entity"],
            "object": p["gold_entity"],
            "predicate": p["predicate"],
        })
    return {"entities": entities, "facts": [], "relations": relations}


def build_queries() -> List[Dict[str, Any]]:
    """Every evaluation query with its class, seed/gold ids, and expectations."""
    queries: List[Dict[str, Any]] = []
    for p in RELATION_PAIRS:
        queries.append({
            "id": p["id"],
            "cls": "relation",
            "query": p["query"],
            "seed_id": f"{p['id']}__seed",
            "gold_id": f"{p['id']}__gold",
            "predicate": p["predicate"],
        })
    for d in DIRECT_CONTROLS:
        queries.append({
            "id": d["id"], "cls": "direct", "query": d["query"],
            "seed_id": d["id"], "gold_id": d["id"], "predicate": "",
        })
    for n in NEGATIVE_CONTROLS:
        queries.append({
            "id": n["id"], "cls": "negative", "query": n["query"],
            "seed_id": n["id"], "gold_id": None, "predicate": "",
        })
    return queries
