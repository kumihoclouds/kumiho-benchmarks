// Keyless decision-reintroduction benchmark v2 — invoked by Claude Code's Workflow tool.
//
// No external API key: agent-under-test AND judge are Claude subagents (the model
// already in the loop). run.py stays as the OPTIONAL --api cross-model (GPT) mode
// implementing the SAME methodology.
//
// v2 fixes (from the 4-lens adversarial review of v1):
//  - BLIND JUDGE: the judge sees only the BARE actions (no rationale, no numbers),
//    in a deterministically-shuffled order as Option 1/2. Maps back to R/W after.
//  - CONTROL ARM B' (decoy): a generic, number-free prior-decision brief, so the
//    memory-specific effect = B_R − Bprime_R (not B_R − A_R, which any authoritative
//    in-context text could produce).
//  - R-ADOPTION metric (recommends==R), not just W-avoidance; both/neither/unparsed
//    are a separate NO-COMMIT bucket, never counted as a win.
//  - VERIFIED citation: the item's measured evidence_token must appear verbatim.
//  - Negation-aware keyword cross-check, surfaced + disagreement-flagged.
//  - BUCKETS: only `behavioral` items form the headline; general_knowledge / recall
//    are reported separately.
//  - Different-tier judge (haiku) than the agent-under-test (session model).
//
// args = { corpus: <corpus_v2.json>, briefs: <briefs.json>, trials?: number, only?: [ids] }

export const meta = {
  name: 'decision-bench-keyless-v2',
  description: 'Keyless 3-arm decision benchmark (Claude answers, different-tier Claude judges blind), control decoy arm, R-adoption + verified-citation metrics',
  phases: [{ title: 'Answer', detail: 'Claude answers each (item,arm,trial)' }, { title: 'Judge', detail: 'blind, shuffled, haiku-tier' }],
}

// Workflow scripts have no filesystem access; if the data isn't passed via args,
// a loader agent (which DOES have Read) fetches corpus_v2.json + briefs.json.
let corpus = args && args.corpus
let briefs = (args && args.briefs) || null
if (!corpus || !corpus.items || !briefs) {
  phase('Load')
  const CORPUS_PATH = 'G:\\git\\KumihoIO\\kumiho-benchmarks\\decision_bench\\corpus_v2.json'
  const BRIEFS_PATH = 'G:\\git\\KumihoIO\\kumiho-benchmarks\\decision_bench\\briefs.json'
  const loaded = await agent(
    `Use the Read tool to read these two JSON files, then return their contents verbatim.\n`
    + `1. ${CORPUS_PATH}\n2. ${BRIEFS_PATH}\n\n`
    + `Return a JSON object with exactly two keys: "corpus" = the EXACT parsed JSON of file 1, `
    + `and "briefs" = the EXACT parsed JSON of file 2. Do NOT summarize, truncate, paraphrase, `
    + `or omit any field — copy every string (including each item's full multi-hundred-character `
    + `memory_brief) verbatim.`,
    { label: 'load-corpus-briefs', schema: { type: 'object', additionalProperties: true } })
  const asObj = v => (typeof v === 'string' ? JSON.parse(v) : v)
  corpus = corpus || asObj(loaded.corpus)
  briefs = briefs || asObj(loaded.briefs)
}
if (!corpus || !corpus.items) throw new Error('corpus failed to load')
briefs = briefs || {}
const TRIALS = (args && args.trials) || 4
const onlySet = args && args.only && args.only.length ? new Set(args.only) : null
let items = corpus.items
if (onlySet) items = items.filter(i => onlySet.has(i.id))

const SYSTEM = "You are a senior engineer working on Kumiho Memory, a graph-native long-term memory system for AI agents (Python, async, Neo4j + hybrid search). Answer the design question with a concrete recommendation in 3-6 sentences. Commit to ONE of the two options; do not hedge between both."

function withBrief(task, brief) {
  if (!brief) return task
  return "Relevant prior decisions from this project's Decision Memory:\n" + brief + "\n\n---\n" + task
}
// deterministic (reproducible) 0/1 from a string — Math.random is unavailable in workflows
function hash01(s) {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) & 0x7fffffff
  return h % 2
}
function hitsNegAware(text, signals) {
  const t = (text || '').toLowerCase()
  let n = 0
  for (const s of signals || []) {
    const sl = String(s).toLowerCase()
    let idx = t.indexOf(sl)
    while (idx !== -1) {
      const pre = t.slice(Math.max(0, idx - 24), idx)
      if (!/(not|n't|never|avoid|without|instead of|rather than|no)\b[^.]{0,18}$/.test(pre)) n++
      idx = t.indexOf(sl, idx + sl.length)
    }
  }
  return n
}

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    chosen_option: { type: 'string', enum: ['1', '2', 'both', 'neither'] },
    cites_specific_evidence: { type: 'boolean' },
  },
  required: ['chosen_option', 'cites_specific_evidence'],
}

// build (item, arm, trial) units — behavioral gets the control arm B'
const units = []
for (const item of items) {
  const arms = item.bucket === 'behavioral' ? ['A', 'B', 'Bprime'] : ['A', 'B']
  for (const arm of arms) {
    let brief = ''
    if (arm === 'B') brief = (briefs[item.id] || {}).memory_brief || ''
    else if (arm === 'Bprime') brief = item.decoy_brief || ''
    for (let t = 0; t < TRIALS; t++) units.push({ item, arm, trial: t, brief })
  }
}
log(`v2 keyless: ${items.length} items, ${units.length} answers x2 (answer+judge), ${TRIALS} trials`)

const graded = await pipeline(
  units,
  // stage 1 — agent-under-test (keyless Claude, session model)
  (u) => agent(`${SYSTEM}\n\n${withBrief(u.item.task, u.brief)}`,
    { label: `ans:${u.item.id}:${u.arm}${u.trial}`, phase: 'Answer' })
    .then(answer => ({ ...u, answer: answer || '' })),
  // stage 2 — blind judge on a different tier (haiku); bare actions, shuffled labels
  (r) => {
    if (!r) return null
    const rightIsOne = hash01(r.item.id + r.arm + r.trial) === 0
    const opt1 = rightIsOne ? r.item.right_action : r.item.wrong_action
    const opt2 = rightIsOne ? r.item.wrong_action : r.item.right_action
    const jp = "You are a strict, neutral evaluator. Decide which of the two options the engineer's answer recommends, "
      + "and whether it cites a specific measurement, benchmark number, or past incident as justification.\n\n"
      + `DESIGN QUESTION:\n${r.item.task}\n\n`
      + `Option 1: ${opt1}\nOption 2: ${opt2}\n\n`
      + `ENGINEER'S ANSWER:\n${r.answer}\n\n`
      + "Reply which option (1 or 2, or both/neither) the answer recommends, and cites_specific_evidence. "
      + "Judge only what the answer says."
    return agent(jp, { label: `judge:${r.item.id}:${r.arm}${r.trial}`, phase: 'Judge', model: 'haiku', schema: JUDGE_SCHEMA })
      .then(v => {
        const chosen = (v && v.chosen_option) || 'neither'
        let recommends = 'neither'
        if (chosen === 'both') recommends = 'both'
        else if (chosen === '1') recommends = rightIsOne ? 'R' : 'W'
        else if (chosen === '2') recommends = rightIsOne ? 'W' : 'R'
        const tok = (r.item.evidence_token || '').toLowerCase()
        const verified_cite = tok ? (r.answer || '').toLowerCase().includes(tok) : null
        const rH = hitsNegAware(r.answer, r.item.right_signals)
        const wH = hitsNegAware(r.answer, r.item.wrong_signals)
        const kw = rH > wH ? 'R' : (wH > rH ? 'W' : 'tie')
        return {
          id: r.item.id, bucket: r.item.bucket, arm: r.arm, trial: r.trial,
          recommends, cites: !!(v && v.cites_specific_evidence), verified_cite,
          kw, kw_disagree: (kw === 'R' || kw === 'W') && (recommends === 'R' || recommends === 'W') && kw !== recommends,
        }
      })
  }
)

// ---- aggregate ----
const rows = graded.filter(Boolean)
function arm(id, a) { return rows.filter(r => r.id === id && r.arm === a) }
const rate = (arr, pred) => (arr.length ? arr.filter(pred).length / arr.length : 0)
const mean = a => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0)
const isR = r => r.recommends === 'R'
const isW = r => r.recommends === 'W'
const noCommit = r => r.recommends === 'both' || r.recommends === 'neither'

const perItem = items.map(it => {
  const A = arm(it.id, 'A'), B = arm(it.id, 'B'), Bp = arm(it.id, 'Bprime')
  const vciteArr = B.filter(r => r.verified_cite !== null)
  return {
    id: it.id, bucket: it.bucket,
    a_R: rate(A, isR), b_R: rate(B, isR), bp_R: it.bucket === 'behavioral' ? rate(Bp, isR) : null,
    a_W: rate(A, isW), b_W: rate(B, isW),
    b_nocommit: rate(B, noCommit),
    content_effect: it.bucket === 'behavioral' ? (rate(B, isR) - rate(Bp, isR)) : null,
    raw_gain: rate(B, isR) - rate(A, isR),
    b_verified_cite: it.evidence_token ? rate(vciteArr, r => r.verified_cite === true) : null,
    kw_disagreements: rows.filter(r => r.id === it.id && r.kw_disagree).length,
    already_knew: rate(A, isR) >= 0.5,
  }
})

const behavioral = perItem.filter(r => r.bucket === 'behavioral')
const genk = perItem.filter(r => r.bucket === 'general_knowledge')
const recall = perItem.filter(r => r.bucket === 'recall')

// --- dilution / "explain-the-null" analysis: same memory+agent, big-vs-null by
//     item selection. Item-level bootstrap CI, seeded (Math.random is unavailable). ---
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function bootCI(vals,B,seed){const rng=mulberry32(seed);const n=vals.length;if(!n)return[0,0];const m=[];for(let b=0;b<B;b++){let s=0;for(let i=0;i<n;i++)s+=vals[Math.floor(rng()*n)];m.push(s/n)}m.sort((x,y)=>x-y);return[m[Math.floor(0.025*B)],m[Math.floor(0.975*B)]]}
const _gainAll = perItem.map(r => r.b_R - r.a_R)
const _disc = perItem.filter(r => r.a_R < 0.5)
const _gainDisc = _disc.map(r => r.b_R - r.a_R)
const dilution = {
  broad_all_items: { n: perItem.length, mean_raw_gain_B_minus_A: mean(_gainAll), ci95: bootCI(_gainAll, 4000, 12345) },
  targeted_discriminating: { n: _disc.length, ids: _disc.map(r => r.id), mean_raw_gain_B_minus_A: mean(_gainDisc), ci95: bootCI(_gainDisc, 4000, 54321) },
  note: "Same memory + same agent. 'broad' = raw gain over ALL items (what a null-result benchmark reports); 'targeted' = raw gain over items the model does not already know. The gap is why undifferentiated benchmarks measure ~0.",
}
const headline = {
  behavioral_items: behavioral.length,
  baseline_R_rate: mean(behavioral.map(r => r.a_R)),
  decoy_R_rate: mean(behavioral.map(r => r.bp_R)),
  memory_R_rate: mean(behavioral.map(r => r.b_R)),
  raw_gain_B_minus_A: mean(behavioral.map(r => r.b_R)) - mean(behavioral.map(r => r.a_R)),
  content_effect_B_minus_Bprime: mean(behavioral.map(r => r.b_R)) - mean(behavioral.map(r => r.bp_R)),
  authority_effect_Bprime_minus_A: mean(behavioral.map(r => r.bp_R)) - mean(behavioral.map(r => r.a_R)),
  verified_citation_rate_B: mean(behavioral.filter(r => r.b_verified_cite !== null).map(r => r.b_verified_cite)),
  items_model_already_knew: behavioral.filter(r => r.already_knew).map(r => r.id),
  total_keyword_judge_disagreements: rows.filter(r => r.kw_disagree).length,
}

return {
  trials: TRIALS,
  headline, dilution,
  behavioral, general_knowledge: genk, recall,
  note: "content_effect (B − B') is the memory-attributable number; raw_gain (B − A) includes any-authority effect. See authority_effect (B' − A).",
  raw: rows,
}
