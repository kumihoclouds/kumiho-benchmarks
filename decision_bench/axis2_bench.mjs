export const meta = {
  name: 'decision-bench-axis2-clean',
  description: 'Axis-2 re-run with a LEAK-FREE corpus: task text never reveals the convention, so the model cannot infer it. Measures conformance to arbitrary project-specific team conventions.',
  phases: [{ title: 'Answer', detail: 'A=none / B=team convention / Bprime=generic advice' }],
}

const TRIALS = 4
// Leak-free: each task states ONLY the intent, never the naming/mechanism convention.
const ITEMS = [
  { id:'gating_config',
    task:"You're adding a runtime toggle for a new 'wide-then-trim' recall mode. In this codebase, how do you expose the toggle? Be concrete about the mechanism.",
    conv:"Prior decisions: feature toggles are environment variables resolved through a config_from_env() helper (env-gated at import time), NOT constructor arguments.",
    decoy:"Prior decisions: make configuration flexible, explicit, and testable.",
    pattern:"config_from_env" },
  { id:'env_flag_naming',
    task:"You're adding an environment-variable feature flag to turn on a new cross-encoder reranker. Give the exact environment variable name you would use.",
    conv:"Prior decisions: every feature flag in this project is named with the prefix KUMIHO_MEMORY_ followed by the feature (e.g. KUMIHO_MEMORY_ONTOLOGY).",
    decoy:"Prior decisions: use clear, descriptive, consistent configuration names.",
    pattern:"KUMIHO_MEMORY_[A-Z]" },
  { id:'kind_naming',
    task:"You're adding a new graph node type that represents a rejected alternative considered within a captured decision. What KIND string (the node's type label) do you name it?",
    conv:"Prior decisions: code-domain node KINDs follow the pattern code_<noun> — e.g. code_decision, code_anchor.",
    decoy:"Prior decisions: name types clearly and consistently.",
    pattern:"code_[a-z]" },
  { id:'stats_return',
    task:"A commit-mining function needs to return summary statistics (how many decisions captured, how many errors, how many skipped) to its caller. Concretely, what do you return?",
    conv:"Prior decisions: mining/ingest/capture functions return a stats dataclass that exposes an .as_dict() method for serialization.",
    decoy:"Prior decisions: return structured, well-typed results from functions.",
    pattern:"as_dict" },
  { id:'endpoint_env',
    task:"Your code needs to connect to the local community-edition server. What is the mechanism or name it uses to discover the server's address? Be specific.",
    conv:"Prior decisions: the CE server endpoint is read from the KUMIHO_SERVER_ENDPOINT environment variable.",
    decoy:"Prior decisions: keep connection configuration explicit and environment-driven.",
    pattern:"KUMIHO_SERVER_ENDPOINT" },
  { id:'space_naming',
    task:"This system isolates code decisions in a separate storage project so they don't mix with conversation memory. For a repository named 'acme-api', what do you name that dedicated project?",
    conv:"Prior decisions: the dedicated code-decision project is named {repo}-code — the repository name with a -code suffix.",
    decoy:"Prior decisions: keep project names clear and scoped to their purpose.",
    pattern:"acme-api-code" },
]

const SYS = "You are a senior engineer working on Kumiho Memory (Python, async, Neo4j + hybrid search). Answer the design question concretely and briefly (1-4 sentences or a short code line). Commit to one concrete choice."
function conforms(ans, pat){ try { return new RegExp(pat, 'i').test(ans || '') } catch(e){ return false } }
function withBrief(task, brief){ return brief ? ("Relevant prior decisions from this project's Decision Memory:\n" + brief + "\n\n---\n" + task) : task }

const units = []
for (const it of ITEMS) for (const arm of ['A','B','Bprime']) for (let t=0;t<TRIALS;t++) units.push({ it, arm, t })

phase('Answer')
const graded = await parallel(units.map(u => () => {
  const brief = u.arm==='B' ? u.it.conv : u.arm==='Bprime' ? u.it.decoy : ''
  return agent(`${SYS}\n\n${withBrief(u.it.task, brief)}`, { label:`${u.it.id}:${u.arm}${u.t}`, phase:'Answer' })
    .then(ans => ({ id:u.it.id, arm:u.arm, conform: conforms(ans||'', u.it.pattern) }))
}))

const rate = arr => arr.length ? arr.filter(g=>g.conform).length/arr.length : 0
const rows = ITEMS.map(it => {
  const of = arm => graded.filter(g => g && g.id===it.id && g.arm===arm)
  return { id:it.id, a:rate(of('A')), bp:rate(of('Bprime')), b:rate(of('B')) }
})
const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : 0
return {
  trials: TRIALS,
  rows,
  headline: {
    baseline_conform_A: mean(rows.map(r=>r.a)),
    decoy_conform_Bprime: mean(rows.map(r=>r.bp)),
    memory_conform_B: mean(rows.map(r=>r.b)),
    content_effect_B_minus_Bprime: mean(rows.map(r=>r.b)) - mean(rows.map(r=>r.bp)),
    raw_gain_B_minus_A: mean(rows.map(r=>r.b)) - mean(rows.map(r=>r.a)),
    clean_items_A_below_0_5: rows.filter(r=>r.a < 0.5).map(r=>r.id),
  },
  note: "Leak-free axis-2. Task never reveals the convention. A = the model's own default (cannot know the arbitrary project choice); B-Bprime isolates the SPECIFIC captured convention from generic 'be consistent' advice.",
}
