# Quantified novelty search — machine-doable backlog rows

Completes three rows of `NOVELTY_SEARCH.md` §0: Semantic Scholar API,
native OpenReview query, ACL Anthology. **Search date: 2026-08-02.**
Single screener (one automated agent session); abstract-level screening
only; no full-text retrieval.

Claim under test (`PAPER_REFRAME.md` §4 contribution 1): *an executable,
pre-declared manipulation check — a direct test that a role prompt
performed its declared information-processing operation, measured
against a length-matched neutral control, before benchmark deltas are
interpreted — is not standard practice in multi-agent / role-prompted
LLM evaluation.*

**Headline result stated up front: the "we found none in N screened
records" sentence does NOT survive in its plain form.** Two records
(one borderline third) satisfy the screening rule below. The supportable
sentence is now: *"In ~640 screened records we found no instance of a
pre-declared, executable, per-role manipulation check measured against a
length-matched neutral control and used to gate interpretation of
benchmark deltas; the nearest predecessors are [§5], and a substantial
role-adherence / persona-consistency literature [§6] measures related
constructs post hoc."* That is a better result than a zero: it names the
neighbours instead of being blindsided by them.

---

## 1. Screening rule (fixed before screening, applied verbatim)

> A record COUNTS as a predecessor candidate if it (a) evaluates
> role/persona-prompted LLM agents AND (b) includes any direct, per-role
> test of whether the prompted operation/behaviour actually occurred
> that is separate from downstream task accuracy.

Anything measuring role adherence, persona consistency, role-playing
fidelity, or prompt compliance **without** such a separate per-role
operation test is recorded as a **near-miss** and treated as related
work the paper must cite.

Disposition subtlety applied throughout: in the role-playing-agent
(RPA) literature, persona fidelity is usually the downstream task
metric itself — the role-play *is* the task. Records whose fidelity
measure is the task outcome are classified near-miss under clause (b)
("separate from downstream task accuracy"). Records that probe role
expression separately from an independent task outcome are candidates
even when they lack the neutral control and pre-declaration that
distinguish this paper's instrument.

---

## 2. Semantic Scholar API

Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search`
with `fields=title,abstract,year,venue,externalIds&limit=100`, fetched
by HTTP GET (curl) after the WebFetch route was rate-limited. The
public unauthenticated pool returned HTTP 429 for most of a ~25-minute
window; requests were retried with jittered backoff. **3 of 8 planned
queries succeeded; 5 remained rate-limited and are recorded as failures,
not as empty results.** Citation-graph forward/backward snowballing was
NOT performed (that sub-item of the §0 row remains open).

| # | Query (URL-encoded into the endpoint above) | Total hits | Screened | Result |
| --- | --- | ---: | ---: | --- |
| S1 | `manipulation check large language model agents` | 34,954 | top 100 (titles+abstracts) | 1 candidate (CUI 2026 empathy study), 1 near-miss |
| S2 | `role prompt verification multi-agent LLM` | 6,536 | top 100 | 0 candidates, 3 near-misses |
| S4 | `treatment fidelity check LLM agents` | 75 | all 75 | 1 candidate (When Roles Fail), 3 near-misses |
| S3 | `persona prompt operation check LLM evaluation` | — | — | **attempted, rate-limited (HTTP 429) on 2026-08-02** after 120+ retries |
| S5 | `role adherence evaluation multi-agent LLM` | — | — | attempted, rate-limited on 2026-08-02 |
| S6 | `persona consistency evaluation large language models` | — | — | attempted, rate-limited on 2026-08-02 |
| S7 | `role-playing fidelity large language model` | — | — | attempted, rate-limited on 2026-08-02 |
| S8 | `validate prompt induced behavior language model before benchmark` | — | — | attempted, rate-limited on 2026-08-02 |

Mitigation of the failed queries: the topics of S5–S7 (role adherence,
persona consistency, role-playing fidelity) were covered by OpenReview
queries O3/O4/O6 and ACL queries A2/A3/A5 below, which returned the
known persona-fidelity literature; S3 and S8 are paraphrases of S1/S2.
The failures still cap what this row can claim and should be re-run
with an API key before submission.

Example full URL (S1):
`https://api.semanticscholar.org/graph/v1/paper/search?query=manipulation%20check%20large%20language%20model%20agents&fields=title,abstract,year,venue,externalIds&limit=100`

Raw JSON responses are preserved in the session scratchpad
(`scratchpad/s2/S{1,2,4}.json`) with a `progress.log` of all 429s.

---

## 3. OpenReview (native API — includes rejected/withdrawn submissions)

Endpoint: `https://api2.openreview.net/notes/search?term=<terms>&content=all&group=all&limit=50`
(api2 responded; api.openreview.net not needed). All 6 queries returned
`count: 10000`, which is a **server-side cap, not a meaningful hit
count**; the API returns the relevance-ranked top 50 notes. Because
`content=all` matches review/comment notes as well as papers, many
returned notes have no title: of 300 returned notes, **77 were
paper-like (had a title)**; all 300 were screened at title level and 20
flagged records at abstract level.

| # | term | Returned | Paper-like | Result |
| --- | --- | ---: | ---: | --- |
| O1 | `manipulation check LLM agents` | 50 | 20 | 0 candidates, 0 near-misses |
| O2 | `role prompt verification multi-agent` | 50 | 10 | 0 candidates (mostly classical MAS verification) |
| O3 | `persona consistency evaluation` | 50 | 7 | 2 near-misses (persona-consistency framework; atomic OOC) |
| O4 | `role adherence LLM` | 50 | 21 | 2 near-misses (PBSuite; instruction-adherence benchmark) |
| O5 | `treatment fidelity language model` | 50 | 2 | 0 relevant |
| O6 | `role-playing fidelity` | 50 | 17 | 1 borderline candidate (InCharacter), 5+ near-misses |

---

## 4. ACL Anthology (via site-restricted web search — substitution note)

**Substitution and why:** the ACL Anthology's native search box is a
Google Custom Search Engine with no queryable API and markup that is
hostile to scraping. In its place, a general web-search tool was run
with `site:aclanthology.org` restriction. This finds Anthology pages
via Google's index rather than the Anthology's own ranking; the search
engine also injected some non-Anthology (arxiv.org) results despite the
restriction, and returns ~7–10 links per query rather than an exhaustive
count. **Hit counts below are links returned and screened, not database
totals — this leg establishes existence/coverage, not exhaustiveness.**

| # | Query (all prefixed `site:aclanthology.org`) | Links returned/screened | Result |
| --- | --- | ---: | --- |
| A1 | `manipulation check role prompt LLM agents` | 10 | 0 candidates; 1 near-miss (Role Vectors) |
| A2 | `persona consistency evaluation role-playing large language models` | 9 | 4 near-misses |
| A3 | `role adherence evaluation dialogue agents LLM` | 7 | 2 near-misses |
| A4 | `"manipulation check" language model` | 7 | **0 hits pairing "manipulation check" with LLM-agent evaluation** (top hit: Wikipedia's psychology entry) |
| A5 | `persona fidelity out-of-character evaluation LLM` | 9 | 3 near-misses |
| A6 | `verify role prompt behavior neutral control multi-agent evaluation` | 9 | 0 candidates |
| A7 | `quantifying persona effect LLM simulations` | 10 | 1 near-miss (Hu & Collier) |

A4 is the sharpest single datum for the method-transfer framing: the
exact phrase "manipulation check" does not co-occur with LLM-agent
evaluation anywhere in the Anthology that Google indexes.

---

## 5. Predecessor candidates (met the rule)

| Record | IDs | Why it counts | What it lacks vs. the paper's instrument | Disposition |
| --- | --- | --- | --- | --- |
| **When Roles Fail: Epistemic Constraints on Advocate Role Fidelity in LLM-Based Political Statement Analysis** (2026) | arXiv:2604.27228 | Multi-agent pipeline with adversarial advocate roles; builds an epistemic stance classifier and four per-role fidelity metrics (RDI, EDD, DDI, ERS) measuring whether roles were actually maintained — separate from task output; self-describes as "the first systematic empirical test of that assumption" | Post-hoc measurement, not pre-declared; no length-matched neutral control; does not gate benchmark-delta interpretation | **Must cite as nearest predecessor.** Its own "first" claim also directly constrains our framing |
| **Do Prompt-Level Empathy Instructions Influence User Experience? Evidence From A Controlled Chatbot Study** (CUI 2026) | doi:10.1145/3816046.3816221 | Controlled comparison of an empathically framed system prompt vs. a **neutral assistant control**, with "an automated manipulation check … run in parallel" before interpreting outcome measures — the psychology instrument, transferred, executable | Single agent, single role; control not stated to be length-matched; HCI user-study outcomes, not capability benchmarks | **Must cite.** Proof the transfer has begun in HCI; our claim narrows to multi-agent/benchmark evaluation |
| *(borderline)* **InCharacter: Evaluating Personality Fidelity in Role-Playing Agents through Psychological Interviews** (ACL 2024) | arXiv:2310.17976 / OpenReview xOFGu3cFUJ | Probes whether the prompted persona is actually expressed via psychological-scale interviews — an instrument separate from the deployment dialogues | RPA setting where persona expression is arguably the product (clause-b ambiguity); no neutral control; no pre-declaration | Borderline under the rule; **second screener must adjudicate.** Cite either way |

---

## 6. Near-misses (related work the paper must cite, not candidates)

| Record | IDs | One-line disposition |
| --- | --- | --- |
| RPA-Check: multi-stage automated evaluation of role-playing agents (2026) | arXiv:2604.11655 | Boolean per-role checklist verification via LLM-judge, but fidelity is the task metric (RPA setting) |
| Why Do Multi-Agent LLM Systems Fail? / MAST (NeurIPS 2025) | arXiv:2503.13657 | "Disobey role specification" is a named failure mode found by post-hoc trace annotation, not a pre-run check |
| Spotting Out-of-Character Behavior (Findings ACL 2025) | arXiv:2506.19352, 2025.findings-acl.1349 | Atomic-level persona-fidelity metrics; fidelity is the evaluated outcome itself |
| Quantifying the Persona Effect in LLM Simulations (ACL 2024) | arXiv:2402.10811, 2024.acl-long.554 | Measures whether persona variables move predictions at population level; no per-role operation check |
| Are Economists Always More Introverted? (persona consistency framework) | OpenReview 6ONPEKF7BP | Cross-task persona-consistency framework; consistency, not operation verification |
| Enhancing Persona Consistency … Contrastive Learning (Findings ACL 2025) | 2025.findings-acl.1344 | Trains for consistency rather than testing whether an operation occurred |
| EmoCharacter (NAACL 2025) | 2025.naacl-long.316 | Emotional-fidelity benchmark for RPAs; fidelity is the task |
| Can LLM Agents Maintain a Persona in Discourse? (EMNLP 2025) | 2025.emnlp-main.1487 | OCEAN-trait adherence in dyadic conversation; adherence metric, no control arm |
| Can Role Vectors Affect LLM Behaviour? (Findings EMNLP 2025) | 2025.findings-emnlp.963 | Activation-level role-vs-generic-baseline contrast — an internal analogue of a role check, not an evaluation gate |
| Doppelganger Method: breaking role consistency (2025) | arXiv:2506.14539 | Adversarial attack on role consistency; measures breakdown, not intended operation |
| Role-Playing Evaluation for LLMs (2025) | arXiv:2505.13157 | General RPA evaluation; fidelity as task metric |
| Personality Consistency in Quantized RPAs (EMNLP-Industry 2024) | 2024.emnlp-industry.19 | Big-Five stability under quantization; consistency, not operation |
| Persistent Personas? (EACL 2026) | 2026.eacl-long.246 | Long-horizon persona retention protocol; no per-role operation check |
| Pluralistic Behavior Suite (2025) | arXiv:2511.05018, OpenReview ro1sa4iuDJ | Multi-turn adherence to custom behavioral policies; compliance stress-test, not manipulation check |
| Traceability and Accountability in Role-Specialized Pipelines (ASEW 2025) | arXiv:2510.07614 | Per-role repair/harm rates — per-role attribution of task outcome, not of operation |
| Representational Collapse in Multi-Agent LLM Committees (2026) | arXiv:2604.03809 | Embedding similarity shows role prompts fail to induce differentiated reasoning — evidence roles may not "fire", measured post hoc |
| Doing What They Say, Not What They Reason (2026) | arXiv:2606.00476 | Process-fidelity (reasoning→action) check, but not about role prompts |
| Diversity Without Fidelity (2026) | arXiv:2604.11840 | Behavioral-fidelity checks for negotiation simulators; sampler-vs-solver roles, no prompt-operation test |
| Breaking the Script (SIGDIAL 2026) | 2026.sigdial-1.8 | Goal-adherence of RPAs under distraction; adherence metric |
| PersonaEval | OpenReview wZbkQStAXj | Tests the *evaluators* of role-play, not the roles |

The persona-consistency / role-adherence literature is real, active, and
adjacent (predicted by `NOVELTY_SEARCH.md` §3.3's persona work). The
paper's related-work section must carry a paragraph on it; the
differentiator to state is: those metrics are (i) post-hoc, (ii)
uncontrolled or baseline-free, (iii) about identity/style rather than a
declared information-processing operation, and (iv) never used as a
pre-registered gate on interpreting benchmark deltas.

---

## 7. Totals

| Leg | Records screened (title level) | Abstract-screened | Candidates | Near-misses |
| --- | ---: | ---: | ---: | ---: |
| Semantic Scholar (3 of 8 queries) | 275 | 12 | 2 | 7 |
| OpenReview (6 queries) | 300 (77 paper-like) | 20 | 1 borderline | 8 |
| ACL Anthology via site-restricted search (7 queries) | ~65 links | snippets | 0 | 9 |
| **Total (with overlap across legs)** | **~640** | **~35** | **2 (+1 borderline)** | **~20 unique** |

---

## 8. Limitations of this search

Single screener, and the screener was an automated agent session — no
second independent screener, no inter-screener agreement statistic.
Screening was at title/abstract level only; no full texts were
retrieved, so a paper that runs a manipulation check but does not
surface it in the abstract would be missed (this failure mode is
plausible precisely because such checks are treated as minor
methodology). Five of eight Semantic Scholar queries never executed
(public-pool rate limiting; recorded above) and citation-graph
snowballing was not performed. The ACL Anthology leg substitutes a
Google `site:` search for a native query, inheriting Google's index
coverage and ranking, and its per-query link counts are not database
hit counts. OpenReview's `count` field is capped at 10,000 and its
`content=all` matching pulls in review comments, so its screened set is
the relevance-ranked top-50 per query, not an exhaustive sweep.
Screening decisions on RPA-literature records turn on a judgment call
(whether persona fidelity is "downstream task accuracy" in that
setting) that a second screener must adjudicate — InCharacter is the
test case. The human-only backlog rows in `NOVELTY_SEARCH.md` §0
(second screener, full-text reading of the near-neighbour set,
PubMed/PsycINFO MeSH) remain open.
