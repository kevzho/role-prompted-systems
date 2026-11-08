# Ethics and governance compliance record

`ETHICS_AND_DATA_GOVERNANCE.md` worked as a checklist rather than as background
reading, per `REMAINING_PROMPTS.md` P20.

Compiled 2026-07-28. Status values are exact:

- **DONE** — implemented and verifiable in the repository
- **CODE READY / UNVERIFIED** — implemented but not executed (see §0)
- **HUMAN GATE** — cannot be completed by engineering work; requires a named
  person, an institution, or a payment
- **BLOCKED** — depends on an earlier item

## 0. Verification status

The test suite runs and passes (see `logistics/HANDOVER.md` §0 for the `TMPDIR`
workaround the full data volume requires). Controls marked **DONE** below are
covered by passing tests; controls marked **CODE READY / UNVERIFIED** are
implemented and exercised only against simulated data, which is a real
limitation for the red-team items in §6 — an injection boundary that holds
against a stub is not thereby shown to hold against a real adversary.

**Nothing here should be represented to an ethics body as operational for a
deployment.** These are research controls on a research pipeline.

---

## 1. Human-subject and clinical input

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E1 | Institutional ethics review, **or** a documented determination that it is not required, **before** any human-subject component | **HUMAN GATE** | Triggered by three components: AdvConsistency expert annotation, creative-divergence human rating, and the terminology review below. None may begin first. |
| E2 | Informed consent, age-appropriate process | **HUMAN GATE** | Consent text must be drafted and approved under E1. |
| E3 | Do not infer or assign diagnoses from model behaviour | **DONE** | No code path assigns a clinical label. `AuditRecordWriter` rejects clinical identifiers as *values*, so a diagnosis cannot reach the audit trail even by accident. |
| E4 | Do not recruit a stigmatised group merely to validate an analogy | **DONE** | No recruitment is based on diagnosis. Annotators are recruited for domain expertise in text, not for lived experience of any condition. |
| E5 | Involve **compensated** people with relevant lived experience and clinical/research expertise in terminology review, and report that they were involved | **HUMAN GATE** | **Not yet done and cannot be faked.** The paper must either report this review or state plainly that it did not happen. See §6. |
| E6 | Publish discomfort, withdrawal, support, and debrief procedures | **HUMAN GATE** | Draft under E1. Withdrawal must be possible after the fact, which constrains retention (E14). |

**E1 is the gate on the whole human half of the study.** Nothing in P8's
annotation or P10's human rating may start before it resolves.

---

## 2. Terminology

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E7 | Function-first names in code, UI, tables, demonstrations | **DONE** | `config.ROLES` carries only functional labels. `CLINICAL_TRACEABILITY` exists solely as a reader's map back to the concept paper. |
| E8 | No "paranoid reasoner", "OCD agent", "manic phase" etc. in prompts or product copy | **CODE READY / UNVERIFIED** | Enforced by the audit writer's denylist and by `tests/test_prompts.py`. Unverified only because the suite has not been run since the P7/P10/P12 additions. |
| E9 | "Impostor phenomenon", not "impostor syndrome"; never equated with Dunning–Kruger | **DONE** | Applied in `references.bib` ([18] replaced with Clance & Imes 1978) and `BIBLIOGRAPHY.md` §1. |
| E10 | Required disclaimer wherever clinical inspiration appears | **DONE** | `config.DISCLAIMER`, reproduced in `README.md`, `PREREGISTRATION.md`, every `REPORT.md`, and `BIBLIOGRAPHY.md` §2. |

---

## 3. Chain-of-thought replacement

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E11 | Never store or expose private chain-of-thought | **DONE** | Structural, not procedural: `AuditRecordWriter` raises on any key outside `AUDIT_FIELD_ALLOWLIST` and on any key matching the hidden-reasoning denylist. A reasoning trace cannot be written even deliberately. |
| E39 | An artifact must not certify itself on a degenerate result | **DONE — was possible, now blocked** | The Shapley efficiency axiom passes vacuously when every coalition value is identical (`0 == 0 − 0`). A `retrain` run that abstained on every item therefore produced an all-zero ablation table that its own self-test certified. A separate **degeneracy check** now reports the number of distinct coalition values and fails loudly; `REPORT.md` renders the failure. See `FINDINGS_EVIDENCE_PATHWAY.md` §8. |
| E12 | Collect the structured decision record instead | **DONE** | The §7 record schema, with claim-level adjudication rules from a closed vocabulary. |
| E13 | State that the record is an audit artifact, not a window into cognition | **DONE** | In `README.md` and the ethics addendum; must also appear in the manuscript (P18). |

---

## 4. Data minimisation, retention, and access

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E14 | Named data owner | **DONE (2026-08-15)** | Named in §4a below. |
| E15 | Role-based access and access logs | **HUMAN GATE** | Depends on where the corpus is hosted, which is not yet decided. |
| E16 | Per-field log allowlists | **DONE** | `AUDIT_FIELD_ALLOWLIST`. |
| E17 | Never log full system prompts, credentials, tool tokens | **DONE** | Denylist patterns in the audit writer; the orchestrator strips before dispatch. |
| E18 | Encrypt in transit and at rest | **PARTIAL / HUMAN GATE** | In transit: yes, all providers are HTTPS. At rest: depends on E15's hosting decision. |
| E19 | Separate content data from experiment metadata | **DONE** | `manifest.json` / `config.json` carry no item content; `audit_records.jsonl` carries content under the allowlist. |
| E20 | Short default retention, documented deletion and legal-hold exceptions | **HUMAN GATE** | Propose 12 months for raw records, indefinite for aggregate results. Must be fixed before annotation (E6 withdrawal depends on it). |
| E21 | Correction/deletion process | **DONE (2026-08-15)** | Owner and contact route named in §4a. |
| E22 | Do not send personal data to six agents because one-agent processing was authorised | **DONE** | No benchmark in the suite contains personal data; every `DatasetPin` declares `personal_data`, and all seven are `False`. |
| E23 | State which providers and regions receive data, whether they retain it, whether it may be used for training | **DONE (2026-08-15)** | Provider fixed to Microsoft; §5 below is finalised. Settled before the first real call. |
| E24 | Consent or another valid basis before using human conversation data | **NOT APPLICABLE** | No human conversation data is used. The sycophancy battery simulates a user; it does not replay one. Recorded here so the absence is deliberate rather than unnoticed. |

---

## 4a. Named owner (E14, E33) — settled 2026-08-15

**Data owner and accountable system owner:** Kevin Zhou (zhouk.2009@gmail.com),
sole investigator at the time of collection.

E33 requires a single accountable owner rather than a diffuse one, and the
requirement applies to this study with particular force: a paper about an agent
council must not let the council stand in for a responsible human. The same
person is accountable for the data (E14) and for the system (E33). Correction
and deletion requests (E21) go to the same address.

This resolves E14, E21, and E33. It does **not** resolve E15 (role-based access
and access logs), E18's at-rest half, or E20 (retention period), all of which
depend on where the corpus is hosted — a decision that is still open and is not
required before the scoped study's collection runs.

---

## 5. Provider disclosure (E23) — **FINAL, settled 2026-08-15 before the first real call**

| Item | Value |
| --- | --- |
| Providers | Microsoft Corporation, via Azure OpenAI / Microsoft Foundry (resource kind `AIServices`). Requests go **directly** to Microsoft; no broker or reseller is in the path. Models served: `gpt-5.6-terra` and `gpt-5.6-luna` (OpenAI lineage) and `claude-sonnet-4-6` (Anthropic lineage). |
| Regions | `eastus2`. Deployment type: Global Standard. |
| Retention by provider | Standard abuse monitoring applies: prompts and completions may be retained by Microsoft for **up to 30 days**. Review is automated by default; human review by authorised Microsoft reviewers occurs only as necessary, under access controls. |
| Training use | Microsoft states that customer prompts and completions are **not** used to train or retrain models, and are not used to improve Microsoft or third-party products without permission. |
| Zero-retention endpoints available? | **Not in effect for this study.** Microsoft offers a Modified Abuse Monitoring exception that suppresses storage of prompts and completions, but it is application-gated and directed at managed enterprise customers with a Microsoft account team. This study runs on a Microsoft for Startups credit without one, so the standard 30-day abuse-monitoring path applies and is disclosed as such. |

**Exposure is bounded by the corpus, not only by the policy.** Per E22, no
benchmark in the suite contains personal data: every `DatasetPin` declares
`personal_data`, and all seven are `False`. What leaves the machine is public
benchmark text (HaluEval, TruthfulQA) plus generated model output. The 30-day
retention window therefore covers no personal data.

**The brokered-API risk this section originally recorded has been resolved
rather than worked around.** The earlier draft warned that an `OPENROUTER_API_KEY`
in the environment meant requests were brokered, so the *upstream* provider had
to be enumerated per model. That key is unset in the run environment and the
route is now direct to a single named provider.

**One open item, recorded rather than assumed.** Claude Sonnet 4.6 is served
through Microsoft Foundry, so Microsoft receives the bytes; whether Anthropic
acts as a subprocessor for Foundry-served Anthropic models has not been
confirmed against Microsoft's subprocessor documentation. This is the same
"one level deeper than the key suggests" question the original draft raised, in
a new form. It affects the second-family manipulation-check run only. If it
cannot be answered before that run, the run is dropped rather than the
disclosure being softened; the single-family tier check does not depend on it.

---

## 6. Red-teaming

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E25 | Prompt leakage between agents | **CODE READY / SIMULATED ONLY** | The §9 battery includes injection delivered *through another agent's output*; the orchestrator treats inter-agent text as untrusted. Passing against the stub bounds nothing about a real adversary. |
| E26 | Cross-agent data propagation | **DONE** | Structural, and this one genuinely transfers: `Council.build_user_content` assembles each agent's input from the item alone, so no agent output is *reachable* from the call graph before the declared communication round. Isolation is a property of the code, not of the fixtures, so the stub-only caveat does not apply. |
| E27 | Membership inference | **NOT STARTED** | Distinct from contamination (`contamination.py`, which measures overlap). Membership inference asks whether an *individual record's presence* is detectable. Given `personal_data=False` across all seven benchmarks, exposure is low, but the check has not been run. |
| E28 | Accidental release in research artifacts | **DONE** | Two layers. The audit writer's allowlist is the first. The second is `release.check_release`, which **refuses to assemble a bundle from a run whose provenance is not `real`** — a simulated bundle is a demonstration, not a research artifact, and publication must not be one accidental call away. Six tests. |
| E38 | Provenance stamps must be earned, not asserted | **DONE — was broken, now fixed** | A `StubProvider` run previously emitted records stamped `provenance: "real"`, and `build_datasets` inferred "real" from the mere existence of a records file. `--strict-real` therefore did **not** exclude fabricated data, contrary to `README.md`. Provenance is now derived from the provider (`Provider.synthetic`, fail-safe default `True`), any synthetic provider taints the whole council, and the dataset layer reads the stamps. See `FINDINGS_EVIDENCE_PATHWAY.md` §7. Four tests. |

---

## 7. Operational safety

| ID | Requirement | Status | Notes |
| --- | --- | --- | --- |
| E29 | Kill switch | **DONE** | `orchestrator/safety.py`. Three independent triggers — sentinel file, environment variable, programmatic — because which one is reachable depends on how the run failed. Checked between items, never mid-item, so a halted run leaves complete records. A halted run reports itself incomplete rather than looking finished. 5 tests. |
| E30 | Single-agent fallback when the council is degraded or under attack | **DONE** | `orchestrator/safety.py`. Fires on quorum failure or an injection rate above threshold. Deliberately does **not** salvage a council answer from survivors — aggregating two records with a synthesiser trained on six is out of distribution and the resulting confidence would be meaningless. Carried confidence is explicitly marked uncalibrated. 6 tests. |
| E31 | Preserve evidence behind rejected dissenting outputs | **DONE** | Rejected candidates persist in the record with their adjudication rule; the synthesiser's rejection reason is a closed-vocabulary value, so a dissent cannot be discarded silently. |
| E32 | Human review for consequential actions, with evidence, uncertainty, and unresolved disagreements | **NOT APPLICABLE / DOCUMENT** | No deployment exists. The manuscript must carry the deployment restriction (§8). |
| E33 | One accountable system owner; a council must not diffuse responsibility | **DONE (2026-08-15)** | Same person as E14; named in §4a. |
| E34 | Measure false alarms, over-refusal, fabricated alternatives, stubbornness | **DONE** | Every one is a declared harm metric on some adapter: `over_flagging_rate` (advconsistency), `abstention_rate` (all), `unsupported_answer_rate` (temporal_qa), `inappropriate_stubbornness` (sycophancy). The adapter base class refuses to construct an adapter with no harm metric. |
| E35 | Treat inter-agent text as untrusted input | **CODE READY / SIMULATED ONLY** | Orchestrator applies the same injection boundaries used for external content, and `CouncilHealth` degrades the council when the flagged-injection rate crosses a threshold. Exercised only against the stub provider: an injection boundary that holds against a fixture is not thereby shown to hold against an adversary. |
| E36 | Prohibit agents diagnosing users, role-playing a disorder, presenting stereotypes as clinical fact | **CODE READY / UNVERIFIED** | No prompt contains clinical vocabulary (`tests/test_prompts.py`); no routing depends on mental-health language (E37). |
| E37 | No mental-health advice or diagnosis routed differently because a user's language matches an analogy | **DONE** | Verified by construction: routing depends only on benchmark identity and coalition mask. There is no content-conditional routing path in the codebase at all — the synthesiser sees structured records, never raw user text. |

---

## 8. Deployment restriction — required manuscript text

To appear in the manuscript, not only here:

> The council must not be used for medical, legal, financial, or other
> high-impact decisions on the basis of benchmark results alone. Domain
> evaluation and accountable human review are separate requirements that this
> work does not satisfy and does not attempt to.

---

## 9. Publication reporting

| Requirement | Status |
| --- | --- |
| Why each clinical construct was selected | **PENDING (P18)** |
| Which part of each source is used as inspiration | **DONE** — `BIBLIOGRAPHY.md` §2 |
| What was deliberately not modelled | **PENDING (P18)** |
| Expert and lived-experience review | **HUMAN GATE (E5)** — report honestly if not done |
| Negative results and regressions | **BLOCKED on P14** — commitment registered in `PREREGISTRATION.md` §2 |
| Privacy and safety incidents | **BLOCKED on P14** |
| Demographic/language subgroup effects | **NOT APPLICABLE** — no benchmark involves people as subjects; record the reason rather than omitting the section |
| Funding, conflicts, provider relationships, model access | **HUMAN GATE** |

---

## 10. The five things a person must do first

Ordered by lead time, longest first.

1. **E1 — institutional ethics determination.** Gates P8 annotation and P10
   human rating, which are themselves the longest-lead items in the study.
   Start immediately; a determination that review is *not* required still has to
   be documented.
2. **E5 — compensated lived-experience and clinical terminology review.** Needs
   recruitment and payment. Cannot be substituted by more careful writing.
3. **E14/E33 — name the data owner and accountable system owner.** One
   sentence, but it blocks E15, E20, and E21.
4. **E23/§5 — provider and region disclosure.** Must be settled *before* the
   first real API call, because the first call is when data starts leaving.
5. **E20 — retention period.** Blocks the consent text (E6), which blocks
   annotation.
