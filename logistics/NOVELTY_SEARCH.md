# Novelty search

Addresses limitation **L2**: the claim "CCA is the first" is unsupported without
a documented search. This document records what was searched, when, by what
means, what was found, and what the findings do to the manuscript's novelty
language.

**Search date:** 2026-07-28.

---

## 0. Honest scope of this search — read before citing it

This is a **first-pass structured search, not a PRISMA-style systematic review.**
The distinction matters, because overclaiming here would repeat exactly the
error L2 identifies.

**What was actually done.** Nine query families were run through a general web
search interface that indexes arXiv, ACL Anthology, OpenReview, PubMed, and
publisher sites. Returned records were screened by title and abstract snippet.
Full texts were **not** retrieved. Citation counts, venue status, and author
lists were **not** independently verified except where
`REFERENCE_CORRECTIONS.md` had already verified them.

**What was not done, and is still required before submission:**

| Required | Status |
| --- | --- |
| Native ACL Anthology query with result counts | **done 2026-08-02** via `site:aclanthology.org` web search (substitution — the Anthology's native search is a Google CSE with no API; see `NOVELTY_SEARCH_QUANT.md` §4) |
| Native OpenReview query (incl. rejected/withdrawn submissions) | **done 2026-08-02** — 6 queries, top-50 each, 300 notes screened; see `NOVELTY_SEARCH_QUANT.md` §3 |
| Semantic Scholar API search with citation-graph forward/backward snowballing | **search done 2026-08-02 for 3 of 8 queries** (275 records screened; 5 queries rate-limited and recorded as failures); **snowballing still not done**; see `NOVELTY_SEARCH_QUANT.md` §2 |
| PubMed/PsycINFO with MeSH terms for the clinical constructs | **not done** |
| Exact screened/included/excluded counts per database | **done 2026-08-02** for the three databases above; see `NOVELTY_SEARCH_QUANT.md` §7. **Key result: 2 (+1 borderline) predecessor candidates found — "none in N screened" is NOT supportable in plain form; see `NOVELTY_SEARCH_QUANT.md` §5** |
| Second independent screener and inter-screener agreement | **not done** |
| Full-text reading of the near-neighbour set (§3) | **not done** |

Consequently this document can support the statement *"a search identified
material predecessors, so absolute novelty language is removed"* — a claim in
the direction of **less** novelty, which a partial search can legitimately
establish. It **cannot** support *"no predecessor exists"*: absence of evidence
from an incomplete search is not evidence of absence. The remaining work in the
table above is a human task and is listed in the handover.

---

## 1. Queries run

Verbatim, in execution order.

| # | Family | Query |
| --- | --- | --- |
| Q1 | Multi-agent diversity | `multi-agent LLM diversity role specialization council debate survey 2025 2026` |
| Q2 | Cognitive architectures; computational psychiatry | `cognitive architecture inspired LLM agents psychiatric disorder metaphor computational psychiatry` |
| Q3 | Role prompting | `arxiv persona role prompting does not improve LLM accuracy ablation label vs operation` |
| Q4 | Self-critique; verifier models | `arxiv LLM verifier self-critique self-refine limitations cannot self-correct reasoning` |
| Q5 | Ensembles; learned routers | `heterogeneous LLM ensemble learned router mixture of experts model routing cost quality frontier` |
| Q6 | Direct-term probe | `"cognitive constraint architecture" LLM agents OR "constrained cognition" multi-agent inspired by mental disorders` |
| Q7 | Activation-level steering | `activation steering representation engineering control vectors LLM behaviour attention temperature intervention` |
| Q8 | Debiasing ensembles; deliberation | `arxiv multi-agent LLM hallucination reduction cross-examination factuality council 2026 benchmark TruthfulQA HaluEval` |
| Q9 | Uncertainty; abstention | `arxiv LLM uncertainty quantification calibration ensemble disagreement selective prediction abstention 2026` |
| Q10 | Sycophancy (outcome-specific) | `arxiv sycophancy LLM user pressure retention benchmark inappropriate stubbornness capitulation` |

**Inclusion rule.** A record is included if it proposes, evaluates, or
critiques (a) role- or persona-differentiated multi-agent LLM systems, (b)
cognitive- or clinical-theory-derived agent architectures, (c) synthesis over
multiple model outputs, or (d) an outcome this study pre-registers.

**Exclusion rule.** Excluded: applied deployments with no architectural claim;
single-agent prompt engineering with no multi-agent component; work on human
group deliberation with no LLM component; blog posts and vendor pages except
where used only to locate a primary record.

---

## 2. The finding that decides the novelty language

**Q6 — the direct-term probe — returned material predecessors.** The exact
phrase "cognitive constraint architecture" did not surface as a named prior
system, but the *construct* did, under other names:

> **Theater of Mind for LLMs: A Cognitive Architecture Based on Global
> Workspace Theory** — arXiv:2604.08206.
> Couples a central broadcast hub with "a heterogeneous swarm of **functionally
> constrained agents**", assigning "strict cognitive constraints to specialized
> agents".

That is the same architectural move CCA proposes: a set of agents each
deliberately restricted to a partial information-processing operation, combined
by a central synthesiser, motivated by a theory of human cognition. It is
derived from Global Workspace Theory rather than from psychopathology, but the
manuscript's novelty claim is about the *architecture*, not about which
cognitive theory supplied the metaphor.

Together with §3.1 and §3.4 below, this is sufficient to conclude:

> **"CCA is the first" must be removed. It is not defensible.**
> **The "categorical, not a matter of degree" framing must also go** — the
> nearest neighbours differ from CCA in degree, along axes (which theory, how
> many agents, how synthesis works) that are continuous.

---

## 3. Related-work matrix

Rated against CCA on four axes: **RD** role differentiation is functional rather
than label-only · **CT** derived from a cognitive/clinical theory · **SY**
learned or structured synthesiser over candidates · **EV** predeclared
manipulation checks + per-role attribution.

### 3.1 Cognitive-architecture councils — the closest neighbours

| Work | RD | CT | SY | EV | Relation to CCA |
| --- | :-: | :-: | :-: | :-: | --- |
| Theater of Mind (arXiv:2604.08206) | ✓ | ✓ | ✓ | ✗ | **Nearest predecessor.** Functionally constrained specialised agents + central broadcast hub, from Global Workspace Theory. Differs in the source theory and in having no per-agent attribution. |
| CoALA (arXiv:2309.02427) | ~ | ✓ | ✗ | ✗ | Framework/blueprint for cognitive-architecture language agents. Establishes the general programme CCA sits inside. |
| United Minds or Isolated Agents? (arXiv:2506.06843) | ~ | ✓ | ✗ | ✗ | Cognitive Load Theory applied to LLM coordination; constraint-as-architecture, no role specialisation. |
| WiseMind (npj Digital Medicine, 2026) | ✓ | ✓ | ✓ | ✗ | Dialectical-Behaviour-Therapy-derived two-agent split ("Reasonable Mind" / "Emotional Mind") for psychiatric assessment. **A clinical theory already generating a multi-agent role decomposition.** |

### 3.2 Psychiatry-inspired AI — the framing is an existing field

| Work | Relation |
| --- | --- |
| Lee et al., *Emergence of psychopathological computations in LLMs* (arXiv:2504.08016) | Already corrected in `REFERENCE_CORRECTIONS.md` [20]. Computational-theoretical + mechanistic-interpretability framework. |
| *Synthetic psychopathology* (established topic area) | Dysfunction-analogous patterns in artificial cognitive systems, drawing on computational psychiatry and clinical nosology. **CCA is an entry in an existing literature, not the opening of one.** |
| Architectural Constraints in LLM-Simulated Cognitive Decline (doi:10.3390/ai7020069) | Manipulates prompt context to induce structurally constrained processing; explicitly separates constrained-memory from generative tasks. |
| BioLLMAgent (arXiv:2603.05016) | RL cognitive model + LLM shell + decision fusion, for computational psychiatry. |

**Consequence.** The introduction cannot present clinical inspiration for AI
architectures as new. It should cite this literature and state what CCA adds.

### 3.3 Role and persona prompting — the evidence runs *against* the mechanism

| Work | Finding | Consequence for CCA |
| --- | --- | --- |
| Zheng et al. (arXiv:2311.10054) | 162 roles in system prompts **do not** improve performance vs no-persona control | Condition 6 (persona-label-only) is predicted null by prior work, not by CCA. Cite it as the prior, not as a CCA finding. |
| Expert Personas Improve Alignment but Damage Accuracy (arXiv:2603.18507) | MMLU 68.0% with expert personas vs **71.6%** baseline — personas *hurt* | Strengthens H3's premise but means a positive H3 result is a replication of a known direction. |
| Persona is a Double-edged Sword (arXiv:2408.08631) | Role-playing can degrade zero-shot reasoning; ensembling role + neutral helps | Anticipates CCA's neutral-control arm. |
| When Does Persona Prompting Actually Help? (arXiv:2605.29420) | Task-dependent; gains concentrated in extraction/STEM | Predicts heterogeneous per-benchmark effects — matches L12's requirement to score every agent on every benchmark. |
| The Prompt Makes the Person(a) (arXiv:2507.16076) | Systematic sociodemographic persona evaluation | Methodological precedent for length-matched persona controls. |

**This is the most useful result of the search for the study itself.** The
literature already establishes that *labels* do not carry performance. That
makes CCA's H3 — gains survive label removal — the load-bearing hypothesis, and
makes a *failed* H3 an uninteresting replication rather than a surprise. The
manuscript should say so in advance.

### 3.4 Debate, councils, and synthesis

| Work | Relation |
| --- | --- |
| **Council Mode** (arXiv:2604.02923) | **The head-to-head comparator.** Heterogeneous frontier models + triage + dedicated consensus model. Reports −35.9% relative hallucination on a 1,200-item HaluEval subset and +7.8 points TruthfulQA vs best single model — *the same two benchmarks CCA pre-registers*. Already listed as [10] in `REFERENCE_CORRECTIONS.md`. CCA condition 10 must be compared against this directly, and CCA's numbers must be reported next to it. |
| Talk Isn't Always Cheap (arXiv:2509.05396) | Failure modes in multi-agent debate. |
| The Deliberative Illusion (arXiv:2606.03032) | Factual attrition and stance homogenisation in deliberation — the mechanism behind CCA's "synthesizer capture" and "evidence-free agreement" probes. |
| Hallucination Cascade (arXiv:2606.07937) | Error propagation across agents; supports the §9 injection-via-agent-output probe. |
| Heterogeneous LLM Debate Under Adversarial Peers (arXiv:2606.19826) | Robustness of heterogeneous councils to bad-faith members. |
| Estornell & Liu (tyranny of the majority) | Minority agents conform to a wrong majority — **exactly the "confidently wrong majority" probe**, already implemented and already failing in `FINDINGS_EVIDENCE_PATHWAY.md` §3. |
| Multi-Agent Systems are Mixtures of Experts (arXiv:2605.25929) | Listed as [11]; input-dependent influence via Friedkin–Johnsen dynamics. |
| DEBATE benchmark (NeurIPS 2025) | 16,428 messages, 107 topics, >2,000 participants. |

### 3.5 The result that most threatens the premise

> **When Does Combining Language Models Help? A Co-Failure Ceiling on Routing,
> Voting, and Mixture-of-Agents Across 67 Frontier Models** — arXiv:2606.27288.

Combination methods are bounded by *correlated* failures across models. This is
limitation **L7** stated as an empirical ceiling and measured at scale. It must
be cited in the introduction, not buried: it sets the prior expectation for H1,
and it means CCA's measured pairwise error correlation is not a diagnostic
afterthought but a **primary** quantity determining whether any gain is
available to be found.

### 3.6 Self-critique and verifiers — supports the council premise

| Work | Finding |
| --- | --- |
| Huang et al. (arXiv:2310.01798) | LLMs cannot reliably self-correct reasoning without external feedback; performance can degrade. |
| Stechly et al. (arXiv:2402.08115) | Poor at all three self-critique roles; stacked errors make the loop worse than one-shot. Gains appear **only with a sound external verifier**. |
| Small LMs Need Strong Verifiers (arXiv:2404.17140) | Weak self-verifiers do not help; strong external ones do. |
| **The Self-Correction Illusion** (arXiv:2606.05976) | Models "correct others but not themselves" — catch errors in external content that they miss in their own traces. |

The last is the strongest *supporting* citation available for CCA's core
premise, and it is currently uncited. If agents critique each other's records
rather than their own, the literature predicts a real gain. This should replace
weaker clinical-analogy motivation in the introduction.

### 3.7 Routers and cost-matching

RouteLLM (arXiv:2406.18665), RouterEval (arXiv:2503.10657), xRouter
(arXiv:2510.08439), UCCI (arXiv:2605.18796). Learned routing achieves a large
fraction of frontier quality at a fraction of cost. **This is the control CCA's
budget-matched arms must beat**, and a six-call council starts far behind it on
the cost axis. Reported router results reaching ~92% of frontier accuracy at
~18× less compute set a demanding bar for H1 under cost-matching.

### 3.8 Uncertainty and abstention

UQ survey (arXiv:2503.15850); conformal abstention (arXiv:2405.01563);
pre-trained UQ heads (arXiv:2505.08200); **cross-model disagreement for UQ
(arXiv:2604.17112)** — the last uses inter-model vs intra-model disagreement as
an epistemic-uncertainty signal, which is the same quantity CCA's synthesiser
has available and, per `FINDINGS_EVIDENCE_PATHWAY.md` §2–§3, currently fails to
route to abstention.

### 3.9 Activation-level steering — bears on the V2 scope problem (L4)

What Can We Actually Steer? (arXiv:2511.18284); Understanding Unreliability of
Steering Vectors (arXiv:2602.17881); Activation Steering Induces Emergent
Misalignment (arXiv:2606.08682).

The steering literature has moved toward *negative* results on reliability and
safety. The manuscript's V2 claims about early-layer attention temperature are
therefore not merely unimplemented (L4) but would be **contested even if
implemented**. Scope the paper to portable V1.

### 3.10 Sycophancy — the pre-registered outcome already has instruments

SycEval (arXiv:2502.08177); **PARROT (arXiv:2511.17220)**, which classifies
outcomes as robust-correct / sycophantic-agreement / reinforced-error /
**stubborn-error**; Pressure, What Pressure? (arXiv:2604.05279), decomposing
sycophancy into pressure capitulation and evidence blindness; Social Sycophancy
(arXiv:2505.13995).

PARROT's four-way classification is a published instrument for exactly the
both-directions measurement `PREREGISTRATION.md` §3 requires of
`confidence_calibrator`. **Adopt it rather than inventing a battery** — P7's
sycophancy adapter should implement PARROT's taxonomy and cite it.

---

## 4. What survives as a novelty claim

Removed:

- ~~"CCA is the first..."~~ — refuted by §2, §3.1, §3.2.
- ~~"categorical, not a matter of degree"~~ — the neighbours differ by degree.
- ~~clinical inspiration for AI architectures as a new idea~~ — §3.2.
- ~~role differentiation improves multi-agent performance~~ — prior work, §3.3/§3.4.

Defensible, stated with "to our knowledge" and subject to the §0 caveat:

1. **Predeclared operations with executable manipulation checks.** No work found
   in §3.1–§3.4 verifies that a role prompt performed its intended
   transformation before interpreting a benchmark result. This is CCA's
   strongest methodological contribution and should lead the paper.
2. **A learned, permutation-invariant synthesiser over *structured* records**
   rather than free text, with claim-level adjudication rules. Council Mode uses
   a consensus *model* over text; CCA's operator consumes a schema and is
   provably order-invariant.
3. **Exact 64-coalition Shapley attribution with item-level bootstrap
   intervals** applied to LLM agent roles. The debate literature reports system
   totals; per-role marginal value with uncertainty was not found.
4. **The label-vs-operation dissociation run as two separate arms** (role-blind
   synthesiser *and* clinical-language-removed prompts). §3.3's prior work
   removes personas; it does not separate these two mechanisms.

Each is a claim about *method*, not about performance. That is the correct
altitude while `LIMITATIONS.md` L1 stands.

---

## 5. Required changes to the manuscript

1. Delete every "first" and "categorical" claim (§2).
2. Add a related-work subsection covering §3.1 and §3.2; state that
   clinical/cognitive-theory-derived agent architectures are an existing field.
3. Cite Council Mode (arXiv:2604.02923) as the direct comparator and report CCA
   against it on HaluEval and TruthfulQA, or state why no comparison was run.
4. Cite the co-failure ceiling (arXiv:2606.27288) in the introduction and tie it
   to the measured error correlation.
5. Replace clinical-analogy motivation for the council premise with
   arXiv:2606.05976 (models correct others but not themselves).
6. Cite the persona literature (§3.3) as establishing the *prior* for H3.
7. Adopt PARROT's four-way sycophancy taxonomy in P7 rather than a bespoke one.
8. Scope to V1; cite §3.9 on steering unreliability when explaining why.
9. Reproduce §0 verbatim as an appendix limitation of the search itself.

---

## Sources

- [Multi-Agent Debate: Framework & Applications](https://www.emergentmind.com/topics/multi-agent-debate-approach)
- [Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate](https://arxiv.org/pdf/2509.05396)
- [The Deliberative Illusion](https://arxiv.org/pdf/2606.03032)
- [Heterogeneous LLM Debate Under Adversarial Peers](https://arxiv.org/pdf/2606.19826)
- [DEBATE benchmark (NeurIPS 2025)](https://neurips.cc/virtual/2025/124579)
- [Council Mode: A Heterogeneous Multi-Agent Consensus Framework](https://arxiv.org/abs/2604.02923)
- [Hallucination Cascade: Error Propagation in Multi-Agent LLM Systems](https://arxiv.org/abs/2606.07937)
- [When Does Combining Language Models Help? A Co-Failure Ceiling](https://arxiv.org/html/2606.27288)
- [Theater of Mind for LLMs: Cognitive Architecture Based on Global Workspace Theory](https://arxiv.org/html/2604.08206v1)
- [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/pdf/2309.02427)
- [United Minds or Isolated Agents? Coordination of LLMs under Cognitive Load Theory](https://arxiv.org/pdf/2506.06843)
- [WiseMind: knowledge-guided multi-agent framework for psychiatric assessment](https://www.nature.com/articles/s41746-026-02559-9)
- [Synthetic Psychopathology in AI](https://www.emergentmind.com/topics/synthetic-psychopathology)
- [Architectural Constraints in LLM-Simulated Cognitive Decline](https://doi.org/10.3390/ai7020069)
- [BioLLMAgent: Simulating Human Decision-Making in Computational Psychiatry](https://arxiv.org/pdf/2603.05016)
- [When "A Helpful Assistant" Is Not Really Helpful](https://arxiv.org/abs/2311.10054)
- [Expert Personas Improve LLM Alignment but Damage Accuracy](https://arxiv.org/pdf/2603.18507)
- [Persona is a Double-edged Sword](https://arxiv.org/html/2408.08631v1)
- [When Does Persona Prompting Actually Help?](https://arxiv.org/html/2605.29420v1)
- [The Prompt Makes the Person(a)](https://arxiv.org/html/2507.16076v2)
- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
- [On the Self-Verification Limitations of LLMs](https://arxiv.org/abs/2402.08115)
- [Small Language Models Need Strong Verifiers to Self-Correct Reasoning](https://arxiv.org/abs/2404.17140)
- [The Self-Correction Illusion: LLMs Correct Others but Not Themselves](https://arxiv.org/html/2606.05976)
- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/html/2406.18665)
- [RouterEval](https://arxiv.org/html/2503.10657v1)
- [xRouter: Training Cost-Aware LLM Orchestration](https://arxiv.org/pdf/2510.08439)
- [UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing](https://arxiv.org/pdf/2605.18796)
- [Multi-Agent Systems are Mixtures of Experts](https://arxiv.org/abs/2605.25929)
- [Uncertainty Quantification and Confidence Calibration in LLMs: A Survey](https://arxiv.org/abs/2503.15850)
- [Complementing Self-Consistency with Cross-Model Disagreement for UQ](https://arxiv.org/html/2604.17112v1)
- [Mitigating LLM Hallucinations via Conformal Abstention](https://arxiv.org/pdf/2405.01563)
- [A Head to Predict and a Head to Question: Pre-trained UQ Heads](https://arxiv.org/pdf/2505.08200)
- [What Can We Actually Steer? A Multi-Behavior Study of Activation Control](https://arxiv.org/pdf/2511.18284)
- [Understanding Unreliability of Steering Vectors in Language Models](https://arxiv.org/pdf/2602.17881)
- [Activation Steering Induces Emergent Misalignment](https://arxiv.org/pdf/2606.08682)
- [SycEval: Evaluating LLM Sycophancy](https://arxiv.org/abs/2502.08177)
- [PARROT: Persuasion and Agreement Robustness Rating of Output Truth](https://arxiv.org/abs/2511.17220v1)
- [Pressure, What Pressure? Sycophancy Disentanglement via Reward Decomposition](https://arxiv.org/html/2604.05279v1)
- [Social Sycophancy: A Broader Understanding of LLM Sycophancy](https://arxiv.org/html/2505.13995v1)
- [Emergence of psychopathological computations in large language models](https://arxiv.org/abs/2504.08016)
