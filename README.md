# Did the Role Fire?

**Manipulation checks for role-prompted multi-agent LLM systems, and an audit of
the checks themselves.**

Kevin Zhou¹, Sparsh Roy¹², Maxim Kudryashov¹
¹ Hopewell Valley Central High School · ² Massachusetts Institute of Technology

*Agent Evaluation Science, Fall 2026 (NYC): extended abstract and poster
(Session 1, Poster 23).*

---

Multi-agent systems give agents roles such as *critic*, *verifier* or *explorer*.
When a benchmark score improves, the improvement is credited to the roles. We
borrow **manipulation checks** from experimental psychology to test whether a
role prompt actually changes behaviour in the way it says it should. For each
of six roles we use:
- an executable check, computed from the agent's structured output;
- a length-matched neutral prompt and a bare role-name prompt as controls;
- a decision rule registered before data collection: Benjamini–Hochberg
  q ≤ 0.05 and paired d_z ≥ 0.40.

## Findings

| | Result | Status |
|---|---|---|
| Registered verdicts | 1 of 6 roles passes on `gpt-5.6-terra`; 2 of 6 on `grok-4.3` | confirmatory |
| Councils vs. one call | Three six-agent councils score 0.839–0.846 vs. 0.861 for a single call; none outperform it | registered |
| Specificity | A prompt moves its own check ≥ 0.20 more than any other check in **1 of 12** role–model cells | exploratory |
| Redundancy | Three checks track claim count (r = 0.79–1.00); local-claim **is** 2 × claim count on this data | exploratory |
| Guards | Two are defeatable. Three were never tested, because the behaviour they guard against barely occurred | exploratory |
| Explore-then-prune "reversal" | Consistent with a mismatch between prompt and check: the prompt records only the hypotheses that survive pruning | exploratory |

So **no role has a validated pass**. A benchmark delta doesn't show that a role did
its job, and neither does a manipulation check until the check itself is tested.
The next step is to preregister positive and negative controls for every check.

## What's here

| Path | Contents |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **Start here.** Setup, every command, and what each one reproduces |
| `cca_synth/` | Code: the six roles and their checks (`prompts/`), collection (`orchestrator/`), analyses (`analysis_*.py`, `main.py`), tests |
| `cca_synth/runs/` | Every collected record and every analysis output |
| `cca_synth/data/` | HaluEval QA and TruthfulQA at pinned revisions, with SHA-256 |
| `cca_synth/scripts/` | The exact collection commands (need credentials) and the offline per-arm pipeline |
| [`paper/`](paper/) | Full manuscript (NeurIPS 2026 EIA workshop submission) with a revision checklist |
| [`extended_abstract/`](extended_abstract/) | AES extended abstract as submitted, where every number comes from, and corrections since submission |
| [`poster/`](poster/) | AES poster (48 × 36 in): every figure generated from `cca_synth/runs/` |
| `logistics/` | Research specification, preregistration and amendments, ethics review, data card, working notes |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r cca_synth/requirements.txt
cd cca_synth
python3 -m pytest -q                                                    # offline, about 30 s
python3 -m prompts.manipulation_check --records runs/checks/check_records.jsonl   # registered verdicts (exits 1 by design)
python3 analysis_cross_role.py --records runs/checks/check_records.jsonl          # the audit
```

No API keys or network access are needed. The commands re-score the shipped
records. Verified on Python 3.13.9. See [`REPRODUCE.md`](REPRODUCE.md) for the
full list.

## Scope

Six roles from one council design, one benchmark (HaluEval QA) and two model
families. The audit (specificity, redundancy, guards) is **exploratory**: it was
computed after the registered analysis, on records that had already been
observed. The margin it uses was fixed in advance; see
`logistics/A4_DRAFT.FREEZE.sha256`.

## Ethics note

The roles were inspired by accounts of human cognition. *These roles are
functional AI processing constraints. They are not diagnostic models,
simulations of lived experience, or claims that mental health conditions confer
a single cognitive mechanism.* See `logistics/ETHICS_COMPLIANCE.md`.

## License and data

The code is MIT-licensed (`LICENSE`). HaluEval is under the MIT License and
TruthfulQA under Apache-2.0; both are redistributed with attribution (see
`LICENSE`). The AES template files in `extended_abstract/` are MIT; the AES mark
is not redistributed.

Model inference was funded by Microsoft for Startups Azure credits.

## Citation

```bibtex
@inproceedings{zhou2026rolefire,
  title     = {Did the Role Fire? Auditing the Manipulation Checks Behind
               Role-Prompted Multi-Agent Systems},
  author    = {Zhou, Kevin and Roy, Sparsh and Kudryashov, Maxim},
  booktitle = {Agent Evaluation Science (AES), Fall 2026, New York},
  note      = {Extended abstract, non-archival},
  year      = {2026},
  url       = {https://github.com/kevzho/role-prompted-systems}
}
```

Contact: zhouk.2009@gmail.com
