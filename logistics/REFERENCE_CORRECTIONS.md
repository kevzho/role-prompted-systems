# Reference corrections and verification notes

Checked 2026-07-26. These notes resolve visible `[TBD]` records and distinguish bibliographic support from conceptual inspiration. A future submission should still run a complete, documented literature review.

## Corrections to incomplete records

### [9] Personality diversity

The PDF lists this as NeurIPS 2024. The verified record is a 2025 NeurIPS workshop paper:

Hengbin Yu, Jingzhe Lin, Shuhang Xu, Xuefeng Cui, Song-Chun Zhu, and Fangwei Zhong. "Enhancing Collective AI Decision Robustness through Personality Diversity." First Workshop on LLM Persona Modeling, NeurIPS 2025.

Official listing: <https://nips.cc/virtual/2025/loc/mexico-city/129942>

The official abstract reports a 95.6% GPI reduction for the paper's OCEAN condition. That is a result in its specific corporate-decision simulation, not a general 95.6% reduction in multi-agent group polarization.

### [10] Council Mode

Shuai Wu, Xue Li, Yanna Feng, Yufang Li, and Zhijun Wang. "Council Mode: Mitigating Hallucination and Bias in LLMs via Multi-Agent Consensus." arXiv:2604.02923, 2026.

Record: <https://arxiv.org/abs/2604.02923>

The work uses heterogeneous frontier models, triage, and a dedicated consensus model. CCA comparisons should control for both model heterogeneity and synthesis strength.

### [11] Multi-agent systems as mixtures of experts

Franka Bause, Jonas Niederle, Martin Pawelczyk, and Rebekka Burkholz. "Multi-Agent Systems are Mixtures of Experts: Who Becomes an Influencer?" arXiv:2605.25929, 2026.

Record: <https://arxiv.org/abs/2605.25929>

The paper analyzes input-dependent influence through Friedkin-Johnsen opinion dynamics. It is relevant to routing and influence, but it does not directly validate CCA's proposed confidence weighting.

### [12] Value diversity

Muhua Huang, Qinlin Zhao, Xiaoyuan Yi, and Xing Xie. "On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity." arXiv preprint, December 2025.

Official Microsoft Research record: <https://www.microsoft.com/en-us/research/publication/on-the-dynamics-of-multi-agent-llm-communities-driven-by-value-diversity/>

The official summary reports benefits with diminishing returns and instability under extreme heterogeneity. CCA should cite both the benefit and this trade-off.

### [20] Psychopathological computations in LLMs

The likely intended source is:

Soo Yong Lee, Hyunjin Hwang, Taekwan Kim, Yuyeong Kim, Kyuri Park, Jaemin Yoo, Denny Borsboom, and Kijung Shin. "Emergence of psychopathological computations in large language models." arXiv:2504.08016, 2025.

Record: <https://arxiv.org/abs/2504.08016>

Use the source's actual title. It proposes a computational-theoretical and mechanistic-interpretability framework; describing it only as "synthetic psychopathology" loses important scope and caveats.

## Incorrect or weak attribution

### [14] ADHD and explore/exploit

The title exists, but the PDF's author attribution to "M. Jepma et al." is incorrect. The verified citation is:

Merideth A. Addicott, John M. Pearson, Julia C. Schechter, Jeffrey J. Sapyta, Margaret D. Weiss, and Scott H. Kollins. "Attention-deficit/hyperactivity disorder and the explore/exploit trade-off." *Neuropsychopharmacology* 46(3):614-621, 2021. Online 2020. DOI: 10.1038/s41386-020-00881-8.

Record: <https://pubmed.ncbi.nlm.nih.gov/33040092/>

The study used 26 adults with ADHD and 23 controls in a six-armed bandit task. It supports a narrow empirical association in that setting, not a universal "shifted explore/exploit ratio." Also, no ADHD-derived role appears in the six-agent table, so either remove the introductory mapping or explain its exclusion.

### [18] Impostor phenomenon

Dunning's review of the Dunning-Kruger effect does not establish impostor phenomenon. Replace it with a source on impostor phenomenon, for example:

Pauline Rose Clance and Suzanne Ament Imes. "The Impostor Phenomenon in High Achieving Women: Dynamics and Therapeutic Intervention." *Psychotherapy: Theory, Research and Practice* 15(3):241-247, 1978. DOI: 10.1037/h0086006.

The paper's proposed uniform confidence penalty is still an engineering hypothesis. The psychological source does not validate subtracting a constant from an LLM's confidence.

## Sources that inspire but do not validate the mechanism

| PDF reference | What it can support | What it cannot establish by itself |
| --- | --- | --- |
| [13] Adams et al. | Computational psychiatry as a modeling field | That OCD is equivalent to a fixed `k`-pass verification loop |
| [15] Frith | Historical weak-central-coherence theory in autism research | An equivalence among autism, savant syndrome, context segmentation, and narrative-error detection |
| [16] Gruber | Positive emotion and reward disturbance in bipolar disorder | A two-temperature generation/pruning algorithm or beneficial "cycling" |
| [17] Kapur | Aberrant salience as an account of psychosis | Low-frequency token upweighting, paranoid verification, or improved rare-answer accuracy |
| [18] replacement | Self-doubt and difficulty internalizing achievement in impostor phenomenon | A calibrated confidence correction for model outputs |

For each role, use two citations where possible:

1. a current review or computational account of the human construct
2. an AI or machine-learning source that directly motivates the implemented operation

Then describe the link as inspiration until the manipulation check and benchmark evidence support a narrower mechanistic statement.

## Literature-review procedure needed for novelty

Before submission:

1. Search arXiv, ACL Anthology, OpenReview, Semantic Scholar, PubMed/PsycINFO, and major proceedings.
2. Publish exact queries covering multi-agent diversity, cognitive architectures, computational psychiatry and AI, debiasing ensembles, role prompting, self-critique, verifier models, uncertainty, and deliberation.
3. Record the search date, screened count, inclusion/exclusion rules, and related-work matrix.
4. Compare CCA against prompt-role specialization without clinical framing, heterogeneous-model ensembles, learned routers, verifier/debate systems, and activation-level steering.
5. Change absolute novelty language if any material predecessor overlaps.

