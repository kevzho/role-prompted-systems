# Invalidated: OLS output-length adjustment (2026-08-23)

`analysis_length_robustness.py` regressed the paired check difference on the
paired output-token difference and read the intercept as "the effect at equal
output length". Adversarial review disqualified the estimator:

1. **Length is post-treatment.** It is caused by the operation prompt, so it is
   a mediator, never a confounder. The intercept is not a corrected estimate.
2. **Placebo failure.** Persona-vs-neutral on gpt-5.6-terra -- two arms neither
   of which carries the operation -- yields spuriously negative adjusted d on
   5 of 6 roles.
3. **Common-slope restriction rejected** for counter_evidence_checker,
   F(1,197) = 76.6 (terra) / 30.1 (grok), so a single "effect at equal length"
   does not exist for the role the conclusion rested on.
4. **Proxy invalid on terra.** Delta-tokens and delta-characters carry opposite
   signs on 3 of 6 roles; explore_then_prune is +31 tokens but -133 characters.
5. **Circular on grok.** output_tokens is a near-deterministic readout of the
   scored record (R^2 = 0.970), so the regression conditions on the outcome.

Moved rather than deleted so the record of what was tried survives. Nothing in
the paper cites these files. A replacement, if wanted, should use
volume-invariant estimands (counterevidence presence rate; check per 100 output
tokens), which are immune to all five defects.
