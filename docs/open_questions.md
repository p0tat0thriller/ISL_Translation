# Open Questions & Caveats (for ISL-expert / superior review)

Living list of decisions deferred until we have Deaf-community / ISL-linguist
input. None block prototype development; all must be resolved before production.

## M4 — Grammar reordering (`scripts/m4_grammar_reorder.py`)

1. **Rules not validated by a Deaf/ISL expert** — this is the spec's stated KEY
   RISK. Current ordering follows the project's own worked example, not a
   signer's judgement. "Technically correct" ≠ "natural." Needs sign-off:
   show ~20 before/after gloss examples, confirm naturalness ≥ 4/5.

2. **Time-last vs time-first.** We place DURATION at the end (per the graph-doc
   worked example). Many sign languages topicalize time to the FRONT
   ("DAY-3, MOLAR PAIN…"). Decide which ISL prefers — it's a one-line change
   to rule R5.

3. **NMM over-flagging.** We attach `furrowed_brow` (intensity) to *every*
   SYMPTOM sign. A signer may reserve it for severe cases only. Tune the R7
   heuristic after expert review.

4. **No number signs.** "3 days" loses the "3" — there are no numeral signs in
   the 81-sign vocabulary yet. Add numerals (and a counting strategy) before
   durations/quantities can be signed faithfully.

## M1 — Clinical NLP (`scripts/m1_clinical_nlp.py`)

8. ~~**Negation scoping is utterance-wide.**~~ **RESOLVED.** Negation is now a
   per-entity, clause-scoped flag (`ClinicalNLP._is_negated`): a cue negates only
   entities before the next comma/`but`/`however`. M4 emits the headshake scoped
   to the negated sign. "no swelling, mild pain" → `SWELLING[negation]` only,
   pain unaffected. (Remaining nuance: full constituency parsing would handle
   rarer scopes better; the clause heuristic covers the common cases.)

9. ~~**Instruction phrases don't map.**~~ **RESOLVED.** Known no-sign phrases
   (`TIER3_PHRASES`: oral hygiene instructions, malocclusion) are tagged by NER
   with `force_decompose=True` and routed straight to M2 by the pipeline,
   bypassing M3's fuzzy match (which had collided "oral hygiene instructions"
   with "oral examination"). OHI → [mouth, tooth, cleaning, brush]. Extend
   `TIER3_PHRASES` as more no-sign clinical phrases surface.

10. ~~**Redundant/lossy temporal.**~~ **PARTLY RESOLVED.** Bare markers
    (since/ago) adjacent to a concrete unit are dropped (`_collapse_duration`):
    "since 3 days" → `DAY`. The "3" is still lost — that needs numeral signs
    (see #4), still open.

## M2 — LLM decomposition (`scripts/m2_decompose.py`)

11. **Vocabulary lacks function/role signs.** Good decompositions constantly
    produce logical/role primitives — `not`, `same`, `together`, `fit`,
    `doctor`, `straight`, `finish` — that aren't in the 81-sign vocab, so a
    correct breakdown (e.g. orthodontist → tooth + doctor + straight) can't
    fully map. Add a small set (~10–15) of function/role signs to the vocabulary
    and source ISL signs for them.
12. **LLM model + cache review pending.** Live decomposition currently defaults
    to `claude-opus-4-8` (overridable via `M2_MODEL` in `.env`). Decide the
    production model, then have the ISL expert review `data/derived/m2_cache.json`
    before any live use (LLM output is non-deterministic — the cache is the
    production source of truth).

## M3 — Primitive mapper (`scripts/m3_primitive_mapper.py`)

5. **`candidates` tier untested.** The 0.55–0.82 "show top-k" band didn't fire
   on the gold set. Needs real medium-confidence cases to validate.

6. **Lexical floor only.** Truly novel synonyms (not in `surface_forms`) need
   the semantic scorer (BioSentVec) slotted in behind `_best_candidates`.

7. **`decompose` fallback strategy.** Confirm the literacy-free cascade with the
   superior: top-k signs → M2 visual decomposition → image/iconic depiction →
   fingerspell (last resort only). Image-depiction route is still a proposal.
