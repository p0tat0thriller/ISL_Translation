# ISL Reverse Translation — Clinical Text → Indian Sign Language

Turns a dentist's typed clinical input (shorthand, abbreviations, plain English)
into a grammatically correct **Indian Sign Language (ISL)** rendering for a Deaf
patient.

> **"Reverse" translation:** the usual direction in sign-language ML is
> sign → text (recognition). This project does the opposite — **text → sign
> production** — for the clinical dental domain. No published system does
> text→ISL for a clinical domain, which is what makes M3 (mapping) and M4
> (grammar) the novel, publishable core.

```
Doctor types:  "pt c/o severe pain UL6 since 3d, worsening on cold"
System emits:  MOLAR LOC_UPPER LOC_LEFT  PAIN[intensity] SEVERE[emphasis]  COLD  DAY
               (+ a non-manual-marker track for facial expressions)
```

---

## Table of contents

1. [Status at a glance](#status-at-a-glance)
2. [Architecture & the hybrid decision](#architecture--the-hybrid-decision)
3. [Repository layout](#repository-layout)
4. [Quick start](#quick-start)
5. [The data foundation (vocabulary spine)](#the-data-foundation-vocabulary-spine)
6. [The pipeline — data contract chain](#the-pipeline--data-contract-chain)
7. [Module reference (input / output / how it feeds the next)](#module-reference)
8. [End-to-end worked example](#end-to-end-worked-example)
9. [Glossary](#glossary)
10. [Open questions & caveats](#open-questions--caveats)

---

## Status at a glance

| Stage | Module | Status | Notes |
|-------|--------|--------|-------|
| Data foundation | `phase0_mining`, `build_vocabulary` | ✅ Built | 81-sign vocabulary + ISKG edges, mined from the NoteChat corpus |
| M1 | Clinical NLP preprocessing | ✅ Built | Pure stdlib; spaCy NER can swap in later |
| M2 | LLM decomposition | ✅ Built | Offline cache + seeds run now; live Claude path optional |
| M3 | ISL primitive mapper *(core)* | ✅ Built | Lexical/fuzzy v1; BioSentVec can swap in later |
| M4 | ISL grammar reordering *(core)* | ✅ Built | 7-rule engine + non-manual markers |
| M5 | Sign renderer | ⬜ Not built | **Blocked**: 0 of 81 ISL signs sourced (needs videos/poses) |

**The entire text→ISL "brain" (M1→M4) runs end-to-end today with no API key and
no sign videos.** Only the visual renderer (M5) is outstanding, and it depends on
real ISL sign data that must be captured/sourced from a Deaf signer or corpus.

---

## Architecture & the hybrid decision

The project follows a **hybrid** path (decided up front):

- **Build the rule-based pipeline first** (M1→M5) as a working system.
- **Structure M3/M4's knowledge as graph data** (the ISKG — a typed node/edge
  table) so the same artifacts double as a knowledge graph, and the rule-based
  output can later be distilled into a Graph Neural Network.

Two source design docs sit in the repo for reference:

- [`Reverse Translation.pdf`](Reverse%20Translation.pdf) — the sequential M1→M5 spec.
- [`graph_based_reverse_translation_architecture.md`](graph_based_reverse_translation_architecture.md)
  — the graph (HCSG/ISKG) reformulation the hybrid path borrows from.

---

## Repository layout

```
ISL Translation/
├── README.md                         ← you are here
├── notechat_dental_conversations_cleaned.csv   ← source corpus (5,430 rows)
├── scripts/
│   ├── phase0_mining.py              ← mine the corpus → vocab candidates, NER seed
│   ├── build_vocabulary.py           ← build the 81-sign vocabulary + ISKG edges
│   ├── m1_clinical_nlp.py            ← M1  ClinicalNLP
│   ├── m2_decompose.py               ← M2  LLMDecomposer (+ pluggable LLM client)
│   ├── m3_primitive_mapper.py        ← M3  ISLPrimitiveMapper
│   ├── m4_grammar_reorder.py         ← M4  ISLGrammarReorderer
│   └── pipeline.py                   ← ISLTranslator — the end-to-end entry point
├── data/
│   ├── derived/                      ← generated artifacts (safe to regenerate)
│   │   ├── isl_vocabulary.json       ← THE vocabulary spine (81 signs + ISKG nodes)
│   │   ├── iskg_edges.json           ← co-occurrence + curated relational edges
│   │   ├── vocab_candidates.csv      ← frequency-ranked dental terms
│   │   ├── doctor_utterances.jsonl   ← real doctor input distribution (M1 test set)
│   │   ├── ner_seed.jsonl            ← weak-labeled spans to bootstrap M1 NER
│   │   ├── m2_cache.json             ← per-term decompositions (M2 source of truth)
│   │   └── *_report.md               ← human-readable build reports
│   └── gold/
│       └── m3_concept_gold.csv       ← 55-concept gold set for M3 accuracy
├── docs/
│   └── open_questions.md             ← living list of caveats / expert-review items
└── .env                              ← (blank) put ANTHROPIC_API_KEY here for M2 live mode
```

**Dependencies:** Python 3.10+. M1, M3, M4, and the data scripts are **pure
standard library**. M2 only needs the `anthropic` package for *live* decomposition
(`pip install anthropic`); without it, M2 runs on the cache + seeds.

---

## Quick start

```bash
# 1. (Optional) regenerate the data foundation from the corpus
python scripts/phase0_mining.py        # → data/derived/{vocab_candidates,ner_seed,...}
python scripts/build_vocabulary.py     # → data/derived/{isl_vocabulary,iskg_edges}.json

# 2. Run the full pipeline (raw clinical text → ISL gloss)
python scripts/pipeline.py

# 3. Run any module's self-test / demo in isolation
python scripts/m1_clinical_nlp.py      # M1 normalization + NER
python scripts/m3_primitive_mapper.py  # M3 mapper + 55-concept gold accuracy
python scripts/m4_grammar_reorder.py   # M4 grammar rules (asserts)
python scripts/m2_decompose.py         # M2 decomposition (offline)
```

Use it programmatically:

```python
from scripts.pipeline import ISLTranslator   # or run from the scripts/ dir
t = ISLTranslator()
result = t.translate("pt c/o severe pain UL6 since 3d, worsening on cold")
print(result["segments"][0]["gloss"])
# MOLAR LOC_UPPER LOC_LEFT PAIN[intensity] SEVERE[emphasis] COLD DAY
```

---

## The data foundation (vocabulary spine)

Everything downstream reads from **one shared artifact**:
`data/derived/isl_vocabulary.json`. It was built data-driven, not hand-picked:

1. `phase0_mining.py` scans all 5,430 NoteChat dental conversations and produces
   a **frequency-ranked** list of dental terms by NER category, plus a corpus of
   real doctor utterances and a weak-labeled NER seed.
2. `build_vocabulary.py` canonicalizes surface variants into **81 primitive
   signs** (`SIGN_001`–`SIGN_081`), assigns IDs by category + descending
   frequency, and derives **co-occurrence edges** directly from the corpus.

Each sign record is the contract every module relies on:

```json
{
  "id": "SIGN_017",
  "term": "pain",
  "category": "SYMPTOM",
  "tier": 1,
  "frequency": 9891,
  "surface_forms": ["pain", "ache", "aching", "throbbing", "soreness", "sore"],
  "grammar_role": "COMMENT",      // ← M4 ordering attribute
  "nmm_eligible": true,           // ← M4 non-manual-marker attribute
  "spatial_loc": null,
  "isl_video_url": null,          // ← M5 fills these once signs are sourced
  "mediapipe_pose_path": null
}
```

- **`surface_forms`** = M3's synonym set (this is how `aching` → the pain sign).
- **`grammar_role` / `nmm_eligible`** = M4's ordering & facial-marker rules,
  encoded as data rather than hard-coded `if`s.
- **`tier`** routes terms: Tier 1 = direct sign, Tier 3 = no sign → M2 decompose.
- **`isl_video_url` / `mediapipe_pose_path`** are `null` placeholders — this is
  exactly the gap M5 fills (see [Status](#status-at-a-glance)).

Categories: ANATOMY, SYMPTOM, PROCEDURE, INSTRUCTION, SEVERITY, DURATION,
TRIGGER, SPATIAL. Special non-sign nodes (SPATIAL_LOC, GRAMMAR_ROLE, NMM types)
complete the graph schema.

---

## The pipeline — data contract chain

The orchestrator `ISLTranslator.translate()` passes a **plain data structure**
across each boundary. The unit of currency changes shape at every hop:

```
 raw string
   │  M1: normalize (UL6→molar, 3d→3 days, pt→patient) → segment → NER
   ▼
 entities  [{concept, label, negated, force_decompose}]
   │  per entity, the orchestrator routes:
   │    force_decompose ─────────────► M2
   │    else  M3.map_concept ─ direct/candidates ─► sign token
   │                          └ decompose ────────► M2 ─► sign tokens
   ▼
 sign tokens  [{sign_id, negated}]
   │  M4: enrich from vocab → clause type → attach modifiers
   │      → 7 ordering rules → non-manual markers → gloss
   ▼
 { ordered[], nmm[], gloss }      ← M5 will consume ordered (pose lookups) + nmm (face)
```

One-line summary of the contracts:

```
string ─M1─► entities{concept,negated,force_decompose}
       ─M3─► {decision, sign_id}      (─M2─► mappings[].sign_id on a miss)
       ─►    tokens[{sign_id,negated}]
       ─M4─► {ordered[], nmm[], gloss}
```

---

## Module reference

### M1 — Clinical NLP preprocessing  (`ClinicalNLP`)

**Purpose:** make messy clinical input legible and tag clinical entities.

**Input:** raw string.
**Output:** `{raw, clean_text, segments[]}` where each segment carries
`entities`, `concepts`, and `context`.

**What it does (stages):**
1. `normalize()` — FDI tooth notation (`UL6`→`upper left first molar`), duration
   shorthand (`3d`→`3 days`), and ~25 dental abbreviations (`pt`, `c/o`, `RCT`,
   `OHI`…).
2. `segment()` — splits multi-complaint input on `. ; \n ?`.
3. `ner()` — longest-match tagging from the vocabulary's `surface_forms`. For
   each entity it also computes:
   - `concept` — the canonical string M3 expects (`upper`→`maxillary`).
   - `negated` — **clause-scoped** (a `no/not` cue before it with no comma/`but`
     between). *Fixes the old utterance-wide negation bug.*
   - `force_decompose` — `True` for known no-sign phrases (`TIER3_PHRASES`),
     routing them straight to M2.
   - bare temporal markers (`since`/`ago`) next to a real unit are dropped.

**Example output (one segment):**
```python
{"text": "...",
 "entities": [
   {"text":"severe","label":"SEVERITY","concept":"severe","negated":False,"force_decompose":False},
   {"text":"pain","label":"SYMPTOM","concept":"pain","negated":False,"force_decompose":False},
   {"text":"first molar","label":"ANATOMY","concept":"first molar",...},
   ... ],
 "concepts": ["severe","pain","maxillary","left side","first molar","days","cold"],
 "context": {"question": False}}
```

**How it feeds the next module:** the orchestrator iterates `entities`, sending
each `concept` to M3 (or M2 if `force_decompose`), and carrying `negated` forward
so it lands on the right sign in M4. `context.question` becomes an M4 NMM.

---

### M3 — ISL primitive mapper  (`ISLPrimitiveMapper`)  *(core contribution)*

**Purpose:** resolve a concept string to a specific `SIGN_ID`, or declare it
un-signable.

**Input:** one concept string.
**Output:** a decision record.

**How it scores:** `similarity()` blends exact match, stemmed-token overlap, and
token-subset back-off, and **discounts pure character overlap** so out-of-vocab
words don't false-match. Three confidence tiers:

| `decision` | confidence | meaning | what happens |
|---|---|---|---|
| `direct` | ≥ 0.82 | one confident sign | emit that `SIGN_ID` |
| `candidates` | 0.55–0.82 | unsure between a few | return top-k signs; **show all**, patient recognizes one (no expert in the loop) |
| `decompose` | < 0.55 | no usable sign | hand the concept to **M2** |

```python
m3.map_concept("aching")
# {"concept":"aching","decision":"direct","confidence":1.0,
#  "sign_id":"SIGN_017","term":"pain","tier":1,
#  "candidates":[{"sign_id":"SIGN_017","term":"pain","score":1.0}]}
```

**Design note — the literacy-free fallback:** there is no `expert review` tier and
fingerspelling is *not* the default. The target patient may be illiterate, so an
unknown term is **decomposed into known signs** (M2) rather than spelled.
Fingerspelling survives only as a flagged last resort.

**Validation:** 55/55 on `data/gold/m3_concept_gold.csv`.

**How it feeds the next module:** `direct`/`candidates` → a sign token
`{sign_id, negated}` for M4. `decompose` → call M2.

---

### M2 — LLM decomposition  (`LLMDecomposer`)

**Purpose:** the literacy-free fallback — rewrite a no-sign term into concrete,
mimeable concepts that *do* have signs. Closes the **M3 ↔ M2 loop**.

**Input:** a concept string. **Output:** decomposition record.

**How it works:**
1. 2-stage chained prompt — *simplify* the term, then *extract* 3–6 visual
   concepts. (Prompts are model-agnostic plain text.)
2. Each extracted concept is run **back through M3** (this is the loop). Concepts
   that still don't map either recurse (depth-limited) or flag a last-resort
   fingerspell.
3. Results are cached per term in `data/derived/m2_cache.json` — the **production
   source of truth** that tames LLM non-determinism. An ISL expert reviews the
   cache once; production never re-calls the LLM for known terms.

**Pluggable LLM, chosen late:** M2 depends only on a tiny `LLMClient.complete()`
interface.
- `OfflineClient` (default) → runs on cache + seeds, no key, no cost.
- `AnthropicClient` → live Claude, reads `ANTHROPIC_API_KEY` and `M2_MODEL` from
  `.env` (default `claude-opus-4-8`). Wired but inactive until you add a key.

```python
m2.decompose("malocclusion")
# {"term":"malocclusion","description":"teeth that do not meet correctly when biting",
#  "concepts":["tooth","bite"],
#  "mappings":[{"concept":"tooth","sign_id":"SIGN_001","term":"tooth","decision":"direct"},
#              {"concept":"bite","sign_id":"SIGN_045","term":"bite","decision":"direct"}],
#  "fully_mappable":True,"source":"seed"}
```

**How it feeds the next module:** the orchestrator emits one sign token per
`mappings[].sign_id` (propagating the originating entity's `negated` flag) → M4.

---

### M4 — ISL grammar reordering  (`ISLGrammarReorderer`)  *(core contribution)*

**Purpose:** turn a bag of signs into ISL word order with non-manual markers.
ISL is **not** signed English — the body part leads (topic), modifiers follow,
time goes last, and meaning is carried by facial expressions.

**Input:** `tokens = [{sign_id, negated}, ...]` + `context`.
**Output:** `{clause_type, ordered[], nmm[], gloss}`.

**The 7 rules** (driven by the vocab's `grammar_role` / `nmm_eligible` attributes):

| Rule | Effect |
|---|---|
| R1 | ANATOMY (topic) leads a complaint clause |
| R2 | SEVERITY follows its sign (`PAIN SEVERE`, not `SEVERE PAIN`) |
| R3 | SPATIAL follows its anatomy (`MOLAR upper left`) |
| R4 | TRIGGER follows the symptom it triggers |
| R5 | DURATION / time goes last |
| R6 | INSTRUCTION clauses use VERB → OBJECT → TIME (verb leads) |
| R7 | Non-manual markers: `furrowed_brow` (intensity), `forward_lean` (emphasis), `headshake` (negation, scoped to the negated sign), `raised_brows` (yes/no question) |

```python
{"clause_type":"complaint",
 "ordered":[{"sign_id":"SIGN_004","term":"molar","category":"ANATOMY"}, ...],
 "nmm":[{"id":"NMM_brow_furrow","type":"intensity","scope":"sign","over":"pain"},
        {"id":"NMM_forward_lean","type":"emphasis","scope":"sign","over":"severe"}],
 "gloss":"MOLAR LOC_UPPER LOC_LEFT PAIN[intensity] SEVERE[emphasis] COLD DAY"}
```

**How it feeds the next module:** `ordered` is the sequence of sign-pose lookups
M5 will animate; `nmm` is the facial-overlay track with per-sign / per-utterance
scope. This is the seam between the symbolic brain and the (unbuilt) renderer.

---

### M5 — Sign renderer  (not built)

**Purpose:** convert `{ordered, nmm}` into a visible ISL avatar (Three.js
skeleton) or stitched video.

**Why it's blocked:** every sign needs real articulation data —
`mediapipe_pose_path` (33 body + 21+21 hand landmarks per frame) extracted from a
real ISL sign video. **0 of 81 signs are sourced.** Binding the 81 `SIGN_ID`s to
real videos + poses is a one-time data-acquisition task for an ISL expert /
corpus, and is the prerequisite for M5.

---

## End-to-end worked example

```
raw   = "pt c/o severe pain UL6 since 3d, worsening on cold"

M1    clean_text = "patient complains of severe pain upper left first molar
                    since 3 days, worsening on cold"
      entities   = severe(SEVERITY) pain(SYMPTOM) upper(SPATIAL→maxillary)
                   left(SPATIAL→left side) first molar(ANATOMY)
                   days(DURATION)   ← "since" collapsed away
                   cold(TRIGGER)

M3    every concept resolves `direct` → 7 sign tokens
      severe→SIGN_059  pain→SIGN_017  maxillary→SIGN_078(loc_upper)
      left side→SIGN_077(loc_left)  first molar→SIGN_004  days→SIGN_066  cold→SIGN_070

M4    clause = complaint
      topic block : MOLAR + spatial → MOLAR LOC_UPPER LOC_LEFT
      comment     : PAIN + severity + trigger → PAIN SEVERE COLD
      time last   : DAY
      NMM         : furrowed_brow over PAIN, forward_lean over SEVERE

OUT   "MOLAR LOC_UPPER LOC_LEFT PAIN[intensity] SEVERE[emphasis] COLD DAY"
```

A negation example shows the per-sign headshake (M1→M4):

```
"no swelling, mild pain in upper right molar"
→ MOLAR LOC_UPPER LOC_RIGHT SWELLING[intensity+negation] PAIN[intensity] MILD
                                       ↑ headshake on SWELLING only — pain is NOT negated
```

A decomposition example shows the M3↔M2 loop:

```
"advised OHI; c/o malocclusion"
  oral hygiene instructions → [mouth, tooth, cleaning, brush]   (M2, fully mappable)
  malocclusion              → [tooth, bite]                     (M2, fully mappable)
→ BRUSH MOUTH TOOTH CLEANING   /   BITE TOOTH
```

---

## Glossary

- **Sign / `SIGN_ID`** — a primitive ISL sign in the vocabulary (`SIGN_001`–`SIGN_081`).
  Currently a *symbolic* identifier; the actual articulation (video/pose) is not
  yet attached.
- **Tier** — 1 = direct sign exists; 2 = compound/spatial; 3 = no sign, must be
  decomposed by M2.
- **NMM (non-manual marker)** — facial expression / posture that carries meaning
  (furrowed brow = intensity, headshake = negation, raised brows = yes/no question,
  forward lean = emphasis).
- **ISKG** — ISL Sign Knowledge Graph: the typed node/edge view of the vocabulary
  (`isl_vocabulary.json` + `iskg_edges.json`) that lets the rule-based system
  later distill into a GNN.
- **Topic–comment order** — ISL grammar where the body part (topic) is stated
  first, then the description/action (comment).

---

## Open questions & caveats

A living list of items deferred for ISL-expert / Deaf-community review lives in
[`docs/open_questions.md`](docs/open_questions.md). Highlights:

- **Grammar rules need Deaf-community validation** — "technically correct" ≠
  "natural." (M4 #1)
- **Numeral signs missing** — "3 days" loses the "3". (M4 #4)
- **Vocabulary needs ~10–15 function/role signs** (`not`, `fit`, `doctor`,
  `straight`…) so every decomposition can fully map. (M2 #11)
- **Semantic mapper (BioSentVec)** can replace M3's lexical floor for novel
  synonyms. (M3 #6)
- **0/81 signs sourced** — the M5 prerequisite.

Resolved so far: M1 negation scoping (#8), instruction-phrase routing (#9), and
temporal redundancy (#10, partial).
```
