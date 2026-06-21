"""
Phase 0 — Data foundation for ISL reverse translation (clinical text -> sign).

Mines the NoteChat dental dataset to produce the assets that BOTH the
rule-based pipeline (M1-M5) and the graph approach (HCSG) need before any
core module can be built:

  1. doctor_utterances.jsonl  - the real production INPUT distribution for M1
                                (every "Doctor:" line, normalized + deduped)
  2. vocab_candidates.csv     - frequency-ranked dental concepts grouped by the
                                M1 NER categories. This is the data-driven seed
                                for M3's "150 primitives" + Tier classification.
  3. ner_seed.jsonl           - weakly-labeled spans (text, [start,end,LABEL])
                                in spaCy offset format, to bootstrap M1 NER.
  4. phase0_report.md         - human-readable summary of what was found.

Pure standard library. Run:  python scripts/phase0_mining.py
"""
from __future__ import annotations
import csv, re, json, sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notechat_dental_conversations_cleaned.csv"
OUT = ROOT / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dental seed lexicon, grouped by the M1 NER categories.
# Multi-word terms first so longest-match wins during tagging.
# This is a SEED, not the final vocabulary; frequencies from the corpus rank it
# and surface gaps (see discovery n-grams in the report).
# ---------------------------------------------------------------------------
LEXICON: dict[str, list[str]] = {
    "ANATOMY": [
        "wisdom tooth", "central incisor", "lateral incisor", "anterior mandible",
        "anterior maxilla", "alveolar bone", "periodontal ligament", "gum line",
        "root canal", "pulp chamber", "enamel surface", "soft tissue",
        "upper left molar", "lower right molar", "baby tooth", "primary tooth",
        "permanent tooth", "back tooth", "front tooth",
        "molar", "premolar", "incisor", "canine", "tooth", "teeth", "gum", "gums",
        "gingiva", "jaw", "mandible", "maxilla", "enamel", "dentin", "pulp",
        "root", "crown", "cavity", "palate", "tongue", "lip", "cheek", "nerve",
        "bone", "cusp", "socket", "sinus", "cementum", "occlusion", "bite",
        "mucosa", "frenum", "saliva", "jawbone", "mouth",
    ],
    "SYMPTOM": [
        "tooth decay", "gum disease", "bad breath", "dry mouth", "tooth loss",
        "cold sensitivity", "hot sensitivity", "bleeding gums", "loose tooth",
        "pain", "ache", "aching", "swelling", "swollen", "bleeding", "sensitivity",
        "sensitive", "sore", "soreness", "inflammation", "inflamed", "abscess",
        "decay", "caries", "infection", "infected", "discoloration", "fracture",
        "hypersensitivity", "throbbing", "ulcer", "lesion", "cyst", "fever",
        "pus", "redness", "numbness", "tenderness", "stiffness", "halitosis",
        "discomfort", "cracked", "chipped", "gingivitis", "periodontitis",
    ],
    "PROCEDURE": [
        "root canal treatment", "root canal", "tooth extraction", "dental implant",
        "dental crown", "dental bridge", "dental filling", "deep cleaning",
        "scaling and polishing", "fluoride treatment", "wisdom tooth removal",
        "local anesthesia", "general anesthesia", "x-ray", "bone graft",
        "gum surgery", "tooth whitening", "fissure sealant",
        "extraction", "filling", "crown", "cleaning", "scaling", "polishing",
        "radiograph", "implant", "bridge", "denture", "anesthesia", "anaesthesia",
        "restoration", "biopsy", "surgery", "treatment", "examination", "fluoride",
        "sealant", "whitening", "braces", "prophylaxis", "rct", "veneer", "pulpotomy",
        "apicoectomy", "curettage", "suture", "splint", "rinse", "checkup",
    ],
    "INSTRUCTION": [
        "open wide", "open your mouth", "close your mouth", "bite down",
        "rinse your mouth", "brush your teeth", "floss daily", "avoid chewing",
        "rinse", "brush", "floss", "avoid", "apply", "swallow", "bite", "chew",
        "spit", "gargle", "relax", "breathe", "hold", "press", "wait",
    ],
    "SEVERITY": [
        "very severe", "extremely painful",
        "severe", "mild", "moderate", "slight", "sharp", "dull", "intense",
        "acute", "chronic", "worsening", "persistent", "constant", "extreme",
        "minor", "significant", "unbearable",
    ],
    "DURATION": [
        "days", "weeks", "months", "years", "hours", "minutes",
        "week", "month", "year", "hour", "minute", "day",
        "since", "ago", "overnight", "recently", "chronic",
    ],
    "TRIGGER": [
        "cold water", "hot water", "cold air", "sweet food", "hot food",
        "cold", "hot", "sweet", "sour", "pressure", "chewing", "biting",
        "air", "touch", "temperature",
    ],
}

# Build longest-first regex per category for offset tagging.
def build_patterns(lex: dict[str, list[str]]):
    pats = {}
    for cat, terms in lex.items():
        terms_sorted = sorted(set(terms), key=len, reverse=True)
        alt = "|".join(re.escape(t) for t in terms_sorted)
        pats[cat] = re.compile(r"\b(" + alt + r")\b", re.IGNORECASE)
    return pats

PATTERNS = build_patterns(LEXICON)

# Tier heuristic: multi-word PROCEDUREs (and a known set) have no atomic ISL sign
# -> Tier 3 (must be decomposed by M2). Single concrete nouns -> Tier 1.
TIER3_HINTS = {"root canal treatment", "root canal", "dental implant", "dental bridge",
               "apicoectomy", "pulpotomy", "bone graft", "fissure sealant", "curettage",
               "scaling and polishing", "fluoride treatment", "prophylaxis"}

def classify_tier(term: str, cat: str) -> int:
    t = term.lower()
    if t in TIER3_HINTS:
        return 3
    if cat == "PROCEDURE" and (" " in t or len(t) > 11):
        return 3
    if " " in t:  # multi-word non-procedure -> mid tier (may need spatial/compound signing)
        return 2
    return 1

STOP_NGRAM = set("the a an and or of to in on for with your you i it is was were are be "
                 "this that have has had will can do does did not no yes good morning "
                 "doctor patient okay well so but as at by from".split())

def iter_rows():
    with open(SRC, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row.get("data") or "", row.get("conversation") or ""

DOCTOR_RE = re.compile(r"Doctor:\s*(.*)")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def normalize(s: str) -> str:
    s = s.replace("﻿", "").replace("’", "'").replace("�", "")
    return re.sub(r"\s+", " ", s).strip()

def main():
    if not SRC.exists():
        sys.exit(f"Source CSV not found: {SRC}")

    doctor_counter: Counter[str] = Counter()
    term_counts: dict[str, Counter] = {cat: Counter() for cat in LEXICON}
    ngram_counts: Counter[str] = Counter()
    ner_examples: list[dict] = []
    n_rows = 0
    n_doctor = 0

    for data, conv in iter_rows():
        n_rows += 1
        full_text = normalize(data + " " + conv)

        # 1) Doctor utterance corpus (production input distribution)
        for m in DOCTOR_RE.finditer(conv):
            utt = normalize(m.group(1))
            if 3 <= len(utt) <= 200:
                doctor_counter[utt] += 1
                n_doctor += 1

        # 2) Term frequency by NER category (over the whole row text)
        for cat, pat in PATTERNS.items():
            for m in pat.finditer(full_text):
                term_counts[cat][m.group(1).lower()] += 1

        # 3) Discovery n-grams (bi/tri-grams) from clinical narrative only
        words = re.findall(r"[a-z]+", data.lower())
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if gram[0] in STOP_NGRAM or gram[-1] in STOP_NGRAM:
                    continue
                ngram_counts[" ".join(gram)] += 1

        # 4) NER seed labels: weak-label sentences from the clinical narrative.
        #    Keep a manageable, high-signal sample (every ~6th row, sentences
        #    that contain >=2 entity hits across categories).
        if n_rows % 6 == 0:
            for sent in SENT_SPLIT.split(normalize(data))[:8]:
                if not (20 <= len(sent) <= 240):
                    continue
                spans = []
                for cat, pat in PATTERNS.items():
                    for m in pat.finditer(sent):
                        spans.append([m.start(), m.end(), cat])
                # resolve overlaps: keep longest span at each position
                spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
                kept, last_end = [], -1
                for st, en, lab in spans:
                    if st >= last_end:
                        kept.append([st, en, lab]); last_end = en
                if len(kept) >= 2:
                    ner_examples.append({"text": sent, "entities": kept})

    # ---- write doctor utterance corpus ----
    with open(OUT / "doctor_utterances.jsonl", "w", encoding="utf-8") as f:
        for utt, c in doctor_counter.most_common():
            f.write(json.dumps({"text": utt, "count": c}, ensure_ascii=False) + "\n")

    # ---- write frequency-ranked vocab candidates ----
    vocab_rows = []
    for cat, counter in term_counts.items():
        for term, c in counter.most_common():
            vocab_rows.append({
                "term": term, "category": cat, "frequency": c,
                "tier": classify_tier(term, cat),
            })
    vocab_rows.sort(key=lambda r: (-r["frequency"], r["category"]))
    with open(OUT / "vocab_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "category", "frequency", "tier"])
        w.writeheader(); w.writerows(vocab_rows)

    # ---- write NER seed ----
    with open(OUT / "ner_seed.jsonl", "w", encoding="utf-8") as f:
        for ex in ner_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ---- discovery n-grams (filter to dental-looking, freq>=25) ----
    discovery = [(g, c) for g, c in ngram_counts.most_common(400) if c >= 25]

    # ---- report ----
    tier_dist = Counter(r["tier"] for r in vocab_rows if r["frequency"] > 0)
    lines = []
    lines.append("# Phase 0 — Data Foundation Report\n")
    lines.append(f"- Source rows processed: **{n_rows:,}**")
    lines.append(f"- Doctor utterances harvested: **{n_doctor:,}** "
                 f"({len(doctor_counter):,} unique)")
    lines.append(f"- Vocabulary candidates (seen >0): "
                 f"**{sum(1 for r in vocab_rows if r['frequency']>0)}** terms across "
                 f"{len(LEXICON)} NER categories")
    lines.append(f"- Tier distribution: "
                 f"Tier1={tier_dist.get(1,0)}, Tier2={tier_dist.get(2,0)}, "
                 f"Tier3={tier_dist.get(3,0)}")
    lines.append(f"- NER seed examples (>=2 entities/sentence): **{len(ner_examples):,}**\n")

    lines.append("## Top concepts by NER category (data-driven primitive seed)\n")
    for cat in LEXICON:
        top = term_counts[cat].most_common(12)
        if not top:
            continue
        pretty = ", ".join(f"{t} ({c:,})" for t, c in top)
        lines.append(f"- **{cat}**: {pretty}")
    lines.append("")

    lines.append("## Most frequent doctor utterances (M1 input register)\n")
    for utt, c in doctor_counter.most_common(20):
        lines.append(f"- `{utt}` — {c:,}")
    lines.append("")

    lines.append("## Discovery n-grams (candidate terms NOT necessarily in seed lexicon)\n")
    lines.append("Use these to extend the lexicon / catch missed dental terms:\n")
    for g, c in discovery[:40]:
        lines.append(f"- {g} — {c:,}")
    lines.append("")

    (OUT / "phase0_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. Wrote 4 artifacts to {OUT}")
    print(f"  rows={n_rows:,}  doctor_utts={n_doctor:,} ({len(doctor_counter):,} uniq)")
    print(f"  vocab_terms_seen={sum(1 for r in vocab_rows if r['frequency']>0)}")
    print(f"  ner_seed={len(ner_examples):,}")

if __name__ == "__main__":
    main()
