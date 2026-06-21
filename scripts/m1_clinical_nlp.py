"""
M1 — Clinical NLP Preprocessing (the pipeline's front door).

Turns raw doctor input — shorthand, dental abbreviations, tooth notation — into
clean text + tagged clinical entities, then hands the ordered entity concepts to
M3 (mapper) and M4 (grammar). This v1 is dictionary/rule-based (pure stdlib) so
it runs with no model download; a spaCy statistical NER fine-tuned on the
`data/derived/ner_seed.jsonl` we mined can swap in behind `ner()` later, exactly
like M3's semantic-scorer hook.

Pipeline stages (per the spec):
  1. Abbreviation expansion   pt c/o -> patient complains of ; RCT -> root canal treatment
  2. Tooth-notation parsing   UL6 -> upper left first molar  (quadrant + FDI position)
  3. Duration normalization   3d -> 3 days
  4. Sentence segmentation    split multi-complaint input into units
  5. NER                      ANATOMY / SYMPTOM / PROCEDURE / INSTRUCTION /
                              SEVERITY / DURATION / TRIGGER / SPATIAL
  6. Context flags            question (?) and negation -> M4 NMM (raised_brows / headshake)

NER reuses the vocabulary spine's surface_forms so M1's tags stay consistent
with what M3 can actually map.

Usage:  python scripts/m1_clinical_nlp.py     (runs full M1 -> M3 -> M4 demo)
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "derived" / "isl_vocabulary.json"

# 1) Dental abbreviation dictionary (curated; the dataset is clean English, so
#    this shorthand layer is sourced here, not mined). Whole-token, case-insens.
ABBREV = {
    "pt": "patient", "c/o": "complains of", "h/o": "history of",
    "o/e": "on examination", "r/o": "rule out", "wrt": "with respect to",
    "rct": "root canal treatment", "ohi": "oral hygiene instructions",
    "la": "local anaesthesia", "ga": "general anaesthesia", "ext": "extraction",
    "tx": "treatment", "fx": "fracture", "abx": "antibiotics",
    "opg": "panoramic radiograph", "iopa": "intraoral periapical radiograph",
    "bop": "bleeding on probing", "gic": "glass ionomer cement filling",
    "nad": "no abnormality detected", "tmj": "temporomandibular joint",
    "prophy": "prophylaxis", "perio": "periodontal", "endo": "root canal treatment",
}

# 2) FDI tooth position -> tooth type (1-8 within a quadrant)
TOOTH_POS = {1: "central incisor", 2: "lateral incisor", 3: "canine",
             4: "first premolar", 5: "second premolar", 6: "first molar",
             7: "second molar", 8: "third molar"}
VERT = {"U": "upper", "L": "lower"}
HORZ = {"L": "left", "R": "right"}
TOOTH_RE = re.compile(r"\b([UL])([LR])([1-8])\b")
DUR_RE = re.compile(r"\b(\d+)\s*([dwmy])\b")
DUR_UNIT = {"d": "day", "w": "week", "m": "month", "y": "year"}

# 6) context cues. Negation is scoped per clause: a cue negates entities up to
#    the next clause boundary (comma / "but" / "however"), not the whole sentence.
NEG_RE = re.compile(r"\b(no|not|without|absent|denies|negative|never)\b", re.I)
BOUNDARY_RE = re.compile(r",|;|\bbut\b|\bhowever\b|\bexcept\b", re.I)

# Multi-word clinical terms with NO atomic ISL sign. Tagged so they reach M3 ->
# decompose -> M2 instead of being silently dropped (open_questions #9). Keys are
# lowercase; values are the NER label. Aligned with M2's seed/cache keys.
TIER3_PHRASES = {
    "oral hygiene instructions": "INSTRUCTION",
    "malocclusion": "SYMPTOM",
}

# Bare temporal markers — dropped when a concrete duration unit is adjacent
# (open_questions #10), so "since 3 days" -> DAY, not SINCE DAY.
DUR_MARKERS = {"since", "ago"}

# spatial surface variants -> the canonical concept M3 knows (vocab surface form)
SPATIAL_ALIAS = {"upper": "maxillary", "lower": "mandibular",
                 "left": "left side", "right": "right side",
                 "anterior": "anterior", "posterior": "posterior"}

_WS = re.compile(r"\s+")


class ClinicalNLP:
    def __init__(self, vocab_path: Path = VOCAB):
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        # NER surface form -> label, from the vocabulary spine + spatial aliases
        self.surface: dict[str, str] = {}
        for s in data["signs"]:
            for f in s["surface_forms"]:
                self.surface.setdefault(f.lower(), s["category"])
        for alias in SPATIAL_ALIAS:
            self.surface.setdefault(alias, "SPATIAL")
        for phrase, label in TIER3_PHRASES.items():          # #9
            self.surface.setdefault(phrase, label)
        forms = sorted(self.surface, key=len, reverse=True)
        self.matcher = re.compile(r"\b(" + "|".join(re.escape(f) for f in forms) + r")\b", re.I)

    # -- stage 1-3: normalization ----------------------------------------------
    @staticmethod
    def _expand_tooth(m: re.Match) -> str:
        v, h, n = m.group(1), m.group(2), int(m.group(3))
        return f"{VERT[v]} {HORZ[h]} {TOOTH_POS[n]}"

    @staticmethod
    def _expand_dur(m: re.Match) -> str:
        n, unit = int(m.group(1)), DUR_UNIT[m.group(2)]
        return f"{n} {unit}{'s' if n != 1 else ''}"

    def normalize(self, raw: str) -> str:
        text = TOOTH_RE.sub(self._expand_tooth, raw)
        text = DUR_RE.sub(self._expand_dur, text)
        # whole-token abbreviation expansion (longest key first)
        for ab in sorted(ABBREV, key=len, reverse=True):
            text = re.sub(r"(?<!\w)" + re.escape(ab) + r"(?!\w)", ABBREV[ab], text, flags=re.I)
        return _WS.sub(" ", text).strip()

    # -- stage 4: segmentation --------------------------------------------------
    @staticmethod
    def segment(text: str) -> list[str]:
        parts = re.split(r"[.;\n]+|\?", text)
        return [p.strip() for p in parts if p.strip()]

    # -- stage 5: NER (longest-match, non-overlapping) --------------------------
    @staticmethod
    def _is_negated(pos: int, cues: list[int], bounds: list[int]) -> bool:
        """True if a negation cue precedes `pos` with no clause boundary between."""
        return any(c < pos and not any(c < b < pos for b in bounds) for c in cues)

    @staticmethod
    def _collapse_duration(ents: list[dict]) -> list[dict]:
        """#10: drop bare temporal markers (since/ago) next to a concrete unit."""
        units = [e for e in ents if e["label"] == "DURATION" and e["concept"] not in DUR_MARKERS]
        out = []
        for e in ents:
            if (e["label"] == "DURATION" and e["concept"] in DUR_MARKERS
                    and any(abs(u["start"] - e["start"]) <= 25 for u in units)):
                continue
            out.append(e)
        return out

    def ner(self, text: str) -> list[dict]:
        spans = [(m.start(), m.end(), m.group(0)) for m in self.matcher.finditer(text)]
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        cues = [m.start() for m in NEG_RE.finditer(text)]        # #8
        bounds = [m.start() for m in BOUNDARY_RE.finditer(text)]
        ents, last = [], -1
        for st, en, txt in spans:
            if st < last:
                continue
            low = txt.lower()
            ents.append({"text": txt, "label": self.surface[low],
                         "concept": SPATIAL_ALIAS.get(low, low),
                         "negated": self._is_negated(st, cues, bounds),
                         # known no-sign phrase -> route straight to M2, skip M3
                         "force_decompose": low in TIER3_PHRASES,
                         "start": st, "end": en})
            last = en
        return self._collapse_duration(ents)

    # -- full M1 ----------------------------------------------------------------
    def process(self, raw: str) -> dict:
        clean = self.normalize(raw)
        segments = []
        for seg in self.segment(clean):
            ents = self.ner(seg)
            segments.append({
                "text": seg,
                "entities": ents,
                "concepts": [e["concept"] for e in ents],
                # negation is now per-entity (entity["negated"]); only the
                # utterance-level question marker stays in context.
                "context": {"question": raw.strip().endswith("?") or "?" in raw},
            })
        return {"raw": raw, "clean_text": clean, "segments": segments}


# ---------------------------------------------------------------------------
def _demo():
    from m3_primitive_mapper import ISLPrimitiveMapper
    from m4_grammar_reorder import ISLGrammarReorderer
    m1, m3, m4 = ClinicalNLP(), ISLPrimitiveMapper(), ISLGrammarReorderer()

    inputs = [
        "pt c/o severe pain UL6 since 3d, worsening on cold",
        "RCT wrt LR6; advised OHI",
        "no swelling, mild pain in upper right molar",
        "Is the pain sharp?",
    ]
    for raw in inputs:
        doc = m1.process(raw)
        print(f"\nRAW   : {raw}")
        print(f"CLEAN : {doc['clean_text']}")
        for seg in doc["segments"]:
            ents = ", ".join(f"{e['text']}={e['label']}" for e in seg["entities"])
            print(f"  seg : {seg['text']}")
            print(f"  NER : {ents or '(none)'}")
            signs = [r for r in m3.map_concepts(seg["concepts"]) if r.get("sign_id")]
            decomp = [r["concept"] for r in m3.map_concepts(seg["concepts"]) if r["decision"] == "decompose"]
            isl = m4.reorder(signs, seg["context"])
            print(f"  ISL : {isl['gloss']}")
            if decomp:
                print(f"  ->M2: decompose {decomp}")


if __name__ == "__main__":
    _demo()
