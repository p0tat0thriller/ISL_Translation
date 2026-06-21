"""
M3 — ISL Primitive Mapper (core, publishable).

Maps a plain-language concept to a specific SIGN_ID from the vocabulary spine,
with a three-tier output tuned for an illiterate Deaf audience (no expert in the
loop, and patients may not know the alphabet):

    direct       -> high confidence: one SIGN_ID
    candidates   -> medium confidence: show the top-k signs and let the patient
                    recognize whichever lands (redundancy replaces expert review)
    decompose    -> no usable sign: route to M2 for a literacy-free VISUAL
                    breakdown into known primitives (e.g. malocclusion ->
                    [tooth, not, fit]). Fingerspelling is kept ONLY as a flagged
                    last resort, since it presupposes literacy.

This v1 is a LEXICAL/FUZZY matcher (pure stdlib): exact + synonym (via the
vocabulary's surface_forms) + morphological + fuzzy string similarity, with a
token-subset back-off (so "upper left molar" -> molar). It is deterministic and
runs with no model downloads. A semantic scorer (BioSentVec / sentence
embeddings) can be slotted in later behind the same `score()` interface to fill
the ISKG SEMANTIC_SIM gap — the tiering and fingerspell logic stay unchanged.

Usage:
    from m3_primitive_mapper import ISLPrimitiveMapper
    m = ISLPrimitiveMapper()
    m.map_concept("aching")            -> {'sign_id': 'SIGN_017', 'term': 'pain', ...}
    m.map_concepts(["tooth", "seal"])  -> [ {...}, {...fingerspell...} ]

CLI (also runs a self-evaluation against data/gold/m3_concept_gold.csv):
    python scripts/m3_primitive_mapper.py
"""
from __future__ import annotations
import csv, json, re, sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "derived" / "isl_vocabulary.json"
GOLD = ROOT / "data" / "gold" / "m3_concept_gold.csv"

# Confidence thresholds (calibratable; see KEY RISK in the spec).
DIRECT_THRESHOLD = 0.82   # >= this  -> one confident SIGN_ID        ("direct")
SHOW_THRESHOLD   = 0.55   # >= this  -> show top-k candidate signs   ("candidates")
                          # below     -> route to M2 decomposition   ("decompose")
TOP_K = 3                 # how many candidate signs to surface when unsure

_NONALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    return _NONALNUM.sub(" ", s.lower()).strip()


def stem(token: str) -> str:
    """Light morphological stemmer (plurals + common verb suffixes)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"          # cavities -> cavity
    for suf in ("ing", "ed", "es", "s"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def similarity(a: str, b: str) -> float:
    """
    Blended lexical similarity in [0,1] between two normalized strings.

    A fuzzy (character) match only counts as evidence when the two strings share
    a STEMMED TOKEN or one token-set is a subset of the other. Pure coincidental
    character overlap (no shared token) is heavily discounted, so out-of-vocab
    words like "seal"/"orthodontist" fall through to fingerspelling instead of
    latching onto a look-alike sign. (Real synonym/OOV precision still wants the
    semantic scorer — this is the lexical floor.)
    """
    if a == b:
        return 1.0
    sa = {stem(t) for t in a.split()}
    sb = {stem(t) for t in b.split()}
    char = max(
        SequenceMatcher(None, a, b).ratio(),
        SequenceMatcher(None, " ".join(sorted(sa)), " ".join(sorted(sb))).ratio(),
    )
    if sa and sb and (sa <= sb or sb <= sa):     # token-subset back-off
        return 0.9
    shared = sa & sb
    if shared:                                    # legitimate token overlap
        return max(len(shared) / len(sa | sb), char)
    return char if char >= 0.9 else round(char * 0.6, 3)  # no token evidence


class ISLPrimitiveMapper:
    def __init__(self, vocab_path: Path = VOCAB):
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.signs = {s["id"]: s for s in data["signs"]}
        # exact-match index: normalized surface form / term -> sign_id
        self.exact: dict[str, str] = {}
        # candidate forms for fuzzy scoring: (normalized_form, sign_id)
        self.forms: list[tuple[str, str]] = []
        for s in data["signs"]:
            keys = set(s["surface_forms"]) | {s["term"].replace("_", " ")}
            for k in keys:
                nk = normalize(k)
                if not nk:
                    continue
                self.exact.setdefault(nk, s["id"])
                self.forms.append((nk, s["id"]))

    # -- scoring hook: swap this for a semantic scorer later --------------------
    def _best_candidates(self, nc: str, k: int = 3):
        scored: dict[str, float] = {}
        for form, sid in self.forms:
            sc = similarity(nc, form)
            if sc > scored.get(sid, 0.0):
                scored[sid] = sc
        ranked = sorted(scored.items(), key=lambda x: -x[1])[:k]
        return [{"sign_id": sid, "term": self.signs[sid]["term"],
                 "score": round(sc, 3)} for sid, sc in ranked]

    @staticmethod
    def fingerspell(concept: str) -> list[str]:
        return [c.upper() for c in concept if c.isalpha()]

    def map_concept(self, concept: str) -> dict:
        nc = normalize(concept)
        # 1) exact / synonym hit -> one confident sign
        if nc in self.exact:
            sid = self.exact[nc]
            s = self.signs[sid]
            return {"concept": concept, "decision": "direct", "confidence": 1.0,
                    "sign_id": sid, "term": s["term"], "tier": s["tier"],
                    "candidates": [{"sign_id": sid, "term": s["term"], "score": 1.0}]}

        cands = self._best_candidates(nc, k=TOP_K)
        best = cands[0] if cands else None

        # 2) confident fuzzy match -> one sign
        if best and best["score"] >= DIRECT_THRESHOLD:
            s = self.signs[best["sign_id"]]
            return {"concept": concept, "decision": "direct", "confidence": best["score"],
                    "sign_id": best["sign_id"], "term": s["term"], "tier": s["tier"],
                    "candidates": cands}

        # 3) unsure between a few -> SHOW all top-k signs; the patient recognizes
        #    whichever lands. No expert in the loop, so redundancy disambiguates.
        if best and best["score"] >= SHOW_THRESHOLD:
            return {"concept": concept, "decision": "candidates", "confidence": best["score"],
                    "sign_id": best["sign_id"], "term": best["term"],
                    "tier": self.signs[best["sign_id"]]["tier"], "candidates": cands}

        # 4) no usable sign -> route to M2 for a literacy-free VISUAL decomposition
        #    into known primitives. Fingerspelling is only a flagged last resort
        #    (an illiterate patient may not know letters).
        return {"concept": concept, "decision": "decompose",
                "confidence": best["score"] if best else 0.0,
                "sign_id": None, "term": None, "tier": None,
                "candidates": cands,
                "last_resort_fingerspell": self.fingerspell(concept)}

    def map_concepts(self, concepts: list[str]) -> list[dict]:
        return [self.map_concept(c) for c in concepts]


# ---------------------------------------------------------------------------
def _fmt(r):
    if r["decision"] == "direct":
        return f"direct      {r['sign_id']} ({r['term']})  conf={r['confidence']:.2f}"
    if r["decision"] == "candidates":
        alts = " / ".join(c["term"] for c in r["candidates"])
        return f"candidates  show: [{alts}]  conf={r['confidence']:.2f}"
    fs = "".join(r["last_resort_fingerspell"])
    return f"decompose   -> M2 visual breakdown  (last-resort FS: {fs})  conf={r['confidence']:.2f}"


def _demo_and_eval():
    m = ISLPrimitiveMapper()
    print(f"Loaded {len(m.signs)} signs, {len(m.exact)} exact keys.\n")

    print("== M3 example: PDF Tier-3 decomposition ['tooth','inside','infected','clean','seal'] ==")
    for r in m.map_concepts(["tooth", "inside", "infected", "clean", "seal"]):
        print(f"  {r['concept']:<10} -> {_fmt(r)}")

    print("\n== M3 example: realistic M1 concepts ==")
    for r in m.map_concepts(["severe", "pain", "upper left molar", "cold", "3 days"]):
        print(f"  {r['concept']:<16} -> {_fmt(r)}")

    print("\n== M3 example: out-of-vocab clinical terms (literacy-free fallback) ==")
    for r in m.map_concepts(["orthodontist", "malocclusion"]):
        print(f"  {r['concept']:<16} -> {_fmt(r)}")

    if not GOLD.exists():
        print("\n(no gold file; skipping eval)")
        return
    rows = list(csv.DictReader(open(GOLD, encoding="utf-8")))
    correct = 0
    by_decision = {"direct": 0, "candidates": 0, "decompose": 0}
    misses = []
    for row in rows:
        exp = row["expected_term"].strip()
        r = m.map_concept(row["concept"])
        by_decision[r["decision"]] += 1
        if exp == "DECOMPOSE":
            ok = r["decision"] == "decompose"
        else:
            ok = r["term"] == exp and r["decision"] in ("direct", "candidates")
        correct += ok
        if not ok:
            misses.append((row["concept"], exp, r["term"] or r["decision"], round(r["confidence"], 2)))
    n = len(rows)
    print(f"\n== Self-evaluation on {n} gold concepts ==")
    print(f"  accuracy: {correct}/{n} = {correct/n:.1%}")
    print(f"  decisions: {by_decision}")
    if misses:
        print("  misses (concept | expected | got | conf):")
        for c, e, g, cf in misses:
            print(f"    {c:<22} {e:<14} {g:<14} {cf}")


if __name__ == "__main__":
    _demo_and_eval()
