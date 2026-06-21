"""
M4 — ISL Grammar Reordering (core, publishable).

Takes a sign sequence in English concept order (from M3) and reorders it into
ISL topic-comment word order, attaching non-manual markers (NMM). ISL is NOT
signed English: the body part (topic) leads, modifiers follow their head, time
goes last, and meaning is carried by facial/posture markers layered on top.

It reads the grammar attributes we baked onto every sign in the vocabulary
spine (`grammar_role`, `nmm_eligible`, `spatial_loc`, `category`), so the rules
are driven by graph data, not hand-wired per sign. In the hybrid plan each rule
corresponds to an ISKG TEMPORAL_ORDER / REQUIRES_NMM edge, so this engine also
produces the silver-standard ordering the GNN will later distil.

Seven rules (from the spec):
  R1  ANATOMY (topic) leads a complaint clause.
  R2  SEVERITY follows the sign it modifies ("PAIN SEVERE", not "SEVERE PAIN").
  R3  SPATIAL location follows its anatomy ("MOLAR upper left").
  R4  TRIGGER follows the symptom it triggers.
  R5  DURATION / time goes last.
  R6  INSTRUCTION clauses use VERB -> OBJECT -> TIME (verb leads; overrides R1).
  R7  NMM insertion: furrowed_brow (intensity) on symptoms, forward_lean
      (emphasis) on severity, raised_brows (yes/no question) and headshake
      (negation) from sentence context.

Input  : list of sign tokens (sign_id strings, or dicts/M3 results carrying
         sign_id) + optional context {"question": bool, "negation": bool}.
Output : {clause_type, ordered:[...], nmm:[...], gloss:str}

Usage:  python scripts/m4_grammar_reorder.py   (runs end-to-end M3 -> M4 demos)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "data" / "derived" / "isl_vocabulary.json"

# role class derived from the vocabulary category
HEAD_CATS = {"ANATOMY", "SYMPTOM", "PROCEDURE", "INSTRUCTION"}
MOD_CATS = {"SPATIAL", "SEVERITY", "TRIGGER"}


class ISLGrammarReorderer:
    def __init__(self, vocab_path: Path = VOCAB):
        data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.signs = {s["id"]: s for s in data["signs"]}

    # -- token enrichment -------------------------------------------------------
    def _enrich(self, tok) -> dict | None:
        sid = tok if isinstance(tok, str) else (tok.get("sign_id") if isinstance(tok, dict) else None)
        if not sid or sid not in self.signs:
            return None                      # skip decompose / fingerspell tokens
        s = self.signs[sid]
        negated = bool(tok.get("negated")) if isinstance(tok, dict) else False
        return {"sign_id": sid, "term": s["term"], "category": s["category"],
                "role": s["grammar_role"], "nmm_eligible": s["nmm_eligible"],
                "spatial_loc": s["spatial_loc"], "negated": negated}

    # -- attachment: each modifier binds to its nearest valid head --------------
    @staticmethod
    def _nearest(idx, toks, head_cats) -> int | None:
        best, best_d = None, 10**9
        for j, t in enumerate(toks):
            if t["category"] in head_cats:
                d = abs(j - idx)
                # tie-break: prefer the FOLLOWING head (English puts modifier first)
                if d < best_d or (d == best_d and j > idx):
                    best, best_d = j, d
        return best

    def _attach(self, toks, clause):
        """Return attach[head_idx] = list of modifier indices (in original order)."""
        attach = {i: [] for i, t in enumerate(toks) if t["category"] in HEAD_CATS}
        for i, t in enumerate(toks):
            cat = t["category"]
            if cat == "SPATIAL":
                h = self._nearest(i, toks, {"ANATOMY"})
            elif cat == "SEVERITY":
                h = self._nearest(i, toks, {"SYMPTOM", "PROCEDURE"})
            elif cat == "TRIGGER":
                h = self._nearest(i, toks, {"SYMPTOM"}) or self._nearest(i, toks, {"PROCEDURE"})
            elif cat == "ANATOMY" and clause == "instruction":
                # R6: anatomy is the OBJECT of the instruction verb
                h = self._nearest(i, toks, {"INSTRUCTION"})
            else:
                continue
            if h is not None and h in attach:
                attach[h].append(i)
        return attach

    @staticmethod
    def _clause_type(toks) -> str:
        cats = {t["category"] for t in toks}
        if "INSTRUCTION" in cats and "SYMPTOM" not in cats:
            return "instruction"
        return "complaint"

    # -- emit ISL order ---------------------------------------------------------
    def _flatten_head(self, h, toks, attach, within):
        """head followed by its modifiers, ordered by the `within` priority."""
        seq = [h]
        mods = sorted(attach.get(h, []), key=lambda j: (within.get(toks[j]["category"], 9), j))
        return seq + mods

    def _emit(self, toks, attach, clause):
        used = set()
        order = []

        def emit_head(h, within):
            for j in self._flatten_head(h, toks, attach, within):
                if j not in used:
                    order.append(j); used.add(j)

        if clause == "instruction":
            # R6: VERB -> OBJECT(anatomy) -> ... ; R5 time last
            for i, t in enumerate(toks):
                if t["category"] == "INSTRUCTION":
                    emit_head(i, {"ANATOMY": 0})
        else:
            # R1: topic block (anatomy + R3 spatial), original order
            for i, t in enumerate(toks):
                if t["category"] == "ANATOMY":
                    emit_head(i, {"SPATIAL": 0})
            # comment block: symptoms/procedures + R2 severity + R4 trigger
            for i, t in enumerate(toks):
                if t["category"] in ("SYMPTOM", "PROCEDURE"):
                    emit_head(i, {"SEVERITY": 0, "TRIGGER": 1})
        # standalone modifiers/instructions not yet placed (defensive)
        for i, t in enumerate(toks):
            if i not in used and t["category"] != "DURATION":
                order.append(i); used.add(i)
        # R5: duration / time goes last
        for i, t in enumerate(toks):
            if t["category"] == "DURATION" and i not in used:
                order.append(i); used.add(i)
        return order

    # -- R7: non-manual markers -------------------------------------------------
    def _nmm(self, ordered_toks, context):
        nmm = []
        for t in ordered_toks:
            if t["category"] == "SYMPTOM":
                nmm.append({"id": "NMM_brow_furrow", "type": "intensity",
                            "scope": "sign", "over": t["term"]})
            if t.get("negated"):     # #8: headshake scoped to the negated sign
                nmm.append({"id": "NMM_headshake", "type": "negation",
                            "scope": "sign", "over": t["term"]})
            if t["category"] == "SEVERITY" and t["term"] in {"severe", "acute", "sharp"}:
                nmm.append({"id": "NMM_forward_lean", "type": "emphasis",
                            "scope": "sign", "over": t["term"]})
        if context.get("question"):
            nmm.append({"id": "NMM_brow_raise", "type": "yn_question",
                        "scope": "utterance", "over": None})
        if context.get("negation"):  # utterance-level fallback (no per-sign flags)
            nmm.append({"id": "NMM_headshake", "type": "negation",
                        "scope": "utterance", "over": None})
        return nmm

    @staticmethod
    def _gloss(ordered_toks, nmm):
        over = {}
        for n in nmm:
            if n["scope"] == "sign":
                over.setdefault(n["over"], []).append(n["type"])
        parts = []
        for t in ordered_toks:
            base = t["term"].upper()
            tags = over.get(t["term"])
            parts.append(f"{base}[{'+'.join(tags)}]" if tags else base)
        prefix = "".join(f"({n['type']}) " for n in nmm if n["scope"] == "utterance")
        return prefix + " ".join(parts)

    # -- public -----------------------------------------------------------------
    def reorder(self, tokens, context=None) -> dict:
        context = context or {}
        toks = [e for e in (self._enrich(t) for t in tokens) if e]
        clause = self._clause_type(toks)
        attach = self._attach(toks, clause)
        order = self._emit(toks, attach, clause)
        ordered_toks = [toks[i] for i in order]
        nmm = self._nmm(ordered_toks, context)
        return {"clause_type": clause,
                "ordered": [{"sign_id": t["sign_id"], "term": t["term"],
                             "category": t["category"]} for t in ordered_toks],
                "nmm": nmm,
                "gloss": self._gloss(ordered_toks, nmm)}


# ---------------------------------------------------------------------------
def _demo():
    from m3_primitive_mapper import ISLPrimitiveMapper
    m3, m4 = ISLPrimitiveMapper(), ISLGrammarReorderer()

    def run(title, concepts, context=None):
        signs = [r for r in m3.map_concepts(concepts) if r.get("sign_id")]
        out = m4.reorder(signs, context)
        eng = " ".join(concepts)
        print(f"\n== {title} ==")
        print(f"  English in : {eng}")
        print(f"  M3 signs   : {[r['term'] for r in signs]}")
        print(f"  ISL out    : {out['gloss']}   [{out['clause_type']}]")
        return out

    # 1) complaint: English puts severity + spatial before the body part;
    #    ISL must lead with the tooth (topic) and trail time.
    out = run("Complaint", ["severe", "pain", "maxillary", "left side", "molar", "cold", "day"])
    seq = [o["term"] for o in out["ordered"]]
    assert seq[0] == "molar", f"R1 topic-first failed: {seq}"
    assert seq.index("severe") > seq.index("pain"), "R2 severity-after-head failed"
    assert seq[-1] == "day", "R5 time-last failed"
    assert any(n["id"] == "NMM_brow_furrow" for n in out["nmm"]), "R7 intensity NMM missing"

    # 2) instruction: verb leads, anatomy is the object (R6)
    out = run("Instruction", ["rinse", "mouth"])
    assert out["clause_type"] == "instruction"
    assert [o["term"] for o in out["ordered"]][:2] == ["rinse", "mouth"], "R6 verb->object failed"

    # 3) yes/no question context -> raised brows over the utterance (R7)
    out = run("Question", ["pain", "molar"], context={"question": True})
    assert any(n["id"] == "NMM_brow_raise" for n in out["nmm"]), "R7 question NMM missing"

    # 4) negation context -> headshake (R7)
    out = run("Negation", ["bleeding", "gum"], context={"negation": True})
    assert any(n["id"] == "NMM_headshake" for n in out["nmm"]), "R7 negation NMM missing"

    print("\nAll M4 rule assertions passed.")


if __name__ == "__main__":
    _demo()
