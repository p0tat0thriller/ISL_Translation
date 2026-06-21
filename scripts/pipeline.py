"""
ISL reverse-translation pipeline — the canonical end-to-end entry point.

Wires the four built modules into one call:

    raw clinical text
      -> M1  clean + NER (per-entity negation, Tier-3 phrase tagging)
      -> M3  concept -> SIGN_ID
           -> M2  decompose unmappable terms into known signs (M3<->M2 loop)
      -> M4  ISL grammar order + non-manual markers
      -> ISL gloss

Negation detected by M1 is carried per-entity all the way to M4, so the
headshake lands on the specific negated sign (not the whole utterance).

Usage:  python scripts/pipeline.py
"""
from __future__ import annotations

from m1_clinical_nlp import ClinicalNLP
from m2_decompose import LLMDecomposer
from m3_primitive_mapper import ISLPrimitiveMapper
from m4_grammar_reorder import ISLGrammarReorderer


class ISLTranslator:
    def __init__(self):
        self.m1 = ClinicalNLP()
        self.m3 = ISLPrimitiveMapper()
        self.m2 = LLMDecomposer(mapper=self.m3)   # share one mapper
        self.m4 = ISLGrammarReorderer()

    def _route_decompose(self, concept, neg, tokens, notes):
        d = self.m2.decompose(concept)                              # M3<->M2 loop
        mapped = [m for m in d["mappings"] if m["sign_id"]]
        tokens.extend({"sign_id": m["sign_id"], "negated": neg} for m in mapped)
        shown = ", ".join(m["term"] for m in mapped) or "(unresolved)"
        notes.append(f"{concept} -> [{shown}]"
                     + ("" if d["fully_mappable"] else "  [partial]"))

    def translate(self, raw: str) -> dict:
        doc = self.m1.process(raw)
        out = []
        for seg in doc["segments"]:
            tokens, notes = [], []
            for ent in seg["entities"]:
                neg = ent.get("negated", False)
                if ent.get("force_decompose"):        # known no-sign phrase
                    self._route_decompose(ent["concept"], neg, tokens, notes)
                    continue
                r = self.m3.map_concept(ent["concept"])
                if r["decision"] in ("direct", "candidates"):
                    tokens.append({"sign_id": r["sign_id"], "negated": neg})
                elif r["decision"] == "decompose":
                    self._route_decompose(ent["concept"], neg, tokens, notes)
            isl = self.m4.reorder(tokens, seg["context"])
            out.append({"text": seg["text"], "gloss": isl["gloss"],
                        "nmm": isl["nmm"], "decompositions": notes})
        return {"raw": raw, "clean_text": doc["clean_text"], "segments": out}


def _demo():
    t = ISLTranslator()
    inputs = [
        "pt c/o severe pain UL6 since 3d, worsening on cold",   # #10 no "SINCE DAY"
        "no swelling, mild pain in upper right molar",          # #8  negate swelling only
        "advised OHI; c/o malocclusion",                        # #9  decompose via M2
    ]
    for raw in inputs:
        res = t.translate(raw)
        print(f"\nRAW   : {raw}")
        print(f"CLEAN : {res['clean_text']}")
        for seg in res["segments"]:
            print(f"  ISL : {seg['gloss']}")
            for n in seg["decompositions"]:
                print(f"  M2  : {n}")


if __name__ == "__main__":
    _demo()
