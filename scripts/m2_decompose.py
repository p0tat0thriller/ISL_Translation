"""
M2 — LLM Decomposition (the literacy-free fallback; closes the M3->M2->M3 loop).

When M3 returns `decompose` for a term that has no ISL sign, M2 breaks it into
concrete, mimeable concepts that DO have signs, then runs each back through M3:

    malocclusion -> "teeth that do not meet correctly" -> [tooth, bite] -> signs

Design (see docs/open_questions.md #7 and the architecture notes):
  * Provider-agnostic: M2 depends only on a tiny `LLMClient.complete()` interface,
    so the LLM choice is a late, swappable decision (offline today, any model later).
  * A 2-stage chained prompt (simplify -> extract concepts); the spec's 3rd
    "map to primitive" call is delegated to M3, which already does mapping.
  * Per-term cache (`data/derived/m2_cache.json`) — the production source of
    truth, controlling LLM non-determinism (spec KEY RISK). An ISL expert
    reviews/edits the cache; production never re-calls the LLM for known terms.
  * Bounded recursion + fingerspell only as a flagged last resort.

Runs now with NO api key (cache + seeds). To enable live decomposition of novel
terms, put `ANTHROPIC_API_KEY=...` (and optionally `M2_MODEL=...`) in `.env`.

Usage:  python scripts/m2_decompose.py
"""
from __future__ import annotations
import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "derived" / "m2_cache.json"
ENV_PATH = ROOT / ".env"
DEFAULT_MODEL = "claude-opus-4-8"   # overridable via M2_MODEL; see claude-api ref


def load_env(path: Path = ENV_PATH) -> None:
    """Minimal .env loader (KEY=VALUE per line); never overrides existing env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Pluggable LLM client — M2 calls only `.complete()`. Swap the implementation to
# change provider/model; M2's logic, cache, and loop are unchanged.
# ---------------------------------------------------------------------------
class LLMClient:
    available = False
    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class OfflineClient(LLMClient):
    """No network. Forces M2 onto cache + seeds only."""
    available = False


class AnthropicClient(LLMClient):
    """Live Claude adapter. Inactive (available=False) until a key is in env/.env."""
    def __init__(self, model: str | None = None):
        load_env()
        self.model = model or os.environ.get("M2_MODEL", DEFAULT_MODEL)
        self._client = None
        self.available = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _lazy(self):
        if self._client is None:
            import anthropic  # imported only when actually used
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, prompt: str) -> str:
        resp = self._lazy().messages.create(
            model=self.model, max_tokens=256, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------------------
# Prompts — model-agnostic plain instructions (work across providers).
# ---------------------------------------------------------------------------
SYS_SIMPLIFY = ("You explain dental terms to a Deaf patient who cannot read. "
                "Use only concrete, visible actions and objects.")
P_SIMPLIFY = ('Explain the dental term "{term}" using only concrete visible '
              "actions and objects. Max 12 words. No jargon. As if explaining "
              "to a non-literate adult.")
SYS_EXTRACT = "You extract simple visual concepts from a description."
P_EXTRACT = ("From this description, extract only things a person can see, "
             "point to, or mime. Return ONLY a JSON array of 3 to 6 single-word "
             'concepts.\n\nDescription: {description}')

# Hand-seeded decompositions (pre-warm the cache for known Tier-3 / OOV terms).
# Chosen to map onto the existing 81-sign vocab where possible.
SEED: dict[str, dict] = {
    "root canal treatment": {"description": "clean the infected inside of a tooth and close it",
                             "concepts": ["tooth", "infection", "clean"]},
    "root canal": {"description": "clean the infected inside of a tooth and close it",
                   "concepts": ["tooth", "infection", "clean"]},
    "oral hygiene instructions": {"description": "how to keep the mouth and teeth clean",
                                  "concepts": ["mouth", "tooth", "clean", "brush"]},
    "malocclusion": {"description": "teeth that do not meet correctly when biting",
                     "concepts": ["tooth", "bite"]},
    "orthodontist": {"description": "a tooth doctor who straightens teeth",
                     "concepts": ["tooth", "doctor", "straight"]},
}

_ARR = re.compile(r"\[.*?\]", re.S)


class LLMDecomposer:
    def __init__(self, client: LLMClient | None = None, mapper=None,
                 cache_path: Path = CACHE_PATH, max_concepts: int = 6, max_depth: int = 1):
        from m3_primitive_mapper import ISLPrimitiveMapper
        self.client = client or OfflineClient()
        self.mapper = mapper or ISLPrimitiveMapper()
        self.cache_path = Path(cache_path)
        self.max_concepts = max_concepts
        self.max_depth = max_depth
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) \
            if self.cache_path.exists() else {}

    def _save(self):
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    @staticmethod
    def _parse_concepts(raw: str) -> list[str]:
        m = _ARR.search(raw)
        if m:
            try:
                arr = json.loads(m.group(0))
                return [str(x).strip().lower() for x in arr if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [w.strip().lower() for w in re.split(r"[,\n]", raw) if w.strip()][:6]

    def _llm_decompose(self, term: str) -> dict:
        desc = self.client.complete(SYS_SIMPLIFY, P_SIMPLIFY.format(term=term)).strip()
        raw = self.client.complete(SYS_EXTRACT, P_EXTRACT.format(description=desc))
        return {"description": desc, "concepts": self._parse_concepts(raw)}

    def _validate(self, concepts, depth, seen):
        """Run each concept through M3; recurse on still-unmappable ones."""
        out, ok = [], True
        for c in concepts:
            r = self.mapper.map_concept(c)
            if r["decision"] in ("direct", "candidates"):
                out.append({"concept": c, "sign_id": r["sign_id"],
                            "term": r["term"], "decision": r["decision"]})
            elif depth < self.max_depth and c.lower() not in seen:
                sub = self.decompose(c, _depth=depth + 1, _seen=seen)
                if sub["fully_mappable"] and sub["mappings"]:
                    out.extend(sub["mappings"])
                else:
                    ok = False
                    out.append({"concept": c, "sign_id": None, "term": None, "decision": "decompose"})
            else:
                ok = False
                out.append({"concept": c, "sign_id": None, "term": None, "decision": "decompose"})
        return out, ok

    def decompose(self, term: str, _depth: int = 0, _seen: set | None = None) -> dict:
        key = term.strip().lower()
        seen = (_seen or set()) | {key}

        if key in self.cache:
            rec, source = self.cache[key], "cache"
        elif key in SEED:
            rec, source = SEED[key], "seed"
            self.cache[key] = {**SEED[key], "source": "seed"}; self._save()
        elif self.client.available:
            rec, source = self._llm_decompose(term), "llm"
            self.cache[key] = {**rec, "source": "llm"}; self._save()
        else:
            return {"term": term, "description": None, "concepts": [], "mappings": [],
                    "fully_mappable": False, "source": "unresolved",
                    "last_resort_fingerspell": [c.upper() for c in term if c.isalpha()]}

        concepts = rec["concepts"][:self.max_concepts]
        mappings, ok = self._validate(concepts, _depth, seen)
        return {"term": term, "description": rec.get("description"),
                "concepts": concepts, "mappings": mappings,
                "fully_mappable": ok, "source": source}


# ---------------------------------------------------------------------------
def _demo():
    client = AnthropicClient()   # live path wired; inactive without a key
    m2 = LLMDecomposer(client=client)
    print(f"LLM live mode: {client.available}  (model={client.model})\n")

    for term in ["root canal treatment", "oral hygiene instructions",
                 "malocclusion", "orthodontist"]:
        r = m2.decompose(term)
        signs = [m["term"] or f"<{m['concept']}?>" for m in r["mappings"]]
        flag = "FULLY MAPPABLE" if r["fully_mappable"] else "partial (needs more signs)"
        print(f"{term}")
        print(f"  desc    : {r['description']}")
        print(f"  concepts: {r['concepts']}  [{r['source']}]")
        print(f"  -> signs: {signs}   {flag}\n")


if __name__ == "__main__":
    _demo()
