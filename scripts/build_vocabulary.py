"""
Build the ISL primitive vocabulary — the shared spine for the hybrid architecture.

This single artifact doubles as:
  * M3's vocabulary table  (concept -> SIGN_ID, with synonyms + tier)
  * the ISKG node table     (typed nodes: SIGN / NMM / SPATIAL_LOC / GRAMMAR_ROLE)

and a companion edge file holds the ISKG relational structure:
  * CO_OCCURRENCE   -- DERIVED FROM THE CORPUS (data-driven, computed here)
  * HYPERNYM / ANTONYM / GRAMMAR_ROLE / REQUIRES_NMM / SPATIAL_AT  -- curated seeds
  * SEMANTIC_SIM / TRANSITION_COST  -- left as explicit PENDING hooks (need
    BioSentVec embeddings / MediaPipe pose data we don't have yet)

Design choice: node-local attributes (synonyms, grammar role, nmm eligibility,
tier) live ON the sign record; relational edges live in the separate edge file.
That keeps M3 able to load a flat lookup table while the GNN distillation step
later consumes the same graph as nodes+edges.

Inputs : data/derived/vocab_candidates.csv  (from phase0_mining.py)
         notechat_dental_conversations_cleaned.csv  (for co-occurrence pass)
Outputs: data/derived/isl_vocabulary.json
         data/derived/iskg_edges.json
         data/derived/vocabulary_report.md

Run: python scripts/build_vocabulary.py   (after phase0_mining.py)
"""
from __future__ import annotations
import csv, re, json, sys, itertools
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notechat_dental_conversations_cleaned.csv"
OUT = ROOT / "data" / "derived"

# ---------------------------------------------------------------------------
# Canonicalization: canonical primitive -> (category, [surface forms]).
# Surface variants (plural, synonym, abbrev) collapse to ONE sign. This IS M3's
# synonym handling, encoded declaratively. Longest surface form wins on tie.
# ---------------------------------------------------------------------------
PRIMITIVES: dict[str, tuple[str, list[str]]] = {
    # ---- ANATOMY (topic candidates) ----
    "tooth":        ("ANATOMY", ["tooth", "teeth"]),
    "molar":        ("ANATOMY", ["molar", "first molar", "second molar", "third molar", "back tooth"]),
    "premolar":     ("ANATOMY", ["premolar", "second premolar"]),
    "incisor":      ("ANATOMY", ["incisor", "central incisor", "lateral incisor", "front tooth"]),
    "canine":       ("ANATOMY", ["canine"]),
    "wisdom_tooth": ("ANATOMY", ["wisdom tooth"]),
    "gum":          ("ANATOMY", ["gum", "gums", "gingiva", "gum line"]),
    "jaw":          ("ANATOMY", ["jaw", "jawbone", "mandible", "maxilla", "anterior mandible", "anterior maxilla"]),
    "bone":         ("ANATOMY", ["bone", "alveolar bone"]),
    "root":         ("ANATOMY", ["root"]),
    "pulp":         ("ANATOMY", ["pulp", "pulp chamber", "nerve"]),
    "crown_anat":   ("ANATOMY", ["enamel", "dentin", "cusp"]),
    "mouth":        ("ANATOMY", ["mouth", "tongue", "lip", "cheek", "palate", "mucosa"]),
    "cavity":       ("ANATOMY", ["cavity", "socket"]),
    "tooth_surface":("ANATOMY", ["soft tissue"]),
    # ---- SYMPTOM (comment) ----
    "pain":         ("SYMPTOM", ["pain", "ache", "aching", "throbbing", "soreness", "sore"]),
    "swelling":     ("SYMPTOM", ["swelling", "swollen"]),
    "lesion":       ("SYMPTOM", ["lesion", "ulcer"]),
    "decay":        ("SYMPTOM", ["decay", "caries", "tooth decay"]),
    "fracture":     ("SYMPTOM", ["fracture", "cracked", "chipped"]),
    "cyst":         ("SYMPTOM", ["cyst"]),
    "bleeding":     ("SYMPTOM", ["bleeding", "bleeding gums"]),
    "infection":    ("SYMPTOM", ["infection", "infected", "abscess", "pus"]),
    "inflammation": ("SYMPTOM", ["inflammation", "inflamed", "gingivitis", "periodontitis", "gum disease"]),
    "sensitivity":  ("SYMPTOM", ["sensitivity", "sensitive", "hypersensitivity", "cold sensitivity", "hot sensitivity"]),
    "discomfort":   ("SYMPTOM", ["discomfort", "tenderness", "numbness"]),
    "bad_breath":   ("SYMPTOM", ["bad breath", "halitosis", "dry mouth"]),
    "discoloration":("SYMPTOM", ["discoloration"]),
    # ---- PROCEDURE (comment) ----
    "examination":  ("PROCEDURE", ["examination", "checkup", "oral examination", "intraoral examination", "clinical examination"]),
    "treatment":    ("PROCEDURE", ["treatment"]),
    "xray":         ("PROCEDURE", ["x-ray", "radiograph", "panoramic radiograph", "computed tomography"]),
    "surgery":      ("PROCEDURE", ["surgery", "gum surgery", "biopsy"]),
    "crown":        ("PROCEDURE", ["crown", "dental crown", "veneer"]),
    "extraction":   ("PROCEDURE", ["extraction", "tooth extraction", "wisdom tooth removal"]),
    "implant":      ("PROCEDURE", ["implant", "dental implant"]),
    "root_canal":   ("PROCEDURE", ["root canal", "root canal treatment", "rct", "pulpotomy", "apicoectomy"]),
    "anesthesia":   ("PROCEDURE", ["anesthesia", "anaesthesia", "local anesthesia", "general anesthesia"]),
    "restoration":  ("PROCEDURE", ["restoration", "filling", "dental filling"]),
    "denture":      ("PROCEDURE", ["denture", "dental bridge", "bridge"]),
    "cleaning":     ("PROCEDURE", ["cleaning", "scaling", "polishing", "deep cleaning", "scaling and polishing", "prophylaxis", "curettage"]),
    "whitening":    ("PROCEDURE", ["whitening", "tooth whitening"]),
    "braces":       ("PROCEDURE", ["braces", "splint"]),
    "sealant":      ("PROCEDURE", ["sealant", "fissure sealant", "fluoride", "fluoride treatment"]),
    "graft":        ("PROCEDURE", ["bone graft", "suture"]),
    # ---- INSTRUCTION (verb, comment) ----
    "open_mouth":   ("INSTRUCTION", ["open wide", "open your mouth"]),
    "close_mouth":  ("INSTRUCTION", ["close your mouth"]),
    "bite":         ("INSTRUCTION", ["bite", "bite down"]),
    "rinse":        ("INSTRUCTION", ["rinse", "rinse your mouth", "gargle", "spit"]),
    "brush":        ("INSTRUCTION", ["brush", "brush your teeth"]),
    "floss":        ("INSTRUCTION", ["floss", "floss daily"]),
    "avoid":        ("INSTRUCTION", ["avoid", "avoid chewing"]),
    "apply":        ("INSTRUCTION", ["apply"]),
    "chew":         ("INSTRUCTION", ["chew"]),
    "wait":         ("INSTRUCTION", ["wait", "hold"]),
    "relax":        ("INSTRUCTION", ["relax", "breathe"]),
    "press":        ("INSTRUCTION", ["press"]),
    # ---- SEVERITY (modifier, follows its sign) ----
    "severe":       ("SEVERITY", ["severe", "very severe", "extreme", "unbearable", "intense", "extremely painful"]),
    "mild":         ("SEVERITY", ["mild", "slight", "minor"]),
    "moderate":     ("SEVERITY", ["moderate"]),
    "sharp":        ("SEVERITY", ["sharp"]),
    "dull":         ("SEVERITY", ["dull"]),
    "chronic":      ("SEVERITY", ["chronic", "persistent", "constant"]),
    "acute":        ("SEVERITY", ["acute"]),
    # ---- DURATION (temporal, end-placed) ----
    "day":          ("DURATION", ["day", "days"]),
    "week":         ("DURATION", ["week", "weeks"]),
    "month":        ("DURATION", ["month", "months"]),
    "year":         ("DURATION", ["year", "years"]),
    "hour":         ("DURATION", ["hour", "hours"]),
    "since":        ("DURATION", ["since", "ago"]),
    # ---- TRIGGER (condition, comment) ----
    "cold":         ("TRIGGER", ["cold", "cold water", "cold air"]),
    "hot":          ("TRIGGER", ["hot", "hot water", "hot food", "temperature"]),
    "sweet":        ("TRIGGER", ["sweet", "sour", "sweet food"]),
    "pressure":     ("TRIGGER", ["pressure", "touch"]),
    "chewing_trig": ("TRIGGER", ["chewing", "biting"]),
    "air":          ("TRIGGER", ["air"]),
    # ---- SPATIAL modifiers (signed in spatial planes; key for ISL) ----
    # Surface forms restricted to UNAMBIGUOUS dental terms; bare "upper"/"lower"/
    # "back"/"front" are excluded — too noisy in general English.
    "loc_left":     ("SPATIAL", ["left side", "left maxillary", "left mandibular", "mandibular left", "maxillary left"]),
    "loc_right":    ("SPATIAL", ["right side", "right maxillary", "right mandibular", "mandibular right", "maxillary right"]),
    "loc_upper":    ("SPATIAL", ["maxillary"]),
    "loc_lower":    ("SPATIAL", ["mandibular"]),
    "loc_anterior": ("SPATIAL", ["anterior", "anterior teeth"]),
    "loc_posterior":("SPATIAL", ["posterior"]),
}

# Special non-SIGN node types completing the ISKG schema.
SPATIAL_LOC_NODES  = ["CENTER", "LEFT", "RIGHT", "HIGH", "LOW"]
GRAMMAR_ROLE_NODES = ["TOPIC", "COMMENT", "QUESTION", "NEGATION"]
NMM_NODES = [
    {"id": "NMM_brow_furrow", "type": "intensity",  "au": "AU4"},
    {"id": "NMM_brow_raise",  "type": "yn_question", "au": "AU1+AU2"},
    {"id": "NMM_headshake",   "type": "negation",    "au": "head_neg"},
    {"id": "NMM_forward_lean","type": "emphasis",    "au": "torso_fwd"},
]

# Category -> default ISL grammar role (M4 Rule 1/2 encoded as graph attribute).
ROLE_BY_CAT = {
    "ANATOMY": "TOPIC", "SYMPTOM": "COMMENT", "PROCEDURE": "COMMENT",
    "INSTRUCTION": "COMMENT", "SEVERITY": "MODIFIER", "DURATION": "TEMPORAL",
    "TRIGGER": "COMMENT", "SPATIAL": "MODIFIER",
}
# Symptom/severity signs carry intensity NMM eligibility.
NMM_ELIGIBLE_CATS = {"SYMPTOM", "SEVERITY"}

# Curated relational seeds -----------------------------------------------------
HYPERNYM = [  # (child, parent) is-a; lets M3 back off to a parent sign
    ("molar", "tooth"), ("premolar", "tooth"), ("incisor", "tooth"),
    ("canine", "tooth"), ("wisdom_tooth", "tooth"),
    ("root_canal", "treatment"), ("extraction", "treatment"),
    ("crown", "treatment"), ("implant", "treatment"), ("cleaning", "treatment"),
    ("gingivitis", "inflammation"),
]
ANTONYM = [("cold", "hot"), ("mild", "severe"), ("acute", "chronic")]

SPACES = re.compile(r"\s+")
def norm(s: str) -> str:
    return SPACES.sub(" ", s.replace("’", "'").replace("﻿", "").replace("�", "")).strip()

def build_surface_index():
    """surface form -> canonical; plus a longest-first matcher."""
    surf2canon, cat_of = {}, {}
    for canon, (cat, forms) in PRIMITIVES.items():
        cat_of[canon] = cat
        for f in forms:
            surf2canon.setdefault(f.lower(), canon)
    all_forms = sorted(surf2canon, key=len, reverse=True)
    matcher = re.compile(r"\b(" + "|".join(re.escape(f) for f in all_forms) + r")\b", re.IGNORECASE)
    return surf2canon, cat_of, matcher

def main():
    if not SRC.exists():
        sys.exit(f"Source CSV not found: {SRC}")

    surf2canon, cat_of, matcher = build_surface_index()

    # ---- single corpus pass: node frequency (term-freq) + CO_OCCURRENCE ----
    # Frequency is counted here (not from phase-0's narrower lexicon) so spatial
    # terms are covered and co-occurrence denominators stay consistent.
    freq = Counter()
    cooc = Counter()
    rows = 0
    with open(SRC, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            text = norm((row.get("data") or "") + " " + (row.get("conversation") or "")).lower()
            present = set()
            for m in matcher.finditer(text):
                c = surf2canon[m.group(1).lower()]
                freq[c] += 1            # term frequency
                present.add(c)          # row-level presence for co-occurrence
            for a, b in itertools.combinations(sorted(present), 2):
                cooc[(a, b)] += 1

    # tier per canonical (max tier among its surface forms via heuristic)
    def tier_of(canon, cat):
        t = 1
        for frm in PRIMITIVES[canon][1]:
            if canon == "root_canal" or (cat == "PROCEDURE" and (" " in frm or len(frm) > 11)):
                t = max(t, 3)
            elif " " in frm:
                t = max(t, 2)
        return t

    # ---- assign stable SIGN_IDs: group by category, freq-desc within ----
    cat_order = ["ANATOMY", "SYMPTOM", "PROCEDURE", "INSTRUCTION",
                 "SEVERITY", "DURATION", "TRIGGER", "SPATIAL"]
    ordered = sorted(PRIMITIVES, key=lambda c: (cat_order.index(cat_of[c]), -freq[c], c))
    sign_id = {c: f"SIGN_{i+1:03d}" for i, c in enumerate(ordered)}

    signs = []
    for c in ordered:
        cat = cat_of[c]
        signs.append({
            "id": sign_id[c],
            "term": c,
            "category": cat,
            "tier": tier_of(c, cat),
            "frequency": freq[c],
            "surface_forms": PRIMITIVES[c][1],          # M3 synonym set
            "grammar_role": ROLE_BY_CAT[cat],           # M4 ordering attr (ISKG)
            "nmm_eligible": cat in NMM_ELIGIBLE_CATS,
            "spatial_loc": (c.replace("loc_", "").upper() if cat == "SPATIAL" else None),
            "isl_video_url": None,                       # TODO: ISL expert capture
            "mediapipe_pose_path": None,                 # TODO: M5 pose extraction
        })

    vocabulary = {
        "meta": {
            "generated_by": "scripts/build_vocabulary.py",
            "source_rows": rows,
            "n_signs": len(signs),
            "id_scheme": "SIGN_NNN, grouped by category then descending corpus frequency",
            "tier_legend": {"1": "direct ISL sign", "2": "compound/spatial", "3": "no sign -> M2 LLM decomposition"},
        },
        "special_nodes": {
            "SPATIAL_LOC": SPATIAL_LOC_NODES,
            "GRAMMAR_ROLE": GRAMMAR_ROLE_NODES,
            "NMM": NMM_NODES,
        },
        "signs": signs,
    }
    (OUT / "isl_vocabulary.json").write_text(json.dumps(vocabulary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- edges ----
    # keep co-occurrence edges with association weight = count / min(freq_a,freq_b)
    cooc_edges = []
    for (a, b), n in cooc.items():
        if n < 50:
            continue
        denom = min(freq[a] or 1, freq[b] or 1)
        cooc_edges.append({"src": sign_id[a], "dst": sign_id[b], "count": n,
                           "weight": round(n / denom, 3)})
    cooc_edges.sort(key=lambda e: -e["count"])

    def as_ids(pairs):
        out = []
        for a, b in pairs:
            if a in sign_id and b in sign_id:
                out.append({"src": sign_id[a], "dst": sign_id[b]})
        return out

    edges = {
        "_meta": {"co_occurrence_min_count": 50, "co_occurrence_edges": len(cooc_edges)},
        "CO_OCCURRENCE": cooc_edges,            # DATA-DERIVED
        "HYPERNYM": as_ids(HYPERNYM),           # curated seed
        "ANTONYM": as_ids(ANTONYM),             # curated seed
        "_pending": {
            "SEMANTIC_SIM": "compute cosine over BioSentVec embeddings (needs model)",
            "TRANSITION_COST": "compute from MediaPipe pose endpoints (needs M5 pose data)",
            "REQUIRES_NMM": "node attribute 'nmm_eligible' is the seed; refine with ISL expert",
            "TEMPORAL_ORDER": "encoded as M4 rules; promote to edges during GNN distillation",
        },
    }
    (OUT / "iskg_edges.json").write_text(json.dumps(edges, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    by_cat = defaultdict(list)
    for s in signs:
        by_cat[s["category"]].append(s)
    L = ["# ISL Vocabulary / ISKG — Build Report\n",
         f"- Signs (primitives): **{len(signs)}**  (corpus rows: {rows:,})",
         f"- Co-occurrence edges (count>=50): **{len(cooc_edges)}**",
         f"- Curated HYPERNYM edges: {len(edges['HYPERNYM'])}, ANTONYM: {len(edges['ANTONYM'])}",
         f"- Tier split: " + ", ".join(f"T{t}={sum(1 for s in signs if s['tier']==t)}" for t in (1,2,3)) + "\n",
         "## Signs by category (id — term — freq — role)\n"]
    for cat in cat_order:
        L.append(f"### {cat}")
        for s in sorted(by_cat[cat], key=lambda x: -x["frequency"]):
            L.append(f"- `{s['id']}` {s['term']} — {s['frequency']:,} — {s['grammar_role']}"
                     + ("  [NMM]" if s["nmm_eligible"] else "")
                     + (f"  [{s['spatial_loc']}]" if s["spatial_loc"] else ""))
        L.append("")
    L.append("## Strongest sign co-occurrences (top 20)\n")
    id2term = {s["id"]: s["term"] for s in signs}
    for e in cooc_edges[:20]:
        L.append(f"- {id2term[e['src']]} ↔ {id2term[e['dst']]} — {e['count']:,} (w={e['weight']})")
    (OUT / "vocabulary_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"Done. {len(signs)} signs, {len(cooc_edges)} co-occurrence edges.")
    print(f"  -> isl_vocabulary.json, iskg_edges.json, vocabulary_report.md in {OUT}")

if __name__ == "__main__":
    main()
