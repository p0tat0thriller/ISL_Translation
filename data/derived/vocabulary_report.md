# ISL Vocabulary / ISKG — Build Report

- Signs (primitives): **81**  (corpus rows: 5,430)
- Co-occurrence edges (count>=50): **1921**
- Curated HYPERNYM edges: 10, ANTONYM: 3
- Tier split: T1=39, T2=28, T3=14

## Signs by category (id — term — freq — role)

### ANATOMY
- `SIGN_001` tooth — 36,759 — TOPIC
- `SIGN_002` mouth — 9,435 — TOPIC
- `SIGN_003` jaw — 9,013 — TOPIC
- `SIGN_004` molar — 8,781 — TOPIC
- `SIGN_005` incisor — 6,500 — TOPIC
- `SIGN_006` root — 6,395 — TOPIC
- `SIGN_007` bone — 6,043 — TOPIC
- `SIGN_008` pulp — 4,697 — TOPIC
- `SIGN_009` premolar — 3,358 — TOPIC
- `SIGN_010` canine — 3,261 — TOPIC
- `SIGN_011` cavity — 2,972 — TOPIC
- `SIGN_012` crown_anat — 2,850 — TOPIC
- `SIGN_013` gum — 2,646 — TOPIC
- `SIGN_014` tooth_surface — 1,189 — TOPIC
- `SIGN_015` wisdom_tooth — 95 — TOPIC

### SYMPTOM
- `SIGN_016` swelling — 10,540 — COMMENT  [NMM]
- `SIGN_017` pain — 9,891 — COMMENT  [NMM]
- `SIGN_018` lesion — 8,887 — COMMENT  [NMM]
- `SIGN_019` infection — 2,784 — COMMENT  [NMM]
- `SIGN_020` inflammation — 2,593 — COMMENT  [NMM]
- `SIGN_021` discomfort — 2,514 — COMMENT  [NMM]
- `SIGN_022` decay — 2,328 — COMMENT  [NMM]
- `SIGN_023` fracture — 2,021 — COMMENT  [NMM]
- `SIGN_024` cyst — 1,923 — COMMENT  [NMM]
- `SIGN_025` bleeding — 1,707 — COMMENT  [NMM]
- `SIGN_026` sensitivity — 1,358 — COMMENT  [NMM]
- `SIGN_027` discoloration — 526 — COMMENT  [NMM]
- `SIGN_028` bad_breath — 115 — COMMENT  [NMM]

### PROCEDURE
- `SIGN_029` examination — 16,956 — COMMENT
- `SIGN_030` treatment — 13,048 — COMMENT
- `SIGN_031` xray — 7,228 — COMMENT
- `SIGN_032` surgery — 6,894 — COMMENT
- `SIGN_033` root_canal — 3,638 — COMMENT
- `SIGN_034` anesthesia — 3,334 — COMMENT
- `SIGN_035` crown — 3,217 — COMMENT
- `SIGN_036` extraction — 3,149 — COMMENT
- `SIGN_037` implant — 2,912 — COMMENT
- `SIGN_038` restoration — 2,256 — COMMENT
- `SIGN_039` denture — 1,643 — COMMENT
- `SIGN_040` cleaning — 1,582 — COMMENT
- `SIGN_041` graft — 564 — COMMENT
- `SIGN_042` braces — 407 — COMMENT
- `SIGN_043` sealant — 153 — COMMENT
- `SIGN_044` whitening — 21 — COMMENT

### INSTRUCTION
- `SIGN_045` open_mouth — 1,347 — COMMENT
- `SIGN_046` bite — 1,252 — COMMENT
- `SIGN_047` avoid — 753 — COMMENT
- `SIGN_048` rinse — 303 — COMMENT
- `SIGN_049` chew — 259 — COMMENT
- `SIGN_050` apply — 232 — COMMENT
- `SIGN_051` brush — 229 — COMMENT
- `SIGN_052` wait — 91 — COMMENT
- `SIGN_053` press — 83 — COMMENT
- `SIGN_054` floss — 73 — COMMENT
- `SIGN_055` relax — 20 — COMMENT
- `SIGN_056` close_mouth — 14 — COMMENT

### SEVERITY
- `SIGN_057` mild — 2,736 — MODIFIER  [NMM]
- `SIGN_058` severe — 2,393 — MODIFIER  [NMM]
- `SIGN_059` chronic — 1,422 — MODIFIER  [NMM]
- `SIGN_060` moderate — 707 — MODIFIER  [NMM]
- `SIGN_061` acute — 395 — MODIFIER  [NMM]
- `SIGN_062` dull — 391 — MODIFIER  [NMM]
- `SIGN_063` sharp — 306 — MODIFIER  [NMM]

### DURATION
- `SIGN_064` year — 11,052 — TEMPORAL
- `SIGN_065` month — 7,074 — TEMPORAL
- `SIGN_066` day — 4,055 — TEMPORAL
- `SIGN_067` since — 3,991 — TEMPORAL
- `SIGN_068` week — 3,873 — TEMPORAL
- `SIGN_069` hour — 584 — TEMPORAL

### TRIGGER
- `SIGN_070` cold — 1,075 — COMMENT
- `SIGN_071` pressure — 883 — COMMENT
- `SIGN_072` chewing_trig — 754 — COMMENT
- `SIGN_073` hot — 581 — COMMENT
- `SIGN_074` air — 265 — COMMENT
- `SIGN_075` sweet — 22 — COMMENT

### SPATIAL
- `SIGN_076` loc_left — 6,774 — MODIFIER  [LEFT]
- `SIGN_077` loc_right — 6,574 — MODIFIER  [RIGHT]
- `SIGN_078` loc_upper — 6,509 — MODIFIER  [UPPER]
- `SIGN_079` loc_lower — 5,624 — MODIFIER  [LOWER]
- `SIGN_080` loc_anterior — 4,780 — MODIFIER  [ANTERIOR]
- `SIGN_081` loc_posterior — 2,631 — MODIFIER  [POSTERIOR]

## Strongest sign co-occurrences (top 20)

- tooth ↔ year — 4,820 (w=0.436)
- examination ↔ year — 4,319 (w=0.391)
- examination ↔ tooth — 3,926 (w=0.232)
- treatment ↔ year — 3,316 (w=0.3)
- tooth ↔ treatment — 3,038 (w=0.233)
- examination ↔ treatment — 2,652 (w=0.203)
- mouth ↔ year — 2,567 (w=0.272)
- pain ↔ year — 2,498 (w=0.253)
- month ↔ year — 2,474 (w=0.35)
- xray ↔ year — 2,411 (w=0.334)
- mouth ↔ tooth — 2,303 (w=0.244)
- pain ↔ tooth — 2,253 (w=0.228)
- since ↔ year — 2,229 (w=0.559)
- molar ↔ year — 2,211 (w=0.252)
- examination ↔ mouth — 2,204 (w=0.234)
- month ↔ tooth — 2,180 (w=0.308)
- tooth ↔ xray — 2,180 (w=0.302)
- jaw ↔ year — 2,136 (w=0.237)
- examination ↔ pain — 2,120 (w=0.214)
- examination ↔ xray — 2,052 (w=0.284)