# Phase 0 — Data Foundation Report

- Source rows processed: **5,430**
- Doctor utterances harvested: **62,353** (47,730 unique)
- Vocabulary candidates (seen >0): **218** terms across 7 NER categories
- Tier distribution: Tier1=157, Tier2=40, Tier3=21
- NER seed examples (>=2 entities/sentence): **3,306**

## Top concepts by NER category (data-driven primitive seed)

- **ANATOMY**: teeth (19,298), tooth (19,201), molar (8,291), root (6,395), bone (5,616), mouth (4,880), mandible (4,341), premolar (3,358), root canal (3,333), pulp (3,274), canine (3,261), crown (3,095)
- **SYMPTOM**: swelling (10,297), pain (9,549), lesion (8,706), caries (2,115), fracture (1,986), cyst (1,923), bleeding (1,615), discomfort (1,500), periodontitis (1,221), infection (1,041), tenderness (807), inflammation (806)
- **PROCEDURE**: examination (16,895), treatment (13,048), radiograph (5,017), surgery (4,671), crown (3,078), extraction (2,771), implant (2,731), biopsy (2,222), root canal (2,132), local anesthesia (1,913), restoration (1,648), denture (1,221)
- **INSTRUCTION**: open your mouth (1,330), bite (1,223), avoid (740), rinse (263), chew (259), apply (232), brush (205), press (83), floss (71), wait (65), swallow (50), bite down (29)
- **SEVERITY**: severe (2,138), mild (1,510), significant (1,321), chronic (1,068), slight (998), moderate (707), acute (395), dull (391), sharp (306), minor (228), constant (177), persistent (177)
- **DURATION**: year (6,766), months (5,376), years (4,286), days (2,385), weeks (2,302), ago (2,134), since (1,857), month (1,698), day (1,670), week (1,571), chronic (1,068), hours (440)
- **TRIGGER**: cold (977), chewing (593), pressure (556), touch (327), hot (287), temperature (276), air (265), biting (174), cold water (128), hot food (16), sweet (9), sweet food (7)

## Most frequent doctor utterances (M1 input register)

- `Good morning, what brings you here today?` — 915
- `Good morning, how can I help you today?` — 498
- `Hello, how are you feeling today?` — 402
- `Hello, how are you today?` — 342
- `Good morning, how are you feeling today?` — 302
- `Hello, how can I help you today?` — 264
- `You're welcome. Take care.` — 250
- `Hello, what brings you here today?` — 209
- `Good morning, how are you today?` — 186
- `You're welcome. If you have any questions or concerns, don't hesitate to ask.` — 148
- `Can you tell me what brings you here today?` — 142
- `Hi, what brings you here today?` — 108
- `You're welcome. If you have any questions or concerns, please don't hesitate to ask.` — 104
- `Hi, how are you feeling today?` — 102
- `Hi, how are you today?` — 93
- `Good morning, what brings you to the clinic today?` — 92
- `You're welcome. Have a good day.` — 88
- `Good morning, what brings you in today?` — 80
- `I see. Can you tell me more about your chief complaint?` — 76
- `Great. If you have any questions or concerns, please don't hesitate to ask.` — 68

## Discovery n-grams (candidate terms NOT necessarily in seed lexicon)

Use these to extend the lexicon / catch missed dental terms:

- year old — 5,068
- examination revealed — 2,832
- chief complaint — 1,642
- clinical examination — 1,636
- old male — 1,597
- medical history — 1,568
- year old male — 1,541
- root canal — 1,525
- old female — 1,511
- central incisor — 1,496
- year old female — 1,440
- intraoral examination — 1,426
- first molar — 1,320
- lateral incisor — 1,175
- left side — 1,007
- radiographic examination — 1,005
- right side — 965
- local anesthesia — 948
- second molar — 919
- oral hygiene — 822
- left maxillary — 807
- maxillary right — 798
- computed tomography — 795
- intraoral examination revealed — 773
- anterior teeth — 757
- soft tissue — 748
- examination showed — 742
- maxillary left — 718
- follow up — 715
- second premolar — 709
- panoramic radiograph — 693
- right maxillary — 692
- third molar — 671
- left mandibular — 658
- oral examination — 639
- right mandibular — 619
- treatment plan — 617
- clinical and radiographic — 616
- clinical examination revealed — 616
- history revealed — 616
