# Enhancement Plan — BCS407 Campus Safety Project

## Overview
This document outlines the required enhancements to the `bcs407-campus-safety` repository
to address professor feedback before final submission.

---

## 1. Emphasize Why This Is a Real Problem

**Current state:** README has a one-sentence project overview with no real-world context.

**Plan:**
- Add an **Abstract / Motivation** section opening the README
- Include real statistics on campus safety incidents (NFPA fire data, campus crime reports)
- Reference OSHA / NFPA standards for fire safety equipment and PPE compliance
- Frame the problem: unattended fire alarms, missing safety signage, unmonitored PPE zones
- Add a brief paragraph on why automated vision-based monitoring fills a gap vs manual inspections

**Deliverables:**
- New `docs/MOTIVATION.md` with real-world problem framing
- Update README with a compelling introduction section

---

## 2. Literature Review — Analytical, Not Descriptive

**Current state:** No literature review exists at all.

**Plan:**
- Add a **Literature Review** section with 8–12 references covering:
  - YOLO-series evolution (YOLOv1 → YOLOv8) — Redmon et al., Bochkovskiy et al.
  - Object detection for safety compliance (helmet detection, fire detection)
  - Class imbalance in object detection — surveys by Johnson et al., Khan et al.
  - Campus safety monitoring systems — specific papers on smart campus IoT/vision
- Each reference should be **critically analyzed** (what they did well, limitations, how this project extends)
- Use proper citation format (IEEE or APA)

**Deliverables:**
- New `docs/LITERATURE_REVIEW.md` with analytical review table
- Inline citations in README

---

## 3. More Literature on the Problem Domain

**Current state:** Zero references to safety/safety-monitoring research.

**Plan:**
- Add references to:
  - Workplace safety compliance monitoring (ISO 45001, OSHA)
  - PPE detection benchmarks (Safety-Helmet-Wearing dataset, Construction Safety dataset)
  - Fire/smoke detection in indoor environments
  - Pedestrian safety in campus environments
- Target: minimum 5–7 domain-specific references

**Deliverables:**
- Integration into `docs/LITERATURE_REVIEW.md` and README

---

## 4. Class Imbalance Problem

**Current state:** `augment_v2.py` partially addresses this with minority-class ratios, but:
- No formal analysis of imbalance severity
- No evaluation of augmentation effectiveness
- `safety_helmet` dominates (~5000+ vs ~480–900 for others by ~10x)

**Plan:**
- Add a **Class Distribution Analysis** script (`code/analyze_distribution.py`):
  - Count per-class images in train/valid/test
  - Generate bar chart of class distribution
  - Compute imbalance ratio (majority/minority)
- Enhance augmentation pipeline:
  - Add oversampling strategy (copy-augment for extreme minority classes)
  - Consider SMOTE-style augmentation or mosaic-based balancing
  - Add `--target-balance` parameter to equalize classes
- Document before/after distribution stats

**Deliverables:**
- `code/analyze_distribution.py` — dataset analysis tool
- Updated `code/augment_v2.py` — enhanced balancing
- Distribution chart saved to `results/plots/class_distribution.png`
- README table showing before/after class counts

---

## 5. Quantitative Analysis in Methodology

**Current state:** README has training config table but no methodological rigor.

**Plan:**
- Add detailed **Methodology** section with:
  - Dataset collection process (sources, total images, annotation tool)
  - Class distribution tables (before and after augmentation)
  - Augmentation pipeline details (which transforms, why chosen)
  - Training hyperparameters with justification (why YOLOv8m, why 100 epochs, why batch=16)
  - Loss function analysis (box loss, class loss, DFL loss curves)
  - Convergence analysis (at what epoch did metrics plateau?)
- Include quantitative dataset statistics:
  ```
  | Class        | Train | Valid | Test | Imbalance Ratio |
  |-------------|-------|-------|------|-----------------| 
  | safety_helmet | 5000+ | 1400+ | 700+ | 10.0x           |
  | emergency_exit| 900+  | 257+  | 128+ | 1.8x            |
  | fire_alarm   | 590+  | 170+  | 85+  | 1.2x            |
  | wet_floor_sign| 480+ | 137+  | 69+  | 1.0x (reference)|
  ```

**Deliverables:**
- Quantitative methodology subsection in README
- `docs/METHODOLOGY.md` with detailed analysis

---

## 6. Discussion Section with Table

**Current state:** No Discussion section exists.

**Plan:**
- Add **Discussion** section including:
  - Summary table of all results (per-class precision, recall, mAP@0.5, mAP@0.5:0.95)
  - Confusion matrix analysis (which classes are confused with which?)
  - Comparison table against related works (benchmark against published PPE detection papers)
  - Failure case analysis (when does the model fail? Why?)
  - Limitations section
  - Future work recommendations

**Deliverables:**
- Discussion section in README
- `docs/DISCUSSION.md` standalone analysis

---

## 7. Evaluation Metrics Deep-Dive

**Current state:** README lists top-line metrics (mAP@0.5, precision, recall) but no detailed analysis.

**Plan:**
- Add per-class evaluation table:
  ```
  | Class        | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | F1-Score |
  |-------------|-----------|--------|---------|--------------|----------|
  | safety_helmet  | ...      | ...    | ...     | ...          | ...      |
  | emergency_exit | ...      | ...    | ...     | ...          | ...      |
  ```
- Compute and report F1-score per class (not in current results)
- Analyze mAP gap between @0.5 and @0.5:0.95 (indicates localization quality)
- Include training/validation loss convergence curves
- Add inference speed analysis (FPS at different resolutions)

**Deliverables:**
- Enhanced results tables in README
- `docs/EVALUATION.md` with detailed metrics analysis
- F1 curve visualization

---

## 8. Ethical Frameworks (IST / Professional Ethics)

**Current state:** Ethics section is 4 bullet points with a "not surveillance" disclaimer.

**Plan:**
- Reference and apply specific ethical frameworks:
  - **IEEE Code of Ethics** — public safety obligation, competence, avoiding harm
  - **ACM Code of Ethics** — privacy, avoiding discrimination, transparency
  - **IST (Information Systems Technology) ethics** — responsible AI deployment
  - **Canadian ethical standards** — PIPEDA (privacy), CSA standards
- Address specific ethical concerns:
  - Privacy: How are campus occupants' identities protected? (blurring, no face storage)
  - Bias: Is the dataset representative? (lighting conditions, camera angles, demographics)
  - Accountability: Who is responsible when the system misses a hazard?
  - Transparency: Can the system explain why it flagged something?
  - Surveillance vs Safety: Clear boundary definition
- Include an Ethics Impact Assessment table

**Deliverables:**
- Comprehensive `docs/ETHICS.md` with framework analysis
- Enhanced ethics section in README with specific framework references
- Ethics Impact Assessment table

---

## 9. Technical Report Structure Fix

**Current state:** README mixes code documentation with academic report content. No proper report structure.

**Plan:**
- Create a proper technical report structure:
  1. Title + Authors + Affiliation
  2. Abstract
  3. Introduction (problem motivation)
  4. Literature Review
  5. Methodology (dataset, model, training, evaluation)
  6. Results & Discussion
  7. Ethical Considerations
  8. Conclusion & Future Work
  9. References
- Move code documentation to `docs/`
- Keep README as project overview + quick-start guide

**Deliverables:**
- `docs/TECHNICAL_REPORT.md` — full academic report
- Restructured README (overview + quick-start)

---

## Implementation Priority & Timeline

| Priority | Task | Effort |
|----------|------|--------|
| 🔴 HIGH  | Class imbalance analysis + fix | 2–3 hours |
| 🔴 HIGH  | Literature review (analytical) | 3–4 hours |
| 🔴 HIGH  | Ethics framework analysis | 2–3 hours |
| 🟡 MED   | Quantitative methodology | 1–2 hours |
| 🟡 MED   | Discussion section with tables | 1–2 hours |
| 🟡 MED   | Evaluation metrics deep-dive | 1–2 hours |
| 🟢 LOW   | Motivation / problem framing | 30 min |
| 🟢 LOW   | Technical report restructure | 1–2 hours |

---

## Files to Create/Modify

**New files:**
- `docs/MOTIVATION.md`
- `docs/LITERATURE_REVIEW.md`
- `docs/METHODOLOGY.md`
- `docs/DISCUSSION.md`
- `docs/EVALUATION.md`
- `docs/ETHICS.md`
- `docs/TECHNICAL_REPORT.md`
- `code/analyze_distribution.py`

**Modified files:**
- `README.md` — restructure and enhance all sections
- `code/augment_v2.py` — enhanced balancing
- `code/setup_v2.py` — add class distribution reporting