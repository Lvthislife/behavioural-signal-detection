# workplace-nlp-risk-signals

Detecting latent behavioural and psychological risk signals in workplace incident reports using persona-based synthetic data, TF-IDF and RoBERTa embeddings, and SHAP explainability.

---

## Overview

This project investigates whether the *way* workplace incidents are reported carries information beyond the incident itself. Standard WHS reporting systems capture what happened. This project asks whether the language used to describe what happened — and what was done about it — reveals something about the reporter's psychological state, reporting effort, and the quality of corrective actions taken.

The core hypothesis: linguistic patterns in incident reports (passive constructions, hedging, emotional suppression, vagueness) are associated with lower reporting effort, poorer corrective action quality, and lower psychological safety — independent of incident type and severity.

The project operationalises this through a persona framework: five synthetic reporter archetypes with distinct behavioural and linguistic profiles, allowing controlled analysis of how reporting style varies across psychological and organisational conditions.

This work was completed as part of a Graduate Certificate in Data Science and AI Programming and forms the basis for a planned extension into frontier model behavioural evaluation — specifically, whether similar linguistic signals emerge in model outputs under conditions of authority pressure, conflicting incentives, or ambiguous institutional constraints.

---

## Personas

The dataset centres on five reporter personas, each representing a distinct safety culture archetype:

| Persona | Profile |
|---|---|
| **Barry Thompson** | Minimiser — under-assesses severity, uses vague language, high model risk |
| **Talia Navarro** | Anxious — tentative language, fear and avoidance signals, variable report depth |
| **Gavin Leung** | Disengaged — emotionally distant, low effort, consistently poor action quality |
| **Kavita Rao** | Conscientious — detailed, proactive reports, consistently high action quality |
| **Mia Chen** | By-the-book — formal, templated language, reliable but sometimes low specificity |

Each persona contributes 100 records to the dataset (500 total), with effort, psychological safety, and action quality labels calibrated to persona type.

---

## Dataset

`synthetic_whs_incidents_persona_dataset.xlsx` — 500 records, 10 fields:

| Field | Description |
|---|---|
| `persona` | Reporter archetype |
| `incident_description` | Free-text incident narrative |
| `corrective_action` | Free-text corrective action taken |
| `psych_safety` | Psychological safety level (low / mixed / medium / high) |
| `effort` | Reporting effort (low / medium / high) |
| `emotion` | Dominant emotional tone |
| `action_quality` | Corrective action quality (good / poor) |
| `incident_timestamp` | Date of incident |
| `closed_date` | Date closed |
| `incident_severity` | Incident severity level |

The dataset is synthetic, generated to preserve the statistical and linguistic characteristics of real WHS incident reports in operational environments. Real incident data was not used due to privacy constraints.

---

## Linguistic Library

`01_Linguistic_Library_Construction.ipynb` builds the domain language foundation for the project:

- Milton Model NLP language patterns (mind reading, nominalisations, presuppositions, modal operators, and others) — drawn from NLP linguistic analysis frameworks and operationalised as detection targets in report text
- WHS terminology extracted from Safe Work Australia glossary sources
- Two output dictionaries: `WHS_Terms` and `NLP_Language_Patterns`
- Word cloud visualisation of domain vocabulary

This notebook establishes the theoretical basis for what constitutes linguistically degraded or psychologically loaded incident report language.

---

## Pipeline

### Notebook 1 — `01_WHS2.ipynb`: Data Load, Validation & Setup
- Dataset load and string standardisation
- Label encoding: effort (low/medium/high → 0/1/2), psych safety (low/mixed/medium/high → 0/1/2/3), action quality (poor/good → 0/1)
- Train/test split preparation
- Foundation for all downstream notebooks

### Notebook 2 — `02_WHS2.ipynb`: EDA & TF-IDF Classification
- Sample text inspection and label distribution analysis
- Violin plots and persona-level comparisons
- TF-IDF vectorisation (unigrams, max 1,000 features) on corrective action text
- Logistic Regression classification for `effort` and `action_quality`
- Classification reports and confusion matrices

### Notebook 3 — `03_WHS2.ipynb`: RoBERTa Embeddings & SHAP Explainability
- Sentence embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`, RoBERTa-based)
- Logistic Regression classifiers for `effort`, `action_quality`, and `psych_safety`
- SHAP explainability with summary plots for each classification target
- Persona-level mean prediction probability analysis
- Comparison of TF-IDF vs RoBERTa performance across targets

### Notebook 4 — `04_WHS2.ipynb`: Final Comparison, Dashboards & Persona Storytelling
- Model performance comparison: TF-IDF + Logistic vs RoBERTa + Logistic
- Dashboard-style visual summaries (F1 scores by model and target)
- Persona-level boxplots for effort and psychological safety distributions
- Behavioural inference and storytelling by archetype

---

## Model Performance Summary

| Model | Action Quality F1 | Effort F1 | Psych Safety F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.81 | 0.79 | 0.76 |
| RoBERTa + Logistic Regression | 0.89 | 0.87 | 0.85 |

RoBERTa embeddings outperform TF-IDF across all three classification targets, with the largest gain on psychological safety — the most latent and linguistically subtle of the three signals.

---

## Tech Stack

- Python 3.12
- `sentence-transformers` (`all-MiniLM-L6-v2`)
- scikit-learn
- SHAP
- pandas, numpy, matplotlib, seaborn
- NLTK
- python-docx, wordcloud

---

## Repository Structure

```
workplace-nlp-risk-signals/
├── README.md
├── requirements.txt
├── 01_Linguistic_Library_Construction.ipynb
├── 01_WHS2.ipynb
├── 02_WHS2.ipynb
├── 03_WHS2.ipynb
├── 04_WHS2.ipynb
└── data/
    └── synthetic_whs_incidents_persona_dataset.xlsx
```

---

## Limitations and Future Directions

The synthetic dataset, while persona-calibrated and linguistically grounded, does not carry the full distributional complexity of real incident corpora. The persona labels are constructed rather than observed, which bounds the generalisability of findings to real reporting populations.

Planned extensions:

- Apply the feature engineering pipeline to real incident report corpora (pending ethics approval and data access)
- Extend the linguistic signal detection framework to frontier language model outputs — testing whether passive constructions, hedging, specificity loss, and tonal suppression emerge when models operate under conditions analogous to authority pressure, conflicting incentives, or ambiguous institutional constraints
- Develop persona-equivalent archetypes for model behavioural profiling under systematic evaluation conditions

---

## Project Context

Completed as part of a Graduate Certificate in Data Science and AI Programming at the University of Technology Sydney. The project sits at the intersection of NLP, organisational behaviour, and safety science, and is intended as a foundation for research into behavioural signal detection in both human-generated and model-generated text.

