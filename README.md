# behavioural-signal-detection

Detecting latent behavioural and psychological risk signals in workplace incident reports using persona-based synthetic data, TF-IDF and RoBERTa embeddings, and SHAP explainability.

---

## Problem Statement

In critical infrastructure organisations, workplace health and safety (WHS) reporting is a vital control for managing risk, maintaining regulatory compliance, and protecting operational continuity. While incident reports are routinely logged, the current WHS process relies heavily on manual review, with limited analytical support to assess either the severity of incidents early or the quality of the recorded corrective actions.

Corrective actions are often treated as administrative requirements — completed post-incident using free-text fields that may be vague, generic, or copied from prior responses. These actions are not consistently assessed for their alignment with the incident cause or their effectiveness in improving controls. As a result, the same types of incidents may recur despite being closed out, leading to preventable risk, audit findings, or regulatory scrutiny.

This project applies machine learning and NLP to:
- Predict the likely severity of WHS incidents from structured and unstructured report data
- Assess the alignment and quality of corrective actions by analysing their relationship to the incident narrative
- Identify patterns of low-value responses (repeated or copy-paste actions) that may indicate ineffective governance or control theatre

---

## Research Questions

**Business question:** How can data-driven methods reduce the cost and effort of WHS incident oversight, by accurately predicting severity and verifying whether corrective actions genuinely improve control effectiveness?

**Data question:** What features — structured and unstructured — can accurately predict incident severity and assess whether corrective actions are appropriate, original, and aligned with the reported incident?

Specific questions:
- Which structured factors (shift, team, incident type) are most associated with high-severity outcomes?
- Can NLP detect copy-paste patterns or vague corrective actions?
- What is the similarity between incident descriptions and action text, and does this reflect alignment?
- Can we build a scoring system that flags poor action quality or misalignment?

---

## Hypothesis

Can ML/NLP models be used to:
- Evaluate the alignment between incident descriptions and corrective actions
- Flag incomplete or poor-quality responses automatically
- Detect systemic issues across sites, teams, or processes
- Support management and governance layers with explainable, real-time advice — not just dashboards

This project tests the hypothesis that machine learning can create a governance intelligence layer within operational WHS risk reporting — making safety reporting not just administrative, but strategic and actionable.

---

## Methodology

| Technique | Purpose | Tools |
|---|---|---|
| Semantic Similarity | Match incident vs. action meaning | Sentence Transformers, TF-IDF baseline |
| Classification | Predict if corrective action is good or poor | Logistic Regression, Random Forest |
| Explainable AI | Show why the model flagged an action | SHAP |
| Persona analysis | Profile behavioural risk by reporter archetype | Custom synthetic dataset |

### TF-IDF vs RoBERTa

| Feature | TF-IDF | RoBERTa (all-MiniLM-L6-v2) |
|---|---|---|
| Method | Counts word frequency | Learns deep sentence meaning |
| Understands synonyms | No | Yes |
| Understands context | No | Yes |
| "Slipped" vs "fall" similarity | Low | High |
| "Chemical spill" vs "toxic leak" | Low | High |
| Compute cost | Fast, low memory | Higher, GPU-beneficial |
| Business explanation | "How many words matched" | "Are the ideas actually the same" |

RoBERTa was selected as the primary embedding model over BERT-base due to memory constraints on consumer hardware (8GB RAM). `all-MiniLM-L6-v2` via `sentence-transformers` provides strong semantic performance within these constraints.

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

`01_Linguistic_Library_Construction.ipynb` builds the domain language foundation for the project across two components:

**Milton Model patterns** — ten NLP language patterns associated with psychologically loaded or evasive language, each defined with description, example, and regex detection hint:

| Pattern | Signal |
|---|---|
| Mind Reading | Assumed knowledge of others' thoughts |
| Lost Performative | Value judgments without attribution |
| Cause and Effect | Implied causation without evidence |
| Complex Equivalence | Unrelated things treated as equivalent |
| Presupposition | Embedded assumptions |
| Universal Quantifiers | Extreme generalised language (always, never) |
| Modal Operators | Necessity/impossibility framing (must, can't) |
| Nominalisation | Process converted to static noun |
| Unspecified Verb | Vague action language (handled, addressed, resolved) |
| Unspecified Referential Index | Missing subject (they said, people told) |

Exported to `milton_model_patterns.csv` for use in rule-based text validation.

**WHS terminology** — extracted from the Safe Work Australia Model Code of Practice, preprocessed with NLTK lemmatisation and custom stopword filtering, and visualised as a word cloud to identify domain-specific vocabulary relevant to incident reporting.

---

## Pipeline

### Notebook 1 — `01_WHS2.ipynb`: Data Load, Validation & Setup
- Dataset load and string standardisation
- Label encoding: effort (low/medium/high → 0/1/2), psych safety (low/mixed/medium/high → 0/1/2/3), action quality (poor/good → 0/1)
- Train/test split preparation

### Notebook 2 — `02_WHS2.ipynb`: EDA & TF-IDF Classification
- Label distribution analysis and persona-level comparisons
- TF-IDF vectorisation (unigrams, max 1,000 features) on corrective action text
- Logistic Regression classification for `effort` and `action_quality`
- Classification reports and confusion matrices

### Notebook 3 — `03_WHS2.ipynb`: RoBERTa Embeddings & SHAP Explainability
- Sentence embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)
- Logistic Regression classifiers for `effort`, `action_quality`, and `psych_safety`
- SHAP explainability with summary plots for each classification target
- Persona-level mean prediction probability analysis

### Notebook 4 — `04_WHS2.ipynb`: Final Comparison, Dashboards & Persona Storytelling
- Model performance comparison: TF-IDF vs RoBERTa
- Dashboard-style visual summaries
- Persona-level boxplots for effort and psychological safety distributions
- Behavioural inference by archetype

---

## Model Performance Summary

| Model | Action Quality F1 | Effort F1 | Psych Safety F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.81 | 0.79 | 0.76 |
| RoBERTa + Logistic Regression | 0.89 | 0.87 | 0.85 |

RoBERTa outperforms TF-IDF across all three targets, with the largest gain on psychological safety — the most latent and linguistically subtle of the three signals.

---

## Tech Stack

- Python 3.12
- `sentence-transformers` (`all-MiniLM-L6-v2`)
- scikit-learn
- SHAP
- pandas, numpy, matplotlib, seaborn
- NLTK, python-docx, wordcloud

---

## Repository Structure

```
behavioural-signal-detection/
├── README.md
├── requirements.txt
├── 01_Linguistic_Library_Construction.ipynb
├── 01_WHS2.ipynb
├── 02_WHS2.ipynb
├── 03_WHS2.ipynb
├── 04_WHS2.ipynb
└── data/
    ├── synthetic_whs_incidents_persona_dataset.xlsx
    ├── milton_model_patterns.csv
    └── whs_glossary_terms.csv
```

---

## Limitations and Future Directions

The synthetic dataset, while persona-calibrated and linguistically grounded, does not carry the full distributional complexity of real incident corpora. Persona labels are constructed rather than observed, which bounds generalisability to real reporting populations.

Planned extensions:
- Apply the pipeline to real incident report corpora (pending ethics approval and data access)
- Extend the linguistic signal detection framework to frontier language model outputs — testing whether passive constructions, hedging, specificity loss, and tonal suppression emerge when models operate under conditions analogous to authority pressure, conflicting incentives, or ambiguous institutional constraints
- Develop persona-equivalent archetypes for model behavioural profiling under systematic evaluation conditions

---

## Project Context

Completed as part of a Graduate Certificate in Data Science and AI Programming at the University of Technology Sydney. The project sits at the intersection of NLP, organisational behaviour, and safety science, and is intended as a foundation for research into behavioural signal detection in both human-generated and model-generated text.

