# behavioural-signal-detection
Detecting linguistic and behavioural risk signals in workplace incident reports using explainable hybrid NLP and linguistic feature engineering

## Overview
This project investigates whether the *way* workplace incidents are reported carries information beyond the incident itself. Standard WHS reporting systems capture what happened. This project asks whether the language used to describe what happened — and what was done about it — reveals something about the quality of organisational safety culture, reporting integrity, and corrective action effectiveness.

The core hypothesis: linguistically degraded incident reports (passive constructions, hedging, emotional suppression, low specificity, copy-pasted corrective actions) are associated with poorer corrective action quality and higher incident severity, independent of incident type and structural metadata.

This work was completed as a capstone project in a Graduate Certificate in Data Science and AI Programming and forms the basis for a planned extension into frontier model behavioural evaluation — specifically, whether similar linguistic signals appear in model outputs under conditions of authority pressure, conflicting incentives, or ambiguous constraints.
---
## Research Questions

1. Can linguistic features extracted from incident description and corrective action text predict incident severity (Low / Moderate / High)?
2. Can the same features distinguish high-quality from low-quality corrective actions?
3. Which linguistic features are most predictive, and what does that suggest about the relationship between language and safety culture?
---
## Dataset

The dataset is synthetic, generated to replicate the statistical and linguistic characteristics of real WHS incident reports in operational energy sector environments (power generation, heavy industry). Real incident data was not used due to privacy constraints.

Synthetic generation was designed to preserve ecological validity:

- 400–500 records per generation run
- 13 incident types drawn from operational energy sector categories (Slip/Trip/Fall, Electrical Safety, Process Safety Event, Chemical Exposure, Working at Height, Vehicle Incident, Environmental Spill, and others)
- Two reporting style regimes: `high_trust` (factual, specific, active voice) and `low_safety` (passive, tentative, emotionally loaded)
- Seven emotional tone categories: Neutral, Fear, Gratitude, Guilt, Blame, Concern, Frustration
- Corrective action quality labels: Good / Poor
- Linguistic flags injected at generation: passive voice, tentative language, emotional tone

Structured fields include: incident ID, timestamp, team, shift, location, incident type, severity, injury flag, reporting style, and generated linguistic metadata.

---

## Pipeline

The project runs across three scripts:

### `01_generate_synthetic_data.py`
Generates the synthetic dataset with structured fields, templated incident descriptions, and injected linguistic features. Templates include passive and tentative variants calibrated to reporting style regime. Outputs `whs_incidents.csv`.

### `02_preprocess_features.py`
NLP preprocessing and feature engineering pipeline using spaCy (`en_core_web_sm`) and scikit-learn:

- Text cleaning, tokenisation, stop word removal, lemmatisation
- TF-IDF vectorisation with unigram/bigram representation (max 5,000 features)
- Cosine similarity between incident description and corrective action (measures semantic alignment)
- Copy-paste flag: detects repeated corrective action text across records
- Action word count: raw word count of corrective action text
- Specificity score: ratio of unique to total lemmas (proxy for vagueness)
- Passive voice detection: spaCy dependency parse (`nsubjpass`, `auxpass`)

Outputs `whs_incidents_processed.csv`.

### `03_eda_train_evaluate_shap.py`
Exploratory data analysis, model training, evaluation, and explainability:

- Two classification targets: incident severity (multiclass) and corrective action quality (binary)
- Baseline: Logistic Regression
- Primary: Random Forest Classifier (sklearn Pipeline with OneHotEncoder, StandardScaler)
- Evaluation metrics: Accuracy, weighted F1, ROC AUC (OvR weighted), confusion matrix
- Explainability: SHAP TreeExplainer with summary bar plots, beeswarm plots, and waterfall plots for individual predictions
- Trained models and encoders serialised via `joblib`

---

## Features Used in Modelling

| Feature | Type | Source |
|---|---|---|
| `similarity_score` | Float | TF-IDF cosine similarity, desc vs action |
| `copy_paste_flag` | Boolean | Duplicate detection on processed action text |
| `action_word_count` | Integer | Raw word count, corrective action |
| `specificity_score` | Float | Unique/total lemma ratio, processed action |
| `spacy_passive_voice_flag` | Boolean | Dependency parse, corrective action |
| `generated_passive_voice_flag` | Boolean | Injected at generation |
| `generated_tentative_flag` | Boolean | Injected at generation |
| `generated_emotional_tone` | Categorical | Injected at generation |
| `incident_type` | Categorical | Structured field |
| `team`, `shift`, `location` | Categorical | Structured fields |
| `injury` | Boolean | Structured field |
---
## Tech Stack

- Python 3.x
- spaCy (`en_core_web_sm`)
- scikit-learn
- SHAP
- pandas, numpy, matplotlib, seaborn
- joblib
---
## Limitations and Future Directions

The synthetic dataset, while ecologically calibrated, does not carry the full distributional complexity of real incident corpora. Linguistic feature injection was rule-based rather than learned, which bounds the generalisability of findings.

Planned extensions:

- Apply the feature engineering pipeline to real incident report corpora (pending ethics approval and data access)
- Extend the linguistic signal detection framework to frontier language model outputs — specifically, testing whether passive constructions, hedging, specificity loss, and tonal suppression emerge when models are evaluated under conditions analogous to authority pressure, conflicting incentives, or ambiguous institutional constraints
- Explore transformer-based embeddings (BERT, domain-adapted variants) as an alternative to TF-IDF for similarity scoring
---

## Project Context

This project was completed as part of a Graduate Certificate in Data Science and AI Programming at the University of Technology Sydney. It joins NLP, organisational behaviour, and safety science, and is intended as a foundation for research into behavioural signal detection in both human-generated and model-generated text.

