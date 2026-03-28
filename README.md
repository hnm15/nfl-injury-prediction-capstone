# Injury Prediction and Player Safety in the NFL
This capstone project was completed as part of my Master’s Degree in Data Science at Boston University (August 2024 – December 2025). Using publicly available National Football League (NFL) data, the project applies data analysis and predictive modeling to examine factors that influence player health and safety. The analysis identifies patterns in common injuries and investigates aspects of gameplay that may increase a player’s risk of injury, with a particular focus on concussions and lower-body injuries. The findings of this project were in turn used to make recommendations to the NFL that involved rule changes and new coaching techniques.

My undergraduate and graduate degrees in Exercise Science provided me with domain knowledge that helped me prioritize meaningful relationships and interpret and contextualize findings.

******## The Problem

The NFL is one of the most physically dangerous professional sports leagues in the world. Research has linked play to early chronic disease and reduced life expectancy, particularly among linemen. While the league has introduced interventions in recent years — guardian caps, the dynamic kickoff, updated contact penalties — independent studies suggest their effectiveness may be overstated. This project applies machine learning and statistical modeling to independently quantify injury risk factors and identify where targeted interventions could have the greatest impact.

---

## Data

Three publicly available NFL datasets were sourced from Kaggle:

| Dataset | Description | Size |
|---|---|---|
| `Video_Review` | Concussions on punt plays during the 2016–2017 NFL season; target: primary impact type (helmet-to-helmet, helmet-to-body, helmet-to-ground) | 37 records |
| `InjuryRecord` | Lower-limb injuries across two regular seasons; includes playing surface and days missed; target: surface type (synthetic vs. natural) | 105 records |
| `Concussion` | Concussion injuries from 2012–2014 NFL seasons; includes position, week of injury, games missed, average playtime before injury; target: reported injury type | 390 records |

> Raw data is not redistributed in this repo. See `data/README.md` for sources and access instructions.

---

## Methods

### Preprocessing & EDA
- Null value removal, deduplication, and feature validation across all three datasets
- `StandardScaler` applied to all numerical features to eliminate scale bias
- One-Hot Encoding applied to categorical variables (position, impact type, surface)
- Univariate analysis (histograms, distribution plots) and bivariate analysis (correlation heatmaps, scatter plots, pair plots, line graphs)
- Multicollinearity identified in `InjuryRecord` and `Concussion` datasets — informed model selection

### Models Used
- **PLSR & PCR** — selected to handle multicollinearity via dimensionality reduction
- **Decision Tree & Random Forest** — selected for non-linear pattern detection and strong handling of categorical targets
- **DBSCAN & Hierarchical Agglomerative Clustering (HAC)** — used to identify latent risk groupings across players and play types

### Validation Strategy
- 80/20 train-test split on all datasets
- 5-fold cross-validation on all supervised models to reduce overfitting risk on small datasets
- Hyperparameter tuning on all models (`max_depth`, `min_samples_leaf`, `min_samples_split` for trees; `n_components` for PLSR/PCR; `eps` and `min_samples` via k-distance plots for DBSCAN)
- Class imbalance in `Video_Review` addressed with `class_weight='balanced'`

### Success Criteria (Pre-Defined)

| Metric | Threshold |
|---|---|
| RMSE | < 0.40 |
| R² | > 0.70 |
| Accuracy | > 70% |
| F1-Score | > 0.70 |
| Pearson r | > 0.70 |

---

## Results

### What Worked
- **PLSR** was the strongest regression model across all three datasets (avg. CV RMSE: 0.5555). The `Concussion` dataset hit an RMSE of **0.3498**, meeting the pre-defined threshold. PLSR outperformed PCR by better extracting signal from weak features — consistent with its known advantage when targets correlate with low-variance directions in the data.
- **Decision Tree & Random Forest on `Concussion`** produced strong classification results: **90.6% CV accuracy** (decision tree) and **84.0%** (random forest), both well above the 70% threshold. Top predictive features were Team (0.27 importance), Season (0.24), and Average Playtime Before Injury (0.13).

### Where Models Fell Short
- `Video_Review` and `InjuryRecord` consistently underperformed across all model types (~42–65% accuracy, RMSE up to 1.41). The primary driver was dataset size — 37 and 105 records respectively are insufficient for stable modeling.
- Clustering silhouette scores were low across all three datasets (range: 0.14–0.49), indicating overlapping rather than well-separated groupings.
- Label encoding (instead of one-hot encoding) on tree models for `Video_Review` and `InjuryRecord` introduced noise — identified as a methodological limitation.

---

## Key Findings

- **Average playtime and snap count before injury** are among the strongest predictors of concussion risk — more time on the field directly increases exposure
- **Team identity and season** were the top predictive features in the `Concussion` dataset, suggesting differences in team training, coaching schemes, or the influence of rule changes over time
- **Tackling and blocking** are the primary mechanisms of concussion on punt plays; helmet-to-body contact led in 2016 while helmet-to-helmet surged in 2017
- **Friendly fire (same-team contact)** showed a moderate correlation with concussion risk on punt plays — a potentially modifiable risk factor
- **Playing surface** (synthetic vs. natural) did not show a strong enough relationship with lower-limb injury outcomes to support costly surface replacement investments based on this data alone
- **Cornerbacks, wide receivers, and safeties** experienced the highest concussion rates in the 2012–2014 dataset

---

## Recommendations

| Recommendation | Supporting Evidence |
|---|---|
| Implement per-player snap count or playtime limits | Average playtime before injury is a top-3 predictive feature |
| Increase positional protection for CBs, WRs, and safeties | These positions had the highest concussion rates in the dataset |
| Investigate reducing punt unit personnel (e.g., 10 vs. 11 players) | Tackling/blocking and friendly fire are primary concussion drivers on punts |
| Explore coaching interventions for special teams technique | Helmet-to-body on tackles and helmet-to-helmet on blocks are addressable contact patterns |
| Deprioritize turf replacement as a near-term safety investment | Surface type did not produce a strong injury relationship in this analysis |
| Collect and analyze return-to-play data post-concussion | CTE risk and repeated concussion likelihood are not captured in current datasets |

---

## Limitations

- All datasets are small (37–390 records), making models prone to overfitting — particularly `Video_Review`
- Most recent dataset was published in 2019, predating the dynamic kickoff, updated helmet standards, and new contact penalties
- Findings are associative, not causal
- The `Concussion` dataset includes identifiable player information; all PII except position was dropped prior to analysis

---

## Future Work

- Collect larger, more recent datasets covering post-2019 rule changes
- A/B testing of coaching interventions (e.g., new blocking/tackling techniques)
- Longitudinal injury trend analysis across full NFL seasons
- Risk stratification modeling by position and play type
- Return-to-play policy analysis and CTE risk modeling

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/nfl-injury-prediction-capstone.git
   cd nfl-injury-prediction-capstone
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets from Kaggle (see `data/README.md`) and place them in the `data/` folder.

4. Open the notebook:
   ```bash
   jupyter notebook notebooks/modeling_notebook.ipynb
   ```
