# Injury Prediction and Player Safety in the NFL

**Overview** 
This capstone project was completed as part of my Master’s Degree in Data Science at Boston University (August 2024 – December 2025). Using publicly available National Football League (NFL) data, the project applies data analysis and predictive modeling to examine factors that influence player health and safety. The analysis identifies patterns in common injuries and investigates aspects of gameplay that may increase a player’s risk of injury, with a particular focus on concussions and lower-body injuries. The findings of this project were in turn used to make recommendations to the NFL that involved rule changes and new coaching techniques.


**The Problem**
The NFL is known as one of the most dangerous sports when it comes to player health and safety, while playing the sport and after the conclusion of one’s career. According to a research study published in 2023, it was discovered that American football players, particularly linemen, were found to be prone to early ageing “characterised by premature burden of chronic disease and reduced healthspan” (Grashow et al., 2023). The NFL has worked to address these issues by introducing rule changes, guardian caps to prevent concussions, redesigning equipment, and procedural changes like the introduction of the dynamic kickoff. These changes have produced success, as the NFL stated in the 2024 season they saw “concussions decreased 17% compared to last season and 12% compared to the 2021-2023 season” (National Football League [NFL], 2025). Although improvements have been made, independent research has shown that the data might not be as strong as the NFL originally reported. For example, Katherine O’Malley in her article “There’s a way to deal with brain injuries in football. It isn’t safety gear”, it was reported that when two lab-based studies and one on-field study replicated the use of guardian caps the results showed no difference in players who wore them versus those who did not wear them. As a result, the threat to player safety remains prevalent in all facets of the game. 

**Datasets**
All three datasets used in this project were accessed online through public downloadable files posted on Kaggle.

| Dataset | Description | Size |
|---|---|---|
| `Video_Review` | Collection of information that was created based on reviewable video evidence that outlines the events that resulted in a concussion during punt players in the NFL 2016-2017 season. | 37 entries |
| `InjuryRecord` | Accounts for lower-limbs injuries that occurred over two NFL regular seasons. Provides information on the surface the game occurred on and the number of days the player missed due to injury.| 105 enteries |
| `Concussion` | Contains a list of concussion injuries that occurred in the NFL from the years 2012 to 2014. | 390 enteries |


**Methods**
### Data Cleaning, Preprocessing, Exploratory Data Analysis (EDA)
- Preprocessing determined no duplicate rows or columns in any of the three datasets
- Missing values or "NaN" values were dropped in all three datasets
- Standarization, using 'StandardScaler' was applied to all numerical features
- One-Hot Encoding was applied to all categorical variables
- Univariate analysis (histograms, distribution plots) and bivariate analysis (heat maps, scatter plots, pair plots, and line graphs) performed on each dataset


**Modeling/Analysis**
- Classification
   - **Decision Tree & Random Forest** selected for insight into non-linear patterns and strong handling of categorical targets, the data type of all three targets.
     
- Regression
   - **PLSR & PCR** selected to handle multicollinearity shown in two of the datasets during EDA.

- Clustering  
   - **DBSCAN & Hierarchical Agglomerative Clustering (HAC)** used to identify if there were any common risk factors or features that may put a player at a higher risk for injury.

### Validation Strategy
- 80/20 train-test split on all datasets
- 5-fold cross-validation on all supervised models to reduce overfitting risk on small datasets. 
- Hyperparameter tuning applied to all models (**PLSR/PCR**: 'n_components'; **Decision Trees/Random Forests**:‘max_depth’, ‘min_samples_leaf’, and ‘min_samples_split'; **DBSCAN**:`eps` and `min_samples` selected via k-distance plots)
- Class imbalance in `Video_Review` combatted with `class_weight='balanced'`

### Success Criteria

- RMSE < 0.40
- R² (Coefficient of Determination) > 0.70
- Accuracy > 70%
- F1-scores > 0.70
- Pearson correlation coefficient > 0.70


**Results**

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

**References**
Abdi, H. (2010). Partial least squares regression and projection on latent structure       
   regression (PLS Regression). WIREs Computational Statistics, 2(1), 97-106.          
   https://doi.org/10.1002/wics.51   
Grashow R., Shaffer-Pancyzk TV., Dairi I., Lee, H., Marengi, D., Baker, J., Weisskop, M. G., 
   Speizer, F. E., Whittington, A. J., Taylor, H. A., Keating, D., Tenforde, D., Guseh, J. S.,    Wasfy, M. M., Zafonte, R., Baggish, A. (2023). Healthspan and chronic disease burden among    young adult and middle-aged male former American-style professional football players.          British Journal of Sports Medicine, 57(3), 166-171. 
Kaggle. (2021). Concussions in the NFL (2012-2014). Kaggle. 
   https://www.kaggle.com/datasets/rishidamarla/concussions-in-the-nfl-20122014/data 
Kaggle. (2019). NFL 1st and Future. Kaggle. 
   https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data
Kaggle. (2019). NFL punt analytics competition dataset. Kaggle. 
   https://www.kaggle.com/competitions/NFL-Punt-Analytics-Competition/overview
Logistic Regression and regularization: Avoiding overfitting and improving generalization. 
   (2023, January 5). Medium. Retrieved June 29, 2025.            
   https://medium.com/@rithpansanga/logistic-regression-and-regularization-avoiding
   overfitting-   and-improving-generalization-e9afdcddd09d 
NFL Football Operations. (2025). Concussions Decrease to Historic Low in 2024 NFL Season. 
   National Football League. https://operations.nfl.com/updates/football-ops/concussions-         decrease-to-historic-low-in-2024-nfl-season/ 
O’Malley, K (2025, February 7). There’s a way to deal with brain injuries in football. It      
   isn’t 
   safety gear. Harvard Public Health. https://harvardpublichealth.org/policy-practice/the-       nfls-concussion-solutions-are-an-illusion/ 
Scikit-learn. (n.d.). Principal Component Regression vs Partial Least Squares Regression. 
   https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html 
The Pennsylvania State University (n.d.). Combining Clusters in the Agglomerative Approach 
   [Lecture Notes]. https://online.stat.psu.edu/stat505/lesson/14/14.4 
