# Injury Prediction and Player Safety in the NFL

### Overview
This capstone project was completed as part of my Master’s Degree in Data Science at Boston University (August 2024 – December 2025). Using publicly available National Football League (NFL) data, the project applies data analysis and predictive modeling to examine factors that influence player health and safety. The analysis identifies patterns in common injuries and investigates aspects of gameplay that may increase a player’s risk of injury, with a particular focus on concussions and lower-body injuries. The findings of this project were in turn used to make recommendations to the NFL that involved rule changes and new coaching techniques.


### The Problem
The NFL is known as one of the most dangerous sports when it comes to player health and safety, while playing the sport and after the conclusion of one’s career. According to a research study published in 2023, it was discovered that American football players, particularly linemen, were found to be prone to early ageing “characterised by premature burden of chronic disease and reduced healthspan” (Grashow et al., 2023). The NFL has worked to address these issues by introducing rule changes, guardian caps to prevent concussions, redesigning equipment, and procedural changes like the introduction of the dynamic kickoff. These changes have produced success, as the NFL stated in the 2024 season they saw “concussions decreased 17% compared to last season and 12% compared to the 2021-2023 season” (National Football League [NFL], 2025). Although improvements have been made, independent research has shown that the data might not be as strong as the NFL originally reported. For example, Katherine O’Malley in her article “There’s a way to deal with brain injuries in football. It isn’t safety gear”, it was reported that when two lab-based studies and one on-field study replicated the use of guardian caps the results showed no difference in players who wore them versus those who did not wear them. As a result, the threat to player safety remains prevalent in all facets of the game. 

### Datasets
All three datasets used in this project were accessed online through public downloadable files posted on Kaggle.

| Dataset | Description | Size |
|---|---|---|
| `Video_Review` | Collection of information that was created based on reviewable video evidence that outlines the events that resulted in a concussion during punt players in the NFL 2016-2017 season. Target:'Primary_Impact_Type'| 37 entries |
| `InjuryRecord` | Accounts for lower-limbs injuries that occurred over two NFL regular seasons. Provides information on the surface the game occurred on and the number of days the player missed due to injury. Target: 'Surface'| 105 enteries |
| `Concussion` | Contains a list of concussion injuries that occurred in the NFL from the years 2012 to 2014. Target: 'Reported_Injury_Type'| 390 enteries |


### Methods
**Data Cleaning, Preprocessing, Exploratory Data Analysis (EDA)**
- Preprocessing determined no duplicate rows or columns in any of the three datasets
- Missing values or "NaN" values were dropped in all three datasets
- Standarization, using 'StandardScaler' was applied to all numerical features
- One-Hot Encoding was applied to all categorical variables
- Univariate analysis (histograms, distribution plots) and bivariate analysis (heat maps, scatter plots, pair plots, and line graphs) performed on each dataset


### Modeling/Analysis 
A total of 18 different machine learning models were evaluated in this project.

- Classification
   - **Decision Tree & Random Forest** selected for insight into non-linear patterns and strong handling of categorical targets, the data type of all three targets.
     
- Regression
   - **PLSR & PCR** selected to handle multicollinearity shown in two of the datasets during EDA.

- Clustering  
   - **DBSCAN & Hierarchical Agglomerative Clustering (HAC)** used to identify if there were any common risk factors or features that may put a player at a higher risk for injury.

**Validation Strategy**
- 80/20 train-test split on all datasets
- 5-fold cross-validation on all supervised models to reduce overfitting risk on small datasets. 
- Hyperparameter tuning applied to all models (**PLSR/PCR**: 'n_components'; **Decision Trees/Random Forests**:‘max_depth’, ‘min_samples_leaf’, and ‘min_samples_split'; **DBSCAN**:`eps` and `min_samples` selected via k-distance plots)
- Class imbalance in `Video_Review` combatted with `class_weight='balanced'`

**Success Criteria**

- RMSE < 0.40
- R² (Coefficient of Determination) > 0.70
- Accuracy > 70%
- F1-scores > 0.70
- Pearson correlation coefficient > 0.70


### Results

| Dataset | Best Machine Learning Model | CV RMSE |
|---|---|---|
| `Video_Review` | PLSR | 0.8241 |
| `Injury_Record` | PLSR | 0.4926 |
| `Concussion` | PLSR | 0.3498 |

| Dataset | Best Machine Learning Model | CV accuracy |
|---|---|---|
| `Video_Review` | Support Vector Machine (with best C) | 0.7586 |
| `Injury_Record` | Support Vector Machine (with best C) | 0.5357 |
| `Concussion` | Support Vector Machine (with best C) | 0.9500 |

| Dataset | Machine Learning Model | Silhouette Scores |
|---|---|---|
| `Video_Review` | K-means | 0.2362 |
| `Injury_Record` | DBSCAN | 0.38 |
| `Concussion` | DBSCAN | 0.49 |

**What Worked**
- **PLSR** was the strongest regression model across all three datasets (avg. CV RMSE: 0.5555).
-  The `Concussion` dataset produced an RMSE of **0.3498**, meeting the success criteria threshold.
- PLSR slightly outperformed **PCR** (avg. CV RMSE 0.5899 across all three datasets) due to its ability to find the best predictor in weak features

- **Decision Tree & Random Forest** performed well with the `Concussion' dataset, producing a **90.6% CV accuracy** (decision tree) and **84.0%** (random forest), both well above the 70% success criteria threshold.
- Top predictive features in predicting concussions in these datasets were “Team”, “Season”, and “Average Playtime Before Injury” which had an importance level of about 0.27, 0.24, and 0.13 respectively.
  

**What Did Not Work**
- All three datasets are small making them prone to overfit, especially 'Video_Review' and 'InjuryRecord'.
   -Model accuacy for 'Video_Review' and 'InjuryRecord' ranged from 40.26%-75.86% with a CV RMSE reaching up to 2.636

- 'Video_Review' and 'InjuryRecord' datasets did not perform well in Classification models.
   -'Video_Review': **Decision Tree**= 42.93% cv accuracy with a standard deviation of 16.7%; **Random Forest**= 45.47% cv accuracy with a standard deviation of 17%
  - 'InjuryRecord': **Decision Tree**= 42.40% cv accuracy with a standard deviation of 8.94%; **Random Forest**= 40.26% cv accuracy with a standard deviation of 9.18%
  - Both datasets were label-encoded when training these models which introduced noise that negatively effected the accuracy of the models

- Clustering silhouette scores were low across all three datasets (range: 0.14–0.49), indicating distinct groups does not satisfy the model.


### Key Findings
-‘Reported Injury Type’ had a strong relationship with features like **number of snaps into the game when concussion occurs and the average playtime before injury**. The longer a player is on the field, the higher their risk for a concussion.
-**A player's position** was strongly correlated with number of concussions. Cornerbacks, wide receivers, and safeties experienced the highest rate of concussions in the 2012-2014 NFL seasons.
-During NFL punt plays, **helmet-to-body** was the most frequent the cause of a concussion in 2016 and **helmet-to-helmet** in 2017.
-During NFL punt plays, **Friendly Fire** was moderately correlated with an increased risk of a concussion.
-**Playing surface** (synthetic vs. natural) did not demonstrate a strong relationship with lower-limb injuries or injury severity.  
-**Team identity and NFL season** were the most influential predictive features in the 'Concussion' dataset.


### Recommendations
-The higher the total snaps a player faced, the higher their risk is for a concussion. Establish player and position specific snap limits and consider shortening the NFL season or length of games to minimize number of repetitive contact.
-Increase protections for cornerbacks, wide receivers, and safeties.
-Explore personnel changes, such as moving to 10 players instead of 11 on punts, to reduce the rate of concussions.
-Explore new tackling techniques to reduce helmet-to-body and helmet-to-helmet contact.
-Surface type was not a strong indicator of lower-limb injuries. Further examination of the relationship between lower-limb injuries and surface type is needed.
-Consider adapting the game of football to reduce contact, investigating changes similar to the rules established in flag football. 


### Limitations
-All three datasets were small (range: 37-390 entries). Machine learning models were  prone to overfitting and produced less accurate results.
-Most recent dataset used was published in 2019. Recent seasons' data, especially since the adaption of the dynamic kickoff, helmet re-design, and new penalties for contact to players head and lower bodies, would provide more accurate insights.


### Future Work
-Collect larger and more recent injury datasets to improve model accuracy.
-Conduct longitudinal study of injury trends in the NFL.
-Implement risk assessment models to determine what features result in highest risk of injury.
-Implement A/B testing of tackling techniques to determine effectiveness.
-Collect and analyze data regarding the NFL’s return-to-play policy and consequential  risk of developing CTE.


### How to Run
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nfl-injury-prediction-capstone.git
   cd nfl-injury-prediction-capstone
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Navigate to the `notebooks/` directory and run the notebooks to reproduce the analysis.

**The datasets used in this project are available through Kaggle. Please see the **Datasets** section in this report for download sources.**


### References 
Abdi, H. (2010). Partial least squares regression and projection on latent structure       
   regression (PLS Regression). WIREs Computational Statistics, 2(1), 97-106.          
   https://doi.org/10.1002/wics.51   

Grashow R., Shaffer-Pancyzk TV., Dairi I., Lee, H., Marengi, D., Baker, J., Weisskop, M. G., 
   Speizer, F. E., Whittington, A. J., Taylor, H. A., Keating, D., Tenforde, D., Guseh, J. S.,    Wasfy, M. M., Zafonte, R., Baggish, A. (2023). 
Healthspan and chronic disease burden among    young adult and middle-aged male former American-style professional football players.          British Journal of Sports Medicine, 57(3), 166-171. 
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
   National Football League. https://operations.nfl.com/updates/football-ops/concussions-decrease-to-historic-low-in-2024-nfl-season/ 
O’Malley, K (2025, February 7). There’s a way to deal with brain injuries in football. It      
   isn’t 
   safety gear. Harvard Public Health. https://harvardpublichealth.org/policy-practice/the-       nfls-concussion-solutions-are-an-illusion/ 
Scikit-learn. (n.d.). Principal Component Regression vs Partial Least Squares Regression. 
   https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html 
The Pennsylvania State University (n.d.). Combining Clusters in the Agglomerative Approach 
   [Lecture Notes]. https://online.stat.psu.edu/stat505/lesson/14/14.4 
