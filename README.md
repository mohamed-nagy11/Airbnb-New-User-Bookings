# Airbnb New User Bookings Prediction
**Predicting the First Destination of New Airbnb Users using Tree-Based Ensembles and Session Behavioral Data.**

[Airbnb New User Bookings Competition | Kaggle](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings)

## Project Overview
This project tackles the Airbnb New User Bookings challenge. The objective is to predict the first country a new user will book their travel to out of 12 possible outcomes (e.g., US, FR, IT, or 'NDF' - No Destination Found). 

Instead of relying solely on sparse demographic data, this project heavily engineers raw behavioral session logs into a unified predictive matrix, utilizing advanced gradient-boosted tree architectures (XGBoost, LightGBM, CatBoost) to map user intent.

## The Thought Process & Strategy

My approach to this problem was grounded in treating user behavior as the primary signal of intent, rather than static demographics.

### 1. Decoding Missing Data (MNAR)
During exploratory data analysis, I found that missing data was not a technical error, but a behavioral signal. Missingness was **Missing Not At Random (MNAR)**.
* Users who hid their `age` or `gender` during onboarding had an overwhelmingly high (~70%) 'No Booking' (NDF) rate. 
* **Action:** Instead of imputing these with means or medians, I engineered an explicit `-unknown-` bucket. This allowed the models to learn that "refusing to provide data" is a strong indicator of low booking intent.

### 2. Demographic Cleaning & Anomaly Detection
The raw dataset contained impossible ages (e.g., 5, 2014, 1920) due to weak historical form validation and user obfuscation.
* **Action:** I established logical domain bounds (18 to 95). Any value outside this range was mathematically treated as `NaN` and routed into the `-unknown-` bucket, preventing the model from learning false demographic signals.

### 3. Session Feature Engineering (The Behavioral Matrix)
The raw session logs contained millions of rows of isolated clickstream events. To make this digestible for machine learning, I aggregated the data:
* Counted the occurrences of the Top 15 most predictive `action_detail` events per user (e.g., `view_search_results`, `p3`).
* Calculated the total `secs_elapsed` per user to gauge overall engagement time.
* Pivoted this data from a long, 1D log into a wide, 2D sparse matrix aligned with the core demographic table.

### 4. Navigating Multicollinearity
I utilized **Cramér's V** to evaluate correlations between high-cardinality categorical variables. The heatmap revealed massive overlaps in information (e.g., strong correlations between `first_browser`, `signup_app`, and `first_device_type`). This discovery heavily influenced the choice of algorithms.

## Methodology & Model Selection

### Why Tree-Based Models?
Linear models (Logistic Regression) and distance-based models (KNN, SVM) were explicitly rejected for this pipeline in favor of Gradient Boosted Decision Trees (GBDT). 
1. **Immunity to Multicollinearity:** Tree models naturally bypass the overlapping information found in the device/browser categories.
2. **Sparse Matrix Efficiency:** The dataset, after One-Hot Encoding and pivoting session actions, became extremely wide and sparse. XGBoost and LightGBM are natively optimized to evaluate sparse matrices without wasting compute power on zeros.
3. **No Feature Scaling Required:** Session times ranged from a few seconds to hours. Tree models evaluate rank order, completely eliminating the need to normalize `secs_elapsed` against `total_actions`.

### The Models Evaluated
* **XGBoost:** Served as the robust baseline, handling the sparse One-Hot Encoded data efficiently.
* **LightGBM:** Utilized for rapid iteration and training speed, though required strict `max_depth` tuning due to its leaf-wise growth causing a tendency to overfit the training set.
* **CatBoost:** Evaluated for its native handling of categorical text (bypassing OHE) and symmetric trees, offering highly stable generalization.

## Evaluation Metric: NDCG@5
The competition and the business objective require a ranked list of recommendations, not just a single binary guess. The model was optimized and evaluated using **Normalized Discounted Cumulative Gain (NDCG@5)**. 

A custom vectorized Python function was built to instantly calculate this metric across the validation set. By using NDCG@5, the model was mathematically penalized for putting highly relevant destinations lower in the top 5 rankings, ensuring the absolute best guess was locked into the #1 spot.

## Results & Key Discoveries

* **The Temporal Data Shift:** The training data (2010–mid 2014) and the test data (post-July 2014) represented a time-based split, not a random shuffle.
* **Test Score > Validation Score:** The final Kaggle public leaderboard score was significantly higher than the local validation score. This anomaly was traced to a distribution shift in the test set, which contained a much higher concentration of 'NDF' (No Destination Found) users. Because the model was highly optimized to predict NDF for low-intent users, its accuracy skyrocketed on the test set.
* **Vectorized Output:** To meet the Kaggle submission requirement of 5 rows per user, a highly optimized NumPy vectorization script was written, reducing submission generation time from minutes to milliseconds.

## How to Use This Repository

### Requirements
* Python 3.8+
* pandas, numpy
* scikit-learn
* xgboost, lightgbm, catboost
* seaborn, matplotlib

### Execution
1. Ensure the raw Airbnb data files (`train_users_2.csv`, `test_users.csv`, `sessions.csv`) are located in the `/data` directory.
2. Run the Jupyter Notebook end-to-end. The pipeline will automatically:
   * Clean and bucket the demographics.
   * Aggregate and pivot the session logs.
   * Merge the datasets.
   * Train the XGBoost model.
   * Generate a strictly formatted `submission.csv` containing the top 5 predicted destinations per user.
