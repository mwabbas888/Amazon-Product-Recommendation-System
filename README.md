# Amazon Product Recommendation System

This repository contains the analysis and implementation of various recommendation systems for Amazon electronic products. The project addresses the challenge of information overload by building personalized product suggestions, demonstrating expertise in collaborative filtering, model evaluation, and leveraging large-scale datasets.

---

## Business Objective

In the era of exponential information growth and overwhelming product choices, e-commerce platforms like Amazon rely heavily on intelligent recommendation systems to keep users engaged and drive sales. This project's objective was to **design and build a robust product recommendation system for Amazon customers based on their historical product ratings**, with the goal of providing personalized and relevant suggestions to enhance user experience and business profitability.

---

## Project Context & Problem Statement

Modern consumers face information overload, making product discovery a significant challenge. Recommendation systems are critical tools that help mitigate this by providing personalized product suggestions. Amazon, a leader in e-commerce, utilizes sophisticated algorithms, including item-to-item collaborative filtering, to intelligently analyze customer preferences and recommend products. This project aims to replicate and evaluate such systems, focusing on electronic product reviews.

---

## Data Overview

The dataset consists of Amazon product reviews, specifically focusing on electronic items. It includes user IDs, product IDs, and corresponding ratings.

* **Initial Data Size:** Over 7.8 million product ratings.
* **Attributes:**
    * `userId`: Unique identifier for each user.
    * `productId`: Unique identifier for each product.
    * `Rating`: The rating (1-5) given by the user to the product.
    * `timestamp`: (This column was dropped as it's not used in this specific problem).
* **Data Reduction:** To manage computational feasibility and ensure meaningful interactions, the dataset was reduced to include:
    * Users who have given at least **50 ratings**.
    * Products that have received at least **5 ratings**.
* **Final Data Size:** 65,290 entries.
* **Key Observations from EDA:**
    * The rating distribution is highly skewed towards higher ratings, with a significant majority being 4-star and 5-star reviews.
    * The dataset is very sparse: while there are 1,540 unique users and 5,689 unique products, only approximately 0.75% of all possible user-product interactions are observed. This sparsity is a key characteristic addressed by the chosen recommendation models.

---

## Recommendation System Approaches & Model Performance

This project explored and evaluated four distinct recommendation system approaches:

### 1. Rank-Based Recommendation System (Popularity-Based)

* **Approach:** Recommends products based on their overall average rating and popularity (number of ratings). This is a non-personalized approach.
* **Key Use Case:** Effective for cold-start scenarios (new users/items) and providing general trending product lists.
* **Output Example (Top 5 products with 50+ minimum interactions):** `['B001TH7GUU', 'B003ES5ZUU', 'B0019EHU8G', 'B006W8U2MU', 'B000QUUFRW']`

### 2. Collaborative Filtering: User-User Similarity (KNNBasic)

* **Approach:** Recommends items to a user based on what "similar" users (users who have rated similar items similarly) have liked. Similarity is calculated using cosine similarity.
* **Baseline Model Performance:**
    * RMSE: ~1.0012
    * Precision@10: 0.855
    * Recall@10: 0.858
    * F1-score@10: 0.856
* **Hyperparameter Tuning:** Optimized `k` (neighbors) and `min_k` (min neighbors), and `sim_options` (similarity measure). Best parameters: `{'k': 40, 'min_k': 6, 'sim_options': {'name': 'cosine', 'user_based': True}}`
* **Optimized Model Performance:**
    * RMSE: ~0.9526 (Improved)
    * Precision@10: 0.847
    * Recall@10: 0.894 (Improved)
    * F1-score@10: 0.870 (Improved)

### 3. Collaborative Filtering: Item-Item Similarity (KNNBasic)

* **Approach:** Recommends items by finding products similar to those a user has already liked. Similarity is calculated between items.
* **Baseline Model Performance:**
    * RMSE: ~0.9950
    * Precision@10: 0.838
    * Recall@10: 0.845
    * F1-score@10: 0.841
* **Hyperparameter Tuning:** Optimized `k`, `min_k`, and `sim_options`. Best parameters: `{'k': 30, 'min_k': 6, 'sim_options': {'name': 'msd', 'user_based': False}}`
* **Optimized Model Performance:**
    * RMSE: ~0.9578 (Improved)
    * Precision@10: 0.839
    * Recall@10: 0.880 (Improved)
    * F1-score@10: 0.859 (Improved)

### 4. Model-Based Collaborative Filtering: Matrix Factorization (SVD)

* **Approach:** Uses Singular Value Decomposition (SVD) to discover latent features that explain user-item interactions. This method is effective for sparse datasets by inferring missing ratings.
* **Baseline Model Performance:**
    * RMSE: ~0.8882
    * Precision@10: 0.853
    * Recall@10: 0.880
    * F1-score@10: 0.866
* **Hyperparameter Tuning:** Optimized `n_epochs` (iterations), `lr_all` (learning rate), and `reg_all` (regularization). Best parameters: `{'n_epochs': 20, 'lr_all': 0.01, 'reg_all': 0.2}`
* **Optimized Model Performance:**
    * **RMSE: ~0.8822 (Best performing model)**
    * **Precision@10: 0.854 (Highest Precision)**
    * **Recall@10: 0.884 (Highest Recall)**
    * **F1-score@10: 0.869 (Highest F1-score)**

---

## Visualizations

Below are key visualizations from the analysis, providing insights into the data and model performance:

**Distribution of Product Ratings:**
*Insight:* Highlights the strong positive bias in ratings, with 4 and 5-star reviews being predominant.
<img src="./visualizations/Distribution of Ratings.png" alt="Distribution of Ratings"/>

*(You can add other relevant visualizations here if you export them from your notebook, e.g., plots of user activity, product popularity, or more detailed model comparison plots if available.)*

---

## Conclusion and Recommendations

The project successfully built and evaluated four different recommendation systems for Amazon product reviews. For this particular sparse dataset, the **Model-Based Collaborative Filtering (SVD) consistently demonstrated the best performance** across all evaluation metrics (RMSE, Precision, Recall, F1-score) after hyperparameter tuning.

**Key Recommendations for Amazon:**

* **Prioritize Model-Based Approaches:** Given the dataset's sparsity and large scale, the SVD model is the most suitable for deployment due to its superior predictive accuracy and efficiency in handling missing data.
* **Utilize Rank-Based Recommendations for Cold-Start:** For new users or new products with insufficient data, a rank-based system can serve as an effective fallback, providing general popularity-based suggestions.
* **Strategic Application of Similarity Models:** While SVD performed best, user-user and item-item similarity models can still offer valuable insights. User-user models are good for an active user community with stable preferences, while item-item models are efficient for large user bases with rapidly changing preferences.
* **Continuous Monitoring & Retraining:** Recommendation systems require constant monitoring and retraining with new data to ensure relevance and accuracy as user preferences and product catalogs evolve.

---

## Technical Details

* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, `surprise` (for recommendation system algorithms: KNNBasic, SVD; and utilities like `Reader`, `Dataset`, `accuracy`, `GridSearchCV`, `train_test_split`), `sklearn.metrics` (for `mean_squared_error`).
* **Dataset:** `amazon_electronics_reviews.csv` (original data was processed from a raw file).
* **Analysis Notebook:** `Recommendation_Systems.ipynb` (the Jupyter Notebook containing all the code).
* **Recommendation System Types Implemented:**
    * Rank-Based (Popularity)
    * User-User Similarity-based Collaborative Filtering (KNNBasic)
    * Item-Item Similarity-based Collaborative Filtering (KNNBasic)
    * Model-Based Collaborative Filtering (Singular Value Decomposition - SVD)
* **Key Techniques:**
    * Large-scale Data Loading and Preprocessing
    * Data Reduction based on interaction thresholds (users >= 50 ratings, products >= 5 ratings)
    * Exploratory Data Analysis (EDA)
    * Custom functions for Precision@k, Recall@k, F1-score@k evaluation
    * Hyperparameter Tuning using `GridSearchCV` and Cross-Validation
    * Model Evaluation and Comparison
    * Prediction for interacted and non-interacted user-product pairs

---
