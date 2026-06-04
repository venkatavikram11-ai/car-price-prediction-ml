# Predicting Car Prices Using Data Analytics and Machine Learning

**Author:** Challagolla Venkata Vikram (Q1116219)  
**Programme:** MSc Data Analytics — Fundamentals of Data Analytics  
**Submission Date:** 18 June 2026  
**Tutor:** Dr. Mohammad Reza Nilchiyan

---

## Overview

This project builds a complete machine learning pipeline to predict and classify automobile prices using the IBM/UCI Automobile dataset. It covers exploratory data analysis, preprocessing, feature engineering, regression, classification, and business interpretation — all in a single end-to-end notebook.

---

## Dataset

- **Source:** IBM/UCI Automobile Dataset (`auto.csv`)
- **Size:** 205 rows × 26 features (201 usable rows after cleaning)
- **Era:** 1980s automobile market data
- **Target variable:** `price` (continuous for regression; binned into Low/Medium/High for classification)

### Key Features Used

| Feature | Description |
|---|---|
| `engine-size` | Engine displacement (cc) |
| `horsepower` | Engine output (HP) |
| `curb-weight` | Vehicle weight (lbs) |
| `width` | Body width (inches) |
| `city-L/100km` | Engineered fuel consumption metric |

---

## Project Structure

The notebook is organised into 7 sequential tasks:

| Task | Description |
|---|---|
| **Task 1** | Exploratory Data Analysis — descriptive stats, histograms, boxplots, correlation heatmap |
| **Task 2** | Preprocessing & Feature Engineering — missing value handling, normalisation, binning, new feature creation |
| **Task 3** | Regression — Linear Regression vs Random Forest across 80/20 and 70/30 splits |
| **Task 4** | Classification — Multinomial Logistic Regression on price bands |
| **Task 5** | Evaluation & Visualisation — actual vs predicted plots |
| **Task 6** | Business Insights & Recommendations |
| **Task 7** | Conclusion, Limitations & Future Work |

---

## Results Summary

### Regression

| Model | Split | R² | MAE |
|---|---|---|---|
| Linear Regression | 80/20 | ~0.75 | — |
| Random Forest | 80/20 | ~0.94 | Lower |
| Linear Regression | 70/30 | Similar | — |
| Random Forest | 70/30 | Slight decrease | — |

### Classification

| Model | Accuracy |
|---|---|
| Logistic Regression (3-class) | ~95% |

**Key finding:** Random Forest outperforms Linear Regression by ~18 percentage points in R², confirming that the price-feature relationship is non-linear. The Logistic Regression classifier achieves 95% accuracy on price band prediction.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

### Google Colab (Recommended)

1. Open the notebook in Google Colab.
2. Upload `auto.csv` via the Files panel (left sidebar).
3. Go to **Runtime → Run all**.

### Local Environment

1. Clone or download this repository.
2. Place `auto.csv` in the same directory as the notebook.
3. Install dependencies (see above).
4. Launch Jupyter and open the notebook:

```bash
jupyter notebook CarPrice_Notebook_Vikram_Q1116219.ipynb
```

---

## Key Insights

- **Engine size** (r = 0.87) and **curb weight** (r = 0.83) are the strongest price predictors.
- **Fuel efficiency** (city-mpg, highway-mpg) is negatively correlated with price — efficient cars tend to be smaller and cheaper.
- **Convertibles and hardtops** command higher median prices than hatchbacks and wagons.
- The price–horsepower relationship is non-linear: below ~100 HP, price barely responds; above ~120 HP it climbs steeply.

---

## Limitations

- Small dataset (201 usable rows) from the 1980s — predictions will not generalise well to the modern market.
- Random Forest R² of 0.94 is likely optimistic on a single train/test split.
- Mean imputation on `normalized-losses` (~20% missing) flattens spread in that column.
- Categorical features (`make`, `body-style`, `drive-wheels`) were not one-hot encoded for the models.

## Future Improvements

- One-hot encode categorical columns for the regression models.
- Replace single train/test splits with k-fold cross-validation.
- Try gradient-boosted models (XGBoost, LightGBM).
- Run a hyperparameter search (GridSearchCV) on the Random Forest.
- Validate on a more recent automobile dataset.

---

## License

This project was submitted as coursework for an MSc Data Analytics programme. All rights reserved by the author.
