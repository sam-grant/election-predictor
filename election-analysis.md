# US presidential election prediction analysis

### Overview

This analysis explores whether macro socioeconomic indicators can predict US presidential election outcomes. Four machine learning classification models were systematically optimised to predict whether the winning party would be Democratic or Republican based on annual economic and social data during election years.

### Data

**Sources:**
- [Election outcomes](https://www.presidency.ucsb.edu/statistics/elections) including winner and party (1960–2024)
- [World Bank](https://databank.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG/1ff4a498/Popular-Indicators) macroeconomic indicators: GDP growth, inflation, foreign investment, etc.
- [Federal Reserve Bank](https://fred.stlouisfed.org/series/UNRATE) unemployment rates, with monthly data aggregated to yearly averages

**Preprocessing:**
- Columns with >50% missing values were dropped
- Linear interpolation filled remaining gaps
- Final dataset: **35 features, 17 samples** (elections from 1960–2020)

The small sample size (n=17) presents a fundamental challenge for model generalisation and necessitates careful validation strategies, including Leave-One-Out Cross Validation (LOOCV).

### Methodology

The analysis followed a systematic optimisation process, evaluating four model variants for each algorithm:

1. **Base**: All 35 features with LOOCV
2. **Selected**: 10 features identified by SelectKBest (ANOVA F-statistic)
3. **Interact**: 15 features including polynomial interaction terms
4. **Tuned**: Hyperparameter optimisation via grid search with LOOCV

#### 1. Initial model comparison (base)

Four classification algorithms were evaluated using Leave-One-Out Cross-Validation (LOOCV):

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression (LR) | 0.647 | 0.833 | 0.661 | 0.647 |
| Random Forest (RF) | 0.706 | 0.771 | 0.735 | 0.706 |
| Support Vector Classifier (SVC) | 0.647 | 0.778 | 0.677 | 0.647 |
| XGBoost (XGB) | 0.588 | 0.625 | 0.579 | 0.588 |

Random Forest demonstrates the highest baseline accuracy (70.6%), whereas Logistic Regression achieved superior class separation (ROC-AUC 0.833).

#### 2. Feature engineering

**Feature selection:**

SelectKBest with ANOVA F-statistic identified the 10 most predictive features from the initial 35. These 10 features improved performance across all models, particularly for Logistic Regression (ROC-AUC increased substantially).

**Interaction terms:**
Polynomial features (degree=2, interaction_only=True) generated 55 features from the 10 base features. SelectKBest then reduced these to 15 most predictive terms, capturing multiplicative relationships between indicators. The final models use **8 unique base features** (4 standalone and 7 appearing in interaction terms):

1. Energy use (kg of oil equivalent per capita)
2. Fertility rate, total (births per woman)
3. Gross capital formation (% of GDP)
4. Inflation, GDP deflator (annual %)
5. Immunization, measles (% of children ages 12-23 months)
6. Population growth (annual %)
7. Revenue, excluding grants (% of GDP)
8. Unemployment (%)

Key interaction terms include:
- Population growth × Fertility rate
- Inflation GDP deflator × Gross capital formation
- Measles immunisation × Energy use
- Population growth × Unemployment

#### 3. Hyperparameter optimisation

A comprehensive grid search with LOOCV was conducted for all four algorithms:

**Logistic Regression** (408 combinations):
- Best parameters: C=0.001, penalty='l2', solver='liblinear'
- Best CV accuracy: **0.882**

**Random Forest** (216 combinations):
- Best parameters: n_estimators=100, max_depth=3, max_features='sqrt'
- Best CV accuracy: **0.882**

**Support Vector Classifier** (3,060 combinations):
- Best parameters: C=1, kernel='linear', gamma='scale'
- Best CV accuracy: **0.882**

**XGBoost** (22,032 combinations):
- Best parameters: n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.6
- Best CV accuracy: **0.765**

![Model Comparison](images/metrics_comparison.png)

### Model convergence

**Three independent algorithms (LR, RF, SVC) converged on identical 88.2% accuracy** after optimisation. This convergence across fundamentally different model architectures -- linear (LR), kernel-based (SVC), and tree-based (RF) -- strongly suggests **88.2% represents the performance ceiling achievable with macro socioeconomic indicators alone**.

## Classifications

The table below shows model predictions with probability of Republican victory in parentheses. Probabilities >0.5 predict Republican; <0.5 predict Democratic.

| Year | Winner            | Party       | LR         | RF         | SVC        | XGB        |
|------|-------------------|-------------|------------|------------|------------|------------|
| 2024 | Donald Trump      | Republican  | ✅ (0.510) | ✅ (0.675) | ✅ (0.982) | ✅ (0.656) |
| 2020 | Joseph Biden      | Democratic  | ✅ (0.493) | ✅ (0.180) | ✅ (0.117) | ✅ (0.291) |
| 2016 | Donald Trump      | Republican  | ❌ (0.498) | ❌ (0.140) | ✅ (0.494) | ❌ (0.276) |
| 2012 | Barack Obama      | Democratic  | ✅ (0.493) | ✅ (0.282) | ✅ (0.111) | ✅ (0.367) |
| 2008 | Barack Obama      | Democratic  | ✅ (0.498) | ✅ (0.387) | ✅ (0.405) | ✅ (0.492) |
| 2004 | George W. Bush    | Republican  | ✅ (0.504) | ✅ (0.705) | ✅ (0.655) | ✅ (0.792) |
| 2000 | George W. Bush    | Republican  | ✅ (0.510) | ✅ (0.660) | ✅ (0.891) | ✅ (0.557) |
| 1996 | William Clinton   | Democratic  | ✅ (0.499) | ✅ (0.489) | ✅ (0.529) | ❌ (0.592) |
| 1992 | William Clinton   | Democratic  | ✅ (0.484) | ✅ (0.146) | ✅ (0.007) | ✅ (0.333) |
| 1988 | George Bush       | Republican  | ✅ (0.509) | ✅ (0.957) | ✅ (0.790) | ✅ (0.837) |
| 1984 | Ronald Reagan     | Republican  | ✅ (0.511) | ✅ (0.655) | ✅ (0.771) | ❌ (0.469) |
| 1980 | Ronald Reagan     | Republican  | ✅ (0.519) | ✅ (0.655) | ✅ (0.811) | ✅ (0.538) |
| 1976 | Jimmy Carter      | Democratic  | ❌ (0.504) | ❌ (0.880) | ❌ (0.669) | ❌ (0.700) |
| 1972 | Richard Nixon     | Republican  | ✅ (0.503) | ✅ (0.680) | ❌ (0.629) | ✅ (0.543) |
| 1968 | Richard Nixon     | Republican  | ✅ (0.505) | ✅ (0.670) | ✅ (0.943) | ✅ (0.674) |
| 1964 | Lyndon B. Johnson | Democratic  | ✅ (0.488) | ✅ (0.130) | ✅ (0.284) | ✅ (0.257) |
| 1960 | John F. Kennedy   | Democratic  | ✅ (0.483) | ✅ (0.120) | ✅ (0.133) | ✅ (0.253) |

#### Key patterns

**1. Model confidence varies dramatically**

Logistic Regression exhibits remarkably low confidence across all predictions, with probabilities clustering tightly around 0.5 (range: 0.483–0.519). The model achieves 88.2% accuracy whilst expressing near-complete uncertainty about individual predictions. In contrast, Random Forest and SVC demonstrate strong confidence for most elections, with probabilities often exceeding 0.65 or falling below 0.35.

**2. The 2016 anomaly**

2016 Trump represents the most challenging prediction:
- **LR**: 0.498 probability Republican (predicted Democratic by 0.2 percentage points)
- **RF**: 0.140 probability Republican (predicted Democratic with 86% confidence)
- **SVC**: 0.494 probability Republican (predicted Republican by 0.6 percentage points) 
- **XGBoost**: 0.276 probability Republican (predicted Democratic)

SVC succeeded where others failed, but just barely. This suggests 2016's outcome was nearly orthogonal to macro socioeconomic indicators.

**3. Consensus predictions**

Elections with near-unanimous model agreement and high confidence:
- **1988 Bush (R)**: RF=0.957, SVC=0.790, XGB=0.837 — strong Republican economic signals
- **1992 Clinton (D)**: RF=0.146, SVC=0.007 — overwhelming Democratic indicators
- **2024 Trump (R)**: SVC=0.982 — exceptionally strong Republican signal
- **1960 Kennedy (D)**: RF=0.120, SVC=0.133 — clear Democratic economic conditions

**4. XGBoost underperformance**

Whilst LR, RF, and SVC each made 2–3 errors, XGBoost misclassified 4 elections (2016 Trump, 1996 Clinton, 1984 Reagan, 1976 Carter). The model's more complex architecture appears to overfit the limited training data despite regularisation.

**5. Near-50% probabilities signal political volatility**

Elections where models assigned probabilities near 0.5 often involved exceptional political circumstances:
- **1996 Clinton**: LR=0.499, RF=0.489 (Clinton's re-election during political scandal)
- **2008 Obama**: LR=0.498 (financial crisis created conflicting economic signals)
- **1976 Carter**: LR=0.504 (post-Watergate upheaval)

These near-equiprobable predictions suggest the macro socioeconomic indicators provided genuinely ambiguous signals, accurately reflecting the political uncertainty of these election years.

### Misclassifications

The three top-performing models made nearly identical errors:

**2016 – Donald Trump (Republican)**
- LR predicted: Democratic (50% confidence Republican)
- RF predicted: Democratic (14% confidence Republican)
- SVC predicted: Republican (49% confidence Republican) ✓
- XGB predicted: Demoncratic (
- Analysis: Economic conditions favoured Democrats. Trump's victory was driven by factors not captured in macro socioeconomic aggregates (populist sentiment, cultural polarisation, media dynamics).

**1976 – Jimmy Carter (Democratic)**
- LR predicted: Republican (50% confidence Republican)
- RF predicted: Republican (88% confidence Republican)
- SVC predicted: Republican (67% confidence Republican)
- Analysis: Post-Watergate and post-Vietnam political upheaval created exceptional circumstances. Carter's outsider candidacy succeeded despite economic indicators favouring Republicans.

**Additional model-specific errors:**
- **1972 Nixon (R)**: SVC incorrectly predicted Democratic (63% confidence)
- **XGBoost** additionally missed 1996 Clinton, 1984 Reagan

### Feature importance analysis

Feature importance was aggregated across all four optimized models. For interaction terms, importance was split equally between the two base features involved. The comparison reveals consistent patterns across different model architectures:

![Feature Importance Comparison](images/final_feature_importance_comparison.png)

**Key findings:**

1. **Gross capital formation** emerges as the dominant predictor across all models, particularly for Random Forest (normalized importance ~0.24). Investment in fixed assets and economic confidence strongly correlate with electoral outcomes.

2. **Unemployment** shows the second-highest importance (0.14-0.16 across models), with consistent rankings suggesting its robust predictive power.

3. **Inflation GDP deflator** ranks highly in tree-based models (RF, XGB) but shows more modest importance in linear models (LR, SVC).

4. **Population growth** and **fertility rate** contribute moderately across all models, capturing demographic trends.

5. Model agreement on feature rankings strengthens confidence that these economic indicators genuinely drive electoral outcomes rather than being artifacts of individual model architectures.

### Economic Conditions and Electoral Outcomes

Box plots comparing the 8 base features used in the final models between Democratic and Republican victories reveal systematic patterns:

![Economic Indicators by Outcome](images/box_2x4_indicators_vs_outcome.png)

Notable patterns:
- **Unemployment** tends to be higher in Democratic victory years (median ~6% vs ~5%)
- **Inflation GDP deflator** shows distinct distributions, with Republican victories associated with lower inflation
- **Gross capital formation** demonstrates clear separation between outcomes, with higher investment rates correlating with Republican victories
- **Population growth** and **fertility rate** show demographic differences between party victory years
- **Energy use** per capita varies between Democratic and Republican administrations

### Conclusions

1. **Macro socioeconomic indicators are strong but imperfect predictors** of US presidential election outcomes, achieving 88.2% accuracy over 17 elections spanning 64 years.

2. **Model convergence signals a performance ceiling.** Three independent algorithms with different architectures all converged on 88.2% accuracy, suggesting this represents the maximum achievable performance from macro socioeconomic data alone. Further improvement requires non-economic features.

3. **Sample size remains the fundamental limitation.** With only 17 observations, models are constrained by limited training data despite using LOOCV and shallow architectures to prevent overfitting.

4. **Feature engineering improved performance substantially.** Interaction terms captured complex relationships between economic variables, whilst feature selection reduced noise. Baseline performance of 58–71% improved to 88.2% through systematic optimisation.

5. **Exceptional political circumstances override economic signals.** Both misclassifications (2016 Trump, 1976 Carter) involved unique political contexts—populist movements and post-Watergate upheaval—that macro socioeconomic indicators cannot capture.

6. **Investment and inflation dominate predictions.** Gross capital formation (~24% normalized importance across models) and inflation measures drive model decisions, suggesting economic stability and business confidence strongly influence electoral preferences. The consistency of these rankings across all four model types (LR, RF, SVC, XGB) validates their importance.

### Limitations and Future Work

**Fundamental limitations:**
- **Temporal autocorrelation:** Elections are not independent; political momentum and demographic shifts create dependencies not addressed by LOOCV.
- **Omitted variable bias:** Cultural factors, foreign policy, candidate quality, campaign effectiveness, and media dynamics are not captured.
- **Limited generalisability:** Predictions for pre-1960 elections or fundamentally different economic regimes would be unreliable.
- **Small sample size:** With n=17, even LOOCV cannot fully mitigate overfitting risk.

**Potential improvements:**
- Incorporate polling data, demographic trends, or sentiment analysis from news/social media
- Expand to state-level predictions (50 states × 17 elections = 850 samples)
- Implement time-series aware validation strategies that respect temporal ordering
- Ensemble economic and non-economic features with regularisation
- Extend dataset backwards to 1900s where feasible, despite data quality challenges
