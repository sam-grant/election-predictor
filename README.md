# US presidential election economic analysis

**Analysis period:** 1960-2024 (17 elections)  

---

## Overview

Can macroeconomic indicators predict US presidential election outcomes? This analysis uses machine learning to with unemployment and inflation rates as predictors, achieving **58.8% accuracy** with logistic regression - some improvement over random chance (50%).

---

## Key Findings

### Top economic predictors
1. **Unemployment Rate** - Most important predictor (importance: 0.977)
2. **Inflation Rate** - Second most important (importance: 0.797)

### Historical patterns
- **High unemployment (>7%)** → Favors Democrats
  - Examples: 2020 Biden (8.1%), 1992 Clinton (7.5%), 1976 Carter (7.7%)
  
- **High inflation (>10%)** → Favors Republicans  
  - Example: 1980 Reagan landslide (13.5% inflation)

### Average economic conditions by winner
| Party | Avg unemployment | Avg inflation |
|-------|------------------|---------------|
| **Democratic** | 6.7% | 2.8% |
| **Republican** | 5.4% | 5.9% |

---

## Methodology

**Data sources:**
- Election results: 1960-2024 presidential elections
- Economic data: FRED (unemployment), World Bank (inflation, GDP, population)

**Machine learning:**
- Models: Logistic Regression, Random Forest, SVM
- Validation: Leave-One-Out Cross-Validation (LOOCV)
- Feature selection: Correlation threshold >30%

**Results:**
| Model | Accuracy |
|-------|----------|
| Logistic Regression | **58.8%** |
| Random Forest | 58.8% |
| SVM (Linear) | 47.1% |

---

## Limitations

- **Small sample**: 17 elections limits statistical power
- **Missing variables**: Candidate charisma, campaign strategy, social issues, foreign policy
- **Correlation ≠ causation**: These are associations, not causal relationships
- **Limited improvement**: 58.8% vs 50% baseline suggests ~40% of variance remains unexplained

---

## Technical details

See `ana/ana.ipynb` for full analysis including:
- Data preprocessing and cleaning
- Feature correlation analysis  
- Model training and evaluation
- Visualisations

**Libraries:** pandas, numpy, scikit-learn, matplotlib