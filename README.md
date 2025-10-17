# US presidential election analysis 

**Date:** December 2024  
**Analysis period:** 1960-2024 (17 presidential elections)

---

## Overview

This analysis examines the relationship between macroeconomic indicators and US presidential election outcomes using machine learning. The study identified **unemployment rates** and **inflation rates** as the most predictive economic factors for election results, achieving **58.8% prediction accuracy** with logistic regression.

---

## Methodology

### Data sources
- **Election data**: Historical presidential election results (1960-2024)
- **Economic indicators**: 
  - Unemployment rates (FRED data)
  - Inflation rates (World Bank data)
  - GDP and population metrics
- **Political classification**: Binary outcome (Left=Democratic, Right=Republican)

### Machine learning approach
- **Feature Selection**: Correlation-based selection (threshold: 30%)
- **Validation**: Leave-One-Out Cross-Validation (LOOCV)
- **Models tested**: Logistic Regression, Random Forest, Support Vector Machine
- **Evaluation metric**: Prediction accuracy

---

## Results

### 1. Economic predictors identified

| Rank | Feature | Correlation with outcome | Model importance |
|------|---------|----------------------------------|------------------|
| 1 | **Unemployment Rate** | High | 0.977 (Logistic) |
| 2 | **Inflation Rate** | High | 0.797 (Logistic) |

### 2. Model performance comparison

| Model | Accuracy | Best feature | Second feature |
|-------|----------|--------------|----------------|
| **Logistic Regression** | **58.8%** | Unemployment (0.977) | Inflation (0.797) |
| Random Forest | 58.8% | Unemployment (0.501) | Inflation (0.499) |
| SVM (Linear) | 47.1% | Unemployment (0.923) | Inflation (0.555) |

---

## Historical analysis

### Economic trends by political outcome

#### Democrat victories (left leaning)
- **Average unemployment**: 6.7%
- **Average inflation**: 2.8%
- **Notable periods**: 1992-1996 (Clinton), 2008-2012 (Obama), 2020 (Biden)

#### Republican victories (right leaning)
- **Average unemployment**: 5.4%
- **Average inflation**: 5.9%
- **Notable periods**: 1980-1988 (Reagan), 2000-2004 (Bush), 2016 (Trump)

### Insights

1. **High unemployment periods** (7%+) tend to favor Democratic candidates
   - 2020: 8.1% unemployment → Biden victory
   - 1992: 7.5% unemployment → Clinton victory
   - 1976: 7.7% unemployment → Carter victory

2. **High inflation periods** (10%+) strongly favor Republican candidates
   - 1980: 13.5% inflation → Reagan landslide victory

3. **Economic stability** periods show mixed results, suggesting other factors become more important

---

## Insights

### Strong economic indicators
- **Unemployment > 7%**: Strong Democratic advantage
- **Inflation > 5%**: Moderate Republican advantage

### Limitations 
- **Accuracy**: 58.8% suggests economic factors explain about 60% of election variance
- **Sample size**: 17 elections limits statistical power, it is challenging to find annual economic data before 1960
- **Missing factors**: Social issues, political policy, candidate charisma, and so on 

---

## Summary

This analysis demonstrates that the **macroeconomic indicators unemployment and consumer price inflation rates are reasonable predictors of US presidential election outcomes**. The 58.8% prediction accuracy, while not perfect, represents some improvement over random chance (50%) and provides some sensible insights which align with economic theory. 

---

**Note**: This is a work in-progress.  