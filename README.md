# Ocean Surface pCO₂ Reconstruction with NGBoost

**Authors** 

**Group 1:** Azam Khan, Bokai He, Sarah Pariser, Zhi Wang

## Contribution Statement:

- **Azam Khan**: Setup pipeline to train ngboost models; wrote visualization functions & analysis
- **Bokai He**: Statistical significance analysis, including p-value & t-test; NGBoost & XGBoost comparison
- **Sarah Pariser**: Created masks; streamlined code/story; contributed to introduction & conclusion
- **Zhi Wang**: Seasonality analysis & visualization



Course: EESC/STAT 4243 – *Climate Prediction Challenges with Machine Learning*  
Spring 2025

---

## Overview

This project investigates the use of **probabilistic machine learning** to reconstruct surface ocean partial pressure of CO₂ (pCO₂) from sparse observations. By implementing **NGBoost**, we quantify uncertainty in pCO₂ estimates and examine how adding more data — either at existing locations or in new regions — improves model confidence and performance.

The project builds on:
- **Gloege et al. (2021)** – quantifying uncertainty in pCO₂ reconstructions
- **Bennington et al. (2022)** – residual-based ML reconstructions

---

## Motivation

Oceans have absorbed about **38%** of anthropogenic CO₂ emissions since the industrial revolution, making them a vital carbon sink. However, future oceanic carbon uptake remains uncertain.

A major barrier to understanding ocean-atmosphere carbon flux is the **sparse and uneven distribution of pCO₂ observations**, particularly in the Southern Hemisphere and high-latitude oceans.

This study addresses two key questions:
1. **Where are reconstructions statistically reliable or uncertain?**
2. **Does adding more observations — in quantity or coverage — improve confidence?**

---

## Objectives

- ✅ Quantify uncertainty and statistical confidence of ML-based pCO₂ reconstructions  
- ✅ Investigate the impact of **different sampling strategies**:
  - Adding more data at existing locations
  - Adding new observations in previously unsampled areas  
- ✅ Identify regions where new data would improve model skill the most  
- ✅ Evaluate model performance via metrics and spatial/temporal diagnostics

---

## Model: NGBoost vs XGBoost

| Feature                    | NGBoost                          | XGBoost                    |
|---------------------------|----------------------------------|----------------------------|
| Output                    | Probabilistic (e.g. N(μ, σ²))    | Point prediction           |
| Uncertainty Estimation    | ✅ Built-in                      | ❌ Not native              |
| Loss Function             | LogScore, CRPS                   | MSE, LogLoss               |
| Use Case                  | Climate, medical, risk-sensitive | General-purpose            |

We use **NGBoost + Normal distribution + LogScore** to predict both the **mean** and **standard deviation** of pCO₂ values.

---

## Sampling Scenarios

We compare the **baseline SOCAT sampling mask** to six enhanced sampling strategies:

### 1.Add More at Existing Locations

| Mask Name              | Description                                     | Increase |
|------------------------|-------------------------------------------------|----------|
| `densify_mean_pattern` | Raise low-sampled locations to global mean      | +14%     |
| `densify_30p`          | Ensure ≥ 7 months per sampled grid cell         | +30%     |
| `densify_50p`          | Ensure ≥ 10 months per sampled grid cell        | +50%     |

### 2.Add Data in New Locations

| Mask Name      | Description                                             | Increase |
|----------------|---------------------------------------------------------|----------|
| `expand_14p`   | Add new grid cells in S. Ocean, Indian, Pacific regions | +14%     |
| `expand_30p`   | 100 new points per basin, moderate sampling             | +30%     |
| `expand_50p`   | 200 new points per basin, dense sampling                | +50%     |

---

## Methodology Summary

- **Sampling Mask Analysis** – visualize SOCAT coverage & define augmentation strategies  
- **Train NGBoost** – on residuals (ESM minus truth) using SOCAT-like masks  
- **Reconstruct pCO₂** – for the globe using trained NGBoost  
- **Inverse Transformation** – recover full pCO₂ from residuals  
- **Evaluate** – spatial metrics (bias, std, corr), uncertainty, p-values  
- **Compare** – baseline vs. augmented masks across key metrics

---

## Peer Review Instructions:

1. Clone the repository:
    ```bash
    git clone https://github.com/spariser/ReconstructOceanCarbonP3G1.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ReconstructOceanCarbonP3G1
    ```
3. Run the cells in the `Project3_data_story.ipynb` notebook to reproduce the analysis.
4. Enter your username in the `username` variable in the first cell of the notebook.
5. Ensure `runthiscell` is set to **-1** as a reviewer to reduce time.



## **Project Structure**


```bash
Project3/
├── lib/                       # Helper scripts
│   ├── __init__.py
│   ├── bias_figure2.py        # Code for bias calculation and visualization
│   ├── corr_figure3.py        # Code for correlation calculation and visualization
│   ├── residual_utils.py      # Prepares data for ML, tools for dataset splitting, model evaluation, and saving files.
│   ├── group1_utils.py       # Group 1: Functions for data preprocessing, including loading and cleaning data, and creating training and test datasets.
│   ├── config.py # Configuration file for the project containing constants
│   └── visualization.py       # Custom plotting class SpatialMap2 for creating high-quality spatial visualizations with colorbars and map features using Cartopy and Matplotlib.
├── notebooks/
│   └── Group1_data_story.ipynb # Main notebook containing full analysis & data story
```

