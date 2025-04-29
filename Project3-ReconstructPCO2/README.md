# Ocean Mixing Group 1: exploring model statistical confidence & adding new data

This project aims to better understand the statistical significance of reconstructing global pCO₂ fields using ML trained on sparse data. It extends upon the analysis carried out by Gloege et al. (2021) “Quantifying Errors in Observationally Based Estimates of Ocean Carbon Sink Variability.” and the method of Bennington et al. (2022) “Explicit Physical Knowledge in Machine Learning for Ocean Carbon Flux Reconstruction: The pCO2-Residual Method” by introducing a new ML method, NGBoost, which gives a probablistic prediction instead of XGBoost’s point prediction. By introducting probablistic information, including standard deviations, scientists can understand the statistical confidence of models and how that confidence changes with the introduction of additional data.

## **Motivation**

Probablistic predictions provide additional understanding of current models based on this sparse and uneven observations versus point-based predictions. By better discerning confidence in pCO2 reconstructions, scientists can more accurately convey recommendations based on the models, such as more accurately answer questions related to carbon budgets and other scientific and socio-economic questions related to climate change. Additionally, this research can inform scientific funding, helping to answer whether more real-world pCO₂ observations are needed for more accurate models.

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
│   └── visualization.py       # Custom plotting class SpatialMap2 for creating high-quality spatial visualizations with colorbars and map features using Cartopy and Matplotlib.
├── notebooks/
│   └── Project3_data_story.ipynb # Main notebook containing full analysis & data story
|   ├── leappersistent_file_management.ipynb # check the size of files and clean up
|   ├── config.py # Configuration file for the project 
```

## **Team & Contribution**

Group 1:
- Sarah Pariser:  Created masks; streamlined code/story; contributed to written introduction & conclusion
- Azam Khan: Setup pipeline to train ngboost models; wrote visualization functions & analysis
- Bokai He: statistical significance analysis, including p-value & t-test; NGBoost & XGBoost comparison
- Victor Wang: Seasonality analysis & visualization

