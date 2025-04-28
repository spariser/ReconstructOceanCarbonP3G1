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
