# UK Multi-Scale Urban Mobility Analysis

## Overview
The repository contains a series of notebooks and scripts for data preprocessing, indicator calculation, spatial modelling (e.g., MGWR), and map visualisation. The analysis is divided into two main components based on the data source.

## Data Availability

Please note: Due to ethical considerations and the private nature of the datasets, **the raw data used in this analysis cannot be published in a public repository.**


## Repository Structure & Files

### Commuting Mobility Analysis

This section focuses on the analysis of the commuting dataset.

* **`OD_commuting.ipynb`**: The primary Jupyter Notebook for the commuting data. It contains the complete workflow, from **data preprocessing and cleaning to the main structural analysis** of the origin-destination flows.
* **`Intra_mobility_indicators_commuting.py`**: A Python script used to calculate intra-city mobility indicators (Centripetality, Anisotropy) from the processed commuting data.

### General Mobility Analysis

This section focuses on the analysis of the general mobility dataset.

* **`OD_general_preprocess.ipynb`**: A Jupyter Notebook for the initial preprocessing and cleaning of the general mobility origin-destination data.
* **`Intra_mobility_indicators_general.py`**: A Python script to calculate intra-city mobility indicators (Centripetality, Anisotropy) from the processed general mobility data.
* **`inter_general.ipynb`**: A Jupyter Notebook for conducting the inter-city network percolation analysis.
* **`MGWR_general.ipynb`**: Implements Multiscale Geographically Weighted Regression (MGWR) to model spatial relationships.
* **`map_visualization_general.ipynb`**: A Jupyter Notebook to create map-based visualisations of the analysis results.



