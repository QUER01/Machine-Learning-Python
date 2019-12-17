#!/bin/sh


conda update conda
conda config --add channels conda-forge
conda create --name TutorialSpatialDataAnalysis -y python=3 pandas numpy matplotlib bokeh seaborn scikit-learn jupyter
pip install ipykernel
conda install --name TutorialSpatialDataAnalysis -y geopandas==0.2 mplleaflet==0.0.5 datashader==0.2.0 geojson cartopy==0.14.2 folium==0.2.1

ipython kernel install --user --name=TutorialSpatialDataAnalysis
jupyter-notebook
