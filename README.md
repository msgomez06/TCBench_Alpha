# TCBench - Alpha Repository
 
Welcome to TCBench, our platform and benchmark dataset for data-driven tropical cyclone studies. For more information on how to use the repository, please [check out the GitHub Wiki](https://github.com/msgomez06/TCBench_Alpha/wiki).

Graphical Summary of TCBench:
![TCBench Graphical Summary](https://github.com/msgomez06/TCBench_Alpha/blob/main/Figure_Summary.png?raw=True)

## Background

Coastal risks and vulnerability to tropical cyclone (TC)-driven heavy rainfall, flooding, and storm surge are increasing due to slower, wetter TCs; poleward expansion of maximum potential TC intensity; and growing population near coasts. However, predicting TC intensity variations remains a challenge even for major global storm forecasting centers due to complex storm dynamics. In particular, the prediction of rapid intensification remains especially challenging. TC intensity has been linked with TC rain rates, but the scaling and spatial variability of this relationship is still being studied. Still, as the climate warms both wind speeds and rainfall rates are projected to increase and thus a strong foundational dataset for tropical cyclone intensity and precipitation predictions is crucial for future risk management and coastal resilience.

We note that the World Meteorological Organization has, as part of its TC Programme, designated ten Regional Specialized Meteorological Centers for operational TC forecasting. While these incorporate satellites, statistical and numerical models to monitor and forecast cyclogenesis and intensification, it is currently difficult to compare ML models that aim to improve TC predictions to the appropriate observations and previous predictions. This is a gap we hope to fill by fusing the archives from several RSMCs into a single data repository.

Machine Learning (ML) models are widely used to improve TC genesis, track, and intensity predictions. However, even though some models are able to outperform NHC operational prediction of RI over the Atlantic and East Pacific ocean basins, most ML groups use standard input lists (e.g., those provided by the NHC). As such, which predictors are most predictive of TC intensity changes remains an open question. Furthermore, an overarching issue in studies applying modern ML to TC intensity prediction is that these studies usually define different targets and evaluation metrics, preventing an objective comparison of different frameworks. This further motivates grouping standard tropical meteorology definitions and evaluation protocols in a unified data repository, e.g., by creating a TC benchmark dataset for ML applications.

The use of benchmark datasets is further motivated by the fact that several scientific domains have shown that defining a benchmark dataset helps identify optimal ML solutions for a specific challenge by enabling cross-institutional collaborations and healthy competition. In atmospheric science, WeatherBench, which frames weather forecasting from global reanalysis as a statistical regression problem, has encouraged scientific reflection on atmospheric dynamics, scale interaction, predictability, and uncertainty that goes beyond the eight entries on its leaderboard. More recent attempts use sophisticated ML methods, such as graph neural networks and adaptive Fourier neural operators, confirming the innovation potential of such benchmark datasets. However, even though ML benchmark datasets for weather and climate applications are progressively appearing in response to community enthusiasm, few focus on extreme events despite their critical importance for operations.

## Where TCBench comes in

Our goal, then, is to provide open, user-friendly data processing tools, evaluation protocols, visualization tools, and baseline prediction models to benefit the atmospheric science and AI communities. By facilitating a unified evaluation of ML models of tropical cyclones, we hope to give the scientific community a clear path towards developing skillful data-driven predictions of tropical cyclones for both present and future climates.

TCBench leverages existing datasets to create an integrated, value-added dataset spanning 1980-Present. Additionally, it provides tools to seamlessly integrate new data sources and pre-process them in a consistent matter across studies. It targets the development and refinement of data-driven models for tropical cyclone prediction, including precipitation downscaling and intensity predictions. Tropical cyclones are some of the highest-impact weather events in affected regions, and we thus strive to support the development of a homogenized, publicly available, database of high-impact events. Our efforts include the optimized use of multiple datasets by fusing reanalysis and observational datasets into a comprehensive benchmark dataset in an expandable framework for experimental flexibility.  

We plan to include state-of-the art uncertainty quantification methods designed for AI models of geospatial data and, following best practices, to contain open-access, AI-ready datasets that follow FAIR principles and will provide clear evaluation protocols for AI models. Finally, tropical cyclone datasets often require complex pre-processing, hindering their use by AI experts without domain knowledge. By providing homogenized data and well-defined evaluation protocols, TCBench will advance the use of AI for Earth sciences. It will facilitate the application and evaluation of sophisticated AI frameworks for tropical cyclone predictions.

In summary, TCBench aims to provide opportunities to study the predictability of tropical cyclones (and changes in behavior associated with changing climate), as well as developing a dataset and evaluation tools that can be used freely by the scientific community.

## Getting Started

TCBench is in its initial stages, so it's currently not meant to be installed. The \_\_init\_\_ files have, however, been made and this might work. Give it a shot if you're feeling brave 😄

### Folder Structure
```
├── dev
│   ├── __init__.py
│   ├── primed_loader.py
│   ├── track_loader.py
│   ├── track_test.py
│   └── utils
│       ├── constants.py
│       ├── __init__.py
│       └── toolbox.py
├── __init__.py
├── LICENSE
├── README.md
└── tracks
```
`\dev` contains all of the python scripts that you will need to test out TCBench's data preprocessing.  
`\dev\utils\constants.py` holds reference values for the operations carried out by TCBench. This includes classes describing the tracks provided by, e.g., IBTrACS, which facilitate track processing.  
`\dev\utils\toolbox.py` includes functions and classes used to process the tracks. Of note is the tc_track class, which is the current cornerstone of TCBench. 
