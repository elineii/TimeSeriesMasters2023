# TimeSeriesMasters2023
One-year project within the Master's program of the National Research University Higher School of Economics "Machine Learning and Highly Loaded Systems".

## Project Description
The project has two objectives: 
1. to investigate possible multi-point-ahead prediction strategies in the presence of one or many time series in a dataset;
2. to finalize an open-source [time series analysis tool](https://github.com/sb-ai-lab/tsururu).

## Data Description:
Since our project is aimed at developing a tool for time series forecasting, we needed to find data on which we could conveniently and clearly demonstrate certain features of the tool.

In the first phase of the project, we looked at quite a lot of public data, which can be categorized into two sources:
1. [Monash Time Series Repository](https://forecastingdata.org/) ([Godahewa (et al.), 2021](https://arxiv.org/abs/2105.06643)):
2. [Datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) that are most commonly found in articles about state-of-the-art architectures.
    - __NOTE__: The data in the Monash Repository is initially provided in .tsv format. There is a function in `data/data_formatting.py` to convert them to .csv format ([source](https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py)). There is also a function that translates datasets from the second item into the required format.

All datasets we consider can be found [here](https://disk.yandex.ru/d/jv8JiSWiSqao5w). 

The list and main characteristics of the datasets we reviewed can be found [here](https://docs.google.com/spreadsheets/d/1JILNPfSjYnumt_GqDQ4rmgW3Zxeg6KEhkaFU21vOVyI/edit#gid=0). And the analysis of these datasets is in `EDA/EDA_other_datasets`.

Among these data we have selected __ETT__ dataset, which "contains the 7 indicators of an electricity transformer in two years, including oil temperature, useful load, etc." and is convenient for analysis, as it contains rather diverse time series, containing both trend and seasonality, and also has rather frequent granularity (15 minutes). Its analysis is located separately in `EDA/EDA_ETT.ipynb`. If necessary, the use of other datasets is assumed (e.g., to consider series with pronounced patterns of seasonality).

## Project Workplan
The project can be divided into 4 parts according to the number of participants. Each of us has more or less independent from each other field of activity and tasks.

1. ML-models and transformations (@laplan)
    - Implement Baseline models in the model module.
    - Implement ML models (linear regression, tree-based) in the model module.
    - Model combination and model ensemble.
    - Combining different transforms, storing parameters of transformers, implementing forward and backward (transform, inverse_transform).
    - Optional: accounting for heuristics (not ML techniques).
2. Neural Networks (@Einstein_30)
    - Consider available architectures for solving time series forecasting problems
    - Implement state-of-the-art neural network architectures in the model module and general learning pipelines. 
3. Hierarchical series and clustering (@s21_fernando)
    - Review the literature on time series hierarchy (building a combination of different models at different levels of the hierarchy that differ in granularity).
    - Implement as a wrapper (preset) on an existing library.
    - Optional: partitioning of time series into useful and not useful.
4. MLOps, library support, strategies (@elineii):
    - Consolidating results and helping with implementation, collecting technical comments and fixing bugs, improving usability. 
    - Service schema and API (Streamlit).
    - MLOps: poetry, pre-commit, actions, etc. 
    - Selection of optimal prediction horizon for recursive and direct strategy mixed with multi-input-multi-output.

## Team Participants
- Kostromina Alina (@elineii)
- Larchenkov Mikhail (@laplan)
- Zuikova Olga (@s21_fernando)
- Karagodin Nikita (@Einstein_30)
