# TimeSeriesMasters2023
One-year project within the Master's program of the National Research University Higher School of Economics "Machine Learning and Highly Loaded Systems".

## Project Description
The project has two objectives: 
1. to investigate possible multi-point-ahead prediction strategies in the presence of one or many time series in a dataset;
2. to finalize an open-source [time series analysis tool](https://github.com/sb-ai-lab/tsururu).

## Data description:
Since our project is aimed at developing a tool for time series forecasting, we needed to find data on which we could conveniently and clearly demonstrate certain features of the tool.

In the first phase of the project, we looked at quite a lot of public data, which can be categorized into two sources:
1. [Monash Time Series Repository](https://forecastingdata.org/) ([Godahewa (et al.), 2021](https://arxiv.org/abs/2105.06643)):
2. [Datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) that are most commonly found in articles about state-of-the-art architectures.
    - __NOTE__: The data in the Monash Repository is initially provided in .tsv format. There is a function in `data/data_formatting.py` to convert them to .csv format (source: https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py). There is also a function that translates datasets from the second item into the required format.

All datasets we consider can be found [here](https://disk.yandex.ru/d/jv8JiSWiSqao5w). 

The list and main characteristics of the datasets we reviewed can be found [here](https://docs.google.com/spreadsheets/d/1JILNPfSjYnumt_GqDQ4rmgW3Zxeg6KEhkaFU21vOVyI/edit#gid=0). And the analysis of these datasets is in `EDA/EDA_other_datasets.ipynb`.

Among these data we have selected __ETT__ dataset, which "contains the 7 indicators of an electricity transformer in two years, including oil temperature, useful load, etc." and is convenient for analysis, as it contains rather diverse time series, containing both trend and seasonality, and also has rather frequent granularity (15 minutes). Its analysis is located separately in `EDA/EDA_ETT.ipynb`.

## Project workplan
*This section should be supplemented.*

## Team participants
- Kostromina Alina (@elineii)
- Larchenkov Mikhail (@laplan)
- Zuikova Olga (@s21_fernando)
- Karagodin Nikita (@Einstein_30)
