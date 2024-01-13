import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional

import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from tslearn.clustering import TimeSeriesKMeans, silhouette_score


def describe_ts(df, id_column="id", value_column="value", date_column="date"):
    freq = pd.infer_freq(df[df[id_column] == df[id_column].unique()[0]][date_column])
    
    if freq == "H":
        freq = "hourly"
    elif freq == "D":
        freq = "daily"
    elif freq == "W":
        freq = "weekly"
    elif freq == "M":
        freq = "monthly"
    elif freq == "Y":
        freq = "yearly"
    else:
        freq = "unknown"
    
    print(f'Число рядов: {df[id_column].nunique()}')
    print(f'Наблюдений в ряде: {df[id_column].value_counts().unique()}')
    print(f'Частота ряда: {freq}')
    print(f'Минимальная дата в ряде: {df[date_column].min()}')
    print(f'Максимальная дата в ряде: {df[date_column].max()}, \n')
    print(f'{df[value_column].describe()}')


def draw_resampled(df, freq, n_series, seed=42, id_column="id", value_column="value", date_column="date"):
    df_resample = df.copy()
    df_resample = df_resample.set_index(date_column)
    df_resample = df_resample.groupby(id_column).resample(freq)[value_column].agg('sum').reset_index()
    
    np.random.seed(seed)
    random_id = np.random.choice(df_resample[id_column].unique(), n_series)
    
    if n_series == 1:
        _ = plt.figure(figsize=(24, 6))
        plt.plot(df_resample[df_resample[id_column] == random_id[0]][date_column], df_resample[df_resample[id_column] == random_id[0]][value_column])
    else:
        _, ax = plt.subplots(n_series, figsize=(24, 20))
        random_id = np.random.choice(df_resample[id_column].unique(), n_series)
        
        for i in range(n_series):
            ax[i].plot(
                df_resample[df_resample[id_column] == random_id[i]][date_column], 
                df_resample[df_resample[id_column] == random_id[i]][value_column]
            )
    plt.show()
    
    
def cluster_ts(
    data: pd.DataFrame, 
    metric: str = "euclidean", 
    optimise_n_clusters: bool = False,
    n_clusters: Optional[int] = 10, 
    plot_silhouette_score: bool = False,
    id_column: str = "id", 
    date_column: str = "date",
    random_state: int = 42,
    n_jobs: int = -1
):
    """Cluster time series using KMeans algorithm.
    If optimise_n_clusters is True, the optimal number of clusters is found using the silhouette score.
    
    Arguments:
        - data: the time series dataframe
        - metric: the metric used for clustering, one of ["euclidean", "dtw", "softdtw"]
        - optimise_n_clusters: whether to optimise the number of clusters
        - n_clusters: the number of clusters if optimise_n_clusters is False 
            or the maximum number of clusters if optimise_n_clusters is True
        - plot_silhouette_score: whether to plot the silhouette score if optimise_n_clusters is True
        - id_column: the name of the id column
        - date_column: the name of the date column
        - random_state: the random state
        - n_jobs: the number of jobs to run in parallel
    
    Returns:
        - data: the dataframe with the cluster column
    """
    def _find_optimal_n_clusters(data_scaled, metric, max_n_clusters, plot_silhouette_score, random_state, n_jobs):
        scores = []
        for n_clusters in range(2, max_n_clusters+1):
            kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state, n_jobs=n_jobs)
            clusters = kmeans.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, clusters)
            scores.append(score)
        if plot_silhouette_score:
            plt.plot(range(2, max_n_clusters+1), scores)
            plt.xlabel('n_clusters')
            plt.ylabel('silhouette_score')
            plt.title('Silhouette Score vs n_clusters')
            plt.show()
        optimal_n_clusters = np.argmax(scores) + 2
        return optimal_n_clusters

    # Подготавливаем данные для кластеризации
    data = data.copy()
    data_scaled = data.pivot(index=date_column, columns=[id_column])
    
    # Масштабируем данные
    ss = StandardScaler()
    data_scaled = ss.fit_transform(data_scaled)
    
    # Кластеризуем TS
    data_scaled = data_scaled.T
    
    if optimise_n_clusters:
        n_clusters = _find_optimal_n_clusters(data_scaled, metric, n_clusters, plot_silhouette_score, random_state, n_jobs)

    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state, n_jobs=n_jobs)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Добавляем новый столбец в исходный датафрейм
    data['cluster'] = np.repeat(clusters, sum(data[id_column] == data[id_column].iloc[0]))
    return data


def plot_clustered_series(data:[pd.DataFrame], n_clusters:[int], id_column="id", value_column="value", date_column="date", n_series=10, seed=42):
    np.random.seed(seed)
    _, axs = plt.subplots(n_clusters, 1, figsize=(24, 4*n_clusters))

    # Проходим по всем кластерам
    for i in range(n_clusters):
        # Выбираем данные, относящиеся к текущему кластеру
        cluster_data = data[data['cluster'] == i]
        random_id = np.random.choice(cluster_data[id_column].unique(), n_series)
        
        for current_id in random_id:
            # Выбираем данные, относящиеся к текущему временному ряду
            current_series = cluster_data[cluster_data['id'] == current_id]
            axs[i].plot(current_series[date_column], current_series[value_column])
    
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axs[i].set_title(f'Кластер {i}')
        axs[i].set_xlabel('Дата')
        axs[i].set_ylabel('Количество')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()


# https://medium.com/vortechsa/detecting-trends-in-time-series-data-using-python-2752be7d1172
def calculate_tau(df,location_col,quantity_col,location_list):
    '''
    This function will loop through the location list and compute tau statistics for each zone.
        
    Parameters:
    df: time series dataframe
    location_col: the group by location column name
    quantity_col: aggregated quantity column name by certain frequency
    location_list: Location to loop through
    
    '''
    location_pvalue_dict = {}
    for location in location_list:
        df2 = df[df[location_col] ==  location].reset_index(drop = True)
        df2[quantity_col] = df2[quantity_col].replace('',0)
        # Extract the time values and reshape them to a 2D array
        X = df2.index.values.reshape(-1, 1)

        # Extract the time series values and reshape them to a 1D array
        y = df2[quantity_col].values.reshape(-1, 1)
        
        # Compute tau statistics
        tau, p_value = stats.kendalltau(X, y)
        
        # Store each location - gradient pairs into dictionary
        location_pvalue_dict[location] = (tau, p_value)
        tau_df = pd.DataFrame(location_pvalue_dict).transpose()
        tau_df = tau_df.rename(columns = {0:'tau',1:'pvalue'})
    return tau_df


def calculate_slope(df,location_col,quantity_col,location_list):
    '''
        This function will loop through the location list and calculate slope of best fit of linear regression.
        
    Parameters:
    df: time series dataframe
    location_col: the group by location column name
    quantity_col: aggregated quantity column name by certain frequency
    location_list: Location to loop through
    
    '''
    location_dict = {}
    for location in location_list:
        df2 = df[df[location_col] ==  location].reset_index(drop = True)
        df2[quantity_col] = df2[quantity_col].replace('',0)
        # Extract the time values and reshape them to a 2D array
        X = df2.index.values.reshape(-1, 1)

        # Extract the time series values and reshape them to a 1D array
        y = df2[quantity_col].values.reshape(-1, 1)
        
        # Fit a linear regression model to the data
        reg = LinearRegression().fit(X, y)

        # Get the prediction
        y_pred = reg.predict(X)
        
        # Get the fitting metrics
        mape = mean_absolute_error(y, y_pred)
        r2 = r2_score(y,y_pred)
        
        # Get the slope of the trend line
        slope = reg.coef_[0][0]
        
        mean = np.mean(y)
        
        # Store each location - gradient pairs into dictionary
        location_dict[location] = (slope,mape,mean,r2)
        lr_df = pd.DataFrame(location_dict).transpose()
        lr_df = lr_df.rename(columns = {0:'gradient',1:'mae',2:'mean',3:'r2_score'})
        lr_df['mape'] = lr_df['mae']/lr_df['mean']
    return lr_df
