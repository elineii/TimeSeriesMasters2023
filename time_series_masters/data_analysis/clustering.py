from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

from tslearn.clustering import TimeSeriesKMeans
from yellowbrick.cluster import SilhouetteVisualizer


def _prepare_data_for_clustering(
    data: pd.DataFrame, 
    date_column: str, 
    id_column: str,
):
    # Prepare data for clustering
    data = data.copy()
    data_scaled = data.pivot(index=date_column, columns=[id_column])
    
    # Scale data
    ss = StandardScaler()
    data_scaled = ss.fit_transform(data_scaled)
    data_scaled = data_scaled.T
    return data_scaled
    

def find_optimal_n_clusters(
    data: pd.DataFrame, 
    metric: str = "euclidean", 
    max_n_clusters: int = 10, 
    id_column: str = "id", 
    date_column: str = "date",
    random_state: int = 42,
    n_jobs: int = -1
):
    """Plot the silhouette score for different number of clusters.
    
    Arguments:
        - data: the time series dataframe
        - metric: the metric used for clustering, one of ["euclidean", "dtw", "softdtw"]
        - max_n_clusters: the maximum number of clusters
        - id_column: the name of the id column
        - date_column: the name of the date column
        - random_state: the random state
        - n_jobs: the number of jobs to run in parallel
        
    """
    data_scaled = _prepare_data_for_clustering(data, date_column, id_column)
    
    n_clusters_cnt = len(range(2, max_n_clusters+1))
    
    fig, ax = plt.subplots(math.ceil(n_clusters_cnt / 2), 2, figsize=(24, 4 * n_clusters_cnt // 2))
    for i in range(2, max_n_clusters+1):
        row = (i - 2) // 2
        col = (i - 2) % 2
        
        kmeans = TimeSeriesKMeans(n_clusters=i, metric=metric, random_state=random_state, n_jobs=n_jobs)
        if math.ceil(n_clusters_cnt / 2) == 1:
            visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[col])
        else:
            visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[row][col])
        visualizer.fit(data_scaled)
    
    
def cluster_ts(
    data: pd.DataFrame, 
    metric: str = "euclidean", 
    n_clusters: int = 10, 
    id_column: str = "id", 
    date_column: str = "date",
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Cluster time series using KMeans algorithm.
    If optimise_n_clusters is True, the optimal number of clusters is found using the silhouette score.
    
    Arguments:
        - data: the time series dataframe
        - metric: the metric used for clustering, one of ["euclidean", "dtw", "softdtw"]
        - n_clusters: the number of clusters
        - id_column: the name of the id column
        - date_column: the name of the date column
        - random_state: the random state
        - n_jobs: the number of jobs to run in parallel
    
    Returns:
        - data: the dataframe with the cluster column
        
    """
    data = data.copy()
    data_scaled = _prepare_data_for_clustering(data, date_column, id_column)
    
    # Кластеризуем TS
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state, n_jobs=n_jobs)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Добавляем новый столбец в исходный датафрейм
    data['cluster'] = np.repeat(clusters, sum(data[id_column] == data[id_column].iloc[0]))
    return data


def plot_clustered_series(
    data: pd.DataFrame, 
    n_clusters: int = 10, 
    id_column: str = "id", 
    value_column: str = "value", 
    date_column: str = "date", 
    n_series: int = 10, 
    seed: int = 42
):
    """Plot clustered time series.
    
    Arguments:
        - data: the time series dataframe
        - n_clusters: the number of clusters
        - id_column: the name of the id column
        - value_column: the name of the value column
        - date_column: the name of the date column
        - n_series: the number of series to plot for each cluster
        - seed: the random seed
        
    """
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