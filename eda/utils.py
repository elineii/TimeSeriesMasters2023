import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from datetime import datetime
from distutils.util import strtobool


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def describe_ts(df, id_column="id", value_column="value", date_column="date"):
    print(f'Число рядов: {df[id_column].nunique()}')
    print(f'Наблюдений в ряде: {df[id_column].value_counts().unique()}')
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
        plt.plot(df_resample[df_resample[id_column] == random_id[0]][date_column],
                 df_resample[df_resample[id_column] == random_id[0]][value_column])
    else:
        _, ax = plt.subplots(n_series, figsize=(24, 20))
        random_id = np.random.choice(df_resample[id_column].unique(), n_series)

        for i in range(n_series):
            ax[i].plot(
                df_resample[df_resample[id_column] == random_id[i]][date_column],
                df_resample[df_resample[id_column] == random_id[i]][value_column]
            )
    plt.show()
    
    
def cluster_ts(data, n_clusters):

    # Преобразуем список временных рядов в датафрейм
    clustering_data_scaled = pd.DataFrame(data['value'].tolist())

    # Масштабируем данные
    ss = StandardScaler()
    clustering_data_scaled = ss.fit_transform(clustering_data_scaled)

    # Кластеризуем TS
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(clustering_data_scaled)

    # Добавляем новый столбец в исходный датафрейм
    data['cluster'] = clusters

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
