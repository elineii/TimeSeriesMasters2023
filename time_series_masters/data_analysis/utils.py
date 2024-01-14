import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def describe_ts(
    df: pd.DataFrame, 
    id_column: str = "id", 
    value_column: str = "value", 
    date_column: str = "date",
):
    """Prints the basic information about the time series dataframe.
    
    Arguments:
        - df: the time series dataframe
        - id_column: the name of the id column
        - value_column: the name of the value column
        - date_column: the name of the date column
    
    """
    freq = pd.infer_freq(df[df[id_column] == df[id_column].unique()[0]][date_column])
    
    if freq == "H":
        freq = "hourly"
    elif freq == "D":
        freq = "daily"
    elif freq == "W" or freq == "W-TUE":
        freq = "weekly"
    elif freq == "M" or freq == "MS":
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


def resample_ts(
    df: pd.DataFrame, 
    freq: str, 
    id_column: str = "id", 
    value_column: str = "value", 
    date_column: str = "date",
) -> pd.DataFrame:
    """Resamples the time series dataframe.
    
    Arguments:
        - df: the time series dataframe
        - freq: the frequency to resample to
        - id_column: the name of the id column
        - value_column: the name of the value column
        - date_column: the name of the date column
    
    Returns:
        - df_resample: the resampled dataframe
    
    """
    df_resample = df.copy()
    df_resample = df_resample.set_index(date_column)
    df_resample = df_resample.groupby(id_column).resample(freq)[value_column].agg('sum').reset_index()
    return df_resample


def draw_resampled(
    df: pd.DataFrame, 
    freq: str, 
    n_series: int, 
    seed: int = 42, 
    id_column: str = "id", 
    value_column: str = "value", 
    date_column: str = "date",
):
    """Draws resampled time series.
    
    Arguments:
        - df: the time series dataframe
        - freq: the frequency to resample to
        - n_series: the number of series to plot
        - seed: the random seed
        - id_column: the name of the id column
        - value_column: the name of the value column
        - date_column: the name of the date column
        
    """
    df_resample = resample_ts(df, freq, id_column, value_column, date_column)
    
    np.random.seed(seed)
    random_id = np.random.choice(df_resample[id_column].unique(), n_series, replace=False)
    
    if n_series == 1:
        _ = plt.figure(figsize=(24, 5))
        plt.plot(df_resample[df_resample[id_column] == random_id[0]][date_column], df_resample[df_resample[id_column] == random_id[0]][value_column])
    else:
        _, ax = plt.subplots(n_series, figsize=(24, n_series*5))
        random_id = np.random.choice(df_resample[id_column].unique(), n_series, replace=False)
        
        for i in range(n_series):
            ax[i].plot(
                df_resample[df_resample[id_column] == random_id[i]][date_column], 
                df_resample[df_resample[id_column] == random_id[i]][value_column]
            )
    plt.show()
    
    
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# https://medium.com/vortechsa/detecting-trends-in-time-series-data-using-python-2752be7d1172
def calculate_slope(
    df: pd.DataFrame,
    location_col: str,
    quantity_col: str,
    location_list: list,
) -> pd.DataFrame:
    '''Loop through the location list and calculate slope of best fit of linear regression.
        
    Arguments:
        - df: time series dataframe
        - location_col: the group by location column name
        - quantity_col: aggregated quantity column name by certain frequency
        - location_list: location to loop through
    
    Returns:
        - lr_df: dataframe with slope, mae, mean, r2_score, mape

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
