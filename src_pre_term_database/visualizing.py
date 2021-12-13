import pandas as pd
import numpy as np
from typing import List, Union
from utils import calculate_percentage
import constants as c
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks
from pathlib import Path
import wfdb
from plotly.subplots import make_subplots
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

numeric_type = Union[int, float, complex, np.number]

RED_COLOR_CODE = 'rgba(222,45,38,0.8)'
GREY_COLOR_CODE = 'rgba(204,204,204,1)'
DARK_BLUE_COLOR_CODE = 'rgb(55, 83, 109)'
LIGHT_BLUE_COLOR_CODE = 'rgb(26, 118, 255)'
TRANSPARENT_COLOR_CODE = 'rgba(255, 255, 255, 0)'
DEFAULT_FONT = dict(family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f')


def _create_color_code_list(values_of_instances_list: List[numeric_type], max_boolean: bool) -> List[str]:
    """"Create a color code list based on the values of the instances. In this function, either the
    maximum value(s) of the list will be assigned a red color and all other values a grey color. Or
    the minimum value(s) will be assigned a red color and all other values a grey color.

    Parameters
    ----------
    values_of_instances_list : List[numeric_type]
        List that contains the values to which you want to assign a color.
    max_boolean : bool
        Either True or False. If true, then the maximum value(s) of the list get plotted in the
        red color. If false, then the minimum value(s) get plotted in the red color.

    Returns
    -------
    color_code_indexes : List[str]
        A list containing the color code for each value (in RGBA color values).
        For instance: the color code 'rgba(204,204,204,1)' corresponds to grey.
    """
    # In case you want the maximum value(s) to be plotted in red
    if max_boolean:
        value = max(values_of_instances_list)

    # In case you want the minimum value(s) to be plotted in red
    if not max_boolean:
        value = min(values_of_instances_list)

    # We want to obtain all indexes that correspond to either the minimum or maximum value
    max_or_min_indexes = [i for i, j in enumerate(values_of_instances_list) if j == value]

    # We want to know to which indexes the non-highest/lowest values correspond because we want
    # to plot these values in a grey color
    non_max_or_min_indexes = [index for index in list(range(0, len(values_of_instances_list))) if
                              index not in max_or_min_indexes]

    color_code_indexes = []

    # For either the minimum/maximum value we assign the red color
    for index in max_or_min_indexes:
        color_code_indexes.insert(index, RED_COLOR_CODE)

    # For all other values we assign the grey color
    for index in non_max_or_min_indexes:
        color_code_indexes.insert(index, GREY_COLOR_CODE)

    return color_code_indexes


def _calculate_missing_percentage_numerical_features(df: pd.DataFrame,
                                                     variable: str) -> numeric_type:
    """"Calculate the percentage of missing values of the numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the numerical variables.
    variable : str
        Name of the numerical variable.

    Returns
    -------
    percentage_missing_col : Union[int, float, complex, np.number]
        The percentage of missing values of 'col'.
    """
    percentage_missing_col = calculate_percentage(df[variable].isna().sum(), len(df))

    return percentage_missing_col


def _calculate_missing_percentage_categorical_features(df: pd.DataFrame,
                                                       variable: str) -> numeric_type:
    """"Calculate the percentage of missing values of the categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the numerical variables.
    variable : str
        Name of the categorical variable.

    Returns
    -------
    percentage_missing_col : Union[int, float, complex, np.number]
        The percentage of missing values of 'variable'.
    """
    # The instances that have a missing value for a categorical feature are denoted by a 'None' value
    percentage_missing_col = calculate_percentage(len(df.query(f'{variable}=="{c.NONE_NAME}"')), len(df))

    return percentage_missing_col


def create_missing_values_plot(df: pd.DataFrame,
                               numeric_features: List[str],
                               categorical_features: List[str]):
    """"Create a bar plot containing the percentage of the missing values for the features in df.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the (numeric and categorical) features for which you want to
        compute the percentage of missing values.
    numeric_features : List[str]
        List of strings that contains the names of the numeric features.
    categorical_features : List[str]
        List of strings that contains the names of the categorical features.

    Returns
    -------
    type : plotly.graph_objs
        Bar plot of the missing values (in percentage) for all the numeric
        and categorical features.
    """
    percentage_missing_numeric_features = []

    for feature in numeric_features:
        percentage_missing_num_feature = _calculate_missing_percentage_numerical_features(df, feature)
        percentage_missing_numeric_features.append(percentage_missing_num_feature)

    percentage_missing_categorical_features = []

    for feature in categorical_features:
        percentage_missing_cat_feature = _calculate_missing_percentage_categorical_features(df, feature)
        percentage_missing_categorical_features.append(percentage_missing_cat_feature)

    total_missing_percentage_features = percentage_missing_numeric_features + percentage_missing_categorical_features

    color_code_list = _create_color_code_list(total_missing_percentage_features, max_boolean=True)

    trace0 = go.Bar(
        x=numeric_features + categorical_features,
        y=total_missing_percentage_features,
        marker=dict(color=color_code_list),
    )

    data = [trace0]
    layout = go.Layout(
        title='Percentage of missing values for each feature on patient level',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Feature',
                font=DEFAULT_FONT
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Percentage',
                font=DEFAULT_FONT
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(yaxis_ticksuffix='%')
    iplot(fig, show_link=False)


def _count_instances_categorical_value(df: pd.DataFrame, variable: str) -> List[numeric_type]:
    """"Count the number of instances for each distinct value of the categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the categorical variable for which you want to
        count the number of instances for each value.
    variable : str
        Name of the categorical feature for which you want to count the number
        of instances for each value.

    Returns
    -------
    num_instances_values : List[numeric_type]
        List that contains the instances count for each unique value of the categorical feature.
    """
    unique_values_col = np.sort(df[f'{variable}'].unique())

    num_instances_values = []
    for unique_value in unique_values_col:
        num_instances_value = len(df.query(f'{variable}=="{unique_value}"'))
        num_instances_values.append(num_instances_value)

    return num_instances_values


def _count_instances_numeric_value(df: pd.DataFrame, variable: str) -> List[numeric_type]:
    """"Count the number of instances for each distinct value of the numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the numeric feature for which you want to
        count the number of instances for each value.
    variable : str
        Name of the numeric feature for which you want to count the number
        of instances for each value.

    Returns
    -------
    num_instances_values : List[numeric_type]
        List that contains the instances count for each unique value of the numeric feature.
    """
    unique_values_col = np.sort(df[f'{variable}'].unique())
    num_instances_values = []
    for unique_value in unique_values_col:
        if np.isnan(unique_value):
            #  In pandas/numpy NaN != NaN. So NaN is not equal itself. So to check if a cell has
            #  a NaN value you can check for cell_value != cell_value -> that is only true for
            #  NaNs (3 != 3 is False but NaN != NaN is True and that query only returns the ones
            #  with True -> the NaNs).
            num_instances_value = len(df.query(f'{variable}!={variable}'))
            num_instances_values.append(num_instances_value)
        else:
            num_instances_value = len(df.query(f'{variable}=={unique_value}'))
            num_instances_values.append(num_instances_value)

    return num_instances_values


def plot_distribution_feature(df: pd.DataFrame, variable: str, categorical: bool = True):
    """"Plot the distribution of a (either categorical or numeric) feature. A bar plot
    will be returned.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which contains the categorical/numeric feature for which you want to plot
        the distribution.
    variable : str
        Name of the categorical/numeric feature for which you want to plot the distribution.
    categorical : bool
        Either true or false. If false -> the feature must be numeric. We make this distinction
        because the method to count the number of instances per value is slightly different
        for numeric features (because of NaNs).

    Returns
    -------
    type : plotly.graph_objs
        Bar plot of the distribution of the feature.
    """
    if categorical:
        y_values = _count_instances_categorical_value(df, variable)
    else:
        y_values = _count_instances_numeric_value(df, variable)

    color_code_list = _create_color_code_list(y_values, max_boolean=False)
    trace0 = go.Bar(
        x=np.sort(df[f'{variable}'].unique()).astype("str"),
        y=y_values,
        marker=dict(color=color_code_list)
    )
    data = [trace0]
    layout = go.Layout(title=f'Distribution of {variable}', xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text=f'{variable}',
            font=DEFAULT_FONT
        )
    ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Number of patients',
                font=DEFAULT_FONT
            )
        )
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


def plot_histogram(df: pd.DataFrame, variable: str, bin_size: numeric_type = 1.0):
    """"Plot the distribution of the categorical feature. A bar plot will be returned.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the feature for which you want to plot a histogram.
    variable : str
        Name of the feature for which you want to plot a histogram.
    bin_size : numeric_type
        Size you want to use to bin each instance in. Default size is 1.

    Returns
    -------
    type : plotly.graph_objs
        Histogram plot of the feature.
    """
    data = [go.Histogram(x=df[f'{variable}'],
                         xbins=dict(start=min(df[f'{variable}']),
                                    end=max(df[f'{variable}']),
                                    size=bin_size))]

    layout = go.Layout(title=f'{variable}',
                       xaxis=go.layout.XAxis(
                           title=go.layout.xaxis.Title(
                               text=f'{variable}',
                               font=DEFAULT_FONT
                           )
                       ),
                       yaxis=go.layout.YAxis(
                           title=go.layout.yaxis.Title(
                               text='Count',
                               font=DEFAULT_FONT)
                       )
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


def plot_boxplot(df: pd.DataFrame, variable: str):
    """"Plot the boxplot of the feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the feature for which you want to plot a boxplot.
    variable : str
        Name of the feature for which you want to plot a boxplot.

    Returns
    -------
    type : plotly.graph_objs
        Boxplot of the feature.
    """
    data = [go.Box(x=df[f'{variable}'])]

    layout = go.Layout(title=f'{variable}', xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text=f'{variable}',
            font=DEFAULT_FONT
        )
    ),
                         yaxis=go.layout.YAxis(
                             title=go.layout.yaxis.Title(
                                 text="",
                                 font=DEFAULT_FONT
                             )
                         )
                         )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


def plot_multiple_boxplots(df: pd.DataFrame, variables: List[str]):
    """"Plot a boxplot of the feature(s) in one figure.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that contains the feature(s) for which you want to plot a boxplot.
    variables : list[str]
        Name(s) of the feature(s) for which you want to plot a boxplot.

    Returns
    -------
    type : plotly.graph_objs
        Boxplot(s) of the feature(s).
    """
    traces_list = []
    for var in variables:
        data_feature = df[f'{var}']
        trace = go.Box(y=data_feature, name=f'{var}')
        traces_list.append(trace)

    layout = go.Layout(xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='feature',
            font=DEFAULT_FONT
        )
    ),
                         yaxis=go.layout.YAxis(
                             title=go.layout.yaxis.Title(
                                 text="value",
                                 font=DEFAULT_FONT
                             )
                         )
                         )

    fig = go.Figure(data=traces_list, layout=layout)
    iplot(fig, show_link=False)


def plot_differences_preterm_and_term_patients(df_preterm: pd.DataFrame,
                                               df_term: pd.DataFrame,
                                               variable: str,
                                               categorical: bool = True):
    """"Plot a bar plot of the preterm patients and the term patients for
    one variable.

    Parameters
    ----------
    df_preterm : pd.DataFrame
        Dataframe that contains the preterm patients.
    df_term : pd.DataFrame
        Dataframe that contains the term patients.
    variable : str
        Name of the variable which you want to plot.
    categorical : bool
        Either true or false. If false -> the feature must be numeric. We make this distinction
        because the method to count the number of instances per value is slightly different
        for numeric features (because of NaNs).

    Returns
    -------
    type : plotly.graph_objs
        Bar plot of both the preterm and term patients for one variable.
    """
    if categorical:
        y_values_preterm = _count_instances_categorical_value(df_preterm, variable)
        y_values_term = _count_instances_categorical_value(df_term, variable)
    else:
        y_values_preterm = _count_instances_numeric_value(df_preterm, variable)
        y_values_term = _count_instances_numeric_value(df_term, variable)

    trace_preterm = go.Bar(
        x=np.sort(df_preterm[f'{variable}'].unique()).astype("str"),
        y=y_values_preterm,
        name='Preterm (<37 weeks) patients',
        marker=dict(color=DARK_BLUE_COLOR_CODE)
    )

    trace_term = go.Bar(
        x=np.sort(df_term[f'{variable}'].unique()).astype("str"),
        y=y_values_term,
        name='Term (>=37 weeks) patients',
        marker=dict(color=LIGHT_BLUE_COLOR_CODE)
    )
    data = [trace_preterm, trace_term]
    layout = go.Layout(
        title=f'Preterm vs term patients for {variable} variable',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=f'{variable}',
                font=DEFAULT_FONT
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='Number of patients',
                font=DEFAULT_FONT
            )
        ),
        legend=dict(
            x=1.0,
            y=0,
            bgcolor=TRANSPARENT_COLOR_CODE,
            bordercolor='#7f7f7f'
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)


def plot_mean_and_std_variables(df_preterm: pd.DataFrame,
                                df_term: pd.DataFrame,
                                variables: List[str]):
    """"Plot a bar plot of the mean and std of the preterm patients and the term
    patients for multiple variables.

    Parameters
    ----------
    df_preterm : pd.DataFrame
        Dataframe that contains the preterm patients.
    df_term : pd.DataFrame
        Dataframe that contains the term patients.
    variables : List[str]
        Name of the variable(s) which you want to plot.

    Returns
    -------
    type : plotly.graph_objs
        Bar plot of the mean and std of both the preterm and term patients for multiple variables.
    """
    y_values_preterm_mean = []
    y_values_preterm_std = []
    y_values_term_mean = []
    y_values_term_std = []

    for variable in variables:
        y_values_preterm_mean.append(df_preterm[f'{variable}'].mean())
        y_values_preterm_std.append(df_preterm[f'{variable}'].std())
        y_values_term_mean.append(df_term[f'{variable}'].mean())
        y_values_term_std.append(df_term[f'{variable}'].std())

    trace_preterm = go.Bar(
        x=variables,
        y=y_values_preterm_mean,
        name='Preterm (<37 weeks) patients',
        marker=dict(
            color=DARK_BLUE_COLOR_CODE
        ),
        error_y=dict(
            type='data',
            array=y_values_preterm_std,
            visible=True
        )
    )
    trace_term = go.Bar(
        x=variables,
        y=y_values_term_mean,
        name='Term (>= 37 weeks) patients',
        marker=dict(
            color=LIGHT_BLUE_COLOR_CODE
        ),
        error_y=dict(
            type='data',
            array=y_values_term_std,
            visible=True
        )
    )
    data = [trace_preterm, trace_term]
    layout = go.Layout(
        title='Statistics preterm vs term patients',
        xaxis=dict(
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Value',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        legend=dict(
            x=1.0,
            y=0,
            bgcolor=TRANSPARENT_COLOR_CODE,
            bordercolor=TRANSPARENT_COLOR_CODE
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)


def plot_ehg_data(path_to_data: str, rec_id: str,
                  time_units: str, df_static_information: pd.DataFrame, **kwargs):
    """"Plot the EHG signal data of one patient (rec_id).

    Parameters
    ----------
    path_to_data : str
        Path to folder with the term-preterm database files.
    rec_id : str
        Name of the record id.
    time_units : str
        The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes',
        and 'hours'.
    df_static_information : pd.DataFrame
        Dataframe that contains the demographic data of the record id.
    kwargs:
        Dictionary of parameters to pass to make_subplots.update_xaxes()

    Returns
    -------
    type : plotly.graph_objs
        Line plot of the EHG signal data of one record id.
    """
    data_path = Path(f'{path_to_data}')
    path_to_signals = data_path / "tpehgdb"

    # TO DO: Write functionality to input a flexible number of channels you want to plot and create
    # the color codes and grid accordingly
    colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)',
              'rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)',
              'rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']

    channel_data = ['channel_1', 'channel_1_filt_0.08_4_hz', 'channel_1_filt_0.3_3_hz', 'channel_1_filt_0.3_4_hz',
                    'channel_2', 'channel_2_filt_0.08_4_hz', 'channel_2_filt_0.3_3_hz', 'channel_2_filt_0.3_4_hz',
                    'channel_3', 'channel_3_filt_0.08_4_hz', 'channel_3_filt_0.3_3_hz', 'channel_3_filt_0.3_4_hz']

    min_value_signal = -0.5
    max_value_signal = 0.7

    line_size = 2
    grid = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (4, 1), (3, 2), (4, 2), (5, 1), (6, 1), (5, 2), (6, 2)]

    # This record object contains all signal data and its properties (such as sampling rate, etc.) of
    # one record id.
    record = wfdb.rdrecord(f'{path_to_signals}/{rec_id}')
    rec_id = int(record.record_name.split('tpehg')[1])

    # The preterm_term_gestation variable contains the gestation length
    preterm_term_gestation = df_static_information.query('rec_id==@rec_id')['gestation'].iloc[0]
    preterm_term_rec_moment = df_static_information.query('rec_id==@rec_id')['gestation_at_rec_time'].iloc[0]

    # Construct time indices for the x-axis
    if time_units == 'samples':
        t = np.linspace(0, record.sig_len-1, record.sig_len)
    else:
        downsample_factor = {'seconds': record.fs, 'minutes': record.fs * 60,
                             'hours': record.fs * 3600}
        t = np.linspace(0, record.sig_len-1, record.sig_len) / downsample_factor[time_units]

    # We plot each channel in a separate subplot
    fig = make_subplots(rows=6, cols=2,
                        subplot_titles=channel_data)

    for index, name in enumerate(record.sig_name):
        fig.add_trace(go.Scatter(x=t, y=record.p_signal[:, index], mode='lines',
                                 name=name, line=dict(color=colors[index], width=line_size),
                                 connectgaps=True),
                      row=grid[index][0],
                      col=grid[index][1])
        fig.update_yaxes(title_text=record.units[index], range=[min_value_signal, max_value_signal])

    fig.update_layout(template='plotly_white', height=1100, showlegend=False,
                      title=dict(
                          text=f'<b>EHG data of patient {rec_id}, gestation: {preterm_term_gestation} wks, '
                               f'rec_moment: {preterm_term_rec_moment} wks</b>',
                          x=0.5,
                          y=0.98,
                          font=dict(
                              family="Arial",
                              size=20,
                              color='#000000'
                          )
                      )
                      )
    # dtick indicates the tick step and is set in such way that we have approx. 5 ticks on the x axis
    if 'range' in kwargs:
        dtick = int(np.diff(kwargs['range']) / 5)
    else:
        dtick = int(max(t) / 5)

    fig.update_xaxes(title_text=f'{time_units}', tick0=0, dtick=dtick, **kwargs)
    fig.show()
