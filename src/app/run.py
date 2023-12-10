import os
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from etna.datasets import generate_periodic_df, TSDataset
from etna.models import CatBoostPerSegmentModel
from etna.transforms import (DateFlagsTransform, DensityOutliersTransform, FourierTransform,
                             LagTransform, LinearTrendTransform, MeanTransform,
                             SegmentEncoderTransform, TimeSeriesImputerTransform)
from etna.pipeline import Pipeline
import etna.metrics as met
from etna.analysis import plot_forecast


# Here we define function to help us load data.
def load_data(file):
    # Get the file extension
    _, ext = os.path.splitext(file.name)

    # Use the appropriate loading function based on the file extension
    if ext.lower() == '.csv':
        data = pd.read_csv(file)
    elif ext.lower() in ['.xls', '.xlsx']:
        data = pd.read_excel(file)
    elif ext.lower() == '.parquet':
        data = pd.read_parquet(file)
    else:
        st.sidebar.write('Unsupported file type')
        data = None

    return data


# This function lets the user save the dataset he has generated.
# IMPORTANT: Cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


# We need to cache results of this function in order not to train the model every time the script is rerun,
# due to specificity of Streamlit rendering.
@st.cache_resource
def create_pipeline(_modell, _transformsl, _horizonl, _train_tsl):
    pipeline = Pipeline(model=_modell, transforms=_transformsl, horizon=_horizonl)
    pipeline.fit(_train_tsl)
    return pipeline


# This is the one way to calculate metrics, but for some reasons, line 61 is calculated incorrectly,
# probably due to some peculiarity of ETNA.metrics, as everything works fine with sklearn.metrics
def calculate_metrics(functions, y_train, y_pred_train, y_test, y_pred):
    d = {'Metric': [],
         'Results on training observations': [],
         'Results on testing observations': []}
    summary = pd.DataFrame(d)

    for func_name in functions:
        metric_train = getattr(met, func_name)(mode='macro', y_true=y_train, y_pred=y_pred_train)
        metric_train = eval(metric_train)
        metric_test = getattr(met, func_name)(mode='macro', y_true=y_test, y_pred=y_pred)
        metric_test = eval(metric_test)
        summary.loc[len(summary.index)] = [func_name, metric_train, metric_test]

    return summary


# Here we start to display first lines of our application
st.image('logo.png')
st.title('Train and Test Your Own CatBoostPerSegment Model')
st.header('Step 1: Data')
st.subheader('Everything starts with data, use sidebar to configure your dataset.')

# Load default data, so that it is always displayed by default
default_data = pd.read_csv('example_dataset.csv')
data = default_data
df = TSDataset.to_dataset(data)
ts = TSDataset(df, freq='D')

# Here we start with letting the user choose between several data options using st.radio
# in order not to overload the main page of the application all the parameters of the dataset are configured
# inside the sidebar.
st.sidebar.header("Dataset configuration:")
data_choice = st.sidebar.radio(
    'Hello there! To get started, let us decide on the type of dataset you would like to use:',
    ('Kick off Easily with Preloaded Data', 'Generate Your Data',
     'Upload Your Own Data')
)

# Here we define some variables that will be used as default values for interface activation
button = True
multiselect = True
freq = 'D'

# If user decides to upload his own data, we display st.file_uploader() and store the uploaded file.
if data_choice == 'Upload Your Own Data':
    uploaded_file = st.sidebar.file_uploader('Upload your data', type=['csv', 'xls', 'xlsx', 'parquet'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.dataframe(data)
        button = False
    else:
        st.sidebar.write('No file uploaded. Default data is displayed.')
        st.dataframe(default_data)

# If user decides to generate his own dataset, we display a menu with dataset parameters
# and then use generate_periodic_df() from etna.
elif data_choice == 'Generate Your Data':
    # Get parameters for generate_periodic_ts
    periods = st.sidebar.number_input(label='Number of Periods:', min_value=10, step=1,
                                      help='Choose the number of timestamps')
    start_time = st.sidebar.date_input(label='Start Time:', help='Enter the start timestamp')
    freq = st.sidebar.selectbox(label='Data Frequency:', options=('D', 'W', 'M', 'B', 'Q'),
                                help='D-calendar day, W-weekly, M-month end, B-business day, Q-quarter end frequency')
    period = st.sidebar.number_input(label='Data Periodicity:', min_value=1, max_value=periods, value=7, step=1,
                                     help='Enter data frequency: x[i+period] = x[i]')
    scale = st.sidebar.number_input(label='Scale', min_value=1, step=1, value=10,
                                    help='Sample data from Uniform[0, scale)')
    n_segments = st.sidebar.slider(label='Number of Segments:', min_value=1, max_value=10, step=1,
                                   help='Choose the number of segments by moving the slider')
    sigma = st.sidebar.number_input(label='Scale of Added Noise:')
    random_seed = st.sidebar.number_input(label='Random Seed:', min_value=1, step=1)

    # Generate data, when user is ready with all the parameters, he clicks on the button.
    if st.sidebar.button(label='Generate data!'):
        data = generate_periodic_df(periods=periods, start_time=start_time, scale=scale, period=period,
                                    n_segments=n_segments, freq=freq, sigma=sigma, random_seed=random_seed)

        button = False
        # Here user can save the dataset, he has just generated, when he clicks on the button,
        # dataset is automatically saved as a .csv file.
        csv = convert_df(data)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='my_generated_df.csv',
            mime='text/csv',
        )
    else:
        st.write('Your generated data will appear below.')
        st.sidebar.write('Press the button when ready to generate data')
    st.dataframe(data)

# But for most cases, when user just wants to explore the app capabilities, he can use
# the default dataset that was preloaded into the application.
elif data_choice == 'Kick off Easily with Preloaded Data':
    data = default_data
    st.dataframe(data)
    button = False

# When the user is finally ready with the data, he should activate the checkbox below,
# this help to logically divide the workflow of the application and get rid of the
# error that occur below, due to the way streamlit renders the python script.
if st.checkbox('Proceed with data displayed above', disabled=button):
    df = TSDataset.to_dataset(data)
    ts = TSDataset(df, freq=freq)
    button = True
    multiselect = False
    st.success('Dataset has been configured!', icon="✅")
else:
    st.warning('You need to configure dataset first.', icon="⚠️")

# At this step we let the user choose the horizon value, that is used in some transformations
# as well as in the train/test split and model training and evaluation.
st.header('Step 2: Choose Horizon')
horizon = st.slider(label='Horizon', min_value=1, max_value=len(ts.to_pandas()), step=1)

# At this step we use st.multiselect() to let the user pick transformations he wants
# to apply to the dataset
st.header('Step 3: Transformations')
# Here we define the list of all possible transformations that can be applied to the data.
possible_transforms = ['DensityOutliersTransform', 'TimeSeriesImputerTransform', 'LinearTrendTransform',
                       'LagTransform', 'DateFlagsTransform', 'FourierTransform',
                       'MeanTransform', 'SegmentEncoderTransform']
options = st.multiselect(label='Select transformations:', options=possible_transforms)
proceed = True

# This part of the code makes sure that at least one transformation was chosen
# based on this we define some variables that account for input widget activity
# this helps to get rid of errors caused by streamlit.
if len(options) == 0:
    'Please, select one or more transformations.'
else:
    proceed = False
activate = True

# When the user is ready with transformations, we proceed with configuring each
# transformation parameters for greater application flexibility.
if st.checkbox('Confirm transformations choice', disabled=proceed):
    # Initialize an empty list to store the transforms
    transforms = []
    # Below starts the long and boring part, I will try to explain what happens here.
    # Iterate through the list of our chosen transformations, and display widgets that
    # are used to tune every transformation from the list of chosen transformations.
    # When the user is ready with selecting parameters of one transformation, he
    # activates the checkbox and this transformation is added to the transforms list with its parameters.
    # This process continues for every chosen transformation.
    for transform in options:
        if transform == 'DensityOutliersTransform':
            st.subheader('DensityOutliersTransform parameters:')
            window_size = st.slider(label='Window size:', min_value=2, max_value=100, step=1, value=15,
                                    help='Size of windows to build')
            distance_coef = st.number_input(label='Distance coefficient:', min_value=0, value=3,
                                            help='Forms distance threshold to determine points are close to each other')
            n_neighbors = st.slider(label='Number of neighbours:', min_value=1, max_value=10, value=3, step=1,
                                    help='Min number of close neighbors of point not to be outlier')
            if st.checkbox('Save transformation parameters', key=1):
                transforms.append(DensityOutliersTransform(in_column='target', window_size=window_size,
                                                           distance_coef=distance_coef, n_neighbors=n_neighbors))
                st.success('Parameters are saved!', icon="✅")
                continue
            else:
                st.write('You need to save transformation parameters first')
        elif transform == 'TimeSeriesImputerTransform':
            st.subheader('TimeSeriesImputerTransform parameters:')
            strategy = st.selectbox(label='Strategy:', options=['constant', 'mean', 'running_mean', 'seasonal',
                                                                'forward_fill'], index=0)
            constant_value = 0
            if strategy == 'constant':
                constant_value = st.number_input(label='Constant value:', value=0,
                                                 help='Value to fill gaps in “constant” strategy')
            default_value = st.number_input(label='Defualt value:', value=0,
                                            help='Value used to impute the NaNs left after applying the imputer')
            seasonality = st.number_input(label='Seasonality:', min_value=0, value=1,
                                          help='The length of the seasonality')

            if st.checkbox('Save transformation parameters', key=2):
                transforms.append(
                    TimeSeriesImputerTransform(in_column='target', strategy=strategy, default_value=default_value,
                                               constant_value=constant_value, seasonality=seasonality))
                st.success('Parameters are saved!', icon="✅")
                continue
            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'LinearTrendTransform':
            st.subheader('LinearTrendTransform parameters:')
            poly_degree = st.slider(label='Polynomial degree:', min_value=1, max_value=50, step=1,
                                    help='Degree of polynomial to fit trend on')

            if st.checkbox('Save transformation parameters', key=3):
                transforms.append(LinearTrendTransform(in_column='target', poly_degree=poly_degree))
                st.success('Parameters are saved!', icon="✅")
                continue

            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'LagTransform':
            st.subheader('LagTransform parameters:')
            lags = st.slider(label='Number of lags:', min_value=1, max_value=len(ts.to_pandas()) - horizon, step=1,
                             help='Generate range of lags from 1 to given value.')
            if st.checkbox('Save transformation parameters', key=4):
                transforms.append(LagTransform(in_column='target', lags=lags, out_column='out_LagTransform'))
                st.success('Parameters are saved!', icon="✅")
                continue
            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'DateFlagsTransform':
            st.subheader('DateFlagsTransform parametes:')
            st.write('Select transformations below:')
            day_number_in_week = st.checkbox(label='day_number_in_week')
            day_number_in_month = st.checkbox(label='day_number_in_month')
            day_number_in_year = st.checkbox(label='day_number_in_year')
            week_number_in_month = st.checkbox(label='week_number_in_month')
            week_number_in_year = st.checkbox(label='week_number_in_year')
            is_weekend = st.checkbox(label='is_weekend')

            if st.checkbox('Save transformation parameters', key=5):
                transforms.append(
                    DateFlagsTransform(day_number_in_week=day_number_in_week, day_number_in_year=day_number_in_year,
                                       day_number_in_month=day_number_in_month, is_weekend=is_weekend,
                                       week_number_in_month=week_number_in_month,
                                       week_number_in_year=week_number_in_year)
                )
                st.success('Parameters are saved!', icon="✅")
                continue

            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'FourierTransform':
            st.subheader('FourierTransform parameters:')
            f_period = st.number_input(label='Period:', min_value=2, value=3,
                                       help='The period of the seasonality to capture in frequency units of time series')
            order = st.slider(label='Order:', min_value=1, max_value=m.ceil(f_period / 2), step=1,
                              help='Upper order of Fourier components to include')
            if st.checkbox('Save transformation parameters', key=6):
                transforms.append(
                    FourierTransform(period=f_period, order=order, out_column='out_FourierTransform')
                )
                st.success('Parameters are saved!', icon="✅")
                continue

            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'MeanTransform':
            st.subheader('MeanTransform parameters:')
            window = st.slider(label='Window size:', min_value=2, max_value=len(ts.to_pandas()) - horizon, step=1,
                               help='Size of window to aggregate')
            seasonality = st.number_input(label='Seasonality:', min_value=1, step=1,
                                          help='Seasonality of lags to compute window’s aggregation with')
            alpha = st.number_input(label='Alpha:', help='Autoregressive coefficient')
            min_periods = st.slider(label='Minimal number of targets:', min_value=1, max_value=30, step=1,
                                    help='Min number of targets in window to compute aggregation.')
            fillna = st.number_input(label='Fill NaNs with:')

            if st.checkbox('Save transformation parameters', key=7):
                transforms.append(
                    MeanTransform(in_column='target', window=window, seasonality=seasonality, fillna=fillna,
                                  min_periods=min_periods, out_column='out_MeanTransform'))
                st.success('Parameters are saved!', icon="✅")
                continue

            else:
                st.write('You need to save transformation parameters first')

        elif transform == 'SegmentEncoderTransform':
            st.subheader('SegmentEncoderTransform parameters:')
            st.write('No parameters can be tuned for this transformation.')
            transforms.append(SegmentEncoderTransform())
            continue

    # Here we check that the user didn't give up and saved at least one transformation so that we can proceed.
    if len(transforms) > 0:
        activate = False

# At this step we proceed with choosing the model
vis = True
st.header('Step 4: Modeling')
st.subheader('At this step you can select parameters of the model.')

# By default, the user is given with interface that lets him set the model hyperparameters,
# but he can use the checkbox below to train and test the default model.
if st.checkbox(label='Use default model', disabled=activate):
    model = CatBoostPerSegmentModel()

# Here we let the user choose the parameters of the model
else:
    iterations = st.number_input(label='Maximum number of trees:', min_value=1, step=1, disabled=activate,
                                 help='The maximum number of trees built when solving machine learning problems.')
    depth = st.slider(label='Depth of the tree:', min_value=2, max_value=16, step=1, disabled=activate)
    learning_rate = st.select_slider(label='Learning rate:', options=[0.2, 0.1, 0.01, 0.001], disabled=activate)
    l2_leaf_reg = st.number_input(label='L2 regularization:', min_value=0, value=0, disabled=activate)
    model = CatBoostPerSegmentModel(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                    l2_leaf_reg=l2_leaf_reg)

# Streamlit reruns python file each time, when some changes are made, but most of the changes
# have nothing to do with model parameters, so there is no need to refit the model each time
# as model training can take a very long time. That is why, we cache every model after training
# and for the situations when user wants to train another model, he can use the button below, to clear model cache.
if st.button(label="Clear model cache"):
    st.cache_resource.clear()
else:
    st.write('You should clear the model cache if you want to retrain the model with different hyperparameters.')

# Here we create a checkbox, when it is activated, model training begins (assuming the model cache is empty)
train = st.checkbox(label='Train the model!', disabled=activate)
if train:
    train_ts, test_ts = ts.train_test_split(test_size=horizon)

    with st.spinner('Training...'):
        # Here we create a final pipeline.
        pipeline = create_pipeline(model, transforms, horizon, train_ts)
    st.success('Training complete!')
    vis = False

# At this step we proceed with metrics calculation and forecasting.
st.header('Step 5: Backtesting and forecasting')
st.subheader('At this step you can evaluate the model performance.')
st.write('First of all, choose the metrics')

# We define the list of all possible matrics that can be calculated.
metrics = ['MAE', 'MAPE', 'MSE', 'MSLE', 'MedAE', 'R2', 'RMSE', 'SMAPE', 'WAPE']
chosen_metrics = st.multiselect(label='Metrics', options=metrics, disabled=vis)

# Here we define a variable that controls printout.
if len(chosen_metrics) == 1:
    per_segment_disabled = False
else:
    per_segment_disabled = True

# By default, we use macro mode for metrics calculations and multiple metrics are allowed at a time.
mode = 'macro'
# In order to make printout more clean, only one per-segment metric is allowed at a time.
per_segment = st.checkbox(label='Use per-segment metric calculation instead of macro.', disabled=per_segment_disabled)
if per_segment:
    mode = 'per-segment'
else:
    st.write('For more clear visualisations, only one per segment metric is allowed at a time.')

# We check is the user has selected at least one metric.
if len(chosen_metrics) > 0:

    # We wil calculate metrics value both for training and testing subsets
    # for this reason we define a dictionary that would store metrics.
    d = {'Metric': [],
         'Results on training observations': [],
         'Results on testing observations': []}
    # We turn our dictionary into dataframe for more clean and intuitive printout.
    summary = pd.DataFrame(d)

    # Here we make predictions both on training and testing data.
    y_train_pred = pipeline.predict(train_ts)
    y_pred = pipeline.forecast()

    # For more effective metric calculation we define a dictionary that maps string to actual function.
    metric_map = {
        'MAE': met.MAE,
        'MAPE': met.MAPE,
        'MSE': met.MSE,
        'MSLE': met.MSLE,
        'MedAE': met.MedAE,
        'R2': met.R2,
        'RMSE': met.RMSE,
        'SMAPE': met.SMAPE,
        'WAPE': met.SMAPE
    }

    metrics = {}
    # here we iterate through chosen metrics and define the metric as well as its parameter.
    for m in chosen_metrics:
        metric_class = metric_map[m]
        metrics[m] = metric_class(mode=mode)

    # Here we iterate through defined metrics and calculate them.
    for m, metric in metrics.items():
        result_train = metric(y_true=train_ts, y_pred=y_train_pred)
        result_test = metric(y_true=test_ts, y_pred=y_pred)
        if per_segment:
            train_visual = pd.DataFrame.from_dict(result_train, orient='index', columns=['values']).T
            test_visual = pd.DataFrame.from_dict(result_test, orient='index', columns=['values']).T
            st.dataframe(train_visual)
            st.dataframe(test_visual)
        else:
            summary.loc[len(summary.index)] = [m, result_train, result_test]

    # For better per segment metrics comparison we use visualisations.
    if per_segment:
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # plot the training performance on the first subplot
        segments_train = list(result_train.keys())
        values_train = list(result_train.values())
        ax1.bar(segments_train, values_train)
        ax1.set_title("Training Performance")
        ax1.set_xlabel("Segments")
        ax1.set_ylabel(f"{chosen_metrics[0]} Values")
        for i, v in enumerate(values_train):
            ax1.text(i, v, str(round(v, 2)), ha='center', va='bottom')

        # plot the testing performance on the second subplot
        segments_test = list(result_test.keys())
        values_test = list(result_test.values())
        ax2.bar(segments_test, values_test)
        ax2.set_title("Testing Performance")
        ax2.set_xlabel("Segments")
        ax2.set_ylabel(f"{chosen_metrics[0]} Values")
        for i, v in enumerate(values_test):
            ax2.text(i, v, str(round(v, 2)), ha='center', va='bottom')

        # adjust the layout and display the figure
        fig.tight_layout()
        st.pyplot(fig)

    # Display the table with calculated metrics.
    else:
        st.dataframe(summary)

# Here we present forecast visualisations.
if not vis:
    st.set_option("deprecation.showPyplotGlobalUse", False)  # Disable the warning
    y_pred = pipeline.forecast()
    # Display forecast and metric
    st.subheader("Plotting Forecast:")
    st.pyplot(
        plot_forecast(
            forecast_ts=y_pred,
            test_ts=test_ts,
            train_ts=train_ts,
            n_train_samples=50,
        )
    )

# References:
# Streamlit library documentation: https://docs.streamlit.io
# ETNA library documentation: https://etna-docs.netlify.app
