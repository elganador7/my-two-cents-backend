import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

from plotly import graph_objects as go
import os

# Function to create sequences of length `seq_length` from the data
def create_sequences(file_name, seq_length, pred_dist):
    # stock_data = pd.read_csv(file_name)
    stock_data = pd.read_pickle(file_name)

    # Convert the 'Open', 'High', 'Low', 'Close', and 'Volume' columns to numpy arrays

    high_prices = stock_data['h'].values
    low_prices = stock_data['l'].values
    volume = stock_data['v'].values
    weighted_avg = stock_data['vw'].values
    time = stock_data["t"].values

    # Combine all five columns into one input array
    data = np.column_stack((high_prices, low_prices))
    time = np.column_stack((time))

    sequences = []
    y_values_min = []
    y_values_max = []
    for i in range(len(data) - seq_length - pred_dist):
        time_gap = int((time[0][i+seq_length+pred_dist] - time[0][i])/60000)
        if time_gap == sequence_length + pred_dist:
            array = data[i:i+seq_length]
            sequences.append([array[j+1]-array[j] for j in range(len(array) -1)])
            min_val = np.min([arr[1] for arr in data[i+seq_length:i+seq_length+pred_dist]])
            max_val = np.max([arr[0] for arr in data[i+seq_length:i+seq_length+pred_dist]])
            range_min = min_val - data[i+seq_length-1][1]
            range_max = max_val - data[i+seq_length-1][0]
            y_values_min.append(range_min)
            y_values_max.append(range_max)
    X = np.array(sequences)
    y_min = np.array(y_values_min)
    y_max = np.array(y_values_max)
    return X, y_min, y_max

# Function to create sequences of length `seq_length` from the data
def create_stepped_sequences(file_name, seq_length, pred_dist):
    # stock_data = pd.read_csv(file_name)
    stock_data = pd.read_pickle(file_name)

    # Convert the 'Open', 'High', 'Low', 'Close', and 'Volume' columns to numpy arrays
    high_prices = stock_data['h'].values
    low_prices = stock_data['l'].values
    volume = stock_data['v'].values
    weighted_avg = stock_data['vw'].values
    time = stock_data["t"].values

    # Combine all five columns into one input array
    data = np.column_stack((high_prices, low_prices))
    time = np.column_stack((time))

    sequences = []
    min_sequences = [[] for a in range(pred_dist)]
    max_sequences = [[] for b in range(pred_dist)]
    for i in range(len(data) - seq_length - pred_dist):
        time_gap = int((time[0][i+seq_length+pred_dist] - time[0][i])/60000)
        if time_gap == sequence_length + pred_dist:
            array = data[i:i+seq_length]
            sequences.append([array[j+1]-array[j] for j in range(len(array) -1)])
            for k in range(pred_dist):
                min_val = np.min([arr[1] for arr in data[i+seq_length:i+seq_length+k+1]])
                max_val = np.max([arr[0] for arr in data[i+seq_length:i+seq_length+k+1]])
                range_min = min_val - data[i+seq_length-1][1]
                range_max = max_val - data[i+seq_length-1][0]
                min_sequences[k].append(range_min)
                max_sequences[k].append(range_max)
    X = np.array(sequences)
    for c in range(pred_dist):
        min_sequences[c] = np.array(min_sequences[c])
        max_sequences[c] = np.array(max_sequences[c])
    return X, min_sequences, max_sequences

def bc_model_train(
    X : np.array,
    labels : np.array,
    epochs : float = 10, 
    model_dir : str = "./default_model",
    sequence_length : float = 10,
):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(sequence_length-1, 2)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, labels, epochs=epochs)
    if not os.path.exists(model_dir):
    # If it doesn't exist, create the directory
        os.makedirs(model_dir)
    model.save(model_dir)
    return model

def lstm_model_train(
    X, y, epochs : float = 10, model_dir : str = "./default_model", sequence_length : float = 10,
):
    # Define and compile the min LSTM model
    min_model = Sequential()
    min_model.add(LSTM(64, input_shape=(sequence_length - 1, 4)))
    min_model.add(Dense(1))

    min_model.compile(optimizer='adam', loss='huber')

    # Train the model with reduced epochs

    min_model.fit(X, y, epochs=epochs, batch_size=1)

    min_model.save(model_dir)

def normalize_data(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X = (X - means) / stds
    return X, means, stds

def gen_binned_bar_chart(
    predicted_df : pd.DataFrame,
    bin_width : float = 0.01,
    x_column : str = "prediction_min",
    y_column : str = "actual_min",
    name_suffix : str = "main"
):
    # Compute the number of bins based on the min and max x values
    x_min = predicted_df[x_column].min()
    x_max = predicted_df[x_column].max()
    num_bins = int((x_max - x_min) / bin_width) + 1

    # Create an array to hold the bin edges
    bin_edges = np.linspace(x_min, x_max, num_bins)

    # Initialize lists to store computed y values
    y_min_values = []
    y_max_values = []
    y_avg_values = []
    y_std_values = []
    y_count_values = []

    # Iterate over each bin and calculate the y values
    for i in range(num_bins - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        
        # Filter the data within the current bin
        bin_data = predicted_df[(predicted_df[x_column] >= bin_start) & (predicted_df[x_column] < bin_end)]
        
        # Compute min, max, average, and standard deviation of y values within the bin
        y_min = bin_data[y_column].min()
        y_max = bin_data[y_column].max()
        y_avg = bin_data[y_column].mean()
        y_std = bin_data[y_column].std()
        y_count = bin_data[y_column].count()
        
        y_min_values.append(y_min)
        y_max_values.append(y_max)
        y_avg_values.append(y_avg)
        y_std_values.append(y_std)
        y_count_values.append(y_count)

    # Create the Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(x=bin_edges[:-1], y=y_min_values, name='Min'))
    fig.add_trace(go.Bar(x=bin_edges[:-1], y=y_max_values, name='Max'))
    fig.add_trace(go.Bar(x=bin_edges[:-1], y=y_avg_values, name='Average'))
    fig.add_trace(go.Bar(x=bin_edges[:-1], y=y_std_values, name='Standard Deviation'))
    fig.add_trace(go.Bar(x=bin_edges[:-1], y=np.array(y_count_values)/1000, name='Count'))

    # Update the layout to have grouped bars
    fig.update_layout(barmode='group', xaxis_title='X Values', yaxis_title='Y Values')

    fig.write_html(f"./{x_column}-{y_column}-{name_suffix}.html")


# Choose the sequence length for input data
sequence_length = 20
pred_distance = 5
epochs = 10

# Create sequences for training
X, y_min_sequences, y_max_sequences = create_stepped_sequences("./train_data.pkl", sequence_length, pred_dist=pred_distance)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
test_X, test_y_min_sequences, test_y_max_sequences = create_stepped_sequences("./test_data.pkl", sequence_length, pred_dist = pred_distance)

min_models = []
max_models = []

for index in range(pred_distance):
    # y_min = y_min_sequences[index]
    # y_min_labels = np.full_like(y_min, 0)
    # y_min_labels[y_min <= -0.05] = 1

    # # Create a model
    # model_min = bc_model_train(
    #     X, y_min_labels, epochs = epochs, model_dir = f'./{index+1}_min_bc_model', sequence_length=sequence_length
    # )

    # min_models.append(model_min)

    # y_max = y_max_sequences[index]
    # y_max_labels = np.full_like(y_max, 0)
    # y_max_labels[y_max >= 0.05] = 1

    # model_max = bc_model_train(
    #     X, y_max_labels, epochs = epochs, model_dir = f'./{index+1}_max_bc_model', sequence_length=sequence_length
    # )

    # max_models.append(model_max)

    model_min = tf.keras.models.load_model(f'./{index+1}_min_bc_model')
    model_max = tf.keras.models.load_model(f'./{index+1}_max_bc_model')

    sequences_test= test_X # - means) / stds

    predicted_values_min = model_min.predict(test_X)
    predicted_values_max = model_max.predict(test_X)

    predicted_values_min = predicted_values_min # * y_min_std + y_min_mean
    predicted_values_max = predicted_values_max # * y_max_std + y_max_mean

    numpy_array = np.column_stack((predicted_values_min, predicted_values_max, test_y_min_sequences[index], test_y_max_sequences[index]))
    new_array = np.hstack((original_array, new_column[:, np.newaxis]))

    predicted_df = pd.DataFrame(numpy_array, columns=["prediction_min", "prediction_max", "actual_min", "actual_max"])


    fig_a = go.Scatter(
        x=predicted_df.index,
        y=predicted_df["prediction_min"],
        name="Prediction Min",
        line={"dash": "dot", "width": 2}
    )

    fig_b = go.Scatter(
        x=predicted_df.index,
        y=predicted_df["actual_min"],
        name="Actual Low",
    )

    fig_c = go.Scatter(
        x=predicted_df.index,
        y=predicted_df["actual_max"],
        name="Actual High",
    )

    fig_d = go.Scatter(
        x=predicted_df.index,
        y=predicted_df["prediction_max"],
        name="Prediction Max",
        line={"dash": "dot", "width": 2, "color" : "yellow"}
    )


    fig_overall = go.Figure(data=[fig_c, fig_d])

    fig_overall.write_html(f"./predictions_{index+1}.html")

    colors = ['green' if val >= 0 else 'red' for val in predicted_df['prediction_min'].diff()]

    fig_g = go.Scatter(
        y=predicted_df["actual_min"],
        x=predicted_df["prediction_min"],
        name="Prediction Min Diff",
        mode="markers",
        marker=dict(color=colors)
    )

    fig_overall_min = go.Figure(data=[fig_g])

    fig_overall_min.write_html(f"./diffs_min_{index+1}.html")

    fig_i = go.Scatter(
        y=predicted_df["actual_max"],
        x=predicted_df["prediction_max"],
        name="Prediction Max Diff",
        mode="markers",
        marker=dict(color=colors)
    )

    fig_overall_min = go.Figure(data=[fig_i])

    fig_overall_min.write_html(f"./diffs_max_{index+1}.html")

    gen_binned_bar_chart(predicted_df=predicted_df, bin_width=0.01, x_column = "prediction_min", y_column="actual_min", name_suffix=index+1)
    gen_binned_bar_chart(predicted_df=predicted_df, bin_width=0.01, x_column = "prediction_max", y_column="actual_max", name_suffix=index+1)
    