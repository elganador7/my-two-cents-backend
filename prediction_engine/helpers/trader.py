import numpy as np
import pandas as pd
import tensorflow as tf

from plotly import graph_objects as go

import requests
import json
import datetime

import yfinance as yf

import time

from prediction_engine.config_files.logger_config import logger

API_KEY = ""

def fetch_stock_data(stock_symbol, start_date = "2023-05-01", end_date = "2023-07-01", limit=500000):
    url = f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}"
    response = requests.get(url)
    loaded_data = json.loads(response.content)
    
    if response.status_code == 200:
        stock_data = loaded_data["results"]

        while "next_url" in loaded_data:
            print(f"{loaded_data['next_url']}?apiKey={API_KEY}")
            response = requests.get(f"{loaded_data['next_url']}?apiKey={API_KEY}")
            loaded_data = json.loads(response.content)
            print(loaded_data)
            stock_data.append(loaded_data["results"])

        return stock_data
    else:
        print(loaded_data)
        return None

def get_stock_price(ticker_name):
    ticker = yf.Ticker(ticker_name)
    price = float(ticker.info['currentPrice'])
    volume = ticker.info['volume']
    timestamp = int(datetime.datetime.now().timestamp() * 1000)
    return price, timestamp, volume

def generate_recommendation_max(
    price : float,
    predicted_values_max = [],
    
):
    # if predicted_value_min > 0.55:
    #     recommendation = "Sell"
    #     last_short_action_price = price
    #     delta_pos = -10 - short_position
    #     short_position = -10
    #     balance += (delta_pos * price)
    #     if predicted_value_min > 0.65:
    #         modifier = "Strong"
    #         delta_pos = -20 - short_position
    #         short_position = -20
    #         balance += (delta_pos * price)
    if predicted_value_max > 0.58:
        recommendation = "Buy"
        last_long_action_price = price
        last_long_action_timestamp = timestamp
        delta_pos = 10 - long_position
        long_position = 10
        balance -= (delta_pos * price)
        if predicted_value_max > 0.65:
            modifier = "Strong"
            delta_pos = 20 - long_position
            long_position = 20
            balance -= (delta_pos * price)
    elif predicted_value_max < 0.5:
        recommendation = "Sell"
        if long_position > 0:
            delta_pos = long_position
            print(f"Sold {long_position} shares at ${price} at a profit of ${price-last_long_action_price}/share")
            long_position = 0
            balance += (delta_pos * price)
                    
    return "Hold"

def group_dataframe(df : pd.DataFrame, column : str,  timestamp : int):
    # Remove any items that are more than 1.2 million less than number x
    df[column] = df[column][df[column]> timestamp - 1200000]

    # Group the dataframe into sets of size 60000
    records = df.to_dict('records')

    stat_array = []
    # Calculate the interval group for the 'timestamp' column
    interval = 60000
    df['timestamp_group'] = (df['timestamp'] - df['timestamp'].min()) // interval

    # Group the DataFrame based on the 'timestamp_group' column
    groups = df.groupby('timestamp_group')

    logger.info(len(groups))

    if len(groups) < 20:
        return None, records
    else: 
        last_min = None
        last_max = None
        
        for group in groups:
            logger.info(group[1]["price"])
            min_val = group[1]["price"].min()
            max_val = group[1]["price"].max()
            if last_max is not None and last_min is not None:
                max_diff = max_val - last_max
                min_diff = min_val - last_min
                stat_array.append(np.array([max_diff, min_diff]))
            last_min = min_val
            last_max = max_val

    # Return the results as a NumPy array
    return np.array(stat_array), records

def trading_loop():
    model_min = tf.keras.models.load_model('./bc_model_five_m_min')
    model_max = tf.keras.models.load_model('./bc_model_five_m_max')

    ticker_name = 'AAPL'

    curr_interval = []
    og_timestamp = int(datetime.datetime.now().timestamp() * 1000)

    stat_array = []

    last_prediction = 0
    last_prediction_time = 0
    last_recommendation = "Zero"
    sequence_length = 30

    balance = 100000
    short_position = 0
    last_short_action_price = 0
    long_position = 0
    last_long_action_price = 0
    last_long_action_timestamp = 0

    # Choose the sequence length for input data
    sequence_length = 20
    pred_distance = 5
    last_min = None
    last_max = None

    print(f"timestamp, close, max, min, timestamp+1000*60*5, predicted_value_min, predicted_value_max, modifier, recommendation, position, balance")

    while True:
        try:
            price, timestamp, volume = get_stock_price(ticker_name)
            if (price > (last_long_action_price + 0.07*(7-(timestamp-last_long_action_timestamp)/60000)) and long_position > 0):
                balance += price*long_position
                print(f"Sold {long_position} shares at ${price} at a profit of ${price-last_long_action_price}/share")
                long_position = 0
        except:
            time.sleep(1.8)
            continue
        curr_interval.append({'price': price, 'timestamp': timestamp, "volume" : volume})
        if timestamp > og_timestamp + 1000*60:
            df = pd.DataFrame.from_records(curr_interval)

            max = df["price"].max()
            min = df["price"].min()
            if last_max is not None and last_min is not None:
                max_diff = max - last_max
                min_diff = min - last_min
                stat_array.append(np.array([max_diff, min_diff]))
            last_max = max
            last_min = min
            curr_interval = []
            og_timestamp = timestamp

            if len(stat_array) == sequence_length-1:
                sequence = np.array(stat_array)
                np.save("stat_array_state.npy", stat_array)
                # Reshape the new sequence to match the model input shape
                new_sequence = np.expand_dims(sequence, axis=0)
                predicted_value_min = model_min.predict(new_sequence, verbose=0)
                predicted_value_max = model_max.predict(new_sequence, verbose=0)
                # Convert the predicted scaled value back to the original scale
                recommendation = "Hold"
                modifier = "Weak"
                    
                # if predicted_value_min > 0.55:
                #     recommendation = "Sell"
                #     last_short_action_price = price
                #     delta_pos = -10 - short_position
                #     short_position = -10
                #     balance += (delta_pos * price)
                #     if predicted_value_min > 0.65:
                #         modifier = "Strong"
                #         delta_pos = -20 - short_position
                #         short_position = -20
                #         balance += (delta_pos * price)
                if predicted_value_max > 0.58:
                    recommendation = "Buy"
                    last_long_action_price = price
                    last_long_action_timestamp = timestamp
                    delta_pos = 10 - long_position
                    long_position = 10
                    balance -= (delta_pos * price)
                    if predicted_value_max > 0.65:
                        modifier = "Strong"
                        delta_pos = 20 - long_position
                        long_position = 20
                        balance -= (delta_pos * price)
                elif predicted_value_max < 0.5:
                    recommendation = "Sell"
                    if long_position > 0:
                        delta_pos = long_position
                        print(f"Sold {long_position} shares at ${price} at a profit of ${price-last_long_action_price}/share")
                        long_position = 0
                        balance += (delta_pos * price)
               
                print(f"{timestamp}, {price}, {max}, {min}, {timestamp+1000*60*5}, {predicted_value_min[0][0]}, {predicted_value_max[0][0]}, {modifier}, {recommendation}, {long_position}, {balance}")
                
                last_predicted_min, last_predicted_max = predicted_value_min, predicted_value_max 
                stat_array = stat_array[1:]

        time.sleep(1.9)