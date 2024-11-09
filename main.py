import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from pykalman import KalmanFilter
from datetime import datetime, timedelta

def generate_signals(z_scores):
    signals = []
    for i in range(1, len(z_scores)):
        prev_z_score = z_scores[i - 1]
        curr_z_score = z_scores[i]
        
        # Ensure valid types for comparison
        if isinstance(prev_z_score, (float, np.float64)) and isinstance(curr_z_score, (float, np.float64)):
            if prev_z_score <= -4 and curr_z_score > -4:
                signals.append('Buy Signal (Buy 4)')
            elif prev_z_score <= -3 and curr_z_score > -3:
                signals.append('Buy Signal (Buy 3)')
            elif prev_z_score <= -1 and curr_z_score > -1:
                signals.append('Buy Signal (Buy 2)')
            elif prev_z_score <= -0.4 and curr_z_score > -0.4:
                signals.append('Buy Signal (Buy 1)')
            elif prev_z_score >= 0.25 and curr_z_score < 0.25:
                signals.append('Take Profit Signal (TP 1)')
            elif prev_z_score >= 0.5 and curr_z_score < 0.5:
                signals.append('Take Profit Signal (TP 2)')
            elif prev_z_score >= 1 and curr_z_score < 1:
                signals.append('Take Profit Signal (TP 3)')
            else:
                signals.append('No Signal')
        else:
            signals.append('No Signal')  # Handle invalid z_scores
    return signals

def calculate_profit(last_buy, closing_price):
    if last_buy is None or closing_price is None or last_buy <= 0:
        return None, None
    profit_dollars = closing_price - last_buy
    profit_percent = (profit_dollars / last_buy) * 100 if last_buy > 0 else 0  # Prevent division by zero
    return profit_percent, profit_dollars

def page_zscore_analysis():
    st.title('Stock Z-Score Analysis Signals')

    df_tickers = pd.read_csv("sp500_tickers.csv")
    tickers = df_tickers['Ticker'].tolist()
    
    data_rows = []
    trades = []

    scanning_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_tickers = len(tickers)
    df_placeholder = st.empty()

    for index, ticker in enumerate(tickers):
        try:
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('longName', stock_info.get('shortName', ticker))

            scanning_placeholder.text(f"Scanning: {company_name} ({ticker})")

            stock_data = yf.download(ticker, start=datetime.today() - timedelta(days=1095), end=datetime.today())
            if stock_data.empty:
                st.warning(f"No data found for ticker: {ticker}. Skipping...")
                continue
            
            close_prices = stock_data["Close"].values
            kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=0, 
                              initial_state_covariance=1, observation_covariance=1, transition_covariance=0.01)

            state_means, _ = kf.filter(close_prices)
            kalman_avg = state_means.flatten()

            std_dev = np.std(close_prices)
            z_score = (close_prices - kalman_avg) / std_dev

            signals = generate_signals(z_score)

            last_buy_signal = None
            last_buy_price = None
            last_buy_date = None
            profit_percent = None
            profit_dollars = None

            for i in range(len(signals)):
                timestamp = stock_data.index[i]
                if signals[i].startswith('Buy'):
                    last_buy_signal = ticker
                    last_buy_price = close_prices[i]
                    last_buy_date = timestamp
                    trades.append({
                        'Company': company_name,
                        'Ticker': ticker,
                        'Buy Price': last_buy_price,
                        'Buy Date': last_buy_date,
                        'Status': 'Open',
                        'Floating Profit': 0  # Initialize Floating Profit
                    })
                elif signals[i].startswith('Take Profit') and last_buy_price is not None:
                    closing_price = close_prices[i]
                    profit_percent, profit_dollars = calculate_profit(last_buy_price, closing_price)
                    trades[-1]['Take Profit Price'] = closing_price
                    trades[-1]['Profit (%)'] = profit_percent
                    trades[-1]['Take Profit Date'] = timestamp
                    trades[-1]['Status'] = 'Closed'
                    trades[-1]['Closing Date'] = timestamp  # Add this line for Closing Date
                    trades[-1]['Floating Profit'] = 0  # Reset Floating Profit for closed trades

                # Calculate Floating Profit for open trades
                if trades and trades[-1]['Status'] == 'Open':
                    current_price = close_prices[i] if close_prices[i] is not None else 0
                    if last_buy_price is not None:  # Ensure valid last buy price
                        trades[-1]['Floating Profit'] = current_price - last_buy_price

            data_rows.append([
                f"{company_name} ({ticker})",
                kalman_avg[-1],
                z_score[-2] if len(z_score) > 1 else None,
                z_score[-1],
                last_buy_price,
                last_buy_date,  # Buy date added here
                close_prices[-1],
                trades[-1].get('Closing Date', None),  # Add Closing Date if it exists
                last_buy_signal,
                signals[-1],
                profit_percent,
                profit_dollars,
                trades[-1]['Floating Profit']  # Add Floating Profit to data rows
            ])

            progress_bar.progress((index + 1) / total_tickers)

            df = pd.DataFrame(data_rows, columns=[
                'Company Name (Ticker)',
                'Kalman Avg',
                'Previous Z-Score',
                'Current Z-Score',
                'Last Buy Price',
                'Last Buy Date',  # New column for last buy date
                'Closing Price',
                'Closing Date',
                'Last Buy Signal',
                'Signal',
                'Profit (%)',
                'Profit ($)',
                'Floating Profit'  # Add the new column for Floating Profit
            ])
            df_placeholder.dataframe(df.style.set_table_attributes('style="width:1800px;"'))  # Set max width

        except Exception as e:
            st.error(f"An error occurred for {ticker}: {e}")

    # Clear the progress bar and scanning message after processing
    progress_bar.empty()
    scanning_placeholder.empty()

    # Trade history DataFrame
    st.write('### Trade History')
    trades_df = pd.DataFrame(trades)
    trades_df['Profit (%)'] = trades_df['Profit (%)'].apply(lambda x: "{:.2f}%".format(x) if x is not None else "N/A")
    trades_df['Closing Date'] = trades_df['Take Profit Date']  # Include Closing Date in trade history
    trades_df['Floating Profit'] = trades_df['Floating Profit'].apply(lambda x: "{:.2f}".format(x) if x is not None else "N/A")  # Format floating profit
    st.dataframe(trades_df.style.set_table_attributes('style="width:1800px;"'))  # Set max width for trades DataFrame

# Call the analysis page function
page_zscore_analysis()
