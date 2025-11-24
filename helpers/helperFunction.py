import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import finnhub
import yfinance as yf
import time
from datetime import datetime, timedelta
import requests
from io import StringIO
from typing import List, Literal

class NewsHelper:
    def __init__(self, api_key):
        # Initialize the client here (all methods will use self.client) 
        self.client = finnhub.Client(api_key=api_key)
    
    def get_company_news(self, ticker, start_date, end_date, chunk_size=7):
        """
        Fetches news for a single ticker. 
        Uses self.client so no API key needed in arguments.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_news_data = []
        current_start = start_dt
        
        # Iterate forward
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            
            print(f"Fetching {ticker}: {current_start.date()} to {current_end.date()}")
            
            attempts = 0
            success = False
            
            while attempts < 3 and not success:
                try:
                    # Use self.client here
                    news_chunk = self.client.company_news(
                        ticker, 
                        _from=current_start.strftime('%Y-%m-%d'), 
                        to=current_end.strftime('%Y-%m-%d')
                    )
                    
                    if news_chunk:
                        chunk_df = pd.DataFrame(news_chunk)
                        
                        if 'datetime' in chunk_df.columns:
                            chunk_df['datetime'] = pd.to_datetime(chunk_df['datetime'], unit='s', errors='coerce')
                            chunk_df = chunk_df.dropna(subset=['datetime'])
                            chunk_df['datetime'] = chunk_df['datetime'].dt.strftime('%Y-%m-%d')
                            chunk_df['ticker'] = ticker
                            all_news_data.append(chunk_df)
                    
                    success = True

                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit hit. Sleeping for 60 seconds...")
                        time.sleep(61)
                        attempts += 1
                    else:
                        print(f"Error fetching chunk: {e}")
                        break

            # Rate Limiting
            time.sleep(1.1)
            current_start = current_end + timedelta(days=1)

        if not all_news_data:
            return pd.DataFrame(columns=['datetime', 'summary', 'ticker'])

        all_news_df = pd.concat(all_news_data, ignore_index=True)
        
        cols_to_keep = ['datetime', 'summary', 'ticker']
        existing_cols = [c for c in cols_to_keep if c in all_news_df.columns]
        
        return all_news_df[existing_cols]

    def get_group_news(self, ticker_list, start_date, end_date, chunk_size=7, save = False):
        """
        Iterates through a list of tickers and returns a concatenated DataFrame.
        """
        group_news_data = []
        total_tickers = len(ticker_list)
        
        print(f"Starting collection for {total_tickers} companies...\n")
        
        for i, ticker in enumerate(ticker_list):
            print(f"--- Processing {ticker} ({i+1}/{total_tickers}) ---")
            
            try:
                # We use self.get_company_news and we don't need to pass the api key
                df = self.get_company_news(ticker, start_date, end_date, chunk_size)
                
                if not df.empty:
                    group_news_data.append(df)
                    print(f"Found {len(df)} articles for {ticker}")
                else:
                    print(f"No data found for {ticker}")
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                
        if not group_news_data:
            print("\nNo news found for any ticker")
            return pd.DataFrame(columns=['datetime', 'summary', 'ticker'])
        
        print("\nConcatenating all data")
        final_df = pd.concat(group_news_data, ignore_index=True)
        final_df = final_df.sort_values(by=['ticker', 'datetime']).reset_index(drop=True)
        
        if save:
            tickers_str = "-".join(ticker_list)
            if len(tickers_str) > 50: tickers_str = "multi_group"
            filename = f"news_{tickers_str}_{start_date}_{end_date}.csv"
            final_df.to_csv(filename, index=False)
            print(f"Saved to {filename}")

        return final_df
    

class TickerHelper:
    # Predefined URLs for common indices
    INDEX_URLS = {
        "SP500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "table_id": "constituents"
        },
        "DOW": {
            "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            "table_id": "constituents"
        },
        "NASDAQ100": {
            "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
            "table_id": "constituents"
        },
        "EURO STOXX 50": {
            "url": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
            "table_id": "constituents"
        }
    }

    @staticmethod
    def get_tickers(index_name_or_url: str):
        """
        Accepts either a predefined key ('SP500', 'DOW', 'NASDAQ100') 
        OR a raw Wikipedia URL.
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # Check if the input is a predefined key
        if index_name_or_url.upper() in TickerHelper.INDEX_URLS:
            info = TickerHelper.INDEX_URLS[index_name_or_url.upper()]
            url = info['url']
            table_attr = {"id": info['table_id']}
        else:
            # Assume its raw URL if not in dict
            url = index_name_or_url
            table_attr = {"id": "constituents"} # Default assumption

        print(f"Fetching tickers from: {url}...")

        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            
            # Wrap HTML string in StringIO
            html_io = StringIO(resp.text)
            
            # Parse tables
            tables = pd.read_html(html_io, attrs=table_attr)
            
            if not tables:
                raise ValueError("No tables found on the page.")
                
            df = tables[0]
            
            # Wikipedia column names vary slightly by page => look for 'Symbol' or 'Ticker'
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
            elif 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
            else:
                # Fallback -> assume first column
                tickers = df.iloc[:, 0].tolist()

            # Clean tickers for yfinance (e.g. change 'BRK.B' to 'BRK-B')
            tickers = [str(t).replace(".", "-") for t in tickers]
            
            print(f"Successfully extracted {len(tickers)} tickers.")
            return tickers

        except Exception as e:
            print(f"Error getting tickers: {e}")
            return []

    @staticmethod
    def get_filtered_stock_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        ohlc_type: Literal['Open', 'High', 'Low', 'Close'] = 'Close'
    ) -> pd.DataFrame:
        
        if not tickers:
            print("Ticker list is empty.")
            return pd.DataFrame()

        print(f"Downloading {ohlc_type} data for {len(tickers)} tickers...")
        
        try:
            # yf.download allows downloading multiple tickers at once
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if data.empty:
                print("No data downloaded.")
                return pd.DataFrame()

            # Handle MultiIndex columns (Open, High, Low, Close, Volume)
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    prices = data[ohlc_type]
                except KeyError:
                    # Fallback if auto_adjust=True changed column names (sometimes becomes just 'Close')
                    # Or if yfinance structure changed
                    prices = data
            else:
                prices = data

            # If result is a Series (single ticker), convert to DataFrame
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
                if len(tickers) == 1:
                    prices.columns = tickers

            # Remove columns with any missing data (NaNs)
            # to ensure strict data quality for models
            original_count = len(prices.columns)
            prices = prices.dropna(axis=1) 
            final_count = len(prices.columns)
            
            dropped = original_count - final_count
            if dropped > 0:
                print(f"Dropped {dropped} tickers due to incomplete data in range.")
            
            print(f"Returning data for {final_count} valid tickers.")
            return prices

        except Exception as e:
            print(f"An error occurred during download: {e}")
            return pd.DataFrame()
        
class FeatureEngineerHelper:
    @staticmethod
    def calculate_daily_returns(df):
        return df.pct_change()

    @staticmethod
    def calculate_excess_return(df_universe, df_benchmark):
        return df_universe.sub(df_benchmark.iloc[:,0], axis = 0)
    
    @staticmethod
    def generate_quantile_labels(data_row, low_q, high_q) -> pd.Series:
        q_low = data_row.quantile(low_q)
        q_high = data_row.quantile(high_q)

        # Baseline -> 1 (Flat/Neutral)
        labels = pd.Series(1, index = data_row.index) 

        labels[data_row > q_high] = 2  # -> Up (Outperformed the high quantile of the universe)
        labels[data_row < q_low] = 0   # -> Down (Underperformed the low quantile of the universe)

        return labels

    
class SentimentHelper:
    @staticmethod
    def sentiment_distribution(df, printPlot = True, printStats = False):
        
        # Check if the 'sentiment' column exists in the dataframe
        if 'sentiment' not in df.columns:
            raise ValueError("No column named 'sentiment' found")

        if printPlot:
            plt.figure(figsize=(8, 6))


            # Plot the sentiment distribution
            df['sentiment'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
            
            # Add labels and title
            plt.title('Sentiment Distribution', fontsize=16)
            plt.xlabel('Sentiment', fontsize=14)
            plt.ylabel('Count', fontsize=14)

            # Show the plot
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()
            plt.show()
        
        if printStats:
            total_observ = len(df)
            pos = len(df[df['sentiment'] == 'positive'])
            neu = len(df[df['sentiment'] == 'neutral'])
            neg = len(df[df['sentiment'] == 'negative'])

            print(f'Total number of observations: {total_observ}')
            print(f'Positive news: {round(pos/total_observ, 4)*100}%')
            print(f'Neutral news: {round(neu/total_observ, 4)*100}%')
            print(f'Negative news: {round(neg/total_observ, 4)*100}%')

        return