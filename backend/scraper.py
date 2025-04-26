import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

def get_stock_data(ticker_symbol, years=5):
    """
    Get historical stock data using yfinance library.
    
    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'NVDA')
    years (int): Number of years of historical data to retrieve
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"Downloading {years} years of data for {ticker_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Download data using yfinance
        data = yf.download(
            ticker_symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=True
        )
        
        # Check if we got data
        if data is None or data.empty:
            print(f"No data retrieved for {ticker_symbol}")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Print raw column names for debugging
        print(f"Raw column names: {list(data.columns)}")
        
        # Rename the columns to flatten the MultiIndex columns
        if isinstance(data.columns[0], tuple):
            # We have MultiIndex columns
            new_columns = []
            for col in data.columns:
                if col[0] == 'Date' or col[0] == 'date':
                    new_columns.append('date')
                else:
                    # For other columns like (Close, NVDA), use just the first part
                    new_columns.append(col[0].lower())
            
            # Rename columns
            data.columns = new_columns
            print(f"Renamed columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

def calculate_metrics(df):
    """
    Calculate additional metrics like moving averages and returns.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with historical stock data
    
    Returns:
    pandas.DataFrame: DataFrame with additional metrics
    """
    if df is None or df.empty:
        return None
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get the appropriate column names based on what's available
    # First try to identify the close price column
    close_column = None
    for possible_name in ['close', 'Close', 'close_nvda', 'Close_nvda', 'Adj Close', 'adj close']:
        if possible_name in result_df.columns:
            close_column = possible_name
            print(f"Using '{close_column}' as close price column")
            break
    
    if close_column is None:
        print(f"Could not find a close price column. Available columns: {list(result_df.columns)}")
        return None
    
    # Calculate moving averages
    result_df['ma_50'] = result_df[close_column].rolling(window=50).mean()
    result_df['ma_200'] = result_df[close_column].rolling(window=200).mean()
    
    # Calculate daily returns
    result_df['daily_return'] = result_df[close_column].pct_change() * 100
    
    # Calculate monthly returns (21 trading days approximation)
    result_df['monthly_return'] = result_df[close_column].pct_change(21) * 100
    
    # Calculate yearly returns (252 trading days approximation)
    result_df['yearly_return'] = result_df[close_column].pct_change(252) * 100
    
    # Calculate volatility (standard deviation of returns over 21 days)
    result_df['volatility_21d'] = result_df['daily_return'].rolling(window=21).std()
    
    return result_df

def save_to_csv(dataframe, filename):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    dataframe (pandas.DataFrame): DataFrame to save
    filename (str): Name of the output file
    """
    dataframe.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    # Set the ticker symbol
    ticker = "NVDA"
    
    # Get data using yfinance
    print(f"Getting 5 years of historical data for {ticker}...")
    df = get_stock_data(ticker, years=5)
    
    if df is not None and not df.empty:
        print(f"Successfully obtained {len(df)} days of data")
        print(f"Final columns in the data: {list(df.columns)}")
        
        # Calculate metrics
        df_with_metrics = calculate_metrics(df)
        
        if df_with_metrics is not None:
            # Save to CSV
            filename = f"{ticker}_5yr_history_{datetime.now().strftime('%Y%m%d')}.csv"
            save_to_csv(df_with_metrics, filename)
            
            # Display summary of results
            print("\nData summary:")
            print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            # Display recent data
            print("\nMost recent data:")
            display_cols = ['date', 'open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                display_cols.append('volume')
            
            # Only include columns that actually exist
            display_cols = [col for col in display_cols if col in df.columns]
            print(df.tail(5)[display_cols])
            
            # Calculate some basic statistics
            latest_price = float(df['close'].iloc[-1])  # Convert to float to avoid formatting issues
            year_ago_index = -252 if len(df) > 252 else 0
            year_ago_price = float(df['close'].iloc[year_ago_index])
            yearly_return = ((latest_price / year_ago_price) - 1) * 100
            
            print(f"\nCurrent price: ${latest_price:.2f}")
            print(f"1-year return: {yearly_return:.2f}%")
            
            if 'volume' in df.columns:
                avg_volume = float(df['volume'].tail(21).mean())
                print(f"Average daily volume (last month): {avg_volume:.0f}")
            
            if 'volatility_21d' in df_with_metrics.columns:
                volatility = float(df_with_metrics['volatility_21d'].iloc[-1])
                print(f"21-day volatility: {volatility:.2f}%")
        else:
            print("Failed to calculate metrics.")
    else:
        print("\nFailed to retrieve data.")

if __name__ == "__main__":
    main()