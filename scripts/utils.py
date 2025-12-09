"""
Utility functions for the NBA game prediction pipeline.
"""

import pandas as pd


def parse_datetime_column(df: pd.DataFrame, column_name: str = 'gameDateTimeEst') -> pd.DataFrame:
    """
    Parse datetime column handling mixed formats (with and without timezone).
    
    Args:
        df: DataFrame with datetime column
        column_name: Name of the datetime column
        
    Returns:
        DataFrame with parsed datetime column (timezone removed)
    """
    df[column_name] = pd.to_datetime(df[column_name], format='mixed', utc=True).dt.tz_localize(None)
    return df
