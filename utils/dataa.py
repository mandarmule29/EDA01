import pandas as pd
import numpy as np


def load_data(file):
    """
    CSV ya Excel file load karta hai
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return None

        # Automatically detect and convert date columns
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass

        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def get_basic_info(df):
    """
    Dataset ki basic information return karta hai
    """
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2  # MB me
    }
    return info


def get_missing_values(df):
    """
    Missing values ka analysis
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    return missing


def get_numeric_summary(df):
    """
    Numeric columns ka statistical summary
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols].describe()
    return None


def get_categorical_summary(df):
    """
    Categorical columns ka summary
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    summary = {}
    for col in cat_cols:
        summary[col] = {
            'unique_values': df[col].nunique(),
            'top_5_values': df[col].value_counts().head(5).to_dict()
        }
    return summary


def detect_outliers(df, column):
    """
    IQR method se outliers detect karta hai
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound


def get_date_columns(df):
    """
    Date columns ko detect karta hai
    """
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
    return date_cols


def get_date_range(df, date_column):
    """
    Date column ka range return karta hai
    """
    if date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return {
            'min_date': df[date_column].min(),
            'max_date': df[date_column].max(),
            'days_range': (df[date_column].max() - df[date_column].min()).days
        }
    return None