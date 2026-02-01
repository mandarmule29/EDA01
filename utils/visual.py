import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Seaborn style set karo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_missing_values(missing_df):
    """
    Missing values ka bar chart
    """
    if missing_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=missing_df, x='Missing_Percentage', y='Column', palette='Reds_r', ax=ax)
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_ylabel('Column Name', fontsize=12)
    ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_distribution(df, column):
    """
    Numeric column ka distribution plot (histogram + KDE)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='#292928', ax=ax, bins=12)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_boxplot(df, column):
    """
    Boxplot for outlier detection
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=df[column], color='#292928', ax=ax)
    ax.set_ylabel(column, fontsize=12)
    ax.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df):
    """
    Correlation matrix ka heatmap
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_categorical_bars(df, column, top_n=10):
    """
    Categorical column ka bar chart (top N values)
    """
    value_counts = df[column].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=value_counts.values, y=value_counts.index, palette='viridis', ax=ax)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    ax.set_title(f'Top {top_n} Values in {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    """
    Scatter plot for two numeric columns
    """
    fig = px.scatter(df, x=x_col, y=y_col,
                     title=f'{x_col} vs {y_col}',

                     color_discrete_sequence=['#ffffff'])
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(size=12)
    )
    return fig


def plot_pairplot_interactive(df, columns):
    """
    Interactive pairplot using Plotly
    """
    fig = px.scatter_matrix(df[columns],
                            dimensions=columns,
                            title='Pairplot of Selected Columns')
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return fig


def plot_time_series(df, date_col, value_col):
    """
    Time series plot for date-based analysis
    """
    temp_df = df[[date_col, value_col]].dropna()
    temp_df = temp_df.sort_values(date_col)

    fig = px.line(temp_df, x=date_col, y=value_col,
                  title=f'{value_col} Over Time',
                  markers=True)
    fig.update_layout(
        xaxis_title=date_col,
        yaxis_title=value_col,
        hovermode='x unified'
    )
    return fig


def plot_sales_by_category(df, category_col, value_col):
    """
    Category-wise sales bar chart
    """
    category_sales = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_sales.values, y=category_sales.index, palette='rocket', ax=ax)
    ax.set_xlabel(f'Total {value_col}', fontsize=12)
    ax.set_ylabel(category_col, fontsize=12)
    ax.set_title(f'{value_col} by {category_col}', fontsize=14, fontweight='bold')

    # Add value labels
    for i, v in enumerate(category_sales.values):
        ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig