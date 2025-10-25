# config.py
# Configuration settings for the analysis

ANALYSIS_CONFIG = {
    'data_cleaning': {
        'handle_missing_values': True,
        'handle_outliers': True,
        'create_features': True
    },
    'analysis': {
        'temporal_analysis': True,
        'category_analysis': True,
        'regional_analysis': True,
        'profitability_analysis': True
    },
    'visualization': {
        'interactive_charts': True,
        'save_html': True,
        'chart_theme': 'plotly_white'
    },
    'output': {
        'save_excel': True,
        'save_csv': True,
        'save_visualizations': True,
        'create_dashboard': True
    }
}

# Column mappings for different dataset structures
COLUMN_MAPPINGS = {
    'sales_columns': ['Sales', 'Revenue', 'Amount'],
    'profit_columns': ['Profit', 'Net_Profit', 'Margin'],
    'date_columns': ['Order_Date', 'Date', 'Transaction_Date'],
    'category_columns': ['Category', 'Product_Category', 'Department'],
    'region_columns': ['Region', 'Territory', 'Area']
}