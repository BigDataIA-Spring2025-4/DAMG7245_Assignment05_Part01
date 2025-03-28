
import os
import pandas as pd
import snowflake.connector
from typing import Dict, Any, List
from io import StringIO
import re


required_env_vars = {
    'SNOWFLAKE_ACCOUNT': os.getenv('SNOWFLAKE_ACCOUNT'),
    'SNOWFLAKE_USER': os.getenv('SNOWFLAKE_USER'),
    'SNOWFLAKE_PASSWORD': os.getenv('SNOWFLAKE_PASSWORD'),
    'SNOWFLAKE_ROLE': os.getenv('SNOWFLAKE_ROLE'),
    'SNOWFLAKE_DB': os.getenv('SNOWFLAKE_DB'),
    'SNOWFLAKE_WAREHOUSE': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'SNOWFLAKE_SCHEMA': os.getenv('SNOWFLAKE_SCHEMA')
}


def get_query_for_analysis_type(analysis_type: str, year: str = None, quarter: List[str] = None) -> str:
    """
    Generate appropriate SQL query based on analysis type
    
    Args:
        analysis_type (str): Type of financial analysis
        year (str, optional): Year to filter
        quarter (List[str], optional): Quarters to filter
    
    Returns:
        str: Constructed SQL query
    """
    # Default query mapping
    queries = {
        'stock_performance': """
        SELECT 
            DATE, 
            OPEN, 
            HIGH, 
            LOW, 
            CLOSE, 
            VOLUME,
            ROUND((CLOSE - LAG(CLOSE) OVER (ORDER BY DATE)) / LAG(CLOSE) OVER (ORDER BY DATE) * 100, 2) AS DAILY_RETURN_PERCENT
        FROM NVDA_HISTORICAL_5Y
        """,
        
        'financial_summary': """
        SELECT 
            DATE, 
            TOTAL_REVENUE, 
            NET_INCOME, 
            EBITDA, 
            DILUTED_EPS
        FROM NVDA_FINANCIALS
        """,
        
        'balance_sheet': """
        SELECT 
            DATE, 
            TOTAL_ASSETS, 
            TOTAL_LIABILITIES_NET_MINORITY_INTEREST, 
            STOCKHOLDERS_EQUITY, 
            WORKING_CAPITAL
        FROM NVDA_BALANCE_SHEET
        """,
        
        'earnings_analysis': """
        SELECT 
            DATE, 
            TOTAL_REVENUE, 
            NET_INCOME, 
            BASIC_EPS, 
            DILUTED_EPS, 
            GROSS_PROFIT
        FROM NVDA_FINANCIALS
        """
    }
    
    # Select base query
    base_query = queries.get(analysis_type, queries['financial_summary'])
    
    # Add year filtering
    if year:
        base_query += f" WHERE SUBSTRING(DATE, 1, 4) = '{year}'"
    
    # Add quarter filtering if applicable
    if quarter and year:
        quarter_conditions = []
        quarter_map = {
            'Q1': "(SUBSTRING(DATE, 6, 2) IN ('01', '02', '03'))",
            'Q2': "(SUBSTRING(DATE, 6, 2) IN ('04', '05', '06'))",
            'Q3': "(SUBSTRING(DATE, 6, 2) IN ('07', '08', '09'))",
            'Q4': "(SUBSTRING(DATE, 6, 2) IN ('10', '11', '12'))"
        }
        
        for q in quarter:
            if q in quarter_map:
                quarter_conditions.append(quarter_map[q])
        
        if quarter_conditions:
            base_query += ' AND (' + ' OR '.join(quarter_conditions) + ')'
    
    base_query += ' ORDER BY DATE DESC LIMIT 100'
    
    return base_query

def generate_analysis_summary(df: pd.DataFrame, analysis_type: str) -> str:
    """
    Generate summary based on analysis type
    
    Args:
        df (pd.DataFrame): DataFrame with analysis results
        analysis_type (str): Type of analysis performed
    
    Returns:
        str: Textual summary of analysis
    """
    summary_generators = {
        'stock_performance': lambda df: f"""
        Stock Performance Analysis:
        - Total trading days: {len(df)}
        - Price Range: ${df['CLOSE'].min():.2f} - ${df['CLOSE'].max():.2f}
        - Average Daily Return: {df['DAILY_RETURN_PERCENT'].mean():.2f}%
        """,
        
        'financial_summary': lambda df: f"""
        Financial Summary:
        - Average Revenue: ${df['TOTAL_REVENUE'].mean():,.2f}
        - Average Net Income: ${df['NET_INCOME'].mean():,.2f}
        - Average Diluted EPS: ${df['DILUTED_EPS'].mean():.2f}
        """,
        
        'balance_sheet': lambda df: f"""
        Balance Sheet Overview:
        - Average Total Assets: ${df['TOTAL_ASSETS'].mean():,.2f}
        - Average Total Liabilities: ${df['TOTAL_LIABILITIES_NET_MINORITY_INTEREST'].mean():,.2f}
        - Average Stockholders Equity: ${df['STOCKHOLDERS_EQUITY'].mean():,.2f}
        """,
        
        'earnings_analysis': lambda df: f"""
        Earnings Analysis:
        - Average Gross Profit: ${df['GROSS_PROFIT'].mean():,.2f}
        - Average Net Income: ${df['NET_INCOME'].mean():,.2f}
        - Average Basic EPS: ${df['BASIC_EPS'].mean():.2f}
        """
    }
    
    # Use appropriate summary generator or default to financial summary
    return summary_generators.get(analysis_type, summary_generators['financial_summary'])(df)


def snowflake_query_agent(
    query: str = None, 
    analysis_type: str = 'financial_summary', 
    year: str = None, 
    quarter: List[str] = None
):
    """
    Comprehensive Snowflake query tool for financial analysis
    
    Args:
        query (str, optional): Custom or natural language query
        analysis_type (str, optional): Type of analysis to perform
        year (str, optional): Filter for specific year
        quarter (List[str], optional): Filter for specific quarters
    
    Returns:
        Dict containing analysis results
    """
    try:
        # Snowflake Connection
        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            role=os.getenv('SNOWFLAKE_ROLE'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DB'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
        )
        
        # Get appropriate SQL query based on analysis type
        sql_query = get_query_for_analysis_type(analysis_type, year, quarter)
        
        # Execute query
        df = pd.read_sql(sql_query, conn)
        
        # Generate insights
        summary = generate_analysis_summary(df, analysis_type)
        
        # Create visualizations
        # visualizations = create_visualizations(df, analysis_type)
        print("query.....->", sql_query)
        return {
            'data': df.to_dict(orient='records'),
            'summary': summary,
            'analysis_type': analysis_type
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'detail': 'Failed to query Snowflake database',
            'query': analysis_type
        }
    
    finally:
        if 'conn' in locals():
            conn.close()
