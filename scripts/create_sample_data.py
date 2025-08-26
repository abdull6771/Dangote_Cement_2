"""Script to create sample financial data for testing."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config import CSV_DIR, ANALYSIS_YEARS
from utils import create_metadata

def create_sample_financial_data():
    """Create sample financial data for testing."""
    
    # Sample revenue data (in millions NGN)
    revenue_data = {
        2018: 928000,
        2019: 1034000,
        2020: 1034000,
        2021: 1383000,
        2022: 1613000,
        2023: 1750000
    }
    
    # Sample profit data (in millions NGN)
    profit_data = {
        2018: 275000,
        2019: 316000,
        2020: 276000,
        2021: 378000,
        2022: 420000,
        2023: 485000
    }
    
    # Sample EBITDA data (in millions NGN)
    ebitda_data = {
        2018: 425000,
        2019: 485000,
        2020: 445000,
        2021: 580000,
        2022: 650000,
        2023: 720000
    }
    
    # Sample assets data (in millions NGN)
    assets_data = {
        2018: 1850000,
        2019: 2100000,
        2020: 2250000,
        2021: 2450000,
        2022: 2680000,
        2023: 2850000
    }
    
    # Create CSV files for each year
    for year in ANALYSIS_YEARS:
        if year in revenue_data:
            # Income Statement
            income_statement = pd.DataFrame({
                'Item': ['Revenue', 'Cost of Sales', 'Gross Profit', 'Operating Expenses', 'EBITDA', 'Net Profit'],
                'Amount': [
                    revenue_data[year],
                    revenue_data[year] * 0.6,  # 60% cost of sales
                    revenue_data[year] * 0.4,  # 40% gross profit
                    revenue_data[year] * 0.15, # 15% operating expenses
                    ebitda_data[year],
                    profit_data[year]
                ],
                'Year': year,
                'Currency': 'NGN',
                'Unit': 'Millions'
            })
            
            income_statement.to_csv(CSV_DIR / f"{year}_income_statement_0.csv", index=False)
            
            # Balance Sheet
            balance_sheet = pd.DataFrame({
                'Item': ['Total Assets', 'Current Assets', 'Non-Current Assets', 'Total Liabilities', 'Total Equity'],
                'Amount': [
                    assets_data[year],
                    assets_data[year] * 0.3,  # 30% current assets
                    assets_data[year] * 0.7,  # 70% non-current assets
                    assets_data[year] * 0.4,  # 40% liabilities
                    assets_data[year] * 0.6   # 60% equity
                ],
                'Year': year,
                'Currency': 'NGN',
                'Unit': 'Millions'
            })
            
            balance_sheet.to_csv(CSV_DIR / f"{year}_balance_sheet_0.csv", index=False)
            
            # Segment Reporting
            segment_data = pd.DataFrame({
                'Segment': ['Nigeria Operations', 'Pan-Africa Operations', 'Other Operations'],
                'Revenue': [
                    revenue_data[year] * 0.75,  # 75% Nigeria
                    revenue_data[year] * 0.20,  # 20% Pan-Africa
                    revenue_data[year] * 0.05   # 5% Other
                ],
                'Profit': [
                    profit_data[year] * 0.80,   # 80% Nigeria
                    profit_data[year] * 0.15,   # 15% Pan-Africa
                    profit_data[year] * 0.05    # 5% Other
                ],
                'Year': year,
                'Currency': 'NGN',
                'Unit': 'Millions'
            })
            
            segment_data.to_csv(CSV_DIR / f"{year}_segment_reporting_0.csv", index=False)
            
            print(f"Created sample data for {year}")

def create_sample_text_data():
    """Create sample text chunks for testing."""
    
    sample_texts = {
        2018: {
            "Chairman's Statement": "The year 2018 was challenging for Dangote Cement due to currency fluctuations and increased competition. However, we maintained our market leadership position in Nigeria and continued expansion in Pan-Africa operations.",
            "CEO's Statement": "Our strategic focus on operational efficiency and cost optimization helped us navigate the difficult market conditions in 2018. We invested heavily in technology and process improvements.",
            "Risk Factors": "Key risks in 2018 included foreign exchange volatility, regulatory changes in key markets, and increased competition from imported cement."
        },
        2019: {
            "Chairman's Statement": "2019 showed improved performance with better market conditions and successful implementation of our efficiency programs. Revenue grew significantly compared to 2018.",
            "CEO's Statement": "We achieved strong operational performance in 2019 through our focus on volume growth and market share expansion across all our operations.",
            "Risk Factors": "Main risks in 2019 were political instability in some African markets, currency devaluation, and supply chain disruptions."
        },
        2020: {
            "Chairman's Statement": "The COVID-19 pandemic presented unprecedented challenges in 2020, but our resilient business model and strong balance sheet helped us weather the storm.",
            "CEO's Statement": "Despite the pandemic, we maintained operations across all markets and implemented comprehensive health and safety protocols for our employees.",
            "Risk Factors": "2020 risks included pandemic-related disruptions, reduced construction activity, and supply chain constraints due to lockdown measures."
        },
        2021: {
            "Chairman's Statement": "2021 marked a strong recovery with record revenue and profitability. Our Pan-Africa strategy continued to deliver results with improved performance across all markets.",
            "CEO's Statement": "We achieved exceptional results in 2021 through operational excellence, market expansion, and successful cost management initiatives.",
            "Risk Factors": "Key risks in 2021 included inflationary pressures, energy cost increases, and ongoing pandemic-related uncertainties."
        },
        2022: {
            "Chairman's Statement": "2022 was another year of strong performance with continued growth in both Nigeria and Pan-Africa operations. We maintained our position as Africa's leading cement producer.",
            "CEO's Statement": "Our focus on sustainability and operational efficiency drove strong results in 2022. We made significant investments in renewable energy and process optimization.",
            "Risk Factors": "Main risks in 2022 were rising energy costs, supply chain inflation, and geopolitical tensions affecting global commodity markets."
        },
        2023: {
            "Chairman
