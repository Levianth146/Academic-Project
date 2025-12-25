# Import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('data_raw_VN.xlsx')
df.rename(columns={'Unnamed: 2': 'Firm ID', 'Unnamed: 4': 'Date'}, inplace=True)

# Part 1: Data Cleaning and Summary
# 1.1. Identify and list all columns with missing values.
missing_values = df.isnull().sum()
print('Columns with missing values:')
print(missing_values[missing_values > 0])

# 1.2. Replace missing values
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    df[column].fillna(df[column].median(), inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    mode_series = df[column].mode()
    if not mode_series.empty:
        df[column].fillna(mode_series[0], inplace=True)

# 1.3. Convert columns
object_columns = df.select_dtypes(include='object').columns
num_columns = object_columns.drop(['Unnamed: 0', 'Unnamed: 1', 'Firm ID', 'Unnamed: 3', 'industry'])
for column in num_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Part 2: Data Integrity Checks
# 2.1. Check duplicates
duplicate_rows = df.duplicated()
print(f"Number of duplicate rows: {duplicate_rows.sum()}")
df.drop_duplicates(inplace=True)

# 2.2. Validate date range
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
invalid_dates = df[(df['Date'] < '2008-01-01') | (df['Date'] > '2023-12-31')]
df = df[(df['Date'] >= '2008-01-01') & (df['Date'] <= '2023-12-31')]


# Part 3: Derived Metrics (Expanded)
# 3.1. Debt Ratio
invalid_assets = (df['Total Assets'].isna()) | (df['Total Assets'] == 0)
df['Debt Ratio'] = np.where(
    invalid_assets,
    np.nan,
    df['Total Liabilities'] / df['Total Assets']
)

# 3.2. Earnings Quality
df['Earnings Quality'] = df['Net Income after Tax'] / df['Operating Expenses - Total']

def classify_earnings_quality(val):
    if pd.isna(val):
        return 'Undefined'
    elif val >= 1.0:
        return 'High Quality'
    elif 0.5 <= val < 1.0:  # FIXED: was 0.5 <= val <= 0.1 (invalid)
        return 'Medium Quality'
    else:
        return 'Low Quality'

df['Earnings Quality Categories'] = df['Earnings Quality'].apply(classify_earnings_quality)

# 3.3. Liquidity Index
numerator = df['Cash & Cash Equivalents - Total'] + df['Working Capital']
denominator = df['Total Current Assets']
invalid_liquidity = numerator.isna() | denominator.isna() | (denominator == 0)
df['Liquidity Index'] = np.where(
    invalid_liquidity,
    np.nan,
    numerator / denominator
)
df['Liquidity Status'] = np.where(df['Liquidity Index'] < 0.1, 'Low Liquidity', 'Normal Liquidity')

# 3.4. Profit Growth
df['Year'] = df['Date'].dt.year
df = df.sort_values(by=['Firm ID', 'Year']).reset_index(drop=True)
df['Profit Growth'] = df.groupby('Firm ID')['Net Income after Tax'].pct_change() * 100
df.loc[df.groupby('Firm ID').head(1).index, 'Profit Growth'] = np.nan

# 3.5. Profit Margin
df['Profit Margin'] = df['Net Income after Tax'] / df['Revenue from Business Activities - Total']


# =============================================================================
# Part 4: Data Aggregation (Expanded)
# =============================================================================

# 4.1. Top 3 industries by average Profit Margin & highest Market Cap firm
average_profit_margin = df.groupby('industry')['Profit Margin'].mean().sort_values(ascending=False).head(3)
print('Top three industries by average Profit Margin:')
print(average_profit_margin)

highest_market_cap = df.loc[df.groupby('industry')['Market Capitalization'].idxmax()]
industry_firm_year = highest_market_cap[['industry', 'Firm ID', 'Year', 'Market Capitalization']]
print('Firm with the highest Market Capitalization in each industry:')
print(industry_firm_year)

# 4.2. Firms with Net Income > median
median_net_income = df['Net Income after Tax'].median()
high_net_income_firms = df[df['Net Income after Tax'] > median_net_income]
avg_assets = high_net_income_firms['Total Assets'].mean()
avg_debt_ratio = high_net_income_firms['Debt Ratio'].mean()
print(f'Average Total Assets: {avg_assets}')
print(f'Average Debt Ratio: {avg_debt_ratio}')

high_debt_risk_firms = high_net_income_firms[high_net_income_firms['Debt Ratio'] > 0.5]
pct_high_debt = (len(high_debt_risk_firms) / len(high_net_income_firms)) * 100
print(f'Percentage with High Debt Risk: {pct_high_debt:.2f}%')

# 4.3. Pivot table
df_valid_age = df[df['age'] >= 0]
pivot_table = df_valid_age.pivot_table(
    index=['age', 'Year'],
    values={
        'Revenue from Goods & Services': 'sum',
        'Net Income after Tax': 'mean',
        'Operating Expenses - Total': 'std'
    },
    aggfunc={
        'Revenue from Goods & Services': 'sum',
        'Net Income after Tax': 'mean',
        'Operating Expenses - Total': 'std'
    }
).reset_index()
print('Pivot table:')
print(pivot_table.head())

# 4.4. 3-Year Moving Average Revenue & declining firms
df['3-Year Moving Average Revenue'] = df.groupby('Firm ID')['Revenue from Business Activities - Total'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

declining_trend = df.groupby('Firm ID').apply(
    lambda g: g['3-Year Moving Average Revenue'].iloc[-3:].is_monotonic_decreasing
)
declining_firms = declining_trend[declining_trend].index.tolist()
print(f'Firms with declining trend: {declining_firms}')


# =============================================================================
# Part 5: Business Insights (Expanded)
# =============================================================================

# 5.1. Top 10 firms by Profit Margin
top_10_firms = df.groupby('Firm ID')['Profit Margin'].mean().sort_values(ascending=False).head(10)
top_10_data = df[df['Firm ID'].isin(top_10_firms.index)]

avg_cogs = top_10_data['Cost of Goods Sold / Sales, % (Pvt)'].mean()
print(f'Average COGS/Sales %: {avg_cogs:.2f}')

lowest_debt = top_10_data.loc[top_10_data.groupby('Firm ID')['Debt Ratio'].idxmin()]
print('Year with lowest Debt Ratio for top 10 firms:')
print(lowest_debt[['Firm ID', 'Year', 'Debt Ratio']])

# 5.2. Market Cap volatility and % change
volatility = df.groupby(['Firm ID', 'industry'])['Market Capitalization'].std().reset_index()
most_volatile = volatility.loc[volatility.groupby('industry')['Market Capitalization'].idxmax()]
print('Most volatile firms by industry:')
print(most_volatile)

avg_mc_by_year = df.groupby(['industry', 'Year'])['Market Capitalization'].mean().reset_index()
earliest_latest = avg_mc_by_year.groupby('industry').agg(
    Earliest_Year=('Year', 'min'),
    Latest_Year=('Year', 'max'),
    Avg_Earliest=('Market Capitalization', lambda x: x.iloc[0]),
    Avg_Latest=('Market Capitalization', lambda x: x.iloc[-1])
).reset_index()
earliest_latest['Percentage Change'] = (
    (earliest_latest['Avg_Latest'] - earliest_latest['Avg_Earliest']) /
    earliest_latest['Avg_Earliest'] * 100
)
print('Percentage change in Market Cap:')
print(earliest_latest[['industry', 'Percentage Change']])

# 5.3. High Liquidity firms
high_liquidity_firms = df[df['Liquidity Index'] > 0.5]
avg_gross_rev = high_liquidity_firms['Gross Revenue from Business Activities - Total'].mean()
print(f'Average Gross Revenue for high liquidity firms: {avg_gross_rev:.2f}')

high_liquidity_firms['Gap'] = (
    high_liquidity_firms['Cash & Cash Equivalents - Total'] -
    high_liquidity_firms['Short-Term Debt & Current Portion of Long-Term Debt']
)
largest_gap_firm = high_liquidity_firms.loc[high_liquidity_firms['Gap'].idxmax(), ['Firm ID', 'Gap']]
print('Firm with largest cash-debt gap:')
print(largest_gap_firm)


# =============================================================================
# Part 6: Advanced Analysis and Challenges (Expanded)
# =============================================================================

# 6.1. Z-score standardization and outliers
columns_to_standardize = ['Net Income after Tax', 'Total Assets', 'Market Capitalization']
for col in columns_to_standardize:
    df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()

outliers = df[
    (df[['Net Income after Tax_zscore', 'Total Assets_zscore', 'Market Capitalization_zscore']] > 2.5)
    .sum(axis=1) >= 2
]
print(f'Number of outlier firms: {len(outliers)}')
avg_depreciation = outliers['Depreciation - Total'].mean()
print(f'Average Depreciation for outliers: {avg_depreciation:.2f}')

# 6.2. Filtered dataset
criteria = (
    (df['Working Capital'] > 0) &
    (df['Debt Ratio'] < df['Debt Ratio'].median()) &
    (df['Profit Margin'] > 0.1)
)
filtered_data = df[criteria]
print(f'Filtered dataset: {filtered_data.shape[0]} rows')
print(f'Number of firms: {filtered_data["Firm ID"].nunique()}')
print(f'Number of industries: {filtered_data["industry"].nunique()}')
print('Average financial metrics:')
print(filtered_data[['Working Capital', 'Debt Ratio', 'Profit Margin']].mean())
filtered_data.to_excel('filtered_dataset.xlsx', index=False)

# 6.3. Yearly Revenue and Profit Analysis
df['Revenue change YoY'] = df.groupby('Firm ID')['Revenue from Business Activities - Total'].pct_change() * 100
df['Profit change YoY'] = df.groupby('Firm ID')['Net Income after Tax'].pct_change() * 100

firm_trends = df.groupby('Firm ID').agg(
    Positive_Revenue_Years=('Revenue change YoY', lambda x: (x > 0).sum()),
    Positive_Profit_Years=('Profit change YoY', lambda x: (x > 0).sum()),
    Total_Years=('Year', 'nunique')
).reset_index()

firm_trends['Trend'] = 'Other'
firm_trends.loc[
    (firm_trends['Positive_Revenue_Years'] / firm_trends['Total_Years'] >= 0.8) &
    (firm_trends['Positive_Profit_Years'] / firm_trends['Total_Years'] >= 0.8),
    'Trend'
] = 'Consistently Growing'

firm_trends.loc[
    ((firm_trends['Positive_Revenue_Years'] / firm_trends['Total_Years']).between(0.4, 0.8)) |
    ((firm_trends['Positive_Profit_Years'] / firm_trends['Total_Years']).between(0.4, 0.8)),
    'Trend'
] = 'Fluctuating'

firm_trends.loc[
    (firm_trends['Positive_Revenue_Years'] / firm_trends['Total_Years'] < 0.4) &
    (firm_trends['Positive_Profit_Years'] / firm_trends['Total_Years'] < 0.4),
    'Trend'
] = 'Declining'

trend_summary = df.merge(firm_trends, on='Firm ID').groupby('Trend').agg(
    Average_Operating_Expenses=('Operating Expenses - Total', 'mean'),
    Total_Market_Capitalization=('Market Capitalization', 'sum'),
    High_Debt_Ratio_Percentage=('Debt Ratio', lambda x: (x > 0.5).mean() * 100)
).reset_index()
print('Trend analysis summary:')
print(trend_summary)

# 6.4. Expense-to-Revenue Ratio YoY
df['Expense to Revenue Ratio'] = df["Operating Expenses - Total"] / df["Revenue from Business Activities - Total"]
df['Ratio change YoY'] = df.groupby('Firm ID')['Expense to Revenue Ratio'].pct_change() > 0

year_summary = df.groupby('Year').agg(
    Percentage_firms_with_Increased_Ratio=('Ratio change YoY', lambda x: (x.sum() / x.count()) * 100)
).reset_index()
years_over_50 = year_summary[year_summary['Percentage_firms_with_Increased_Ratio'] > 50]
print('Years with >50% firms increasing expense ratio:')
print(years_over_50)

# Save final data
df.to_excel('completed_final_data.xlsx', index=False)
print("✅ Analysis complete. Output saved.")