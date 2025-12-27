# Financial Data Analysis: Vietnamese Corporate Dataset 

A comprehensive data cleaning, transformation, and financial insight extraction project on a 15-year panel dataset of Vietnamese firms across multiple industries.

## 📊 Overview

This project processes raw financial statements of **50+ Vietnamese firms** (2008–2023) to:
- Clean and validate financial data
- Engineer key financial metrics (Debt Ratio, Earnings Quality, Liquidity Index, etc.)
- Derive business insights through aggregation, trend analysis, and segmentation

The dataset includes balance sheet, income statement, and market data (e.g., Market Capitalization, Total Assets, Net Income, Operating Expenses).

---

## 🔧 Key Features

### ✅ Data Preprocessing
- Handled missing values using **median (numerical)** and **mode (categorical)**
- Converted misformatted string columns to numeric
- Removed duplicates and invalid dates (outside 2008–2023)

### 📈 Derived Financial Metrics
- **Debt Ratio** = Total Liabilities / Total Assets  
- **Earnings Quality** = Net Income after Tax / Operating Expenses  
  → Classified as *High*, *Medium*, or *Low*
- **Liquidity Index** = (Cash + Working Capital) / Total Current Assets  
  → Flagged firms with *Low Liquidity* (< 0.1)
- **Profit Growth** = Year-over-year % change in Net Income (firm-level)

### 📉 Advanced Analysis
- **Industry benchmarking**: Top 3 industries by Profit Margin
- **Firm-level segmentation**: Consistently Growing vs. Fluctuating vs. Declining
- **Outlier detection**: Z-score standardization (Net Income, Total Assets, Market Cap)
- **Filtered healthy firms**: Positive Working Capital, Debt Ratio < median, Profit Margin > 0.1
- **Expense-to-Revenue Ratio**: Year-over-year trend analysis

---

## 🛠️ Technologies Used

- **Language**: Python 3.10+
- **Libraries**: `pandas`, `numpy`
- **Input**: `data_raw_VN.xlsx` (corporate financial statements)
- **Output**:  
  - `completed_final_data.xlsx` (cleaned + enriched dataset)  
  - `filtered_dataset.xlsx` (firms meeting financial health criteria)

---

## 📌 How to Run

1. Clone this repository
2. Ensure `data_raw_VN.xlsx` is in the same directory
3. Run the script:
   ```bash
   python financial_analysis_vn.py
