# 🛒 Multi-Store Sales Forecasting

This project focuses on forecasting daily item-level sales across 50 products and 10 stores using time series modeling. Two approaches were tested — **LightGBM** and **LSTM** — to evaluate which algorithm performs better in a retail sales prediction context.

---

## 📌 Project Overview

Retail companies rely on accurate demand forecasting to optimize inventory, reduce costs, and improve customer satisfaction. This notebook presents a practical solution for:

- Forecasting daily sales for each `(store, item)` pair
- Comparing tree-based (LightGBM) and deep learning (LSTM) approaches
- Analyzing forecast accuracy at a granular level

While both models were evaluated, **LightGBM** yielded consistently better overall performance and was chosen as the final model for deployment.

---

## 🧠 Models Used

| Model     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| LightGBM  | Fast and scalable gradient boosting framework, strong with engineered features |
| LSTM      | Recurrent neural network architecture for modeling temporal dependencies    |

---

## 📊 Dataset

The dataset contains daily sales records per store and per item.  
**Note**: Due to licensing restrictions, the original dataset cannot be shared.

### Expected Schema

| Column | Type   | Description              |
|--------|--------|--------------------------|
| date   | date   | Transaction date         |
| store  | int    | Store ID (1–10)          |
| item   | int    | Item ID (1–50)           |
| sales  | int    | Number of items sold     |

### 📂 Sample Code to Load Your Dataset

```python
import pandas as pd

# Load your own sales dataset
df = pd.read_csv("your_sales_data.csv")

# Ensure it contains the following columns
# ['date', 'store', 'item', 'sales']
df['date'] = pd.to_datetime(df['date'])
df.head()

```

---

## 🔍 Key Results

* LightGBM captured overall trends and seasonality well, with reasonable error margins at an aggregate level.
* LSTM showed potential on specific store-item combinations but underperformed in general due to data sparsity and high variance.
* Final evaluation favored **LightGBM** based on RMSE and generalization to unseen days.

---

## 🚫 Disclaimer

This notebook is based on a private dataset that cannot be redistributed.
If you use this notebook with your own data, make sure it adheres to the expected schema.

---
