# Nigerian Food Prices: Data Cleaning and Exploratory Analysis

A beginner data science project in Python. I took a deliberately messy dataset of monthly food prices across 8 Nigerian states, cleaned it step by step, and explored it to answer questions about how prices changed from 2023 to 2025.

> **Important:** the dataset is **synthetic practice data**, generated for learning. The numbers show my method, not real Nigerian market prices. The next project repeats this workflow on real data.

---

## Project goals

1. Practise the full cleaning workflow on realistic, messy data.
2. Answer business-style questions with clear charts.
3. Document every problem found and how it was solved, so anyone can follow and reproduce it.

## The dataset

| File | Description |
|---|---|
| `nigeria_food_prices_practice.csv` | Raw data: 975 rows, 5 columns |
| `nigeria_food_prices_clean.csv` | Cleaned data: 960 rows, 5 columns |

**Columns:** `date` (month), `state`, `item`, `price_ngn` (price in naira), `markets_surveyed` (number of markets in the survey).
**Coverage:** 6 items (rice, beans, garri, tomatoes, vegetable oil, eggs), 8 states, January 2023 to December 2025.

---

## Problems found and how I solved them

### 1. Audit first
Before changing anything, I inspected the data with `shape`, `info()`, `dtypes`, `isna().sum()`, `duplicated().sum()`, `value_counts()` and `describe()`. I worked on a copy (`df_clean`) so the raw data stayed untouched.

### 2. The problems

| # | Problem | How I found it | Fix |
|---|---|---|---|
| 1 | **15 duplicate rows** | `df.duplicated().sum()` | `drop_duplicates()`, then `reset_index(drop=True)` |
| 2 | **Prices stored as text**, some with a `₦` sign and commas, so `min()` and `max()` gave nonsense (text sorts character by character) | `dtypes`, then a `str.contains` filter on the `₦` symbol | Removed `₦` and `,` with `str.replace`, then converted with `pd.to_numeric` |
| 3 | **Inconsistent state names** (`Lagos`, `LAGOS`, `lagos`): 17 spellings for 8 states | `value_counts()` on `state` | Standardised the casing with `.str` methods and fixed `Abuja FCT` |
| 4 | **Dates stored as text** | `dtypes` | `pd.to_datetime`, then checked 36 unique months (3 years x 12) |
| 5 | **One impossible price:** a litre of vegetable oil at ₦9,999,999 (about 3,400x the typical price) | `sort_values('price_ngn', ascending=False)` | Set that single price to `NaN`, kept the rest of the valid row |
| 6 | **42 missing prices** (43 after the outlier fix) | `isna().sum()` | Filled each gap with the **median price of the same item** |
| 7 | **26 missing `markets_surveyed`** (25 after removing duplicates) | `isna().sum()` | Filled with the overall median (22), since it is a count and not tied to an item |

### 3. Key code

```python
# Remove duplicates
df_clean = df.drop_duplicates().reset_index(drop=True)

# Prices: strip symbols, then convert to numbers
df_clean['price_ngn'] = df_clean['price_ngn'].str.replace('₦', '').str.replace(',', '')
df_clean['price_ngn'] = pd.to_numeric(df_clean['price_ngn'])

# Dates
df_clean['date'] = pd.to_datetime(df_clean['date'])

# Outlier -> missing
df_clean.loc[df_clean['price_ngn'] > 100000, 'price_ngn'] = np.nan

# Fill prices with the median of the same item
df_clean['price_ngn'] = df_clean['price_ngn'].fillna(
    df_clean.groupby('item')['price_ngn'].transform('median')
)

# Fill markets_surveyed with the median
df_clean['markets_surveyed'] = df_clean['markets_surveyed'].fillna(
    df_clean['markets_surveyed'].median()
)
```

### 4. Why the median, not the mean?
One extreme value can drag a mean far from the typical price. The median stays in the middle, so it is the safer choice for filling gaps.

### 5. Verification after cleaning
- `shape` is `(960, 5)`
- `isna().sum()` is 0 for every column
- `duplicated().sum()` is 0
- `price_ngn` is `float64` and `date` is `datetime64`
- Maximum price is about ₦22,549 (a tomato basket), a believable value

---

## Analysis and findings

**Q1. Which item rose most from 2023 to 2025?**

| Item | Rise (%) | Rise (₦) |
|---|---|---|
| Garri (1kg) | **79.2%** | 470 |
| Tomatoes (basket) | 78.3% | **7,473** |
| Rice (1kg) | 78.2% | 827 |
| Eggs (crate) | 76.6% | 2,007 |
| Beans (1kg) | 74.1% | 971 |
| Vegetable oil (1L) | 68.7% | 1,533 |

- Garri rose fastest **in percent**, but tomatoes added the most **in naira**, because they started from a much higher price. The two rankings answer different questions, so I report both.
- Every item rose by roughly 70 to 80 percent, so the spread between items is narrow.

**Q2. Which state had the highest average tomato price?**
Abuja FCT and Lagos were highest, and Oyo was lowest, but the gaps were small (about ₦11,800 to ₦13,500). With only a few rows per state per year, I would not draw strong conclusions.

**Q3. Is there seasonality in tomato prices?**
The monthly average climbs from January to December, but that line mixes a seasonal pattern with the overall price rise over three years. To separate them, the next step is one line per year.

### Charts
Add your exported images to an `images/` folder and link them here:
- `images/price_trends.png`: average price per item, 2023 to 2025
- `images/pct_rise.png`: percentage rise per item
- `images/naira_rise.png`: naira rise per item
- `images/tomato_by_state.png`: average tomato price by state
- `images/tomato_seasonality.png`: tomato price by month

---

## Limitations

- The data is synthetic, so findings illustrate the method and should not be read as real market facts.
- Each state-year group has few rows, so small differences between states may be noise.
- Filling missing prices with item medians assumes gaps are random.

## What I learned

- Check every result a second way. A column that looks fine can still be text.
- A line that does not assign its result changes nothing, so verify after each step.
- Percent change and absolute change can rank items differently.
- State findings with numbers, and state your caveats.

## How to run

```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook EDA.ipynb
```
Place `nigeria_food_prices_practice.csv` in the same folder as the notebook, then run the cells from top to bottom. When reloading the cleaned CSV, convert `date` back with `pd.to_datetime`, because CSV files store dates as text.

## Tools

Python, pandas, NumPy, Matplotlib, Jupyter (VS Code)

## Next steps

1. Repeat the workflow on a real Nigerian dataset (NBS, Kaggle, data.humdata.org).
2. Separate seasonality from trend in tomato prices.
3. Machine learning: price prediction with regression, then forecasting.

## Author

[Your Name] | [LinkedIn link] | [Email]
