import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Parameters
tickers = {
    'BTC': 'BTC-USD',
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
}
# Date range: last 4 years
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')

# 1) Download OHLC data for each series, with Date as index
dfs = {}
for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker})…")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print(f"  → No data for {ticker}, skipping.")
        continue
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index.name = 'Date'
    dfx = df.copy()
    dfx.index = pd.to_datetime(dfx.index)
    dfx.index.name = 'Date'
    dfx.columns = ['Open', 'High', 'Low', 'Close']
    dfx = dfx.sort_index()
    dfx = dfx[~dfx.index.duplicated()]
    dfs[name] = dfx
    print(f"  → Retrieved {len(dfx)} rows for {name}")

# 2) Compute Gold–Silver Ratio (GSR) via inner join of closes
if 'GOLD' in dfs and 'SILVER' in dfs:
    paired = pd.concat(
        [dfs['GOLD']['Close'], dfs['SILVER']['Close']],
        axis=1,
        join='inner',
        keys=['GOLD_Close', 'SILVER_Close']
    )
    paired.index.name = 'Date'
    ratio_series = paired['GOLD_Close'] / paired['SILVER_Close']
    gsr_df = pd.DataFrame({'GSR': ratio_series})
    gsr_df.index.name = 'Date'
    dfs['GSR'] = gsr_df
    print(f"Computed GSR with {len(gsr_df)} rows")
else:
    raise ValueError("GOLD or SILVER data missing, cannot compute GSR.")

# 3) Merge BTC & GSR on Date index
merged = dfs['BTC'].join(dfs['GSR'], how='inner')

# 4) Optionally append GOLD & SILVER OHLC
merged = (
    merged
    .join(dfs['GOLD'].add_suffix('_GOLD'),   how='inner')
    .join(dfs['SILVER'].add_suffix('_SILVER'), how='inner')
)

# 5) Export CSVs
# Reset index for CSV export
btc_out = dfs['BTC'].reset_index()
gsr_out = dfs['GSR'].reset_index()
merged_out = merged.reset_index()

btc_out.to_csv("BTC.csv", index=False)
gsr_out.to_csv("GSR.csv", index=False)
merged_out.to_csv("Aligned_BTC_GSR.csv", index=False)

print("\n✅ Saved CSVs:")
print("  • BTC.csv")
print("  • GSR.csv")
print("  • Aligned_BTC_GSR.csv")