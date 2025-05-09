import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# === 0. Parameters ===
WINDOW             = 90          # look‐back window
Z_THRESH           = 2.0         # z‐score threshold
TOTAL_CAPITAL      = 1e6         # USD
RISK_PER_TRADE_PCT = 0.02        # 2% of capital per trade
FEE_RATE           = 0.0005      # 0.05% per side
SLIPPAGE_RATE      = 0.001       # 0.10% per side
VAR_CONFIDENCE     = 0.95        # 95% VaR

# === 1. Load Data ===
df = pd.read_csv("Aligned_BTC_GSR.csv", parse_dates=["Date"])
df = df.sort_values("Date").set_index("Date")

# === 2. Compute Returns & GSR Changes ===
df["r_btc"] = np.log(df["Close"] / df["Close"].shift(1))
df["d_gsr"] = df["GSR"].pct_change()
df = df.dropna(subset=["r_btc", "d_gsr"])

# === 3. Initialize Columns ===
for col in ["alpha","beta","resid","mu","sd","z","signal","position"]:
    df[col] = np.nan
df["signal"]   = 0
df["position"] = 0

# === 4. Rolling Regression & Signals ===
for t in range(WINDOW, len(df)-1):
    win = df.iloc[t-WINDOW:t]
    Y, X = win["r_btc"], sm.add_constant(win["d_gsr"])
    model = sm.OLS(Y, X).fit()
    
    a, b = model.params
    df.at[df.index[t], "alpha"] = a
    df.at[df.index[t], "beta"]  = b
    
    resid = Y - model.predict(X)
    mu, sd = resid.mean(), resid.std()
    df.at[df.index[t], "mu"] = mu
    df.at[df.index[t], "sd"] = sd
    
    eps = df["r_btc"].iat[t] - (a + b * df["d_gsr"].iat[t])
    z   = (eps - mu) / sd
    df.at[df.index[t], "resid"] = eps
    df.at[df.index[t], "z"]     = z
    
    sig = 1 if z < -Z_THRESH else (-1 if z > Z_THRESH else 0)
    df.at[df.index[t],   "signal"]   = sig
    df.at[df.index[t+1], "position"] = sig

# === 5. Volatility‐Scaled Position Sizing ===
risk_per_trade = TOTAL_CAPITAL * RISK_PER_TRADE_PCT
df["pos_size"] = risk_per_trade / df["sd"]
# clean up any NaN or infinite
df["pos_size"] = df["pos_size"].fillna(0).replace([np.inf, -np.inf], 0)

# === 6. Raw PnL (residual‐units) ===
df["resid_shift"] = df["resid"].shift(1).fillna(0)
df["pnl_units"]  = df["position"] * (df["resid"] - df["resid_shift"])
df["pnl_units"]  = df["pnl_units"].fillna(0)

# === 7. Fees & Slippage ===
df["pos_usd"]        = df["position"] * df["pos_size"]
df["trade_notional"] = df["pos_usd"].diff().abs().fillna(0)
df["fees"]           = df["trade_notional"] * FEE_RATE
df["slippage"]       = df["trade_notional"] * SLIPPAGE_RATE

# === 8. Net PnL in USD ===
df["pnl_$"]     = df["pnl_units"] * df["pos_size"] - df["fees"] - df["slippage"]
df["pnl_$"]     = df["pnl_$"].fillna(0)
df["cum_pnl_$"] = df["pnl_$"].cumsum()

# === 9. Performance Metrics ===
daily_ret  = df["pnl_$"] / TOTAL_CAPITAL
mean_ret   = daily_ret.mean()
std_ret    = daily_ret.std(ddof=1)
sharpe_net = mean_ret / std_ret * np.sqrt(252)

running_max  = df["cum_pnl_$"].cummax()
drawdown     = df["cum_pnl_$"] - running_max
max_drawdown = drawdown.min()

var_dollar = -np.percentile(df["pnl_$"], (1 - VAR_CONFIDENCE) * 100)
var_pct    = -np.percentile(daily_ret,    (1 - VAR_CONFIDENCE) * 100)

total_pnl  = df["cum_pnl_$"].iloc[-1]
return_pct = total_pnl / TOTAL_CAPITAL * 100

print(f"[{datetime.now().date()}] Net Total PnL        = ${total_pnl:,.2f}")
print(f"Net Return              = {return_pct:.2f}%")
print(f"Annualized Sharpe (net) = {sharpe_net:.2f}")
print(f"Max Drawdown            = ${max_drawdown:,.2f}")
print(f"{int(VAR_CONFIDENCE*100)}% VaR (dollar)        = ${var_dollar:,.2f}")
print(f"{int(VAR_CONFIDENCE*100)}% VaR (percent)       = {var_pct:.2%}")

# === 10. Plots ===
plt.figure(figsize=(10,4))
plt.plot(df.index, df["cum_pnl_$"], label="Cumulative Net PnL")
plt.title(f"Cumulative Net PnL — {WINDOW}-Day Rolling | Threshold ±{Z_THRESH}")
plt.ylabel("USD"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.fill_between(df.index, drawdown, 0, color="red")
plt.title("Underwater Curve (Drawdowns)")
plt.ylabel("Drawdown (USD)"); plt.tight_layout(); plt.show()
