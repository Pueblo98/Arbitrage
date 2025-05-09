import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# ========== PARAMETERS ==========
WINDOW   = 30
Z_THRESH = 1.5
CAPITAL  = 1_000_000      # USD total capital
RISK_PC  = 0.02           # 2 % of capital per trade
FEE_RT   = 0.0005         # 0.05 % per side
SLIP_RT  = 0.001          # 0.10 % per side
VAR_P    = 0.95           # 95 % VaR
np.random.seed(42)        # repeatable randomness

# ========== LOAD DATA ==========
df = (pd.read_csv("Aligned_BTC_GSR.csv", parse_dates=["Date"])
        .sort_values("Date")
        .set_index("Date"))

# returns & GSR changes
df["r_btc"] = np.log(df["Close"] / df["Close"].shift(1))
df["d_gsr"] = df["GSR"].pct_change()
df = df.dropna(subset=["r_btc", "d_gsr"])

# helper columns
for col in ["alpha","beta","resid","mu","sd","z",
            "signal","position"]:
    df[col] = 0.0

# ========== ROLLING REGRESSION & STRATEGY SIGNALS ==========
for t in range(WINDOW, len(df) - 1):
    win   = df.iloc[t-WINDOW:t]
    model = sm.OLS(win["r_btc"], sm.add_constant(win["d_gsr"])).fit()
    a, b  = model.params

    resid_series = win["r_btc"] - model.predict(sm.add_constant(win["d_gsr"]))
    mu, sd       = resid_series.mean(), resid_series.std()

    eps = df["r_btc"].iat[t] - (a + b * df["d_gsr"].iat[t])
    z   = (eps - mu) / sd if sd > 0 else 0.0

    df.iloc[t, df.columns.get_loc("alpha")] = a
    df.iloc[t, df.columns.get_loc("beta")]  = b
    df.iloc[t, df.columns.get_loc("mu")]    = mu
    df.iloc[t, df.columns.get_loc("sd")]    = sd
    df.iloc[t, df.columns.get_loc("resid")] = eps
    df.iloc[t, df.columns.get_loc("z")]     = z

    sig = 1 if z < -Z_THRESH else (-1 if z > Z_THRESH else 0)
    df.iloc[t,   df.columns.get_loc("signal")]   = sig
    df.iloc[t+1, df.columns.get_loc("position")] = sig

# ========== RANDOM-WALK BENCHMARK (THRESHOLD-AWARE) ==========
rand_pos = []
current  = 0.0
for z in df["z"]:
    if abs(z) > Z_THRESH and current == 0.0:
        # entry: pick random leverage in (-2, +2)
        current = np.random.uniform(-2, 2)
    elif abs(z) <= Z_THRESH and current != 0.0:
        # exit when back inside band
        current = 0.0
    rand_pos.append(current)
df["rand_position"] = rand_pos

# ========== RESIDUAL SHIFT ==========
df["resid_shift"] = df["resid"].shift(1).fillna(0)

# ========== VOL-SCALED NOTIONAL ==========
risk_per_trade = CAPITAL * RISK_PC
df["pos_size"] = (risk_per_trade / df["sd"]).replace([np.inf, -np.inf], 0).fillna(0)

# ========== STRATEGY PnL ==========
df["pnl_units"] = df["position"] * (df["resid"] - df["resid_shift"])
df["pnl_units"] = df["pnl_units"].fillna(0)
df["pos_usd"]        = df["position"] * df["pos_size"]
df["trade_notional"] = df["pos_usd"].diff().abs().fillna(0)
df["fees"]           = df["trade_notional"] * FEE_RT
df["slippage"]       = df["trade_notional"] * SLIP_RT
df["pnl_$"]     = df["pnl_units"] * df["pos_size"] - df["fees"] - df["slippage"]
df["cum_pnl_$"] = df["pnl_$"].cumsum()

# ========== RANDOM-WALK PnL ==========
df["rand_pnl_units"] = df["rand_position"] * (df["resid"] - df["resid_shift"])

df["rand_pos_usd"]        = df["rand_position"] * df["pos_size"]
df["rand_trade_notional"] = df["rand_pos_usd"].diff().abs().fillna(0)
df["rand_fees"]           = df["rand_trade_notional"] * FEE_RT
df["rand_slippage"]       = df["rand_trade_notional"] * SLIP_RT

df["rand_pnl_$"]     = (df["rand_pnl_units"] * df["pos_size"]
                        - df["rand_fees"] - df["rand_slippage"])
df["rand_pnl_$"]     = df["rand_pnl_$"].fillna(0)
df["rand_cum_pnl_$"] = df["rand_pnl_$"].cumsum()

# ========== METRICS ==========
tot_strat = df["cum_pnl_$"].iloc[-1]
tot_rand  = df["rand_cum_pnl_$"].iloc[-1]

print(f"[{datetime.now().date()}] Strategy PnL = ${tot_strat:,.2f}")
print(f"[{datetime.now().date()}] Random Walk PnL = ${tot_rand:,.2f}")

# ========== PERFORMANCE METRICS (strategy & random walk) ==========

def perf(cum_pnl_series, daily_pnl_series, capital, label=""):
    """Return a dict of performance stats for the given PnL series."""
    daily_ret = daily_pnl_series / capital
    ann_ret   = daily_ret.mean() * 252
    ann_vol   = daily_ret.std(ddof=1) * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan

    running_max = cum_pnl_series.cummax()
    max_dd      = (cum_pnl_series - running_max).min()     # USD

    var_95_pct  = -np.percentile(daily_ret, 5) * 100       # percentage VaR

    return {
        "Label"      : label,
        "Total PnL $"   : cum_pnl_series.iloc[-1],
        "Return  %"     : 100 * cum_pnl_series.iloc[-1] / capital,
        "Sharpe (ann.)" : sharpe,
        "Max DD  $"     : max_dd,
        "95% VaR %"     : var_95_pct,
    }

strat_stats = perf(df["cum_pnl_$"], df["pnl_$"],  CAPITAL, "Strategy")
rand_stats  = perf(df["rand_cum_pnl_$"], df["rand_pnl_$"], CAPITAL, "Random walk")

# pretty‐print
for s in (strat_stats, rand_stats):
    print(f"\n=== {s['Label']} ===")
    for k, v in s.items():
        if k == "Label": continue
        print(f"{k:>15}: {v:,.2f}")

# ========== PLOT ==========
plt.figure(figsize=(10,5))
plt.plot(df.index, df["cum_pnl_$"],      label="Strategy")
plt.plot(df.index, df["rand_cum_pnl_$"], label="Random Walk (±2 random dir)", linestyle="--")
plt.title("Cumulative Net PnL: Strategy vs. Threshold-Aware Random Walk")
plt.ylabel("USD")
plt.legend()
plt.tight_layout()
plt.show()
