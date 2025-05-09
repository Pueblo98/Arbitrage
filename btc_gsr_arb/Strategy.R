library(ggplot2)
library(dplyr)
library(zoo)
library(tseries)

# === 1. Parameters (tweak anything here) ===
window_size <- 30    # look-back window in days
z_thresh    <- 1     # entry/exit z-score threshold

# === 2. Working Directory ===
setwd("/Users/lucasportela/Documents/btc_arb

# === 3. Helper Function ===
prepare_data <- function(path) {
  read.csv(path, stringsAsFactors = FALSE) %>%
    mutate(
      Date   = as.Date(Date),
      across(c(Open, High, Low, Close), ~ suppressWarnings(as.numeric(.)))
    ) %>%
    filter(!is.na(Date), !is.na(Close))  # drop incomplete rows
}

# === 4. Load & Prepare Data ===
KROP_data <- prepare_data("KROP.csv")  # first ETF
PBJ_data  <- prepare_data("PBJ.csv")   # second ETF

# === 5. Merge & Initialize ===
aligned <- inner_join(KROP_data, PBJ_data, by = "Date", suffix = c("_KROP","_PBJ")) %>%
  arrange(Date) %>%
  mutate(
    resid_roll     = NA_real_,  # today's spread/residual
    alpha_roll     = NA_real_,  # intercept from hedge regression
    beta_roll      = NA_real_,  # slope from hedge regression
    mu_roll        = NA_real_,  # mean of residuals over window
    sd_roll        = NA_real_,  # sd of residuals over window
    threshold_roll = NA_real_,  # z_thresh * sd
    z_roll         = NA_real_,  # z-score of today's residual
    signal_roll    = 0,         # raw trade signal (+1, -1, 0)
    position_roll  = 0          # position for next day
  )
n <- nrow(aligned)

# === 6. Rolling Estimation & Signal Generation ===
for (i in (window_size + 1):(n - 1)) {
  calib <- aligned[(i - window_size):(i - 1), ]  # calibration window
  
  # 6.1 Fit linear regression: PBJ_t ~ alpha + beta * KROP_t
  mod <- lm(Close_PBJ ~ Close_KROP, data = calib)
  a   <- coef(mod)[1]; b <- coef(mod)[2]
  aligned$alpha_roll[i] <- a
  aligned$beta_roll[i]  <- b
  
  # 6.2 Compute residual stats on calibration window
  resid_calib <- calib$Close_PBJ - (a + b * calib$Close_KROP)
  mu <- mean(resid_calib); s <- sd(resid_calib)
  aligned$mu_roll[i]        <- mu
  aligned$sd_roll[i]        <- s
  aligned$threshold_roll[i] <- z_thresh * s
  
  # 6.3 Today's residual & z-score
  today_resid          <- aligned$Close_PBJ[i] - (a + b * aligned$Close_KROP[i])
  aligned$resid_roll[i] <- today_resid
  aligned$z_roll[i]     <- (today_resid - mu) / s
  
  # 6.4 Trade signal: +1 = go long PBJ, -1 = go short PBJ
  sig <- ifelse(aligned$z_roll[i] < -z_thresh,  1,
                ifelse(aligned$z_roll[i] >  z_thresh, -1, 0))
  aligned$signal_roll[i]     <- sig
  aligned$position_roll[i + 1] <- sig  # position enters next day
}

# === 7. Compute PnL & Track Positions ===
aligned <- aligned %>%
  mutate(
    pnl_roll    = ifelse(row_number() > 1,
                         position_roll * (resid_roll - lag(resid_roll)),
                         0),           # daily PnL
    pnl_roll    = ifelse(is.na(pnl_roll), 0, pnl_roll),
    cumPnL_roll = cumsum(pnl_roll)      # cumulative PnL
  ) %>%
  mutate(
    prev_pos   = lag(position_roll, default = 0),             # yesterday's position
    entry_roll = prev_pos == 0 & position_roll != 0,          # mark entries
    exit_roll  = prev_pos != 0 & position_roll == 0          # mark exits
  )

# === 8. Statistical Tests ===
adf_roll <- adf.test(na.omit(aligned$resid_roll[(window_size + 1):n]))
print(adf_roll)  # test residual stationarity

pnl_test <- aligned$pnl_roll[(window_size + 2):n]  # drop initial zeros
tt_roll  <- t.test(pnl_test, mu = 0, alternative = "greater")
print(tt_roll)    # test positive mean PnL

# === 9. Plotting Results ===
# 9.1 Cumulative PnL
ggplot(aligned, aes(Date, cumPnL_roll)) +
  geom_line(size = 1) +
  labs(
    title    = sprintf("Cumulative PnL: %d-Day Rolling Z-Score", window_size),
    subtitle = sprintf("Threshold = ±%.2f σ", z_thresh),
    x        = "Date", y = "Cumulative PnL"
  ) +
  theme_minimal()

# 9.2 PBJ vs Hedge Line + markers
aligned <- aligned %>% mutate(hedgeLine_roll = alpha_roll + beta_roll * Close_KROP)

ggplot(aligned, aes(Date)) +
  geom_line(aes(y = Close_PBJ), size = 1) +      # actual PBJ price
  geom_line(aes(y = hedgeLine_roll), linetype = "dashed", size = 1) +  # fitted hedge
  geom_point(data = filter(aligned, entry_roll), aes(y = Close_PBJ), shape = 24, size = 2) +  # ▲ entry
  geom_point(data = filter(aligned, exit_roll),  aes(y = Close_PBJ), shape = 25, size = 2) +  # ▼ exit
  labs(
    title    = sprintf("PBJ vs %d-Day Rolling Hedge Line", window_size),
    subtitle = "▲ entries, ▼ exits",
    x        = "Date", y = "Price"
  ) +
  theme_minimal()

# === 10. Scale PnL & Report Final Results ===
position_size <- 10000   # $10k per trade
portfolio     <- 1e6     # $1M total capital

aligned <- aligned %>%
  mutate(
    # scaled daily PnL (handle initial NA residuals)
    pnl_roll    = ifelse(row_number() > 1,
                         position_roll * (resid_roll - lag(resid_roll)) * position_size,
                         0),
    pnl_roll    = ifelse(is.na(pnl_roll), 0, pnl_roll),  # replace NA with 0
    cumPnL_roll = cumsum(pnl_roll)        # scaled cumulative PnL
  )

# Final metrics
final_profit <- tail(aligned$cumPnL_roll, 1)
return_pct   <- final_profit / portfolio * 100
message(sprintf("Total Profit = $%.2f", final_profit))
message(sprintf("Return       = %.2f%%", return_pct))

