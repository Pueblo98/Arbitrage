# Load required libraries
library(ggplot2)
library(dplyr)
library(zoo)
library(tseries)



# Set working directory
setwd("/Users/simon/OneDrive/Desktop/quant/btc_arb")

# exploratory analysis
# Function to prepare data
prepare_data <- function(file_path) {
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  data$Date <- as.Date(data$Date)
  data <- data %>% mutate(across(c(Open, High, Low, Close), ~ suppressWarnings(as.numeric(.))))
  data <- data[!is.na(data$Date) & !is.na(data$Close), ]
  return(data)
}

# Load and prepare datasets
BTC_data <- prepare_data("BTC.csv")
SFTB_data  <- prepare_data("SFTB.csv")

# Plot Close Prices
ggplot(BTC_data, aes(x = Date, y = Close)) +
  geom_line(color = "#1F77B4", size = 1) +
  labs(title = "BTC close", x = "Date", y = "Price (USD)") +
  theme_minimal()

ggplot(SFTB_data, aes(x = Date, y = Close)) +
  geom_line(color = "#1F77B4", size = 1) +
  labs(title = "SFTB Close Price", x = "Date", y = "Price (USD)") +
  theme_minimal()

# Extract Close prices
BTC_close <- BTC_data$Close
SFTB_close  <- SFTB_data$Close

# Summary statistics
BTC_stats <- round(c(
  Mean   = mean(BTC_close),
  Median = median(BTC_close),
  StdDev = sd(BTC_close),
  Min    = min(BTC_close),
  Max    = max(BTC_close)
), 4)
SFTB_stats <- round(c(
  Mean   = mean(SFTB_close),
  Median = median(SFTB_close),
  StdDev = sd(SFTB_close),
  Min    = min(SFTB_close),
  Max    = max(SFTB_close)
), 4)
stats_df <- data.frame(
  Statistic = names(BTC_stats),
  BTC      = BTC_stats,
  SFTB       = SFTB_stats,
  row.names = NULL
)
print(stats_df)

# Standardize (z-scores)
BTC_z <- as.numeric(scale(BTC_close))
SFTB_z  <- as.numeric(scale(SFTB_close))

# Histograms with robust breaks
hist(BTC_z, breaks = "Sturges", main = "BTC Distribution (Standardized)", xlab = "Z-score")
hist(SFTB_z, breaks = "Sturges", main = "SFTB Distribution (Standardized)", xlab = "Z-score")

# Overlaid histogram
all_z <- c(BTC_z, SFTB_z)
hist(BTC_z, breaks = "Sturges", freq = FALSE, col = rgb(1,0,0,0.5), xlim = range(all_z),
     main = "BTC vs SFTB (Standardized)", xlab = "Z-score")
hist(SFTB_z, breaks = "Sturges", freq = FALSE, col = rgb(0,0,1,0.5), add = TRUE)

# Histogram in price space using z-axis ticks
BTC_mean <- mean(BTC_close)
BTC_sd   <- sd(BTC_close)
hist(BTC_z, breaks = "Sturges", freq = FALSE, axes = FALSE,
     main = "BTC Price Distribution", xlab = "Price (USD)")
z_ticks <- seq(floor(min(BTC_z)), ceiling(max(BTC_z)), by = 1)
axis(1, at = z_ticks, labels = round(BTC_mean + z_ticks * BTC_sd, 2))
axis(2)
box()

# SFTB custom axis
SFTB_mean <- mean(SFTB_close)
SFTB_sd   <- sd(SFTB_close)
hist(SFTB_z, breaks = "Sturges", freq = FALSE, axes = FALSE,
     main = "SFTB Price Distribution", xlab = "Price (USD)")
z_ticks <- seq(floor(min(SFTB_z)), ceiling(max(SFTB_z)), by = 1)
axis(1, at = z_ticks, labels = round(SFTB_mean + z_ticks * SFTB_sd, 2))
axis(2)
box()

# Scatter plots with regression
df_BTC <- data.frame(Time = seq_along(BTC_close), Price = BTC_close)
plot(df_BTC$Time, df_BTC$Price, pch = 19,
     xlab = "Time Index", ylab = "BTC Price (USD)",
     main = "BTC Price Scatter with Fit")
fit_BTC <- lm(Price ~ Time, data = df_BTC)
abline(fit_BTC, col = 'blue', lwd = 2)
df_SFTB <- data.frame(Time = seq_along(SFTB_close), Price = SFTB_close)
plot(df_SFTB$Time, df_SFTB$Price, pch = 19,
     xlab = "Time Index", ylab = "SFTB Price (USD)",
     main = "SFTB Price Scatter with Fit")
fit_SFTB <- lm(Price ~ Time, data = df_SFTB)
abline(fit_SFTB, col = 'blue', lwd = 2)

# Bar chart of summary stats
stats_matrix <- as.matrix(stats_df[, c("BTC","SFTB")])
rownames(stats_matrix) <- stats_df$Statistic
barplot(stats_matrix, beside = TRUE,
        legend.text = TRUE, args.legend = list(x = 'topright'),
        main = "Summary Statistics: BTC vs SFTB", xlab = "Statistic", ylab = "Value")

# Density plots
dens_BTC <- density(BTC_close)
dens_SFTB  <- density(SFTB_close)
plot(dens_BTC, main = "Density Estimates", xlab = "Price (USD)",
     ylim = range(c(dens_BTC$y, dens_SFTB$y)))
lines(dens_SFTB, lty = 2)
legend('topright', legend = c('BTC','SFTB'), lty = c(1,2))

# Q-Q plot
qqplot(BTC_close, SFTB_close, main = "Q-Q Plot BTC vs SFTB",
       xlab = "BTC Quantiles", ylab = "SFTB Quantiles")
abline(0,1, col = 'red')

# Differences and ratios
hist(SFTB_close - BTC_close, main = "SFTB - BTC Differences", xlab = "Difference (USD)")
hist(SFTB_close / BTC_close, main = "SFTB / BTC Ratios", xlab = "Ratio")

# Correlation
print(cor(BTC_close, SFTB_close))




#-------------------------------------------------------------------
# Spread, Returns, and Predictive Analysis
#-------------------------------------------------------------------
# Compute spread and returns
aligned <- inner_join(BTC_data, SFTB_data, by = "Date", suffix = c("_BTC","_SFTB"))
aligned <- aligned %>% mutate(
  Spread    =  Close_BTC- Close_SFTB,
  SpreadPct = Spread / Close_SFTB,
  Ret_BTC  = c(NA, diff(log(Close_SFTB))),
  Ret_SFTB   = c(NA, diff(log(Close_BTC)))
) %>%
  filter(!is.na(Spread), !is.na(Ret_BTC), !is.na(Ret_SFTB))

# Plot spread over time
ggplot(aligned, aes(x = Date, y = Spread)) +
  geom_line() +
  labs(title = "Price Spread: SFTB - BTC", y = "Spread (USD)", x = "Date") +
  theme_minimal()

#Is this the same?
ggplot(aligned, aes(x = Date, y = Spread)) +
  geom_line() +
  labs(title = "Price Spread: BTC - SFTB", y = "Spread (USD)", x = "Date") +
  theme_minimal()

# Rolling correlation of returns
# Use base R cbind for rollapply input
aligned$RollCor <- zoo::rollapply(
  cbind(aligned$Ret_SFTB, aligned$Ret_BTC),
  width = 60,
  FUN = function(x) cor(x[,1], x[,2], use = "complete.obs"),
  by.column = FALSE,
  fill = NA,
  align = "right"
)
# ADF test on spread
adf_spread <- tseries::adf.test(aligned$Spread)
print(adf_spread)

# Johansen cointegration test using base subset
coint_test <- urca::ca.jo(
  aligned[, c("Close_SFTB", "Close_BTC")],
  type = "trace",
  K = 2
)
print(summary(coint_test))

# Granger causality VAR - use base R for data selection
var_data <- na.omit(aligned[, c("Ret_SFTB", "Ret_BTC")])
var_model <- vars::VAR(var_data, p = 5)
print(vars::causality(var_model, cause = "Ret_SFTB"))
print(vars::causality(var_model, cause = "Ret_BTC"))

# Mean reversion half-life
hl_df <- na.omit(aligned %>% mutate(dSpread = Spread - lag(Spread)))
hl_model <- lm(dSpread ~ lag(Spread), data = hl_df)
half_life <- -log(2) / coef(hl_model)[2]
print(paste("Half-life:", round(half_life, 2), "days"))

# Simple stat arb backtest
sd_spread <- sd(aligned$Spread)
aligned <- aligned %>% mutate(
  Position = case_when(
    Spread >  sd_spread ~ -1,
    Spread < -sd_spread ~  1,
    TRUE               ~  0
  ),
  PnL = Position * (lag(Spread) - Spread)
) %>%
  mutate(PnL = coalesce(PnL, 0))
# Calculate expected value (mean PnL) and risk (SD)
expected_value <- mean(aligned$PnL, na.rm = TRUE)
sd_pnl         <- sd(aligned$PnL, na.rm = TRUE)
sharpe         <- expected_value / sd_pnl
# Print key metrics
print(paste("Expected Value (Mean PnL):", round(expected_value, 4)))
print(paste("Sharpe Ratio:", round(sharpe, 4)))

# Plot cumulative PnL using ggplot2
aligned$cumPnL <- cumsum(aligned$PnL)
ggplot(aligned, aes(x = Date, y = cumPnL)) +
  geom_line() +
  labs(title = "Cumulative PnL of Stat Arb", y = "Cumulative PnL", x = "Date") +
  theme_minimal()

# Compute spread and returns
aligned <- inner_join(BTC_data, SFTB_data, by = "Date", suffix = c("_BTC","_SFTB"))
aligned <- aligned %>% mutate(
  Spread    = Close_SFTB - Close_BTC,
  SpreadPct = Spread / Close_BTC,
  Ret_BTC  = c(NA, diff(log(Close_BTC))),
  Ret_SFTB   = c(NA, diff(log(Close_SFTB)))
) %>%
  filter(!is.na(Spread), !is.na(Ret_BTC), !is.na(Ret_SFTB))

# Plot the spread over time
ggplot(aligned, aes(x = Date, y = Spread)) +
  geom_line(color = "darkgreen", size = 1) +
  labs(title = "Spread between SFTB and BTC", x = "Date", y = "Spread (USD)") +
  theme_minimal()

#Residual
# 1) Add a numeric time index
aligned$t <- as.numeric(aligned$Date)

# 2) Fit a trend model to the spread
trend_mod <- lm(Spread ~ t, data = aligned)

# 3) Extract the residuals (detrended spread)
aligned$residual <- resid(trend_mod)

# 4) Compute SD of the residuals
sigma <- sd(aligned$residual, na.rm = TRUE)

# 5) Plot
library(ggplot2)
p <- ggplot(aligned, aes(x = Date, y = residual)) +
  geom_line(color = "steelblue") +
  geom_hline(yintercept = c(-sigma, sigma),
             linetype = "dashed", color = "darkred") +
  labs(
    title = "Detrended Spread Residuals (SFTB–BTC)",
    subtitle = "Dashed lines = ±1 SD of residuals",
    x = "Date",
    y = "Residual (USD)"
  ) +
  theme_minimal()

print(p)






#arb part
# === PARAMETERS (adjust here) ===
window_size <- 7    # rolling look-back in days
z_thresh    <- 1   # entry/exit z-score threshold
portfolio <- 100000
# ================================
# Portfolio sizing
risk_fraction <- 0.1  # 10% of portfolio used per trade
position_size <- portfolio * risk_fraction


# Function to prepare data
prepare_data <- function(file_path) {
  read.csv(file_path, stringsAsFactors = FALSE) %>%
    mutate(
      Date = as.Date(Date),
      across(c(Open, High, Low, Close), ~ suppressWarnings(as.numeric(.)))
    ) %>%
    filter(!is.na(Date), !is.na(Close))
}

# Load and prepare datasets
BTC_data <- prepare_data("BTC.csv")
SFTB_data  <- prepare_data("SFTB.csv")

# Merge into one data frame and initialize columns
aligned <- aligned %>%
  mutate(
    capital = portfolio + lag(cumPnL_roll, default = 0),  # this needs cumPnL_roll to exist
    dynamic_position_size = capital * risk_fraction,
    pnl_roll = ifelse(row_number() > 1,
                      dynamic_position_size * position_roll * (resid_roll - lag(resid_roll)),
                      0),
    pnl_roll = ifelse(is.na(pnl_roll), 0, pnl_roll)
  )

# Now calculate cumulative PnL *after* pnl_roll exists
aligned <- aligned %>%
  mutate(
    cumPnL_roll = cumsum(pnl_roll)
  )

  
  

n <- nrow(aligned)

# 90-day rolling estimation and next-day signal
for (i in (window_size+1):(n-1)) {
  calib <- aligned[(i-window_size):(i-1), ]
  
  # 1) Fit hedge line on calibration window
  mod <- lm(Close_SFTB ~ Close_BTC, data = calib)
  a   <- coef(mod)[1]
  b   <- coef(mod)[2]
  aligned$alpha_roll[i] <- a
  aligned$beta_roll[i]  <- b
  
  # 2) Compute calibration residual stats
  resid_calib <- calib$Close_SFTB - (a + b * calib$Close_BTC)
  mu <- mean(resid_calib)
  s  <- sd(resid_calib)
  aligned$mu_roll[i]        <- mu
  aligned$sd_roll[i]        <- s
  aligned$threshold_roll[i] <- z_thresh * s
  
  # 3) Today's residual & z-score
  today_resid          <- aligned$Close_SFTB[i] - (a + b * aligned$Close_BTC[i])
  aligned$resid_roll[i] <- today_resid
  aligned$z_roll[i]     <- (today_resid - mu) / s
  
  # 4) Generate today's signal (enter next day)
  sig <- ifelse(aligned$z_roll[i] < -z_thresh,  1,
                ifelse(aligned$z_roll[i] >  z_thresh, -1, 0))
  aligned$signal_roll[i]     <- sig
  aligned$position_roll[i+1] <- sig
}

# Compute PnL and cumulative PnL with corrected sign
aligned <- aligned %>%
  mutate(
    pnl_roll    = ifelse(row_number() > 1,
                         position_roll * (resid_roll - lag(resid_roll)),
                         0),
    pnl_roll    = ifelse(is.na(pnl_roll), 0, pnl_roll),
    cumPnL_roll = cumsum(pnl_roll)
  ) %>%
  # Define entry/exit based on position changes
  mutate(
    prev_pos   = lag(position_roll, default = 0),
    entry_roll = prev_pos == 0 & position_roll != 0,
    exit_roll  = prev_pos != 0 & position_roll == 0
  )

# 1) ADF test on rolling residuals
adf_roll <- adf.test(na.omit(aligned$resid_roll[(window_size+1):n]))
print(adf_roll)

# 2) One-sample t-test on rolling PnL (drop initial zeros)
pnl_test <- aligned$pnl_roll[(window_size+2):n]
tt_roll  <- t.test(pnl_test, mu = 0, alternative = "greater")
print(tt_roll)

# 3) Plot cumulative PnL
ggplot(aligned, aes(Date, cumPnL_roll)) +
  geom_line(color = "darkorange", size = 1) +
  labs(
    title    = sprintf("Cumulative PnL: %d-Day Rolling Z-Score Strategy", window_size),
    subtitle = sprintf("Threshold = ±%.2f σ", z_thresh),
    x        = "Date",
    y        = "Cumulative PnL (USD)"
  ) +
  theme_minimal()

# 4) Plot SFTB vs. rolling hedge line with entry/exit markers
aligned <- aligned %>%
  mutate(hedgeLine_roll = alpha_roll + beta_roll * Close_BTC)

ggplot(aligned, aes(x = Date)) +
  geom_line(aes(y = Close_SFTB),      color = "blue",  size = 1) +
  geom_line(aes(y = hedgeLine_roll), color = "red",   linetype = "dashed", size = 1) +
  geom_point(
    data = filter(aligned, entry_roll),
    aes(x = Date, y = Close_SFTB),
    shape = 24, color = "darkgreen", fill = "green", size = 2
  ) +  # ▲ entries
  geom_point(
    data = filter(aligned, exit_roll),
    aes(x = Date, y = Close_SFTB),
    shape = 25, color = "darkred",   fill = "red",   size = 2
  ) +  # ▼ exits
  labs(
    title    = sprintf("SFTB vs %d-Day Rolling Hedge Line", window_size),
    subtitle = sprintf("Entries = ▲ green, Exits = ▼ red (Threshold ±%.2fσ)", z_thresh),
    x        = "Date",
    y        = "Price (USD)"
  ) +
  theme_minimal()

# 5) Plot rolling residual spread with dynamic thresholds & entries/exits
ggplot(aligned, aes(Date, resid_roll)) +
  geom_line(color = "steelblue") +
  geom_line(aes(y =  threshold_roll), linetype = "dashed", color = "grey50") +
  geom_line(aes(y = -threshold_roll), linetype = "dashed", color = "grey50") +
  geom_point(
    data = filter(aligned, entry_roll),
    aes(Date, resid_roll),
    shape = 24, color = "darkgreen", fill = "green", size = 2
  ) +
  geom_point(
    data = filter(aligned, exit_roll),
    aes(Date, resid_roll),
    shape = 25, color = "darkred",   fill = "red",   size = 2
  ) +
  labs(
    title = sprintf("%d-Day Rolling Residual Spread\n(Entries = ▲ green, Exits = ▼ red)", window_size),
    x     = "Date",
    y     = "Residual Spread (USD)"
  ) +
  theme_minimal()



# At the end, report your profit and % return
final_profit <- tail(aligned$cumPnL_roll, 1)
return_pct   <- final_profit / portfolio * 100

message(sprintf("Total Profit = $%.2f", final_profit))
message(sprintf("Portfolio Return = %.2f%%", return_pct))



# Remove any NA or first-zero rows from PnL
valid_pnl <- aligned$pnl_roll[aligned$pnl_roll != 0]

# 1. Mean daily PnL
mean_pnl <- mean(valid_pnl)

# 2. Daily standard deviation
sd_pnl <- sd(valid_pnl)

# 3. Daily Sharpe ratio (risk-free rate assumed = 0)
sharpe_daily <- mean_pnl / sd_pnl

# Optional: annualized Sharpe ratio (assuming 252 trading days)
sharpe_annual <- sharpe_daily * sqrt(252)

# Print results
message(sprintf("Mean Daily PnL     = $%.2f", mean_pnl))
message(sprintf("Daily SD of PnL    = $%.2f", sd_pnl))
message(sprintf("Daily Sharpe Ratio = %.4f", sharpe_daily))
message(sprintf("Annualized Sharpe  = %.4f", sharpe_annual))


