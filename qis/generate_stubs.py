import csv
import math
import numpy as np
from datetime import date, timedelta

rng = np.random.default_rng(42)

# ── 1. contract_specs.csv ──────────────────────────────────────────────────
specs = [
    ("CL","WTI Crude Oil","energy","NYMEX",1000,0.01,"USD","cash"),
    ("GC","Gold","metals","COMEX",100,0.10,"USD","physical"),
    ("NG","Natural Gas","energy","NYMEX",10000,0.001,"USD","cash"),
    ("HG","Copper","metals","COMEX",25000,0.0005,"USD","cash"),
    ("ZC","Corn","agriculture","CBOT",5000,0.25,"USD","physical"),
    ("ZW","Wheat","agriculture","CBOT",5000,0.25,"USD","physical"),
    ("LC","Live Cattle","livestock","CME",40000,0.00025,"USD","cash"),
]
with open("C:/user/Rohan/claude/qis/tests/stubs/reference/contract_specs.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticker","name","sector","exchange","lot_size","tick_size","currency","settlement"])
    w.writerows(specs)
print("contract_specs.csv done")

# ── 2. production_weights.csv ──────────────────────────────────────────────
base = {"CL":0.28,"GC":0.04,"NG":0.06,"HG":0.03,"ZC":0.07,"ZW":0.04,"LC":0.06}
tickers_w = list(base.keys())
rows_pw = []
for yr in range(2010, 2025):
    noise = rng.uniform(-0.005, 0.005, len(tickers_w))
    raw = np.array([base[t] for t in tickers_w]) + noise
    raw = np.clip(raw, 0.01, None)
    raw = raw / raw.sum() * 0.58  # keep 42% for OTHER
    other = round(1.0 - float(raw.sum()), 6)
    for t, wt in zip(tickers_w, raw):
        rows_pw.append((t, yr, round(float(wt), 6)))
    rows_pw.append(("OTHER", yr, other))

with open("C:/user/Rohan/claude/qis/tests/stubs/reference/production_weights.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticker","year","weight"])
    w.writerows(rows_pw)
print(f"production_weights.csv done ({len(rows_pw)} rows)")

# ── 3. roll_schedule.csv ───────────────────────────────────────────────────
roll_months = {
    "CL": list(range(1,13)),
    "GC": [2,4,6,8,10,12],
    "NG": list(range(1,13)),
    "HG": [3,5,7,9,12],
    "ZC": [3,5,7,9,12],
    "ZW": [3,5,7,9,12],
    "LC": [2,4,6,8,10,12],
}
month_codes = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}

rows_rs = []
for ticker, months in roll_months.items():
    for yr in range(2010, 2025):
        for m in months:
            next_m = m + 1 if m < 12 else 1
            next_yr = yr if m < 12 else yr + 1
            contract_month = f"{ticker}{month_codes[next_m]}{str(next_yr)[-2:]}"
            rows_rs.append((ticker, yr, m, 5, 9, contract_month))

with open("C:/user/Rohan/claude/qis/tests/stubs/reference/roll_schedule.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticker","year","month","roll_start_bday","roll_end_bday","contract_month"])
    w.writerows(rows_rs)
print(f"roll_schedule.csv done ({len(rows_rs)} rows)")

# ── 4. OHLCV files ─────────────────────────────────────────────────────────
params = {
    "CL": (80.0,   0.35),
    "GC": (1100.0, 0.15),
    "NG": (5.5,    0.55),
    "HG": (3.5,    0.25),
    "ZC": (380.0,  0.30),
    "ZW": (500.0,  0.30),
    "LC": (85.0,   0.15),
}

# Business days 2010-01-04 to 2024-12-31
d = date(2010, 1, 4)
end_date = date(2024, 12, 31)
bdays = []
while d <= end_date:
    if d.weekday() < 5:
        bdays.append(d)
    d += timedelta(days=1)
n = len(bdays)
print(f"Business days: {n}")

# Store closes for options lookup
price_lookup = {}

for ticker, (s0, ann_vol) in params.items():
    daily_vol = ann_vol / math.sqrt(252)
    log_rets = rng.normal(-0.5 * daily_vol**2, daily_vol, n)
    closes = s0 * np.exp(np.cumsum(log_rets))
    closes = np.maximum(closes, s0 * 0.10)

    open_noise  = rng.normal(0, daily_vol * 0.3, n)
    range_noise = np.abs(rng.normal(0, daily_vol * 0.5, n))

    opens = np.concatenate([[s0], closes[:-1]]) * np.exp(open_noise)
    opens = np.maximum(opens, s0 * 0.05)
    highs = np.maximum(opens, closes) * (1 + range_noise)
    lows  = np.minimum(opens, closes) * (1 - range_noise)
    lows  = np.maximum(lows, s0 * 0.01)

    vols = rng.integers(50_000, 500_001, n)
    base_oi = rng.integers(150_000, 400_001, n).astype(float)
    for i, dv in enumerate(bdays):
        if 5 <= dv.day <= 12:
            frac = (dv.day - 4) / 8.0
            base_oi[i] *= max(0.6, 1 - frac * 0.35)
    oi = base_oi.astype(int)

    price_lookup[ticker] = {dv: closes[i] for i, dv in enumerate(bdays)}

    path = f"C:/user/Rohan/claude/qis/tests/stubs/futures_ohlcv/{ticker}.csv"
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["date","open","high","low","close","volume","open_interest"])
        for i, dv in enumerate(bdays):
            cw.writerow([
                dv.isoformat(),
                round(float(opens[i]),  4),
                round(float(highs[i]),  4),
                round(float(lows[i]),   4),
                round(float(closes[i]), 4),
                int(vols[i]),
                int(oi[i]),
            ])
    print(f"{ticker}.csv done ({n} rows)")

# ── 5. Options chain stubs ─────────────────────────────────────────────────
opt_dates = []
for yr in range(2010, 2025):
    for mo in range(1, 13):
        first = date(yr, mo, 1)
        while first.weekday() >= 5:
            first += timedelta(days=1)
        opt_dates.append(first)

def make_expiry(d):
    nm = d.month + 1 if d.month < 12 else 1
    ny = d.year if d.month < 12 else d.year + 1
    first = date(ny, nm, 1)
    days_to_fri = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_fri + 14)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black76(F, K, T, sigma, is_call):
    if T <= 0 or F <= 0 or K <= 0:
        return max((F - K) if is_call else (K - F), 0.001)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return max(F * norm_cdf(d1) - K * norm_cdf(d2), 0.001)
    else:
        return max(K * norm_cdf(-d2) - F * norm_cdf(-d1), 0.001)

def delta76(F, K, T, sigma, is_call):
    if T <= 0:
        return 1.0 if (is_call and F > K) else 0.0
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0

rng3 = np.random.default_rng(99)
opt_header = ["date","expiry","strike","price","implied_vol","delta","volume","open_interest"]

opt_configs = [
    ("CL", "CL", 5.0,   0.30, 0.02),
    ("GC", "GC", 50.0,  0.15, 0.01),
]

for ticker, lookup_key, strike_step, atm_iv, skew in opt_configs:
    spot_map = price_lookup[lookup_key]
    for is_call, suffix in [(True, "calls"), (False, "puts")]:
        rows = []
        for d in opt_dates:
            F = spot_map.get(d)
            if F is None:
                for offset in range(1, 6):
                    cand = d + timedelta(days=offset)
                    if cand in spot_map:
                        F = spot_map[cand]
                        break
            if F is None:
                continue
            T = 30.0 / 365.0
            expiry = make_expiry(d)
            atm_strike = round(F / strike_step) * strike_step
            for step in range(-3, 4):
                K = atm_strike + step * strike_step
                if K <= 0:
                    continue
                # skew: OTM options have higher IV
                moneyness_step = step if is_call else -step
                iv = max(atm_iv + abs(moneyness_step) * skew * (1 if moneyness_step < 0 else 0.5), 0.05)
                price = black76(F, K, T, iv, is_call)
                dlt   = delta76(F, K, T, iv, is_call)
                vol_val = int(rng3.integers(100, 5001))
                oi_val  = int(rng3.integers(500, 20001))
                rows.append((
                    d.isoformat(),
                    expiry.isoformat(),
                    round(K, 4),
                    round(price, 4),
                    round(iv, 4),
                    round(dlt, 4),
                    vol_val,
                    oi_val,
                ))
        path = f"C:/user/Rohan/claude/qis/tests/stubs/options_chain/{ticker}_{suffix}.csv"
        with open(path, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(opt_header)
            cw.writerows(rows)
        print(f"{ticker}_{suffix}.csv done ({len(rows)} rows)")

print("All stub files generated successfully.")
