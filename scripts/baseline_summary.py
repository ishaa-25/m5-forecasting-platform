import pandas as pd

results = {
    'CA_1': {'ets_add': 0.064553, 'seasonal_naive': 0.094714, 'arima_211': 0.154123},
    'TX_1': {'ets_add': 0.089280, 'seasonal_naive': 0.104925, 'arima_211': 0.129214},
    'WI_1': {'ets_add': 0.078225, 'seasonal_naive': 0.108912, 'arima_211': 0.188104},
}

df = pd.DataFrame(results).T
df['avg'] = df.mean(axis=1)

print("Per-store WAPE:")
print(df.round(4).to_string())
print()

avgs = df['avg']
print(f"Average WAPE across 3 stores:")
for model in ['ets_add','seasonal_naive','arima_211']:
    print(f"  {model:20s}: {avgs[model]:.4f}")

print()
ets_vs_naive_pct = (avgs['seasonal_naive'] - avgs['ets_add']) / avgs['seasonal_naive'] * 100
print(f"ETS beats SeasonalNaive by: {ets_vs_naive_pct:.1f}%")

arima_vs_naive_pct = (avgs['arima_211'] - avgs['seasonal_naive']) / avgs['seasonal_naive'] * 100
print(f"ARIMA is worse than SeasonalNaive by: {arima_vs_naive_pct:.1f}%")

rmsse = {
    'CA_1': {'ets_add': 0.441895, 'seasonal_naive': 0.630202, 'arima_211': 1.056592},
    'TX_1': {'ets_add': 0.605072, 'seasonal_naive': 0.697689, 'arima_211': 0.945431},
    'WI_1': {'ets_add': 0.554070, 'seasonal_naive': 0.780785, 'arima_211': 1.272077},
}
df_r = pd.DataFrame(rmsse).T
print()
print("Average RMSSE across 3 stores:")
print(df_r.mean().round(4).to_string())
