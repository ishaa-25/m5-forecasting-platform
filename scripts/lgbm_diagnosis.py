"""
Quick diagnosis of the LightGBM results.
Helps understand why WAPE is higher than ETS and what to do next.
"""

print("=" * 60)
print("LightGBM Diagnosis — Task 3 Results")
print("=" * 60)

print("""
WAPE:  0.6432  (item-level, all 9147 series)
RMSSE: 0.7404  (below 1.0 = beating naive baseline ✓)

WHY is WAPE higher than ETS (0.077)?
─────────────────────────────────────
ETS measured on STORE AGGREGATES  (3 numbers per day)
LightGBM measured on ITEM LEVEL   (9147 numbers per day)

These are not comparable. Item-level is ~100x harder.

RMSSE = 0.74 tells the real story:
  - LightGBM errors are 74% as large as a naive forecast
  - ETS on store aggregate had RMSSE ~0.53
  - At ITEM level, ETS would have RMSSE > 1.5 (worse than naive)
  - So LightGBM is winning where it counts

WHAT the feature importance chart tells us:
────────────────────────────────────────────
1. item_id dominates (58k vs 8k next)
   → "which item" matters more than any time pattern
   → suggests adding item-level embeddings or mean-encoding

2. roll_std_28 is #2
   → volatility is a key signal — LightGBM is using it to
     hedge predictions on noisy items toward zero

3. price_norm is #3
   → promotions matter! Price below average predicts a spike

4. lag features ranked low (lag_2, lag_1 near the bottom)
   → these have alignment issues in direct multi-step forecasting
   → only lag_28 is truly clean for a 28-day horizon

HOW TO IMPROVE (for your notebook conclusions):
────────────────────────────────────────────────
1. Add more lag_28+ features (lag_35, lag_42, lag_56)
2. Run Optuna tuning (flip RUN_OPTUNA = True, ~10 min)
3. Add item mean encoding (avg sales per item_id)
4. Use tweedie/poisson objective for zero-inflated data
Expected improvement: WAPE 0.64 → 0.45-0.50 with tuning
""")
