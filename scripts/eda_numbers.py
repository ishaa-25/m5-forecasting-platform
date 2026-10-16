import duckdb
import pandas as pd

con = duckdb.connect()
parquet = "data/processed/m5_long.parquet"
con.execute(f"CREATE VIEW m5 AS SELECT * FROM read_parquet('{parquet}')")

print("=" * 55)
print("PLOT 2: Daily aggregate sales per store")
print("=" * 55)
r = con.execute("""
    SELECT store_id,
           ROUND(AVG(total_sales), 0) as avg_daily,
           ROUND(MIN(total_sales), 0) as min_daily,
           ROUND(MAX(total_sales), 0) as max_daily
    FROM (
        SELECT date, store_id, SUM(sales) as total_sales
        FROM m5 GROUP BY 1, 2
    )
    GROUP BY store_id ORDER BY store_id
""").df()
print(r.to_string(index=False))

print()
print("=" * 55)
print("PLOT 3: Average sales by day of week")
print("=" * 55)
r2 = con.execute("""
    SELECT weekday, ROUND(AVG(sales), 4) as avg_sales
    FROM m5 GROUP BY weekday
    ORDER BY CASE weekday
        WHEN 'Saturday'  THEN 0
        WHEN 'Sunday'    THEN 1
        WHEN 'Monday'    THEN 2
        WHEN 'Tuesday'   THEN 3
        WHEN 'Wednesday' THEN 4
        WHEN 'Thursday'  THEN 5
        ELSE 6 END
""").df()
print(r2.to_string(index=False))

print()
print("=" * 55)
print("PLOT 4: Intermittency (zero-sales days)")
print("=" * 55)
r3 = con.execute("""
    SELECT
      SUM(CASE WHEN zero_rate > 0.5 THEN 1 ELSE 0 END) as series_gt50pct_zeros,
      SUM(CASE WHEN zero_rate > 0.9 THEN 1 ELSE 0 END) as series_gt90pct_zeros,
      COUNT(*) as total_series
    FROM (
      SELECT item_id, store_id,
             SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as zero_rate
      FROM m5 GROUP BY item_id, store_id
    )
""").df()
print(r3.to_string(index=False))

print()
print("=" * 55)
print("PLOT 5: Price vs Sales correlation by department")
print("=" * 55)
r4 = con.execute("""
    SELECT dept_id, ROUND(CORR(sell_price, sales), 4) as price_sales_corr
    FROM m5
    WHERE sell_price IS NOT NULL
    GROUP BY dept_id
    ORDER BY price_sales_corr
""").df()
print(r4.to_string(index=False))

print()
print("=" * 55)
print("PLOT 6: Event day vs Normal day sales lift")
print("=" * 55)
r5 = con.execute("""
    SELECT
        CASE WHEN event_name_1 != 'none' THEN 'Event day' ELSE 'Normal day' END as day_type,
        ROUND(AVG(sales), 4) as avg_sales,
        COUNT(*) as n_rows
    FROM m5
    GROUP BY 1
""").df()
print(r5.to_string(index=False))

normal = float(r5[r5.day_type == "Normal day"]["avg_sales"])
event  = float(r5[r5.day_type == "Event day"]["avg_sales"])
lift   = (event - normal) / normal * 100
print(f"\nEvent-day lift: {lift:+.1f}%")
