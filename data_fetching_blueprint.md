# Data Fetching Blueprint: Multi-Source Engine

This blueprint outlines a modular architecture for fetching financial data from multiple sources (e.g., Yahoo Finance, NSE, Investing.com, Crypto APIs) while ensuring robustness, caching, and data alignment.

## Core Architecture Principles

1.  **Modularity**: Each source is an independent "Fetcher" module.
2.  **Robustness**: Multi-level fallbacks (Live -> Cache -> Fallback File).
3.  **Efficiency**: Incremental synchronization; download only what's missing.
4.  **Consistency**: Uniform schema across all sources before merging.

---

## 1. Modular Fetcher Interface

Instead of hardcoding APIs in one function, use a Class-based or Registry-based approach.

### Base Interface (Pythonic Example)

```python
class BaseFetcher:
    """Standard interface for all data sources."""
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
        self.csv_path = f"data/{name}_daily.csv"

    def get_last_date(self):
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return pd.to_datetime(df['date']).max()
        return None

    def fetch(self, start_date):
        raise NotImplementedError("Each fetcher must implement fetch()")
```

---

## 2. Synchronization & Caching Strategy

The engine follows a "Gap-Fill" logic to minimize bandwidth and API hits.

- **Check Current State**: Read cached CSV and get the `last_date`.
- **Fetch Window**: Set `start_date = last_date + 1 day`.
- **Incremental Merge**: Append new data to existing dataframe and `drop_duplicates(subset='date')`.
- **Persistence**: Save immediately to CSV/Parquet to prevent data loss on crash.

---

## 3. Implementation Blueprint (Data Engine Sync)

```python
class DataEngineSync:
    def __init__(self, fetchers):
        self.fetchers = fetchers # List of Fetcher objects

    def sync_all(self):
        results = {}
        for fetcher in self.fetchers:
            last_date = fetcher.get_last_date()
            if not last_date:
                # Full download if no cache exists
                data = fetcher.fetch(start_date="2015-01-01")
            else:
                # Incremental sync
                data = fetcher.fetch(start_date=last_date)
            
            results[fetcher.name] = self._merge_and_save(fetcher, data)
        return results

    def _merge_and_save(self, fetcher, new_data):
        # Implementation of the merging logic...
        pass
```

---

## 4. Data Alignment & Merging Protocol

When merging multiple sources (e.g., Nifty from NSE vs. S&P 500 from Yahoo), timelines often clash due to different holidays or timezones.

### The Alignment Pipeline:
1.  **Anchor Dataset**: Choose the primary asset (e.g., Nifty) as the "Date Master."
2.  **Left Join**: Merge all other sources onto the Anchor's date index.
3.  **Forward Fill (`ffill`)**: Fill gaps for sources that were closed (e.g., US market closed on US Holiday while India was open).
4.  **Backfill (`bfill`)**: Initial cleanup for the start of the series.
5.  **Scaling**: (Optional) Normalize features if they have vastly different ranges.

---

## 5. Live Snapshot Injection

To make the engine "Real-Time Ready," the engine should:
1.  Load the **Historical Master** (all days up to yesterday).
2.  Perform a **Live Hit** (1-minute interval) for the current 1-5 symbols.
3.  **Inject** the live row into the dataframe as the "TODAY" row.
4.  Allow calculations (RSI, Moving Averages) to run on this ephemeral data.

---

## How to Add a New Source
1.  **Create a new Fetcher class** (e.g., `BinanceFetcher`).
2.  **Map the symbol** in your [utils.py](file:///c:/Users/hp/Desktop/David-V2/utils.py) or config file.
3.  **Register the fetcher** in the [load_all_data](file:///c:/Users/hp/Desktop/David-V2/data_engine.py#153-317) sequence.
4.  The engine will automatically handle the caching, merging, and ffilling.
