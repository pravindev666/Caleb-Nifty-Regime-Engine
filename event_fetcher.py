import os, json, warnings, argparse, requests
from datetime import date, timedelta, datetime
import pandas as pd

warnings.filterwarnings("ignore")
DATA_DIR    = "data"
EVENTS_CSV  = os.path.join(DATA_DIR, "events.csv")
EVENTS_JSON = os.path.join(DATA_DIR, "events_today.json")
HEADERS     = {'User-Agent': 'Mozilla/5.0'}

def _today(): return date.today()

def get_nifty_expiry_dates(months_ahead=6):
    events = []
    today, end = _today(), _today() + timedelta(days=months_ahead*30)
    d = today
    while d <= end:
        if d.weekday() == 3:
            next_thu = d + timedelta(days=7)
            is_monthly = next_thu.month != d.month
            events.append({'date': d, 'label': 'Monthly Expiry' if is_monthly else 'Weekly Expiry',
                           'type': 'danger' if is_monthly else 'warn', 'impact': 'HIGH' if is_monthly else 'MEDIUM',
                           'tip': 'Avoid selling expiry week.' if is_monthly else 'Options expire today.'})
        d += timedelta(days=1)
    return events

def get_hardcoded_events():
    today, events = _today(), []
    for d, lbl, t in [(date(2026,4,9), "RBI MPC", "danger"), (date(2026,6,5), "RBI MPC", "danger"),
                      (date(2026,4,29), "FOMC", "warn"), (date(2026,6,17), "FOMC", "warn")]:
        if d >= today: events.append({'date': d, 'label': lbl, 'type': t, 'impact': 'HIGH', 'tip': f'{lbl} decision day.'})
    return events

def fetch_nse_holidays():
    events = []
    try:
        r = requests.get("https://www.nseindia.com/api/holiday-master?type=trading", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            for item in r.json().get('FO', []):
                dt = datetime.strptime(item['tradingDate'], '%d-%b-%Y').date()
                if dt >= _today(): events.append({'date': dt, 'label': f"NSE: {item['description']}", 'type': 'danger', 'impact': 'HIGH'})
    except: pass
    return events

def fetch_all_events():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(get_nifty_expiry_dates() + get_hardcoded_events() + fetch_nse_holidays())
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[df['date'] >= _today()].drop_duplicates(subset=['date','label']).sort_values('date')
    df.to_csv(EVENTS_CSV, index=False)
    return df

def load_events(days_ahead=30):
    if not os.path.exists(EVENTS_CSV): fetch_all_events()
    df = pd.read_csv(EVENTS_CSV)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df[(df['date'] >= _today()) & (df['date'] <= _today() + timedelta(days=days_ahead))].to_dict('records')

def is_event_day(d=None):
    if d is None: d = _today()
    if not os.path.exists(EVENTS_CSV): fetch_all_events()
    df = pd.read_csv(EVENTS_CSV)
    df['date'] = pd.to_datetime(df['date']).dt.date
    matched = df[(df['date'] == d) & (df['impact'] == 'HIGH')]
    return not matched.empty, matched['label'].iloc[0] if not matched.empty else None

if __name__ == "__main__":
    print(fetch_all_events())
