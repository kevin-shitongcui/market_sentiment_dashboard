"""Run original analytics unchanged, and serialize its in-memory results for GitHub Pages."""
import json
import math
import runpy
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'site'
DATA = SITE / 'data'
DATA.mkdir(parents=True, exist_ok=True)

# Original script calculates all results and creates the PDF, unchanged.
state = runpy.run_path(str(ROOT / 'legacy_report.py'))


def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()[:10]
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def timeseries(s):
    return [[str(date.date()), clean(value)] for date, value in s.dropna().items()]


rows = []
for result in state['results']:
    fields = ('bucket', 'ratio_name', 'short_name', 'purpose', 'current_ratio',
              'current_date', 'avg_10y', 'avg_5y', 'avg_3y', 'avg_1y',
              'deviation_from_10y', 'deviation_from_5y', 'deviation_from_3y',
              'deviation_from_1y', 'one_vs_three_pct', 'three_vs_five_pct',
              'five_vs_ten_pct', 'momentum_value', 'momentum_tstat',
              'momentum', 'acceleration_value', 'acceleration_z', 'acceleration',
              'zscore_3y', 'percentile_3y', 'positioning', 'history_years',
              'smart_headline', 'smart_comment', 'summary_interpretation')
    row = {key: clean(result.get(key)) for key in fields}
    row['history'] = timeseries(result['series'])
    row['momentum_history'] = timeseries(result['momentum_series'])
    row['acceleration_history'] = timeseries(result['acceleration_series'])
    rows.append(row)

em = state.get('EM_BOND_DECOMP')
em_data = None
if em:
    em_data = {
        'holdings_as_of': clean(em['holdings_as_of']),
        'weight_source': em['weight_source'],
        'fx_weight_coverage': clean(em['fx_weight_coverage']),
        'missing_fx_currencies': em['missing_fx_currencies'],
        'canada_zfs_weight': clean(em['canada_zfs_weight']),
        'canada_zfm_weight': clean(em['canada_zfm_weight']),
        'lemb_target_duration': clean(em['lemb_target_duration']),
        'period_attribution': clean(em['period_attribution']),
        'currency_weights': clean(em['used_currency_weights'].to_dict()),
        'series': {k: timeseries(em[k]) for k in (
            'em_unhedged_cad', 'local_rates_carry', 'fx_proxy_cad',
            'canada_bonds', 'local_vs_canada_index')}
    }

payload = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'market_date': max((x['current_date'] for x in rows if x['current_date']), default=None),
    'indicators': rows,
    'overview': {
        'overall': state['OVERALL_COMMENT'],
        'factors': state['PORTFOLIO_COMMENT'],
        'risk': state['RISK_COMMENT']
    },
    'em': em_data
}
out = DATA / 'dashboard.json'
tmp = out.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
tmp.replace(out)
pdf = Path(state['OUTPUT_PDF'])
if pdf.exists():
    shutil.copyfile(pdf, SITE / 'market_sentiment_report.pdf')
print(f'Exported {len(rows)} indicators ({out.stat().st_size / 1024:.0f} KiB)')
