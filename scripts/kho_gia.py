"""Kho giá ngày dùng chung: data/gia/<MÃ>.csv, mỗi lần chỉ tải thêm phiên mới.

Chạy:  python scripts/kho_gia.py            (cập nhật tăng dần)
       python scripts/kho_gia.py --full     (tải lại toàn bộ lịch sử)
       python scripts/kho_gia.py FPT HPG    (chỉ vài mã)

Cột CSV: date,open,high,low,close,volume. Cổ phiếu tính nghìn đồng, chỉ số tính điểm.
Nguồn chính DNSE entrade (giá đã điều chỉnh), dự phòng VPS histdatafeed.
Khi giá cũ bị điều chỉnh lại (chia cổ tức, phát hành thêm), phần chồng lấn lệch > 0,5%
thì mã đó được tải lại toàn bộ để chuỗi giá nhất quán.
Chỉ in tóm tắt, không in dữ liệu.
"""
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'gia')
START = int(dt.datetime(2000, 1, 1).timestamp())
OVERLAP_DAYS = 15
TOL = 0.005
INDEXES = ['VNINDEX', 'VN30', 'HNXINDEX', 'UPCOMINDEX']
UA = {'User-Agent': 'Mozilla/5.0 (tin-hieu-ta; kho gia)', 'Accept': 'application/json'}
COLS = ['open', 'high', 'low', 'close', 'volume']


def _frame(j, src):
    if not j or not j.get('t'):
        return None
    d = pd.DataFrame({'date': j['t'], 'open': j['o'], 'high': j['h'], 'low': j['l'], 'close': j['c'], 'volume': j['v']})
    d['date'] = pd.to_datetime(d.date, unit='s', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.strftime('%Y-%m-%d')
    d = d.drop_duplicates('date', keep='last').set_index('date').sort_index().astype(float)
    d = d[(d.close > 0) & (d.open > 0)]
    d.attrs['src'] = src
    return d


def fetch_dnse(sym, is_index, t0, t1):
    kind = 'index' if is_index else 'stock'
    r = requests.get(f'https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}',
                     params={'symbol': sym, 'resolution': '1D', 'from': t0, 'to': t1}, headers=UA, timeout=30)
    r.raise_for_status()
    return _frame(r.json(), 'DNSE')


def fetch_vps(sym, is_index, t0, t1):
    r = requests.get('https://histdatafeed.vps.com.vn/tradingview/history',
                     params={'symbol': sym, 'resolution': 'D', 'from': t0, 'to': t1}, headers=UA, timeout=30)
    r.raise_for_status()
    return _frame(r.json(), 'VPS')


def fetch(sym, is_index, t0):
    t1 = int(time.time()) + 86400
    err = None
    for f in (fetch_dnse, fetch_vps):
        for attempt in range(3):
            try:
                d = f(sym, is_index, t0, t1)
                if d is not None and len(d):
                    return d
                break
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(1 + attempt * 2)
    return err


def save(sym, d):
    d = d.copy()
    d[['open', 'high', 'low', 'close']] = d[['open', 'high', 'low', 'close']].round(4)
    d['volume'] = d.volume.round(0).astype('int64')
    d.to_csv(os.path.join(OUT, f'{sym}.csv'), index_label='date')


def update(sym, is_index, full):
    p = os.path.join(OUT, f'{sym}.csv')
    old = None
    if os.path.exists(p) and not full:
        old = pd.read_csv(p, index_col='date').astype(float)
    if old is None or old.empty:
        d = fetch(sym, is_index, START)
        if not isinstance(d, pd.DataFrame):
            return sym, 'loi', str(d)
        save(sym, d)
        return sym, 'moi', f'{d.attrs["src"]} {d.index[0]}..{d.index[-1]} {len(d)}'
    last = old.index[-1]
    t0 = int((dt.datetime.strptime(last, '%Y-%m-%d') - dt.timedelta(days=OVERLAP_DAYS)).timestamp())
    new = fetch(sym, is_index, t0)
    if not isinstance(new, pd.DataFrame):
        return sym, 'loi', str(new)
    common = old.index.intersection(new.index)
    if len(common):
        dev = (new.loc[common, 'close'] / old.loc[common, 'close'] - 1).abs().max()
        if dev > TOL:
            d = fetch(sym, is_index, START)
            if not isinstance(d, pd.DataFrame):
                return sym, 'loi', str(d)
            save(sym, d)
            return sym, 'dieu_chinh', f'lệch {dev:.1%}, tải lại {len(d)} phiên'
    add = new.index.difference(old.index)
    if not len(add) and not len(common):
        return sym, 'giu', last
    d = pd.concat([old.drop(common), new.loc[common.union(add)]]).sort_index()
    save(sym, d)
    return sym, 'them' if len(add) else 'giu', f'+{len(add)}'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    full = '--full' in sys.argv
    os.makedirs(OUT, exist_ok=True)
    uni = json.load(open(os.path.join(ROOT, 'scripts', 'universe.json')))
    i = uni['columns'].index('symbol')
    stocks = [r[i] for r in uni['rows']]
    jobs = [(s, s in INDEXES) for s in (args or INDEXES + stocks)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: update(j[0], j[1], full), jobs))

    by = {}
    for s, st, _ in res:
        by.setdefault(st, []).append(s)
    lasts = {}
    for f in os.listdir(OUT):
        if f.endswith('.csv'):
            with open(os.path.join(OUT, f), 'rb') as fh:
                fh.seek(-200, 2) if os.path.getsize(fh.name) > 200 else None
                lasts[f[:-4]] = fh.read().decode().strip().splitlines()[-1].split(',')[0]
    session = lasts.get('VNINDEX') or max(lasts.values())
    behind = sorted(s for s, l in lasts.items() if l < session)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime('%Y-%m-%d %H:%M')
    lines = [
        f'phien_cuoi={session}',
        f'cap_nhat_luc={now} (gio VN)',
        f'so_ma={len(lasts)}',
        'nguon=DNSE entrade (du phong VPS histdatafeed); gia co phieu nghin dong, da dieu chinh; chi so tinh diem',
        f'cham_phien={",".join(behind) if behind else "khong"}',
        f'loi={",".join(sorted(by.get("loi", []))) or "khong"}',
    ]
    open(os.path.join(OUT, 'cap_nhat.txt'), 'w').write('\n'.join(lines) + '\n')
    print(' | '.join(f'{k}: {len(v)}' for k, v in sorted(by.items())))
    for s, st, msg in res:
        if st in ('loi', 'dieu_chinh') or (st == 'moi' and s in INDEXES + ['FPT']):
            print(f'  {s} {st}: {msg}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
