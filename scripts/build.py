"""Tải giá ngày, chấm 11 tiêu chí, quét 4 tổ hợp mua và xuất JSON cho trang web.

Chạy:  python scripts/build.py
Biến môi trường:
  OFFLINE_DIR   thư mục CSV (t,o,h,l,c,v) để chạy thử không cần mạng
  OFFLINE_DIV   chia giá CSV cho số này để ra nghìn đồng (mặc định 1000; chỉ số không chia)
  LIQ_MIN       ngưỡng GTGD TB20 tỷ đồng (mặc định 1)
"""
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from indicators import CRIT, CRIT_LABEL, COMBOS, REGIME_NOTE, criteria, regimes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'data')
HIST = os.path.join(OUT, 'history')
LIQ_MIN = float(os.environ.get('LIQ_MIN', '1'))
DAYS = 800                      # lịch sử tải về (đủ cho MA200 và làm nóng chỉ báo)
VN = dt.timezone(dt.timedelta(hours=7))
UA = {'User-Agent': 'Mozilla/5.0 (tin-hieu-ta; daily build)', 'Accept': 'application/json'}


# ---------------- tải giá ----------------
def _frame(j, src):
    if not j or not j.get('t'):
        return None
    d = pd.DataFrame({k: j[k] for k in ('t', 'o', 'h', 'l', 'c', 'v')})
    d['t'] = pd.to_datetime(d.t, unit='s', utc=True).dt.tz_convert('Asia/Ho_Chi_Minh').dt.normalize().dt.tz_localize(None)
    d = d.drop_duplicates('t', keep='last').set_index('t').sort_index().astype(float)
    d = d[(d.c > 0) & (d.o > 0)]
    d.attrs['src'] = src
    return d


def fetch_dnse(sym, is_index, t0, t1):
    kind = 'index' if is_index else 'stock'
    u = f'https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}'
    r = requests.get(u, params=dict(symbol=sym, resolution='1D', **{'from': t0, 'to': t1}), headers=UA, timeout=20)
    r.raise_for_status()
    return _frame(r.json(), 'DNSE')


def fetch_vps(sym, is_index, t0, t1):
    u = 'https://histdatafeed.vps.com.vn/tradingview/history'
    r = requests.get(u, params=dict(symbol=sym, resolution='D', **{'from': t0, 'to': t1}), headers=UA, timeout=20)
    r.raise_for_status()
    return _frame(r.json(), 'VPS')


def fetch(sym, is_index=False):
    t1 = int(time.time()) + 86400
    t0 = t1 - DAYS * 86400
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
    print(f'  ! {sym}: không tải được ({err})', flush=True)
    return None


def load_offline(sym, is_index):
    p = os.path.join(os.environ['OFFLINE_DIR'], f'{sym}.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=['t']).drop_duplicates('t').set_index('t').sort_index().astype(float)
    if not is_index:
        div = float(os.environ.get('OFFLINE_DIV', '1000'))
        d[['o', 'h', 'l', 'c']] /= div
    d = d[(d.c > 0) & (d.o > 0)]
    d.attrs['src'] = 'offline'
    return d


def get(sym, is_index=False):
    d = load_offline(sym, is_index) if os.environ.get('OFFLINE_DIR') else fetch(sym, is_index)
    # chạy trong giờ giao dịch: bỏ nến hôm nay chưa chốt (ATC đóng 14:45, chờ đến 15:05)
    now = dt.datetime.now(VN)
    if d is not None and len(d) and now.hour * 60 + now.minute < 15 * 60 + 5 \
            and d.index[-1] == pd.Timestamp(now.date()):
        src = d.attrs.get('src'); d = d.iloc[:-1]; d.attrs['src'] = src
    return d


# ---------------- tính toán ----------------
def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def scan_symbol(meta, df, ix_close, session):
    if df is None or len(df) < 210:
        return None, 'thiếu lịch sử'
    if df.index[-1] != session:
        return None, f'chưa có phiên {session.date()} (phiên cuối {df.index[-1].date()})'
    X = criteria(df, ix_close)
    n = len(X); last = X.iloc[-1]; prev = X.iloc[-2]
    c = df.c.values
    st = X.Supertrend.values
    liq = float(last.GTGD20)
    row = dict(
        s=meta['symbol'], name=meta.get('name'), ex=meta.get('exchange'), sec=meta.get('sector'),
        c=r2(c[-1]), chg=r2((c[-1] / c[-2] - 1) * 100), liq=r2(liq, 1),
        score=int(last.Diem), dscore=int(last.Diem - prev.Diem),
        crit=[int(bool(last[k])) for k in CRIT],
        rsi=r2(last.rsi, 1), adx=r2(last.adx, 1), cci=r2(last.cci, 0), st=r2(last.st_line),
        macd_up=bool(last.macd_up), macd_dn=bool(last.macd_dn),
        st_flip_dn=bool(st[-2] and not st[-1]), st_flip_up=bool(st[-1] and not st[-2]),
        ok_liq=liq >= LIQ_MIN, combos={},
    )
    for key, cb in COMBOS.items():
        on = X[cb['crit']].all(axis=1).values
        liqok = X.GTGD20.values >= LIQ_MIN
        ent = np.where(on & ~np.r_[False, on[:-1]] & liqok)[0]
        info = dict(on=bool(on[-1]))
        if len(ent):
            e = ent[-1]
            open_ = bool(st[e:].all())
            info.update(e=str(X.index[e].date()), age=int(n - 1 - e), pe=r2(c[e]),
                        ret=r2((c[-1] / c[e] - 1) * 100, 1), open=open_)
            # vừa bị bán phiên này: vị thế còn mở đến phiên trước, Supertrend lật xuống hôm nay
            info['exit'] = bool(e <= n - 2 and st[e:n - 1].all() and not st[-1])
        row['combos'][key] = info
    return row, None


def main():
    t_start = time.time()
    uni = json.load(open(os.path.join(os.path.dirname(__file__), 'universe.json')))
    metas = [dict(zip(uni['columns'], r)) for r in uni['rows']]
    ix = get('VNINDEX', True)
    if ix is None or len(ix) < 260:
        sys.exit('Không tải được VN-Index, dừng.')
    session = ix.index[-1]
    print('Phiên', session.date(), 'nguồn', ix.attrs.get('src'), flush=True)

    with ThreadPoolExecutor(max_workers=int(os.environ.get('WORKERS', '6'))) as ex:
        frames = list(ex.map(lambda m: get(m['symbol']), metas))

    rows, skipped, srcs = [], [], {}
    for m, df in zip(metas, frames):
        if df is not None:
            srcs[df.attrs.get('src')] = srcs.get(df.attrs.get('src'), 0) + 1
        r, why = scan_symbol(m, df, ix.c, session)
        if r:
            rows.append(r)
        else:
            skipped.append(dict(s=m['symbol'], why=why))
    if len(rows) < float(os.environ.get('MIN_FRAC', '0.5')) * len(metas):
        sys.exit(f'Chỉ chấm được {len(rows)}/{len(metas)} mã, không ghi đè dữ liệu cũ.')

    reg, dd, m50, m200 = regimes(ix)
    ic = ix.c
    regime = reg.iloc[-1]
    hist_reg = [dict(d=str(d.date()), c=r2(v), g=g) for d, v, g in zip(ic.index[-260:], ic.values[-260:], reg.values[-260:])]
    market = dict(
        session=str(session.date()), vnindex=r2(ic.iloc[-1]), chg=r2((ic.iloc[-1] / ic.iloc[-2] - 1) * 100),
        regime=regime, regime_note=REGIME_NOTE.get(regime, ''), dd=r2(dd.iloc[-1] * 100, 1),
        ma50=r2(m50.iloc[-1]), ma200=r2(m200.iloc[-1]), history=hist_reg,
    )
    liq_rows = [r for r in rows if r['ok_liq']]
    dist = [sum(1 for r in liq_rows if r['score'] == k) for k in range(12)]

    def events(kind):
        out = []
        for r in liq_rows:
            ks = [k for k, v in r['combos'].items() if (kind == 'buy' and v.get('age') == 0 and v.get('open'))
                  or (kind == 'sell' and v.get('exit'))]
            if ks:
                out.append(dict(s=r['s'], sec=r['sec'], c=r['c'], chg=r['chg'], score=r['score'], liq=r['liq'], combos=ks))
        return sorted(out, key=lambda x: (-len(x['combos']), -x['liq']))

    buys, sells = events('buy'), events('sell')
    now = dt.datetime.now(VN)
    latest = dict(
        generated=now.strftime('%Y-%m-%d %H:%M'), liq_min=LIQ_MIN, n_universe=len(metas), n_scored=len(rows),
        n_liq=len(liq_rows), skipped=skipped, sources=srcs, criteria=[dict(k=k, label=CRIT_LABEL[k]) for k in CRIT],
        combos={k: dict(name=v['name'], crit=v['crit']) for k, v in COMBOS.items()},
        market=market, dist=dist, buys=buys, sells=sells, rows=rows,
    )
    os.makedirs(HIST, exist_ok=True)
    with open(os.path.join(OUT, 'latest.json'), 'w', encoding='utf-8') as f:
        json.dump(latest, f, ensure_ascii=False, separators=(',', ':'))

    # nhật ký tín hiệu theo phiên (nhỏ gọn) và chỉ mục
    day = dict(session=market['session'], regime=regime, vnindex=market['vnindex'], buys=buys, sells=sells,
               dist=dist, n_liq=len(liq_rows))
    with open(os.path.join(HIST, f"{market['session']}.json"), 'w', encoding='utf-8') as f:
        json.dump(day, f, ensure_ascii=False, separators=(',', ':'))
    idx_p = os.path.join(HIST, 'index.json')
    days = json.load(open(idx_p)) if os.path.exists(idx_p) else []
    days = sorted(set(days) | {market['session']}, reverse=True)
    json.dump(days, open(idx_p, 'w'))

    print(f"Xong {len(rows)} mã ({len(liq_rows)} đủ thanh khoản), bỏ qua {len(skipped)}; "
          f"mua mới {len(buys)}, bán {len(sells)}; nguồn {srcs}; {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
