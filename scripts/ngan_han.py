"""Bộ tín hiệu ngắn hạn (giữ tối đa 30 phiên, cắt lỗ 7%), backtest 25/11/2026 trên 199 mã theo dõi.

G1 = MACD cắt lên signal + Stochastic(14,6,5) %K cắt lên %D dưới 25 + EMA9 cắt lên EMA26, cả ba trong 10 phiên.
G2 = G1 + RSI vượt MA20-RSI, chạm lại, vượt lần nữa (4 điều kiện trong 10 phiên).
Mua giá mở cửa phiên sau tín hiệu. Bán giá mở cửa phiên thứ 30 kể từ ngày mua, hoặc phiên sau khi giá
đóng cửa thấp hơn giá mua 7% (sớm nhất phiên thứ 3, hàng về T+2). Mỗi mã giữ một lệnh mỗi bộ.
"""
import numpy as np
import pandas as pd

HOLD = 30           # bán mở cửa phiên b + HOLD
STOP = 0.07         # cắt lỗ theo giá đóng cửa so với giá mua
WIN = 10            # cửa sổ các điều kiện cùng xảy ra
FB, FS = 0.0015, 0.0025
WARM = 60

STRATS = {
    'G1': dict(name='MACD + Stoch<25 + EMA9/26 trong 10 phiên', cond=['M1', 'M3', 'M5'],
               bt=dict(n=2203, win=46.2, avg_win=15.4, avg_loss=-7.3, exp=3.2, exp_is=4.5, exp_oos=1.5, base_is=2.5, base_oos=0.2, hold=23)),
    'G2': dict(name='MACD + Stoch<25 + RSI chạm lại MA20 + EMA9/26 trong 10 phiên', cond=['M1', 'M3', 'M4', 'M5'],
               bt=dict(n=641, win=48.5, avg_win=14.9, avg_loss=-7.1, exp=3.6, exp_is=5.1, exp_oos=1.6, base_is=2.5, base_oos=0.2, hold=24)),
}
COND_LABEL = {'M1': 'MACD(12,26,9) cắt lên signal', 'M3': 'Stochastic %K14 làm mượt 6, %D 5: %K cắt lên %D, cả hai < 25',
              'M4': 'RSI14 cắt lên MA20 của RSI, trong 20 phiên quay về chạm MA (cách ≤ 1 điểm), rồi vượt lên lần nữa',
              'M5': 'EMA9 cắt lên EMA26'}


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def _sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def _xup(a, b):
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _rsi(c, n=14):
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    return 100 - 100 / (1 + up / np.where(dn == 0, 1e-12, dn))


def _rsi_retest(r, m, touch=1.0, fall=3.0, tmax=20):
    n = len(r); out = np.zeros(n, bool); st = 0; t1 = 0
    for t in range(1, n):
        if np.isnan(m[t]) or np.isnan(m[t - 1]):
            continue
        d = r[t] - m[t]
        if st > 0 and (d < -fall or t - t1 > tmax):
            st = 0
        if st == 0:
            if r[t] > m[t] and r[t - 1] <= m[t - 1]:
                st = 1; t1 = t
        elif st == 1:
            if d <= touch:
                st = 2
        elif d > touch and r[t] > r[t - 1]:
            out[t] = True; st = 0
    return out


def conditions(df):
    """df có cột o, h, l, c. Trả về dict M1, M3, M4, M5 (mảng bool, sự kiện tại đóng cửa)."""
    h, l, c = (df[k].values.astype(float) for k in ('h', 'l', 'c'))
    macd = _ema(c, 12) - _ema(c, 26); sig = _ema(macd, 9)
    R = _rsi(c); RM = _sma(R, 20)
    ll = pd.Series(l).rolling(14).min().values; hh = pd.Series(h).rolling(14).max().values
    rng = hh - ll
    raw = 100 * (c - ll) / np.where(rng == 0, np.nan, rng)
    K = _sma(raw, 6); D = _sma(K, 5)
    M = dict(M1=_xup(macd, sig), M3=_xup(K, D) & (K < 25) & (D < 25), M4=_rsi_retest(R, RM), M5=_xup(_ema(c, 9), _ema(c, 26)))
    for k in M:
        M[k][:WARM] = False
    return M


def entries(M, cond):
    act = np.ones(len(M[cond[0]]), bool); ev = np.zeros_like(act)
    for k in cond:
        act &= pd.Series(M[k].astype(np.int8)).rolling(WIN, min_periods=1).max().values.astype(bool)
        ev |= M[k]
    return act & ev


def simulate(o, c, ent):
    """Trả về (lệnh đã đóng [(b, s, lý do)], vị thế mở (b, pend) hoặc None)."""
    n = len(c); trades = []; pos = False; b = 0; pend = False
    for t in range(1, n):
        if pos and (t >= b + HOLD or (pend and t >= b + 3)):
            trades.append((b, t, 'Đủ 30 phiên' if t >= b + HOLD and not pend else 'Cắt lỗ 7%')); pos = False
        elif not pos and ent[t - 1]:
            pos = True; b = t; pend = False
        if pos and c[t] < o[b] * (1 - STOP):
            pend = True
    return trades, ((b, pend) if pos else None)


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def scan(metas, dfs, session, liq_min=1.0, recent=60):
    """dfs: {mã: DataFrame o,h,l,c,v theo ngày}. Trả về dict cho latest.json."""
    sec = {m['symbol']: m.get('sector') for m in metas}
    out = dict(buys=[], holding=[], sells=[], closed=[], stats={})
    agg = {k: [] for k in STRATS}
    for s, df in dfs.items():
        if df is None or len(df) < WARM + 30 or df.index[-1] != session:
            continue
        o = df.o.values.astype(float); c = df.c.values.astype(float); n = len(c); dates = df.index
        liq = float(pd.Series(c * df.v.values).rolling(20).mean().iloc[-1] / 1e6)   # tỷ đồng (giá nghìn đồng)
        M = conditions(df)
        base = dict(s=s, sec=sec.get(s), c=_r(c[-1]), chg=_r((c[-1] / c[-2] - 1) * 100), liq=_r(liq, 1), ok_liq=liq >= liq_min)
        for k, sp in STRATS.items():
            ent = entries(M, sp['cond'])
            trades, op = simulate(o, c, ent)
            for b, t, why in trades:
                r = o[t] * (1 - FS) / (o[b] * (1 + FB)) - 1
                agg[k].append(r)
                if t >= n - recent:
                    out['closed'].append(dict(s=s, g=k, buy_d=str(dates[b].date()), buy_px=_r(o[b]), sell_d=str(dates[t].date()),
                                              sell_px=_r(o[t]), days=int(t - b), ret=_r(r * 100, 1), why=why))
            if op:
                b, pend = op
                held = n - b                       # số phiên đã giữ, tính cả phiên mua
                ret = c[-1] * (1 - FS) / (o[b] * (1 + FB)) - 1
                sell_time = n >= b + HOLD
                sell_stop = pend and n >= b + 3
                row = dict(base, g=k, buy_d=str(dates[b].date()), buy_px=_r(o[b]), days=int(held), ret=_r(ret * 100, 1),
                           left=int(max(b + HOLD - n, 0)), stop_px=_r(o[b] * (1 - STOP)), pend=bool(pend))
                out['holding'].append(row)
                if sell_time or sell_stop:
                    out['sells'].append(dict(row, why='Cắt lỗ 7%' if pend else 'Đủ 30 phiên'))
            elif ent[-1]:
                conds = [kk for kk in sp['cond'] if M[kk][-WIN:].any()]
                out['buys'].append(dict(base, g=k, conds=conds))
    for k, rs in agg.items():
        rs = np.array(rs)
        if len(rs):
            w = rs[rs > 0]; lo = rs[rs <= 0]
            out['stats'][k] = dict(n=int(len(rs)), win=_r(len(w) / len(rs) * 100, 1), exp=_r(rs.mean() * 100, 2),
                                   avg_win=_r(w.mean() * 100, 1) if len(w) else None, avg_loss=_r(lo.mean() * 100, 1) if len(lo) else None,
                                   start=str(min(d.index[WARM] for d in dfs.values() if d is not None and len(d) > WARM).date()))
    out['buys'].sort(key=lambda x: (x['g'] != 'G2', -(x['liq'] or 0)))
    out['holding'].sort(key=lambda x: (x['g'], x['buy_d']))
    out['closed'].sort(key=lambda x: x['sell_d'], reverse=True)
    out['strats'] = {k: dict(name=v['name'], cond=v['cond'], bt=v['bt']) for k, v in STRATS.items()}
    out['cond_label'] = COND_LABEL
    out['rule'] = dict(hold=HOLD, stop=STOP * 100, win=WIN)
    return out


# ---------------- danh mục tối đa 7 mã ----------------
MAXPOS = 7
PF_START = '2025-01-02'     # bắt đầu mô phỏng danh mục trên web (cố định để vị thế không đổi theo cửa sổ dữ liệu)
RANK_NOTE = 'Khi tín hiệu nhiều hơn chỗ trống: ưu tiên mã có G2, rồi GTGD TB20 cao hơn; chỉ mua mã GTGD TB20 ≥ ngưỡng.'


def portfolio(dfs, ix_close, session, liq_min=1.0, start=PF_START, n_max=MAXPOS):
    """Danh mục G1/G2: tối đa n_max mã, mỗi lệnh 1/n_max giá trị danh mục, tiền mặt 0%.
    Backtest 2014–2026 (199 mã): CAGR 17,2%/năm, MaxDD −45%, VN-Index 10,0%, −45%."""
    syms = [s for s, d in dfs.items() if d is not None and len(d) >= WARM + 30 and d.index[-1] == session]
    cal = ix_close.index[(ix_close.index <= session)]
    cal = cal[cal >= cal[0]]
    T = len(cal); n = len(syms)
    O = np.full((T, n), np.nan); C = O.copy(); LQ = np.zeros((T, n)); S1 = np.zeros((T, n), bool); S2 = S1.copy()
    for j, s in enumerate(syms):
        d = dfs[s]; pos = cal.get_indexer(d.index); ok = pos >= 0
        M = conditions(d)
        O[pos[ok], j] = d.o.values[ok]; C[pos[ok], j] = d.c.values[ok]
        LQ[pos[ok], j] = (pd.Series(d.c.values * d.v.values).rolling(20).mean().values / 1e6)[ok]
        S1[pos[ok], j] = entries(M, STRATS['G1']['cond'])[ok]; S2[pos[ok], j] = entries(M, STRATS['G2']['cond'])[ok]
    C = pd.DataFrame(C).ffill().values
    t0 = int(np.searchsorted(cal.values, np.datetime64(start)))
    cash = 1.0; hold = {}; trades = []; nav = []

    def rank(t, cand):
        return sorted(cand, key=lambda j: (-int(S2[t, j]), -np.nan_to_num(LQ[t, j])))

    def value(t):
        return cash + sum(h['q'] * C[t, j] for j, h in hold.items())

    for t in range(t0, T):
        for j in list(hold):
            h = hold[j]
            if (t >= h['b'] + HOLD or (h['pend'] and t >= h['b'] + 3)) and np.isfinite(O[t, j]):
                v = h['q'] * O[t, j] * (1 - FS); cash += v
                trades.append(dict(s=syms[j], g=h['g'], buy_d=str(cal[h['b']].date()), buy_px=_r(O[h['b'], j]), sell_d=str(cal[t].date()),
                                   sell_px=_r(O[t, j]), days=int(t - h['b']), ret=_r((v / h['cost'] - 1) * 100, 1),
                                   why='Cắt lỗ 7%' if h['pend'] else 'Đủ 30 phiên'))
                del hold[j]
        if t > t0:
            cand = [j for j in np.where((S1[t - 1] | S2[t - 1]) & (LQ[t - 1] >= liq_min) & np.isfinite(O[t]))[0] if j not in hold]
            nv = cash + sum(h['q'] * C[t - 1, j] for j, h in hold.items())
            for j in rank(t - 1, cand)[:max(n_max - len(hold), 0)]:
                amt = min(nv / n_max, cash)
                if amt <= nv * 0.02:
                    break
                cash -= amt
                hold[j] = dict(q=amt / (O[t, j] * (1 + FB)), b=t, pend=False, cost=amt, g='G2' if S2[t - 1, j] else 'G1')
        for j, h in hold.items():
            if C[t, j] < O[h['b'], j] * (1 - STOP):
                h['pend'] = True
        nav.append(value(t))
    # lệnh phiên kế tiếp
    tn = T
    sells = [j for j, h in hold.items() if tn >= h['b'] + HOLD or (h['pend'] and tn >= h['b'] + 3)]
    free = n_max - len(hold) + len(sells)
    cand = [j for j in np.where((S1[T - 1] | S2[T - 1]) & (LQ[T - 1] >= liq_min))[0] if j not in hold]
    ranked = rank(T - 1, cand)
    navT = nav[-1] if nav else 1.0
    positions = []
    for j, h in sorted(hold.items(), key=lambda x: x[1]['b']):
        px = C[T - 1, j]
        positions.append(dict(s=syms[j], g=h['g'], buy_d=str(cal[h['b']].date()), buy_px=_r(O[h['b'], j]), c=_r(px),
                              ret=_r((px * (1 - FS) / (O[h['b'], j] * (1 + FB)) - 1) * 100, 1), days=int(T - h['b']),
                              left=int(max(h['b'] + HOLD - T, 0)), stop_px=_r(O[h['b'], j] * (1 - STOP)),
                              w=_r(h['q'] * px / navT * 100, 1), selling=j in sells,
                              why=('Cắt lỗ 7%' if h['pend'] else 'Đủ 30 phiên') if j in sells else None))
    vi = ix_close.reindex(cal[t0:]).ffill().values
    eq = np.array(nav); r = np.r_[0, eq[1:] / eq[:-1] - 1] if len(eq) > 1 else np.zeros(1)
    tr = np.array([x['ret'] for x in trades]) if trades else np.array([])
    yrs = max(len(eq) / 250, 1e-9)
    summ = dict(start=str(cal[t0].date()), ret=_r((eq[-1] - 1) * 100, 1), vni_ret=_r((vi[-1] / vi[0] - 1) * 100, 1),
                cagr=_r((eq[-1] ** (1 / yrs) - 1) * 100, 1) if yrs >= 1 else None,
                maxdd=_r((eq / np.maximum.accumulate(eq) - 1).min() * 100, 1), vni_maxdd=_r((vi / np.maximum.accumulate(vi) - 1).min() * 100, 1),
                n_trades=len(tr), win=_r((tr > 0).mean() * 100, 0) if len(tr) else None, avg=_r(tr.mean(), 1) if len(tr) else None,
                cash=_r(cash / navT * 100, 1))
    step = max(1, len(eq) // 300)
    series = [[str(cal[t0 + i].date()), _r(eq[i] * 100, 2), _r(vi[i] / vi[0] * 100, 2)] for i in range(0, len(eq), step)]
    if len(eq) and series[-1][0] != str(cal[T - 1].date()):
        series.append([str(cal[T - 1].date()), _r(eq[-1] * 100, 2), _r(vi[-1] / vi[0] * 100, 2)])
    return dict(maxpos=n_max, rank_note=RANK_NOTE, summary=summ, positions=positions,
                pend_buy=[dict(s=syms[j], g='G2' if S2[T - 1, j] else 'G1', liq=_r(LQ[T - 1, j], 1)) for j in ranked[:max(free, 0)]],
                skipped=[dict(s=syms[j], g='G2' if S2[T - 1, j] else 'G1', liq=_r(LQ[T - 1, j], 1)) for j in ranked[max(free, 0):]],
                pend_sell=[p for p in positions if p['selling']], trades=trades[::-1][:120], series=series,
                bt=dict(cagr=17.2, maxdd=-45.3, vni_cagr=10.0, vni_maxdd=-45.3, cagr_is=23.2, cagr_oos=3.3, vni_is=14.9, vni_oos=2.2, rand_oos=2.0))
