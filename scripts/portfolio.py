"""Danh mục mua mẫu theo 4 tổ hợp, mô phỏng giấy (không giao dịch thật).

Quy tắc (giống backtest):
- Tín hiệu tại đóng cửa phiên t: mã vừa bật ít nhất 1 trong 4 tổ hợp, giá trên Supertrend, GTGD TB20 >= ngưỡng.
- Mua giá mở cửa phiên t+1, tối đa MAXPOS mã, mỗi mã 1/MAXPOS NAV, lô 100 cổ phiếu, phí mua 0,15%.
  Nhiều tín hiệu hơn số chỗ trống: ưu tiên mã bật nhiều tổ hợp hơn, rồi thanh khoản cao hơn.
- Bán khi giá đóng cửa dưới Supertrend: bán giá mở cửa phiên sau, sớm nhất phiên thứ 3 sau ngày mua (T+2), phí bán 0,25%.
- Tiền mặt không tính lãi. Không margin.
"""
import math

import numpy as np
import pandas as pd

FB, FS = 0.0015, 0.0025
MAXPOS = 10
CAPITAL = 1_000_000_000      # 1 tỷ đồng
MIN_HOLD = 3                 # bán sớm nhất ở phiên mua + 3


def new_state(start):
    return dict(start=start, capital=CAPITAL, cash=float(CAPITAL), pos={}, pend_buy=[], pend_sell={},
                trades=[], nav=[], last=start)


def build_panel(sessions, X_by_sym, df_by_sym, liq_min):
    """Bảng rộng theo phiên: giá mở/đóng, số tổ hợp vừa bật, Supertrend, thanh khoản."""
    O, C, SIG, ST, LIQ, CB = {}, {}, {}, {}, {}, {}
    from indicators import COMBOS
    for s, X in X_by_sym.items():
        df = df_by_sym[s]
        trig = pd.DataFrame({k: X[v['crit']].all(axis=1) & ~X[v['crit']].all(axis=1).shift(fill_value=False)
                             for k, v in COMBOS.items()})
        ok = X.Supertrend & (X.GTGD20 >= liq_min)
        sig = trig.sum(axis=1).where(ok, 0)
        O[s] = df.o; C[s] = df.c; SIG[s] = sig; ST[s] = X.Supertrend; LIQ[s] = X.GTGD20
        CB[s] = trig.apply(lambda r: ''.join(k for k, v in r.items() if v), axis=1).where(ok, '')
    ix = pd.DatetimeIndex(sessions)
    f = lambda d: pd.DataFrame(d).reindex(ix)  # noqa: E731
    return dict(o=f(O), c=f(C).ffill(), traded=f(O).notna(), sig=f(SIG).fillna(0), st=f(ST).fillna(True).astype(bool),
                liq=f(LIQ).fillna(0), cb=f(CB).fillna(''), idx={d: i for i, d in enumerate(ix)}, sessions=ix)


def _nav(state, P, d):
    v = state['cash']
    for s, p in state['pos'].items():
        px = P['c'].at[d, s] if s in P['c'].columns else np.nan
        v += p['qty'] * 1000 * (px if np.isfinite(px) else p['entry_px'])
    return v


def step(state, P, d):
    i = P['idx'][d]; ds = str(d.date())
    # 1. khớp lệnh bán ở giá mở cửa
    for s in list(state['pend_sell']):
        p = state['pos'][s]
        if s not in P['o'].columns or i < _ei(P, p) + MIN_HOLD or not P['traded'].at[d, s]:
            continue
        px = float(P['o'].at[d, s])
        proceeds = p['qty'] * px * 1000 * (1 - FS)
        state['cash'] += proceeds
        cost = p['qty'] * p['entry_px'] * 1000 * (1 + FB)
        state['trades'].append(dict(s=s, combos=p['combos'], buy_d=p['entry_d'], buy_px=p['entry_px'], sell_d=ds, sell_px=round(px, 2),
                                    qty=p['qty'], ret=round((proceeds / cost - 1) * 100, 2), pnl=round(proceeds - cost),
                                    days=i - _ei(P, p), why=state['pend_sell'][s]))
        del state['pos'][s]; del state['pend_sell'][s]
    # 2. khớp lệnh mua ở giá mở cửa
    for od in state['pend_buy']:
        s = od['s']
        if s in state['pos'] or s not in P['o'].columns or not P['traded'].at[d, s]:
            continue
        px = float(P['o'].at[d, s])
        qty = math.floor(min(od['budget'], state['cash']) / (px * 1000 * (1 + FB)) / 100) * 100
        if qty <= 0:
            continue
        state['cash'] -= qty * px * 1000 * (1 + FB)
        state['pos'][s] = dict(qty=qty, entry_px=round(px, 2), entry_d=ds, combos=od['combos'], sig_d=od['sig_d'])
    state['pend_buy'] = []
    # 3. định giá cuối phiên
    nav = _nav(state, P, d)
    state['nav'].append([ds, round(nav), round(state['cash'])])
    # 4. tín hiệu bán: giá đóng cửa dưới Supertrend
    for s in state['pos']:
        if s not in state['pend_sell'] and (s not in P['st'].columns or not P['st'].at[d, s]):
            state['pend_sell'][s] = f'Supertrend lật xuống {ds}'
    # 5. tín hiệu mua mới
    slots = MAXPOS - len(state['pos'])
    if slots > 0:
        row = P['sig'].loc[d]
        cand = [s for s in row.index[row.values > 0] if s not in state['pos']]
        cand.sort(key=lambda s: (-row[s], -P['liq'].at[d, s]))
        reserved = 0.0
        for s in cand[:slots]:
            budget = min(nav / MAXPOS, state['cash'] - reserved)
            if budget < 0.3 * nav / MAXPOS:
                break
            reserved += budget
            state['pend_buy'].append(dict(s=s, budget=budget, combos=P['cb'].at[d, s], sig_d=ds))
    state['last'] = ds
    return state


def run(state, P, after=None):
    for d in P['sessions']:
        if after is not None and d <= pd.Timestamp(after):
            continue
        step(state, P, d)
    return state


def summary(state, P, vni):
    nav = pd.Series({pd.Timestamp(x[0]): x[1] for x in state['nav']})
    out = dict(start=state['start'], last=state['last'], capital=state['capital'], cash=round(state['cash']))
    if len(nav):
        v = vni.reindex(nav.index).ffill()
        base_v = vni[vni.index <= pd.Timestamp(state['start'])]
        v0 = float(base_v.iloc[-1]) if len(base_v) else float(v.iloc[0])
        out['nav'] = round(float(nav.iloc[-1]))
        out['ret'] = round((nav.iloc[-1] / state['capital'] - 1) * 100, 2)
        out['vni_ret'] = round((v.iloc[-1] / v0 - 1) * 100, 2)
        out['maxdd'] = round(float((nav / nav.cummax() - 1).min()) * 100, 2)
        out['vni_maxdd'] = round(float((v / v.cummax() - 1).min()) * 100, 2)
        yrs = (nav.index[-1] - pd.Timestamp(state['start'])).days / 365.25
        if yrs > 0.5:
            out['cagr'] = round(((nav.iloc[-1] / state['capital']) ** (1 / yrs) - 1) * 100, 2)
            out['vni_cagr'] = round(((v.iloc[-1] / v0) ** (1 / yrs) - 1) * 100, 2)
        out['series'] = [[str(d.date()), round(n / state['capital'] * 100, 2), round(float(x) / v0 * 100, 2)]
                         for d, n, x in zip(nav.index, nav.values, v.values)]
        out['exposure'] = round(float(np.mean([1 - x[2] / x[1] for x in state['nav']])) * 100, 1)
    last = P['sessions'][-1]
    pos = []
    for s, p in state['pos'].items():
        px = float(P['c'].at[last, s]) if s in P['c'].columns else p['entry_px']
        pos.append(dict(s=s, combos=p['combos'], qty=p['qty'], entry_d=p['entry_d'], entry_px=p['entry_px'], px=round(px, 2),
                        value=round(p['qty'] * px * 1000), ret=round((px * (1 - FS) / (p['entry_px'] * (1 + FB)) - 1) * 100, 2),
                        days=P['idx'][last] - _ei(P, p), selling=s in state['pend_sell']))
    out['positions'] = sorted(pos, key=lambda x: x['entry_d'])
    out['pending_buy'] = [dict(s=o['s'], combos=o['combos'], sig_d=o['sig_d'], budget=round(o['budget'])) for o in state['pend_buy']]
    out['pending_sell'] = [dict(s=s, why=w, sellable=P['idx'][last] + 1 >= _ei(P, state['pos'][s]) + MIN_HOLD)
                           for s, w in state['pend_sell'].items()]
    tr = state['trades']
    out['n_trades'] = len(tr)
    if tr:
        r = np.array([t['ret'] for t in tr])
        out['win_rate'] = round(float((r > 0).mean()) * 100, 1)
        out['avg_ret'] = round(float(r.mean()), 2)
        out['avg_win'] = round(float(r[r > 0].mean()), 2) if (r > 0).any() else None
        out['avg_loss'] = round(float(r[r <= 0].mean()), 2) if (r <= 0).any() else None
        out['avg_days'] = round(float(np.mean([t['days'] for t in tr])), 1)
    out['trades'] = tr[::-1]
    return out


def _ei(P, p):
    """Chỉ số phiên mua trong bảng hiện tại (cửa sổ dữ liệu dịch mỗi ngày nên không lưu chỉ số)."""
    return P['idx'].get(pd.Timestamp(p['entry_d']), -10**6)
