"""11 tiêu chí TA theo từng phiên (trạng thái tại đóng cửa t). Cùng cách tính với bảng chấm điểm và backtest.

Giá đầu vào: DataFrame cột o, h, l, c, v (index là ngày), giá đơn vị nghìn đồng.
"""
import numpy as np
import pandas as pd

CRIT = ['XH_ngan', 'XH_trung', 'XH_dai', 'Supertrend', 'RSI>MA20RSI', 'MACD>Signal', 'Vol>TB20',
        'DI+>DI-&ADX>25', 'Stoch%K>%D', 'RS>EMA20', 'CCI>100']

CRIT_LABEL = {
    'XH_ngan': 'XH ngắn hạn: Giá > MA10 > MA20',
    'XH_trung': 'XH trung hạn: Giá > MA10 > MA20 > MA50',
    'XH_dai': 'XH dài hạn: Giá > MA10 > MA20 > MA50 > MA200',
    'Supertrend': 'Giá trên Supertrend (10, 3)',
    'RSI>MA20RSI': 'RSI(14) trên MA20 của RSI',
    'MACD>Signal': 'MACD(12,26) trên đường Signal(9)',
    'Vol>TB20': 'Khối lượng trên trung bình 20 phiên',
    'DI+>DI-&ADX>25': 'DI+ > DI- và ADX(14) > 25',
    'Stoch%K>%D': 'Stochastic(14,3,3) %K trên %D',
    'RS>EMA20': 'RS (giá CP / VN-Index) trên EMA20 của RS',
    'CCI>100': 'CCI(20) > 100',
}

# 4 tổ hợp mua tốt nhất trong backtest 2014–11/2026, thoát khi Supertrend lật xuống
COMBOS = {
    'A': {'name': 'XH dài + Supertrend + DI/ADX', 'crit': ['XH_dai', 'Supertrend', 'DI+>DI-&ADX>25']},
    'B': {'name': 'XH dài + MACD + DI/ADX', 'crit': ['XH_dai', 'MACD>Signal', 'DI+>DI-&ADX>25']},
    'C': {'name': 'XH dài + DI/ADX + CCI>100', 'crit': ['XH_dai', 'DI+>DI-&ADX>25', 'CCI>100']},
    'D': {'name': 'XH dài + Vol>TB20 + DI/ADX', 'crit': ['XH_dai', 'Vol>TB20', 'DI+>DI-&ADX>25']},
}


def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def true_range(h, l, c):
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def supertrend(h, l, c, n=10, m=3):
    """Trả về (hướng +1/-1, đường Supertrend)."""
    atr = rma(true_range(h, l, c), n)
    hl2 = (h + l) / 2
    ub = (hl2 + m * atr).values
    lb = (hl2 - m * atr).values
    cc = c.values
    fu = ub.copy(); fl = lb.copy(); d = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or cc[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or cc[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1 and cc[i] < fl[i]:
            d[i] = -1
        elif d[i - 1] == -1 and cc[i] > fu[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
    line = np.where(d == 1, fl, fu)
    return pd.Series(d, c.index), pd.Series(line, c.index)


def criteria(df, idx_close):
    c, h, l, v = df.c, df.h, df.l, df.v
    ma = {n: c.rolling(n).mean() for n in (10, 20, 50, 200)}
    X = pd.DataFrame(index=df.index)
    X['XH_ngan'] = (c > ma[10]) & (ma[10] > ma[20])
    X['XH_trung'] = X.XH_ngan & (ma[20] > ma[50])
    X['XH_dai'] = X.XH_trung & (ma[50] > ma[200])
    st_dir, st_line = supertrend(h, l, c)
    X['Supertrend'] = st_dir == 1
    d = c.diff()
    rsi = 100 - 100 / (1 + rma(d.clip(lower=0), 14) / rma((-d).clip(lower=0), 14))
    X['RSI>MA20RSI'] = rsi > rsi.rolling(20).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    X['MACD>Signal'] = macd > sig
    X['Vol>TB20'] = v > v.rolling(20).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.), c.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.), c.index)
    atr = rma(true_range(h, l, c), 14)
    pdi = 100 * rma(pdm, 14) / atr
    ndi = 100 * rma(ndm, 14) / atr
    adx = rma(100 * (pdi - ndi).abs() / (pdi + ndi), 14)
    X['DI+>DI-&ADX>25'] = (pdi > ndi) & (adx > 25)
    ll = l.rolling(14).min(); hh = h.rolling(14).max()
    k = (100 * (c - ll) / (hh - ll)).rolling(3).mean()
    X['Stoch%K>%D'] = k > k.rolling(3).mean()
    rs = c / idx_close.reindex(c.index).ffill()
    X['RS>EMA20'] = rs > rs.ewm(span=20, adjust=False).mean()
    tp = (h + l + c) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    X['CCI>100'] = cci > 100
    X = X.fillna(False).astype(bool)
    X.loc[ma[200].isna(), :] = False          # cần đủ 200 phiên mới chấm
    X['Diem'] = X[CRIT].sum(axis=1)
    X['GTGD20'] = (c * 1000 * v).rolling(20).mean() / 1e9   # tỷ đồng
    # giá trị chỉ báo để hiển thị
    X['rsi'] = rsi; X['adx'] = adx; X['cci'] = cci; X['st_line'] = st_line
    X['ma20'] = ma[20]; X['ma50'] = ma[50]; X['ma200'] = ma[200]
    X['macd_up'] = (macd > sig) & (macd.shift() <= sig.shift())
    X['macd_dn'] = (macd < sig) & (macd.shift() >= sig.shift())
    return X


REGIME_NOTE = {
    'Tăng trưởng': 'Giai đoạn backtest cho lợi thế rõ nhất với 4 tổ hợp (vượt rổ ~+1,8%/lệnh ngoài mẫu).',
    'Tạo đáy': 'Backtest: không quy tắc nào giữ được lợi thế ngoài mẫu, mua đều rổ tốt hơn.',
    'Đỉnh/điều chỉnh': 'Backtest: các tổ hợp mất lợi thế ngoài mẫu ở giai đoạn này.',
    'Suy giảm': 'Backtest: các tổ hợp mất lợi thế ngoài mẫu ở giai đoạn này, nên thận trọng.',
}


def regimes(ix):
    """Giai đoạn VN-Index, chỉ dùng thông tin đến ngày t.
    Tăng trưởng: Giá > MA200, MA50 > MA200, cách đỉnh 1 năm không quá 10%.
    Tạo đáy: không phải tăng trưởng, đã giảm >=15% từ đỉnh 1 năm, nhưng giá đã lên trên MA50.
    Suy giảm: không thuộc hai nhóm trên và giá < MA200.
    Đỉnh/điều chỉnh: còn lại."""
    c = ix.c
    m50 = c.rolling(50).mean(); m200 = c.rolling(200).mean()
    dd = c / c.rolling(250, min_periods=60).max() - 1
    g = pd.Series('Đỉnh/điều chỉnh', index=c.index, dtype=object)
    up = (c > m200) & (m50 > m200) & (dd > -0.10)
    bot = ~up & (dd <= -0.15) & (c > m50)
    down = ~up & ~bot & (c < m200)
    g[down] = 'Suy giảm'; g[bot] = 'Tạo đáy'; g[up] = 'Tăng trưởng'
    g[m200.isna()] = None
    return g, dd, m50, m200
