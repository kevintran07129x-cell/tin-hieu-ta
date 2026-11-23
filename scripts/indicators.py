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
    'Tăng trưởng': 'VN-Index đang trong nhịp tăng (đã hồi ≥ 6% từ đáy gần nhất) và trên MA50. Giai đoạn thuận lợi cho các tổ hợp mua theo xu hướng.',
    'Tạo đáy': 'VN-Index vừa hồi ≥ 6% từ đáy nhưng còn dưới MA50: nhịp tăng mới chớm, dễ bị rung lắc.',
    'Đỉnh/điều chỉnh': 'VN-Index đã giảm ≥ 6% từ đỉnh gần nhất nhưng còn trên MA50: thị trường đang điều chỉnh, nên thận trọng với lệnh mới.',
    'Suy giảm': 'VN-Index trong nhịp giảm (≥ 6% từ đỉnh) và dưới MA50: tín hiệu mua dễ thất bại, nên thận trọng.',
}

SWING = 0.06   # ngưỡng lật nhịp: giảm 6% từ đỉnh -> nhịp giảm, tăng 6% từ đáy -> nhịp tăng


def swing_state(c, x=SWING):
    """+1 nhịp tăng, -1 nhịp giảm. Chỉ dùng giá đóng cửa đến ngày t (không nhìn trước)."""
    st, ext, out = 1, c.iloc[0], []
    for v in c.values:
        if st == 1:
            ext = max(ext, v)
            if v <= ext * (1 - x):
                st, ext = -1, v
        else:
            ext = min(ext, v)
            if v >= ext * (1 + x):
                st, ext = 1, v
        out.append(st)
    return pd.Series(out, c.index)


def regimes(ix):
    """Giai đoạn VN-Index theo nhịp sóng 6% và MA50 (đổi 23/11/2026, thay cách cũ MA50/MA200 bị trễ ~3–5 tuần).
    Tăng trưởng: nhịp tăng và giá > MA50.   Tạo đáy: nhịp tăng, giá <= MA50, MA50 còn dốc xuống.
    Đỉnh/điều chỉnh: nhịp giảm nhưng giá >= MA50, hoặc nhịp tăng mà giá thủng MA50 đang đi lên.   Suy giảm: nhịp giảm và giá < MA50.
    Kiểm tra VN-Index 2014–2026: trễ trung bình ~11 phiên sau đỉnh/đáy thật (cách cũ ~22), ~5 lần đổi nhịp/năm."""
    c = ix.c
    m50 = c.rolling(50).mean(); m200 = c.rolling(200).mean()
    dd = c / c.rolling(250, min_periods=60).max() - 1
    sw = swing_state(c)
    g = pd.Series(None, index=c.index, dtype=object)
    g[(sw == 1) & (c > m50)] = 'Tăng trưởng'
    falling = m50.diff(10) < 0                 # MA50 còn dốc xuống: vừa đi lên từ nhịp giảm
    g[(sw == 1) & (c <= m50) & falling] = 'Tạo đáy'
    g[(sw == 1) & (c <= m50) & ~falling] = 'Đỉnh/điều chỉnh'   # nhịp tăng nhưng đã thủng MA50
    g[(sw == -1) & (c >= m50)] = 'Đỉnh/điều chỉnh'
    g[(sw == -1) & (c < m50)] = 'Suy giảm'
    g[m50.isna()] = None
    return g, dd, m50, m200
