# Tín hiệu TA hằng ngày

Trang tĩnh chấm 11 tiêu chí phân tích kỹ thuật cho khoảng 270 mã cổ phiếu Việt Nam và quét 4 tổ hợp mua tốt nhất từ backtest 2014–11/2026. GitHub Actions tự chạy sau phiên ATC (15:40 và 17:30 giờ VN, thứ Hai đến thứ Sáu), không cần máy tính bật.

- `scripts/build.py`: tải giá ngày (API công khai DNSE, dự phòng VPS), chấm điểm, ghi `docs/data/latest.json` và nhật ký `docs/data/history/`.
- `scripts/indicators.py`: cách tính 11 tiêu chí, 4 tổ hợp, giai đoạn VN-Index.
- `scripts/universe.json`: danh sách mã (sửa file này để thêm hoặc bớt mã).
- `docs/index.html`: trang web.

Chạy tay: tab **Actions** → *Cập nhật tín hiệu TA* → **Run workflow**.

## 11 tiêu chí (mỗi tiêu chí 1 điểm, trạng thái tại giá đóng cửa)
1. XH ngắn hạn: Giá > MA10 > MA20
2. XH trung hạn: thêm MA20 > MA50
3. XH dài hạn: thêm MA50 > MA200
4. Giá trên Supertrend (10, 3)
5. RSI(14) trên MA20 của RSI
6. MACD(12,26) trên Signal(9)
7. Khối lượng trên TB20
8. DI+ > DI- và ADX(14) > 25
9. Stochastic(14,3,3) %K trên %D
10. RS (giá / VN-Index) trên EMA20 của RS
11. CCI(20) > 100

## 4 tổ hợp mua (bật khi đủ 3 điều kiện, GTGD TB20 ≥ 1 tỷ; thoát khi Supertrend lật xuống)
- A: XH dài + Supertrend + DI/ADX
- B: XH dài + MACD + DI/ADX
- C: XH dài + DI/ADX + CCI>100
- D: XH dài + Vol>TB20 + DI/ADX

## Danh mục mua mẫu (mô phỏng giấy)
Vốn 1 tỷ, tối đa 10 mã, mỗi mã 1/10 giá trị danh mục. Mã vừa bật tổ hợp thì mua giá mở cửa phiên sau; giá đóng cửa dưới Supertrend thì bán giá mở cửa phiên sau (sớm nhất phiên thứ 3 sau ngày mua). Phí mua 0,15%, bán 0,25%.
- `docs/data/portfolio_state.json`: trạng thái danh mục theo dõi thật (xóa file này để bắt đầu lại từ phiên mới nhất).
- `docs/data/portfolio.json`: số liệu cho trang, gồm cả mô phỏng quá khứ khoảng 2,5 năm với danh sách mã hiện tại.

Công cụ tham khảo, không phải khuyến nghị đầu tư.
