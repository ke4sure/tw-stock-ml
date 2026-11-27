# 台股選股模型 v3(學習用)

每個交易日自動下載資料、重新訓練模型,並更新網站。
網站內容:今日排名、你關注的股票、預測 vs 實際的每日紀錄、模型健康度、歷史回測、個股對照。

⚠ 僅供學習研究,不是投資建議。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `stock_model.py` | 主程式:下載資料 → 建特徵 → 滾動訓練 → 輸出網站到 `docs/` |
| `dashboard_template.html` | 網站版型 |
| `requirements.txt` | 需要的 Python 套件 |
| `.github/workflows/daily.yml` | GitHub 每天自動執行的設定 |
| `run_local.bat` | 想在自己電腦跑時,雙擊這個檔案 |
| `data/` | 自動產生:FinMind 資料快取 |
| `docs/` | 自動產生:網站(`index.html`)、預測紀錄 |

## 設定 GitHub 自動網站(只要做一次)

1. **註冊 GitHub**:到 https://github.com 按 Sign up,用 Email 註冊並完成驗證。
2. **建立 repository**:右上角「+」→ New repository → 名稱填 `tw-stock-ml` → 選 **Public** → 按 Create repository。
3. **上傳檔案**:在新 repository 頁面點「uploading an existing file」,把 `stock_model.py`、`dashboard_template.html`、`requirements.txt`、`README.md`、`.gitignore`、`run_local.bat` 拖進去 → 按 Commit changes。
4. **建立自動排程檔**:Add file → Create new file → 檔名輸入 `.github/workflows/daily.yml`(打 `/` 會自動變資料夾)→ 把本機 `daily.yml` 的內容全部貼上 → Commit changes。
5. **放入 FinMind token(選擇性但建議)**:Settings → Secrets and variables → Actions → New repository secret → Name 填 `FINMIND_TOKEN`,Secret 貼上 token → Add secret。
6. **第一次手動執行**:Actions 分頁 → 左邊點「每日更新預測」→ 右邊 Run workflow → Run workflow。第一次約 20~60 分鐘(要下載全部歷史資料)。
7. **開啟網站**:跑完(綠色勾勾)後,Settings → Pages → Source 選 Deploy from a branch → Branch 選 `main`、資料夾選 `/docs` → Save。約 1~2 分鐘後,頁面上方會出現網址:`https://你的帳號.github.io/tw-stock-ml/`。

之後每個交易日台灣時間約 18:30 自動更新(GitHub 排程可能延遲幾十分鐘)。

## 修改股票清單

編輯 `stock_model.py` 最上面的 `INDUSTRIES`(股票和產業)與 `WATCHLIST`(關注股票)。在 GitHub 網頁上點檔案 → 鉛筆圖示就能直接改。

## 在自己電腦執行

1. 安裝套件:`pip install -r requirements.txt`
2. (選擇性)把 FinMind token 存成同資料夾的 `finmind_token.txt`
3. 雙擊 `run_local.bat`,跑完會自動打開網站

## 模型說明

- 股票池:約 130 檔台股大型股
- 特徵:價量(動能、波動、流動性)、月營收、財報(EPS、毛利率、營益率)、估值(本益比、股價淨值比、殖利率)、三大法人買賣超、大盤 / 費半 / 匯率
- 所有基本面資料都用「公布日」對齊,避免偷看未來
- 模型:LightGBM + Ridge 綜合分數;另提供扣掉產業與 beta 影響的「產業中立」分數
- 短期(20 交易日)與長期(約 1 年)兩個預測
- 評估:walk-forward 滾動驗證、Rank IC、五分組、扣成本回測
- 已知限制:股票池用「現在」的大型股,有倖存者偏差,回測會略為樂觀
