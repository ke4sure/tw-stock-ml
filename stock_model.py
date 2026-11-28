"""
台股機器學習選股模型 v4
- 價量(yfinance)+ 基本面 / 估值 / 籌碼(FinMind)
- 短期(20 交易日)與長期(250 交易日)兩個模型
- LightGBM 迴歸 + Ridge + LambdaRank 排序 → 扣掉產業 / beta / 規模影響 → 5 日平滑 = 主要分數
- 回測模擬實際持股(換股緩衝、依實際換手算成本),並用 bootstrap / PSR 估計結果的可信度
- 輸出網站到 docs/(GitHub Pages 會直接顯示)

執行:python stock_model.py
FinMind token:環境變數 FINMIND_TOKEN,或同資料夾的 finmind_token.txt(沒有也能跑,只是比較慢)
⚠ 僅供學習研究,不是投資建議。
"""
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FM_DIR = DATA_DIR / "finmind"
DOCS = ROOT / "docs"

# ============================================================
# 1. 設定
# ============================================================
INDUSTRIES = {
    "半導體": {
        "2330": "台積電", "2303": "聯電", "2454": "聯發科", "3711": "日月光投控", "2379": "瑞昱",
        "3034": "聯詠", "3443": "創意", "3661": "世芯-KY", "5269": "祥碩", "6415": "矽力-KY",
        "2408": "南亞科", "2344": "華邦電", "2337": "旺宏", "2449": "京元電子", "6239": "力成",
        "3035": "智原", "6531": "愛普", "6271": "同欣電", "5347": "世界先進", "6488": "環球晶",
        "5483": "中美晶", "3105": "穩懋", "8299": "群聯", "5274": "信驊", "3529": "力旺",
        "6147": "頎邦", "3374": "精材", "4966": "譜瑞-KY", "6223": "旺矽", "3680": "家登",
        "6510": "精測",
    },
    "電腦/伺服器/網通": {
        "2317": "鴻海", "2382": "廣達", "3231": "緯創", "6669": "緯穎", "2357": "華碩",
        "2353": "宏碁", "2324": "仁寶", "2356": "英業達", "2376": "技嘉", "2377": "微星",
        "4938": "和碩", "2395": "研華", "2345": "智邦", "3017": "奇鋐", "3324": "雙鴻",
        "3653": "健策", "2301": "光寶科", "2385": "群光", "3533": "嘉澤", "2059": "川湖",
        "6121": "新普", "3211": "順達",
    },
    "電子零組件/光學": {
        "2308": "台達電", "2327": "國巨", "2492": "華新科", "3037": "欣興", "4958": "臻鼎-KY",
        "2383": "台光電", "2368": "金像電", "3044": "健鼎", "2313": "華通", "3008": "大立光",
        "3406": "玉晶光", "2474": "可成", "2354": "鴻準", "6176": "瑞儀", "8069": "元太",
        "3481": "群創", "2409": "友達", "2360": "致茂", "6409": "旭隼", "3036": "文曄",
        "2347": "聯強", "3702": "大聯大",
    },
    "金融": {
        "2881": "富邦金", "2882": "國泰金", "2891": "中信金", "2886": "兆豐金", "2884": "玉山金",
        "2885": "元大金", "2892": "第一金", "2880": "華南金", "2883": "凱基金", "2887": "台新金",
        "2890": "永豐金", "5880": "合庫金", "2801": "彰銀", "5876": "上海商銀", "5871": "中租-KY",
    },
    "電信": {"2412": "中華電", "3045": "台灣大", "4904": "遠傳"},
    "傳產/原物料/重電": {
        "1301": "台塑", "1303": "南亞", "1326": "台化", "6505": "台塑化", "2002": "中鋼",
        "1101": "台泥", "1102": "亞泥", "1402": "遠東新", "2105": "正新", "1605": "華新",
        "1504": "東元", "1503": "士電", "1513": "中興電", "1519": "華城", "1560": "中砂",
        "2049": "上銀", "1590": "亞德客-KY", "2404": "漢唐",
    },
    "民生/紡織/其他": {
        "1216": "統一", "2912": "統一超", "2207": "和泰車", "1476": "儒鴻", "1477": "聚陽",
        "9910": "豐泰", "9904": "寶成", "9921": "巨大", "8464": "億豐", "1795": "美時",
        "3293": "鈊象",
    },
    "航運/交通/營建": {
        "2603": "長榮", "2609": "陽明", "2615": "萬海", "2618": "長榮航", "2610": "華航",
        "2633": "台灣高鐵", "9945": "潤泰新", "2542": "興富發",
    },
    "ETF": {"0050": "元大台灣50"},
}
STOCKS = {c: n for g in INDUSTRIES.values() for c, n in g.items()}
INDUSTRY_OF = {c: ind for ind, g in INDUSTRIES.items() for c in g}
WATCHLIST = ["0050", "2330", "2324", "4958", "6488", "3008", "1560"]

MARKET_TICKER, SOX_TICKER, FX_TICKER = "^TWII", "^SOX", "TWD=X"
START = "2015-01-01"
FM_START = "2014-01-01"
TRAIN_YEARS = 4
TOP_N = 15
COST = 0.004
MIN_STOCKS = 30
SHORT = {"name": "短期", "horizon": 20, "retrain": 20}
LONG = {"name": "長期", "horizon": 250, "retrain": 60}


# ============================================================
# 2. 資料下載
# ============================================================
def download_prices():
    import yfinance as yf

    def _dl(tickers):
        raw = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                          group_by="column")
        out = []
        for field in ["Close", "Volume"]:
            x = raw[field]
            if isinstance(x, pd.Series):
                x = x.to_frame(tickers[0])
            x = x.copy()
            x.columns = [str(c).split(".")[0] for c in x.columns]
            out.append(x)
        return out

    def _one(t):
        raw = yf.download(t, start=START, auto_adjust=True, progress=False)
        s = raw["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.dropna()

    codes = list(STOCKS)
    close, vol = _dl([c + ".TW" for c in codes])
    ok = [c for c in codes if c in close.columns and close[c].notna().sum() > 250]
    closes, vols = [close[ok]], [vol[ok]]
    retry = [c for c in codes if c not in ok]
    if retry:
        c2, v2 = _dl([c + ".TWO" for c in retry])
        ok2 = [c for c in retry if c in c2.columns and c2[c].notna().sum() > 250]
        closes.append(c2[ok2])
        vols.append(v2[ok2])
        ok += ok2
    failed = [c for c in codes if c not in ok]
    if failed:
        print("⚠ 抓不到股價(已略過):", failed)
    close = pd.concat(closes, axis=1).sort_index().dropna(how="all")
    volume = pd.concat(vols, axis=1).reindex(close.index)
    market = _one(MARKET_TICKER).reindex(close.index).ffill()
    return close, volume, market, _one(SOX_TICKER), _one(FX_TICKER)


FM_URL = "https://api.finmindtrade.com/api/v4/data"


def load_token():
    t = os.environ.get("FINMIND_TOKEN", "").strip()
    f = ROOT / "finmind_token.txt"
    if not t and f.exists():
        t = f.read_text(encoding="utf-8-sig").strip()
    return t


def finmind_get(dataset, data_id, start, token):
    params = {"dataset": dataset, "data_id": data_id, "start_date": start}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for _ in range(10):
        try:
            r = requests.get(FM_URL, params=params, headers=headers, timeout=60)
            js = r.json()
        except Exception as e:  # 網路錯誤
            print("  連線錯誤,30 秒後重試:", e)
            time.sleep(30)
            continue
        status = js.get("status", r.status_code)
        if status == 200:
            return pd.DataFrame(js.get("data", []))
        msg = str(js.get("msg", ""))
        if status in (402, 429) or "limit" in msg.lower():
            print(f"  FinMind 達到每小時上限,暫停 10 分鐘後繼續…({msg})")
            time.sleep(600)
            continue
        print(f"  FinMind 錯誤 {dataset} {data_id}: {status} {msg}")
        return pd.DataFrame()
    return pd.DataFrame()


def _tf_inst(df):
    df = df.copy()
    df["net"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")
    p = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum").fillna(0)
    out = pd.DataFrame(index=p.index)
    get = lambda k: p[k] if k in p.columns else 0  # noqa: E731
    out["foreign"] = get("Foreign_Investor") + get("Foreign_Dealer_Self")
    out["trust"] = get("Investment_Trust")
    out["dealer"] = get("Dealer_self") + get("Dealer_Hedging")
    return out.reset_index()


# dataset: (多久更新一次[小時], 重抓最近幾天, 去重複用的欄位, 轉換函式)
FM_DATASETS = {
    "TaiwanStockPER": (12, 10, ["date"], None),
    "TaiwanStockInstitutionalInvestorsBuySell": (12, 10, ["date"], _tf_inst),
    "TaiwanStockMonthRevenue": (24 * 5, 70, ["revenue_year", "revenue_month"], None),
    "TaiwanStockFinancialStatements": (24 * 7, 200, ["date", "type"], None),
    "TaiwanStockCashFlowsStatement": (24 * 7, 200, ["date", "type"], None),
    "TaiwanStockBalanceSheet": (24 * 7, 200, ["date", "type"], None),
}


def update_finmind(codes, token):
    meta_f = FM_DIR / "_meta.json"
    FM_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    now = datetime.now()
    n_req = 0
    for ds, (every_h, overlap, keys, tf) in FM_DATASETS.items():
        folder = FM_DIR / ds
        folder.mkdir(exist_ok=True)
        meta.setdefault(ds, {})
        todo = [c for c in codes if c not in meta[ds]
                or (now - datetime.fromisoformat(meta[ds][c])).total_seconds() > every_h * 3600]
        if todo:
            print(f"更新 {ds}:{len(todo)} 檔")
        for c in todo:
            f = folder / f"{c}.csv"
            old = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame()
            if old.empty or "date" not in old:
                start = FM_START
            else:
                start = (pd.to_datetime(old["date"]).max() - timedelta(days=overlap)).strftime("%Y-%m-%d")
            new = finmind_get(ds, c, start, token)
            n_req += 1
            if not new.empty:
                if tf is not None:
                    new = tf(new)
                new = new.astype(str)
                merged = pd.concat([old, new], ignore_index=True)
                merged = merged.drop_duplicates(subset=[k for k in keys if k in merged], keep="last")
                merged.sort_values("date").to_csv(f, index=False)
            meta[ds][c] = now.isoformat(timespec="seconds")
            if n_req % 20 == 0:
                meta_f.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    meta_f.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"FinMind 本次呼叫 {n_req} 次")


def read_fm(ds, codes):
    frames = []
    for c in codes:
        f = FM_DIR / ds / f"{c}.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"stock_id": str})
            if not d.empty:
                d["ticker"] = c
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# 3. 基本面 / 估值 / 籌碼特徵(全部用「公布後才能用」的日期對齊)
# ============================================================
try:
    pd.date_range("2020-01-01", periods=2, freq="QE")
    QFREQ = "QE"
except ValueError:
    QFREQ = "Q"


def fundamental_events(codes):
    """回傳 [(DataFrame[ticker, avail, 特徵...]), ...],每個都用 avail 日期對齊"""
    events = []

    # 月營收:次月 10 日前公布 → 保守用次月 11 日才可用
    rev = read_fm("TaiwanStockMonthRevenue", codes)
    if not rev.empty and {"revenue", "revenue_year", "revenue_month"} <= set(rev.columns):
        rev["period"] = pd.to_datetime(dict(year=rev["revenue_year"].astype(int),
                                            month=rev["revenue_month"].astype(int), day=1))
        rev["revenue"] = pd.to_numeric(rev["revenue"], errors="coerce")
        out = []
        for t, g in rev.groupby("ticker"):
            s = g.drop_duplicates("period", keep="last").set_index("period")["revenue"].sort_index()
            s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
            r3 = s.rolling(3).sum()
            f = pd.DataFrame({
                "rev_yoy": s / s.shift(12) - 1,
                "rev_yoy_3m": r3 / r3.shift(12) - 1,
                "rev_mom": s / s.shift(1) - 1,
                "rev_vs_high": s / s.rolling(12, min_periods=6).max(),
            })
            f["rev_yoy_accel"] = f["rev_yoy_3m"] - f["rev_yoy_3m"].shift(3)
            f["avail"] = f.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
            f["ticker"] = t
            out.append(f.dropna(subset=["rev_yoy"]))
        if out:
            events.append(pd.concat(out))

    # 財報:Q1~Q3 季底後 45 天、Q4 季底後 90 天公布
    fs = read_fm("TaiwanStockFinancialStatements", codes)
    if not fs.empty and {"type", "value"} <= set(fs.columns):
        fs["date"] = pd.to_datetime(fs["date"])
        fs["value"] = pd.to_numeric(fs["value"], errors="coerce")
        out = []
        for t, g in fs.groupby("ticker"):
            p = g.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
            p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq=QFREQ))
            col = lambda k: p[k] if k in p.columns else pd.Series(np.nan, index=p.index)  # noqa: E731
            eps, revq = col("EPS"), col("Revenue")
            gm = col("GrossProfit") / revq.replace(0, np.nan)
            om = col("OperatingIncome") / revq.replace(0, np.nan)
            f = pd.DataFrame({
                "eps_yoy": ((eps - eps.shift(4)) / (eps.shift(4).abs() + 0.1)).clip(-5, 5),
                "gross_margin": gm,
                "gm_chg_yoy": gm - gm.shift(4),
                "op_margin": om,
                "om_chg_yoy": om - om.shift(4),
            }, index=p.index)
            lag = np.where(f.index.month == 12, 90, 45)
            f["avail"] = f.index + pd.to_timedelta(lag, unit="D")
            f["ticker"] = t
            out.append(f.dropna(how="all", subset=["eps_yoy", "gross_margin", "op_margin"]))
        if out:
            events.append(pd.concat(out))
    # 現金流量表(年初累計 → 單季 → 近四季)+ 資產負債表(季底餘額),公布時間同財報
    cf = read_fm("TaiwanStockCashFlowsStatement", codes)
    bs = read_fm("TaiwanStockBalanceSheet", codes)
    if not cf.empty and not bs.empty and {"type", "value"} <= set(cf.columns) & set(bs.columns):
        for d in (cf, bs):
            d["date"] = pd.to_datetime(d["date"])
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
        cf_types = ["CashFlowsFromOperatingActivities", "PropertyAndPlantAndEquipment", "NetIncomeBeforeTax"]
        bs_types = ["TotalAssets", "Liabilities", "CashAndCashEquivalents", "CurrentAssets",
                    "CurrentLiabilities", "OrdinaryShare"]
        bs_by = dict(tuple(bs[bs["type"].isin(bs_types)].groupby("ticker")))
        out = []
        for t, g in cf[cf["type"].isin(cf_types)].groupby("ticker"):
            if t not in bs_by:
                continue
            p = g.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
            qr = pd.date_range(p.index.min(), p.index.max(), freq=QFREQ)
            p = p.reindex(qr)
            q = p - p.shift(1)                      # 單季 = 本季累計 - 上季累計
            q[p.index.month == 3] = p[p.index.month == 3]  # Q1 本身就是單季
            col = lambda df, k: df[k] if k in df.columns else pd.Series(np.nan, index=df.index)  # noqa: E731
            ocf_q = col(q, "CashFlowsFromOperatingActivities")
            fcf_q = ocf_q + col(q, "PropertyAndPlantAndEquipment").fillna(0)  # 資本支出為負數
            ni_q = col(q, "NetIncomeBeforeTax")
            b = bs_by[t].pivot_table(index="date", columns="type", values="value", aggfunc="last").reindex(qr)
            assets = col(b, "TotalAssets").replace(0, np.nan)
            ocf_ttm, fcf_ttm = ocf_q.rolling(4).sum(), fcf_q.rolling(4).sum()
            f = pd.DataFrame({
                "ocf_ttm": ocf_ttm, "fcf_ttm": fcf_ttm,
                "shares": col(b, "OrdinaryShare") / 10,          # 面額 10 元
                "ocf_to_assets": ocf_ttm / assets,
                "accruals": (ni_q.rolling(4).sum() - ocf_ttm) / assets,   # 越低 = 獲利越有現金支撐
                "cash_to_assets": col(b, "CashAndCashEquivalents") / assets,
                "debt_ratio": col(b, "Liabilities") / assets,
                "current_ratio": col(b, "CurrentAssets") / col(b, "CurrentLiabilities").replace(0, np.nan),
                "fcf_pos_8q": (fcf_q > 0).astype(float).where(fcf_q.notna()).rolling(8, min_periods=4).mean(),
            }, index=qr)
            lag = np.where(f.index.month == 12, 90, 45)
            f["avail"] = f.index + pd.to_timedelta(lag, unit="D")
            f["ticker"] = t
            out.append(f.dropna(subset=["ocf_to_assets", "debt_ratio"], how="all"))
        if out:
            events.append(pd.concat(out))
    return events


def daily_fm_features(close, volume, codes):
    """估值(本益比等)與法人買賣超:每日資料,當天收盤後就能用"""
    idx = close.index
    feats = {}
    per = read_fm("TaiwanStockPER", codes)
    if not per.empty:
        per["date"] = pd.to_datetime(per["date"])
        for k in ["PER", "PBR", "dividend_yield"]:
            if k in per:
                per[k] = pd.to_numeric(per[k], errors="coerce")
        for t, g in per.groupby("ticker"):
            g = g.drop_duplicates("date", keep="last").set_index("date").reindex(idx).ffill(limit=5)
            pe, pb = g.get("PER"), g.get("PBR")
            d = pd.DataFrame(index=idx)
            if pe is not None:
                d["earn_yield"] = np.where(pe.isna(), np.nan, np.where(pe > 0, 1 / pe.replace(0, np.nan), 0))
            if pb is not None:
                d["book_to_price"] = np.where(pb > 0, 1 / pb, np.nan)
            if pe is not None and pb is not None:
                d["roe_proxy"] = np.where(pe > 0, pb / pe, np.nan)
            if "dividend_yield" in g:
                d["div_yield"] = g["dividend_yield"]
            feats[t] = d

    inst = read_fm("TaiwanStockInstitutionalInvestorsBuySell", codes)
    if not inst.empty:
        inst["date"] = pd.to_datetime(inst["date"])
        for t, g in inst.groupby("ticker"):
            if t not in volume:
                continue
            g = g.drop_duplicates("date", keep="last").set_index("date").reindex(idx)
            v = volume[t]
            d = feats.get(t, pd.DataFrame(index=idx))
            for k in ["foreign", "trust", "dealer"]:
                if k in g:
                    x = pd.to_numeric(g[k], errors="coerce")
                    d[f"{k}_5"] = x.rolling(5, min_periods=3).sum() / v.rolling(5).sum()
                    d[f"{k}_20"] = x.rolling(20, min_periods=10).sum() / v.rolling(20).sum()
            feats[t] = d
    return feats


# ============================================================
# 4. 建立資料表
# ============================================================
def pct(s, n=1):
    return s.pct_change(n, fill_method=None)


def rsi(s, n=14):
    diff = s.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    down = (-diff.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)


def lag_to_tw(series, tw_index):
    s = series.copy()
    s.index = s.index + pd.Timedelta(days=1)
    return s.reindex(s.index.union(tw_index)).ffill().reindex(tw_index)


MKT_FEATS = ["mkt_ret_5", "mkt_ret_20", "mkt_gap_ma60", "mkt_vol_20",
             "sox_ret_5", "sox_ret_20", "fx_ret_20"]


# 防守分數:現金多、負債低、自由現金流穩定、波動小、配息高(金融股與 ETF 財報結構不同,不計算)
DEFENSIVE = {"fcf_yield": 1, "cash_to_assets": 1, "debt_ratio": -1, "fcf_pos_8q": 1,
             "current_ratio": 1, "vol_60": -1, "beta_60": -1, "div_yield": 1}


def defensive_score(data):
    parts = [(data[k] if w > 0 else 1 - data[k]) for k, w in DEFENSIVE.items() if k in data]
    if not parts:
        return np.nan
    raw = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    n_ok = pd.concat(parts, axis=1).notna().sum(axis=1)
    raw[(n_ok < len(parts) * 0.6) | data["industry"].isin(["金融", "ETF"])] = np.nan
    return raw.groupby(data["date"]).rank(pct=True)


def build_panel(close, volume, market, sox_raw, fx_raw):
    sox, fx = lag_to_tw(sox_raw, close.index), lag_to_tw(fx_raw, close.index)
    mret = pct(market)
    mkt = pd.DataFrame({
        "mkt_ret_5": pct(market, 5), "mkt_ret_20": pct(market, 20),
        "mkt_gap_ma60": market / market.rolling(60).mean() - 1,
        "mkt_vol_20": mret.rolling(20).std(),
        "sox_ret_5": pct(sox, 5), "sox_ret_20": pct(sox, 20), "fx_ret_20": pct(fx, 20),
    }, index=close.index)

    codes = list(close.columns)
    fm_daily = daily_fm_features(close, volume, codes)
    rows = []
    for t in codes:
        c, v = close[t], volume[t]
        r = pct(c)
        dollar = c * v
        df = pd.DataFrame(index=close.index)
        df["ret_1"], df["ret_5"], df["ret_20"] = r, pct(c, 5), pct(c, 20)
        df["ret_60"], df["ret_120"] = pct(c, 60), pct(c, 120)
        df["mom_12_1"] = c.shift(20) / c.shift(250) - 1
        df["dist_high_250"] = c / c.rolling(250, min_periods=120).max() - 1
        df["gap_ma20"] = c / c.rolling(20).mean() - 1
        df["gap_ma60"] = c / c.rolling(60).mean() - 1
        df["gap_ma120"] = c / c.rolling(120).mean() - 1
        df["rsi_14"] = rsi(c)
        df["vol_20"], df["vol_60"] = r.rolling(20).std(), r.rolling(60).std()
        df["max_ret_20"] = r.rolling(20).max()
        df["skew_60"] = r.rolling(60).skew()
        df["beta_60"] = r.rolling(60).cov(mret) / mret.rolling(60).var()
        df["log_dollar_vol"] = np.log(dollar.rolling(20).mean() + 1)
        df["raw_close"] = c
        df["volu_ratio"] = v.rolling(5).mean() / v.rolling(60).mean()
        df["amihud"] = (r.abs() / dollar.replace(0, np.nan)).rolling(20).mean()
        if t in fm_daily:
            df = df.join(fm_daily[t])
        for h in (SHORT["horizon"], LONG["horizon"]):
            df[f"fut_{h}"] = c.shift(-h) / c - 1
        df["ticker"] = t
        rows.append(df[c.notna()])
    data = pd.concat(rows)
    data.index.name = "date"
    data = data.reset_index()

    # 基本面(as-of 對齊:每天只看得到已公布的最新一筆)
    data["date"] = data["date"].astype("datetime64[ns]")
    for ev in fundamental_events(codes):
        ev = ev.reset_index(drop=True)
        ev["avail"] = ev["avail"].astype("datetime64[ns]")
        ev = ev.sort_values("avail")
        cols = [c for c in ev.columns if c not in ("avail", "ticker")]
        data = data.sort_values("date")
        data = pd.merge_asof(data, ev[["avail", "ticker"] + cols].rename(columns={"avail": "date"}),
                             on="date", by="ticker", direction="backward",
                             tolerance=pd.Timedelta(days=200))

    # 現金流殖利率 = 近四季現金流 / 市值(用當天收盤價)
    if {"fcf_ttm", "shares"} <= set(data.columns):
        mcap = data["raw_close"] * data["shares"]
        data["fcf_yield"] = data["fcf_ttm"] / mcap
        data["ocf_yield"] = data["ocf_ttm"] / mcap
    data = data.drop(columns=[c for c in ["fcf_ttm", "ocf_ttm", "shares"] if c in data.columns])
    for k in ["fcf_yield", "debt_ratio", "cash_to_assets"]:
        if k in data:
            data["raw_" + k] = data[k]

    fut_cols = [c for c in data.columns if c.startswith("fut_")]
    stock_feats = [c for c in data.columns if c not in ["date", "ticker", "raw_close"] + fut_cols
                   and not c.startswith("raw_")]
    # 全部缺值的欄位(例如 FinMind 沒抓到)就不用
    stock_feats = [c for c in stock_feats if data[c].notna().mean() > 0.05]

    n = data.groupby("date")["ticker"].transform("count")
    data = data[n >= MIN_STOCKS].copy()
    data["raw_ret_20"] = data["ret_20"]
    data["raw_beta"] = data["beta_60"]
    for k in ["rev_yoy", "earn_yield", "foreign_20"]:
        if k in data:
            data["raw_" + k] = data[k]
    data[stock_feats] = data.groupby("date")[stock_feats].rank(pct=True)
    data = data.merge(mkt, left_on="date", right_index=True, how="left")
    for h in (SHORT["horizon"], LONG["horizon"]):
        data[f"target_{h}"] = data.groupby("date")[f"fut_{h}"].rank(pct=True)
    data["industry"] = data["ticker"].map(INDUSTRY_OF)
    data["defensive"] = defensive_score(data)
    return data.sort_values(["date", "ticker"]).reset_index(drop=True), stock_feats


# ============================================================
# 5. 模型
# ============================================================
from sklearn.linear_model import Ridge  # noqa: E402

SMOOTH_DAYS = 5      # 分數平滑:取最近幾個交易日的平均(降低每天排名大跳動)
BUFFER_MULT = 2      # 換股緩衝:持股只要還在前 TOP_N × 2 名就續抱

try:
    from lightgbm import LGBMRanker, LGBMRegressor

    _LGB_COMMON = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                       min_child_samples=200, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                       random_state=42, n_jobs=-1, deterministic=True, force_row_wise=True)

    def make_gbm():
        return LGBMRegressor(**_LGB_COMMON)

    def make_ranker():
        return LGBMRanker(objective="lambdarank", **_LGB_COMMON)
    GBM_NAME = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make_gbm():
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             max_leaf_nodes=31, min_samples_leaf=200,
                                             random_state=42)
    make_ranker = None
    GBM_NAME = "HistGradientBoosting"


def fit_models(tr, stock_feats, h):
    """訓練三個模型:LightGBM 迴歸、Ridge 線性、LambdaRank 排序(直接學「誰排前面」)"""
    feats = stock_feats + MKT_FEATS
    y = tr[f"target_{h}"]
    models = {"feats": feats, "stock_feats": stock_feats,
              "gbm": make_gbm().fit(tr[feats], y),
              "ridge": Ridge(alpha=10.0).fit(tr[stock_feats].fillna(0.5), y)}
    if make_ranker is not None:
        trs = tr.sort_values(["date", "ticker"])
        rel = np.minimum((trs[f"target_{h}"] * 5).astype(int), 4)  # 0~4 五個等級
        groups = trs.groupby("date", sort=True).size().values
        models["rank"] = make_ranker().fit(trs[feats], rel, group=groups)
    return models


def predict_models(models, df):
    out = {"gbm": models["gbm"].predict(df[models["feats"]]),
           "ridge": models["ridge"].predict(df[models["stock_feats"]].fillna(0.5))}
    if "rank" in models:
        out["rank"] = models["rank"].predict(df[models["feats"]])
    return out


def add_scores(df):
    """綜合分數(各模型排名平均)→ 扣掉產業、beta、規模影響 → 產業中立分數"""
    df = df.reset_index(drop=True)
    g = df.groupby("date")
    parts = []
    for m in ["gbm", "ridge", "rank"]:
        if m in df:
            df[m + "_r"] = g[m].rank(pct=True)
            parts.append(m + "_r")
    df["ens"] = df[parts].mean(axis=1)
    out = np.full(len(df), np.nan)
    for _, d in df.groupby("date", sort=False):
        ii = d.index.values
        X = pd.get_dummies(d["industry"]).astype(float)
        X["beta"] = d["beta_60"].fillna(0.5).values
        X["size"] = d["log_dollar_vol"].fillna(0.5).values
        y = d["ens"].values
        coef, *_ = np.linalg.lstsq(X.values, y, rcond=None)
        out[ii] = y - X.values @ coef
    df["neutral"] = out
    df["neutral_r"] = df.groupby("date")["neutral"].rank(pct=True)
    return df


def smooth_scores(df, col="neutral_r", days=SMOOTH_DAYS):
    """每檔股票取最近 days 個交易日分數的平均,再重新排名"""
    df = df.sort_values(["ticker", "date"])
    df["smooth_raw"] = df.groupby("ticker")[col].transform(lambda s: s.rolling(days, min_periods=1).mean())
    df["smooth"] = df.groupby("date")["smooth_raw"].rank(pct=True)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


KEEP = ["date", "ticker", "industry", "beta_60", "log_dollar_vol", "ret_20", "defensive"]


def walk_forward(data, stock_feats, cfg):
    h, step = cfg["horizon"], cfg["retrain"]
    lab = data.dropna(subset=[f"target_{h}"])
    dates = np.sort(lab["date"].unique())
    first = np.searchsorted(dates, dates[0] + np.timedelta64(365 * TRAIN_YEARS, "D"))
    preds, t0 = [], time.time()
    steps = list(range(first, len(dates), step))
    for k, i in enumerate(steps):
        end = dates[i - h]
        tr = lab[(lab["date"] >= end - np.timedelta64(365 * TRAIN_YEARS, "D")) & (lab["date"] < end)]
        te = lab[lab["date"].isin(dates[i:i + step])]
        if len(tr) < 5000 or te.empty:
            continue
        models = fit_models(tr, stock_feats, h)
        out = te[KEEP + [f"fut_{h}"]].copy().rename(columns={f"fut_{h}": "fut"})
        for m, v in predict_models(models, te).items():
            out[m] = v
        preds.append(out)
        if k % 20 == 0:
            print(f"  [{cfg['name']}] {k + 1}/{len(steps)},{time.time() - t0:.0f} 秒")
    pred = smooth_scores(add_scores(pd.concat(preds)))
    pred["reversal"] = 1 - pred["ret_20"].fillna(0.5)
    pred["fut_rank"] = pred.groupby("date")["fut"].rank(pct=True)
    return pred


MAIN = "主要(中立+平滑)"
MAIN_BT = "主要策略(含換股緩衝)"


def score_names():
    names = {"smooth": MAIN, "neutral": "產業中立(未平滑)", "ens": "綜合(未中立)",
             "gbm": GBM_NAME, "ridge": "Ridge 線性"}
    if make_ranker is not None:
        names["rank"] = "LambdaRank 排序"
    names["defensive"] = "防守分數(規則型)"
    names["reversal"] = "跌深反彈(基準)"
    return names


def buffered_backtest(pred, reb_dates, col="smooth"):
    """模擬實際持股:每期只換掉跌出前 TOP_N×BUFFER_MULT 名的股票,交易成本依實際換股比例計算"""
    held, rets, turns = [], [], []
    for d in reb_dates:
        day = pred[pred["date"] == d].set_index("ticker")
        order = day[col].sort_values(ascending=False).index.tolist()
        keep_zone = set(order[:TOP_N * BUFFER_MULT])
        new = [t for t in held if t in keep_zone]
        for t in order:
            if len(new) >= TOP_N:
                break
            if t not in new:
                new.append(t)
        turnover = 1.0 if not held else len(set(new) - set(held)) / TOP_N
        rets.append(day.loc[new, "fut"].mean() - COST * turnover)
        turns.append(turnover)
        held = new
    return rets, turns


def evaluate(pred, h, market):
    names = score_names()
    ic_rows = []
    for m, name in names.items():
        p = pred.groupby("date")[m].rank(pct=True)
        ric = pred.assign(_p=p).groupby("date").apply(lambda d: d["_p"].corr(d["fut_rank"]))
        ic_rows.append({"model": name, "key": m, "ric": ric.mean(),
                        "icir": ric.mean() / ric.std(), "pos": (ric > 0).mean()})
    reb_dates = np.sort(pred["date"].unique())[::h]
    reb = pred[pred["date"].isin(reb_dates)].copy()
    reb["q"] = np.ceil(reb.groupby("date")["smooth"].rank(pct=True) * 5).clip(1, 5).astype(int)
    quint = reb.groupby(["date", "q"])["fut"].mean().unstack().mean()
    mkt_fut = market.shift(-h) / market - 1
    rec = []
    for d in reb_dates:
        day = pred[pred["date"] == d]
        r = {"date": d, "全部股票平均": day["fut"].mean(), "加權指數": float(mkt_fut.loc[d])}
        for m, name in names.items():
            r[name] = day.nlargest(TOP_N, m)["fut"].mean() - COST
        rec.append(r)
    bt = pd.DataFrame(rec).set_index("date")
    rets, turns = buffered_backtest(pred, reb_dates)
    bt.insert(0, MAIN_BT, rets)
    return pd.DataFrame(ic_rows), quint, bt, float(np.mean(turns[1:])) if len(turns) > 1 else None


def simulate(excess, h, n_boot=5000, seed=7):
    """模擬驗證:
    1) Bootstrap:把每期超額報酬重新抽樣 5000 次,估計「長期平均超額報酬 > 0」的機率與 95% 區間
    2) PSR(Probabilistic Sharpe Ratio, Bailey & López de Prado):考慮樣本數、偏態、峰態後,Sharpe > 0 的機率
    """
    x = pd.Series(excess).dropna().values
    n = len(x)
    if n < 8:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    per_year = 252 / h
    sr = x.mean() / x.std(ddof=1)
    skew = pd.Series(x).skew()
    kurt = pd.Series(x).kurt() + 3
    from math import erf, sqrt
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = sr * sqrt(n - 1) / denom
    psr = 0.5 * (1 + erf(z / sqrt(2)))
    return {"n": n, "p_pos": float((means > 0).mean()),
            "ci_lo": float(np.percentile(means, 2.5) * per_year),
            "ci_hi": float(np.percentile(means, 97.5) * per_year),
            "ann_excess": float(x.mean() * per_year), "psr": float(psr)}


# ============================================================
# 6. 網站輸出
# ============================================================
def _c(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, nd)


def build_site(close, short_pred, short_eval, long_eval, latest, importance):
    DOCS.mkdir(exist_ok=True)
    today = pd.Timestamp(latest["date"].iloc[0])
    today_str = today.strftime("%Y-%m-%d")
    h = SHORT["horizon"]

    # 預測紀錄(同一天重跑會覆蓋)
    hist_f = DOCS / "predictions_history.csv"
    log = latest[["ticker", "name", "rank", "main_pct", "raw_pct", "long_pct", "close"]].copy()
    log.insert(0, "date", today_str)
    prev_rank = {}
    if hist_f.exists():
        hist = pd.read_csv(hist_f, dtype={"ticker": str, "date": str})
        older = hist[hist["date"] < today_str]
        if not older.empty:
            last = older[older["date"] == older["date"].max()]
            prev_rank = dict(zip(last["ticker"], last["rank"]))
        hist = pd.concat([hist[hist["date"] != today_str], log], ignore_index=True)
    else:
        hist = log
    hist = hist.reindex(columns=log.columns)
    hist.to_csv(hist_f, index=False, encoding="utf-8-sig")

    idx = close.index
    live = []
    for d, g in hist.groupby("date"):
        p = idx.get_indexer([pd.Timestamp(d)])[0]
        if p < 0:
            continue
        top = g.nsmallest(TOP_N, "rank")["ticker"].tolist()
        if p + h < len(idx):
            fut = close.iloc[p + h] / close.iloc[p] - 1
            a, t = fut.reindex(g["ticker"]).mean(), fut.reindex(top).mean()
            live.append({"date": d, "top": _c(t), "all": _c(a), "diff": _c(t - a), "left": 0})
        else:
            live.append({"date": d, "top": None, "all": None, "diff": None,
                         "left": int(p + h - (len(idx) - 1))})
    live.sort(key=lambda r: r["date"], reverse=True)
    done = [r for r in live if r["diff"] is not None]

    ic, quint, bt, turnover = short_eval
    cum = (1 + bt).cumprod()
    years = len(bt) * h / 252
    ann = cum.iloc[-1] ** (1 / years) - 1
    vol = bt.std() * np.sqrt(252 / h)
    win = (bt.sub(bt["全部股票平均"], axis=0) > 0).mean()
    yearly = (1 + bt).groupby(bt.index.year).prod() - 1
    excess = bt[MAIN_BT] - bt["全部股票平均"]
    sim = simulate(excess, h)

    lic = long_eval[0] if long_eval is not None else None

    reb_dates = np.sort(short_pred["date"].unique())[::h]
    reb = short_pred[short_pred["date"].isin(reb_dates)].sort_values("date")
    stocks_hist = {t: {"name": STOCKS.get(t, t),
                       "dates": [d.strftime("%Y-%m-%d") for d in g["date"]],
                       "score": [_c(v, 3) for v in g["smooth"]],
                       "actual": [_c(v, 3) for v in g["fut_rank"]],
                       "ret": [_c(v) for v in g["fut"]]}
                   for t, g in reb.groupby("ticker")}

    def pe(r):
        ey = r.get("raw_earn_yield")
        return _c(1 / ey, 1) if ey is not None and pd.notna(ey) and ey > 0 else None

    payload = {
        "date": today_str,
        "generated": (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
        "horizon": h, "topn": TOP_N, "n": len(latest), "smooth_days": SMOOTH_DAYS,
        "main": MAIN, "main_bt": MAIN_BT, "turnover": _c(turnover, 3),
        "today": [{
            "rank": int(r["rank"]), "code": r["ticker"], "name": r["name"], "ind": r["industry"],
            "score": _c(r["main_pct"], 3), "raw": _c(r["raw_pct"], 3), "long": _c(r["long_pct"], 3),
            "chg": (int(prev_rank[r["ticker"]]) - int(r["rank"])) if r["ticker"] in prev_rank else None,
            "ret20": _c(r["raw_ret_20"]), "rev": _c(r.get("raw_rev_yoy")), "pe": pe(r),
            "foreign": _c(r.get("raw_foreign_20")), "close": _c(r["close"], 2),
            "def": _c(r.get("defensive"), 3), "fcfy": _c(r.get("raw_fcf_yield")),
            "debt": _c(r.get("raw_debt_ratio"), 3), "cash": _c(r.get("raw_cash_to_assets"), 3),
        } for _, r in latest.iterrows()],
        "ic": [{"model": r["model"], "ric": _c(r["ric"]), "icir": _c(r["icir"], 3),
                "pos": _c(r["pos"], 3)} for _, r in ic.iterrows()],
        "long_ic": None if lic is None else [{"model": r["model"], "ric": _c(r["ric"]),
                                             "pos": _c(r["pos"], 3)} for _, r in lic.iterrows()],
        "quint": [_c(v) for v in quint.values],
        "summary": [{"name": k, "ann": _c(ann[k]), "vol": _c(vol[k]),
                     "sharpe": _c(ann[k] / vol[k], 2) if vol[k] else None,
                     "win": _c(win[k], 3)} for k in bt.columns],
        "yearly": {"years": [int(y) for y in yearly.index],
                   "model": [_c(v) for v in yearly[MAIN_BT]],
                   "raw": [_c(v) for v in yearly["產業中立(未平滑)"]],
                   "all": [_c(v) for v in yearly["全部股票平均"]]},
        "bt": {"dates": [d.strftime("%Y-%m-%d") for d in bt.index],
               **{k: [_c(v) for v in cum[name]] for k, name in
                  [("main", MAIN_BT), ("raw", "產業中立(未平滑)"),
                   ("all", "全部股票平均"), ("market", "加權指數")]},
               "excess": [_c(v) for v in excess]},
        "sim": None if sim is None else {k: _c(v, 4) for k, v in sim.items()},
        "live": live,
        "live_win": _c(np.mean([r["diff"] > 0 for r in done])) if done else None,
        "live_n": len(done),
        "stocks": stocks_hist,
        "importance": importance,
    }
    tpl = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    latest.to_csv(DOCS / "latest_prediction.csv", index=False, encoding="utf-8-sig")
    print(f"網站已輸出到 {DOCS / 'index.html'}")


# ============================================================
# 7. 主流程
# ============================================================
def main():
    t0 = time.time()
    print("== 1/5 下載股價 ==")
    close, volume, market, sox, fx = download_prices()
    stale_cut = close.index[-1] - pd.Timedelta(days=5)
    stale = [c for c in close if close[c].last_valid_index() < stale_cut]
    print(f"股價 {close.shape[1]} 檔,{close.index[0].date()} ~ {close.index[-1].date()}")

    print("== 2/5 更新 FinMind 基本面 / 籌碼 ==")
    token = load_token()
    print("FinMind token:", "有" if token else "沒有(每小時上限較低,會比較慢)")
    update_finmind([c for c in close.columns if c != "0050"], token)

    print("== 3/5 建立特徵 ==")
    data, stock_feats = build_panel(close, volume, market, sox, fx)
    print(f"特徵 {len(stock_feats)} 個個股 + {len(MKT_FEATS)} 個大盤,共 {len(data)} 筆")

    print("== 4/5 滾動訓練與評估 ==")
    short_pred = walk_forward(data, stock_feats, SHORT)
    short_eval = evaluate(short_pred, SHORT["horizon"], market)
    print(short_eval[0][["model", "ric", "pos"]].round(4).to_string(index=False))
    try:
        long_pred = walk_forward(data, stock_feats, LONG)
        long_eval = evaluate(long_pred, LONG["horizon"], market)
        print("長期模型:")
        print(long_eval[0][["model", "ric", "pos"]].round(4).to_string(index=False))
    except Exception as e:
        print("長期模型資料不足,略過:", e)
        long_eval = None

    print("== 5/5 今日預測與網站 ==")
    dates = np.sort(data["date"].unique())
    today = dates[-1]
    recent = data[data["date"].isin(dates[-SMOOTH_DAYS:]) & (~data["ticker"].isin(stale))].copy()
    latest = data[(data["date"] == today) & (~data["ticker"].isin(stale))].copy().reset_index(drop=True)
    importance = None
    for cfg in (SHORT, LONG):
        h = cfg["horizon"]
        lab = data.dropna(subset=[f"target_{h}"])
        end = lab["date"].max()
        models = fit_models(lab[lab["date"] >= end - np.timedelta64(365 * TRAIN_YEARS, "D")], stock_feats, h)
        tmp = recent[KEEP].copy()
        for m, v in predict_models(models, recent).items():
            tmp[m] = v
        tmp = smooth_scores(add_scores(tmp))
        tmp = tmp[tmp["date"] == today].set_index("ticker")
        if cfg is SHORT:
            latest["main_pct"] = latest["ticker"].map(tmp["smooth"])
            latest["raw_pct"] = latest["ticker"].map(tmp["neutral_r"])
            gbm = models["gbm"]
            if hasattr(gbm, "feature_importances_"):
                imp = pd.Series(gbm.feature_importances_, index=models["feats"]).sort_values(ascending=False).head(15)
                importance = {"names": list(imp.index), "values": [int(v) for v in imp.values]}
        else:
            latest["long_pct"] = latest["ticker"].map(tmp["smooth"])
    latest["rank"] = latest["main_pct"].rank(ascending=False, method="first").astype(int)
    latest["name"] = latest["ticker"].map(STOCKS)
    latest["close"] = latest["ticker"].map(close.loc[today])
    latest = latest.sort_values("rank")
    for k in ["raw_rev_yoy", "raw_earn_yield", "raw_foreign_20", "raw_fcf_yield",
              "raw_debt_ratio", "raw_cash_to_assets", "defensive"]:
        if k not in latest:
            latest[k] = np.nan
    build_site(close, short_pred, short_eval, long_eval, latest, importance)
    print(f"完成,共花 {(time.time() - t0) / 60:.1f} 分鐘")


if __name__ == "__main__":
    main()
