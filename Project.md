Süper. Aşağıda **orijinal plandaki detayları koruyup**, önerdiğimiz geliştirmeleri ekleyerek **revize edilmiş gereksinim dokümanı**ni veriyorum. Dosya adları, bölümler ve beklenen çıktılar senin planınla **uyumlu**, fakat artık **sürekli aksiyon (PPO/SAC)**, **T+1 yürütme**, **gerçekçi maliyet/slipaj**, **walk-forward CV**, **çoklu risk metriği** ve **paper-trading→live** basamakları da standart hâle getirildi.

---

# Proje Gereksinim Dökümanı (Revize): RL Tabanlı BTC/USD Trading Botu

**Proje Başlığı:** BTC/USD için Pekiştirmeli Öğrenme (RL) Tabanlı Al-Sat Botu (Prototip)

**Proje Amacı:**
BTC/USD paritesi üzerinde alım-satım kararlarını verebilen, `Gymnasium` tabanlı özel bir trading ortamı ve `stable-baselines3` ile eğitilen bir RL ajanı geliştirmek.

> Not: Orijinal plandaki **DQN/Discrete(3)** seçeneği korunur; **varsayılan** ise **PPO/SAC + sürekli aksiyon** olacaktır.

**Temel Teknolojiler (Zorunlu):**

* Python 3.10+
* `gymnasium`, `stable-baselines3[extra]` (PPO, SAC ve DQN)
* `pandas` (veya opsiyonel `polars`)
* `numpy`, `ta`, `matplotlib`
* (Önerilir) `optuna` (HPO), `mlflow` veya `wandb` (deney takibi)
* (Paper/Live için) `ccxt`
* (Opsiyonel) `vectorbt` veya `backtrader` (ek backtest/benchmark)

---

## 1) Veri Yönetimi ve Hazırlık — `data_manager.py`

**1.1 Veri Yükleme**

* BTC/USD için 1 saatlik (veya 4 saatlik) **OHLCV** verisi CSV’den **veya** Parquet’ten yüklenmeli.
* Zaman dilimi ve timezone netleştirilmeli (UTC önerilir).

**1.2 Feature Engineering (orijinal + ekler)**

* Orijinal: RSI(14), MACD(12,26,9), Bollinger(20 üst/orta/alt), ATR(14).
* Ekler:

  * **Rejim özellikleri**: Volatilite (rolling std, ATR normalizasyonu), trend (EMA spread z-score), “hafta içi/sonu” gibi takvimsel sinyaller (opsiyonel).
  * **Leakage önlemi**: Tüm indikatörler yalnızca **geçmiş bar** bilgisiyle hesaplanır (forward fill yok).

**1.3 Normalizasyon**

* Gözlem değişkenleri için **rolling z-score / min-max** opsiyonu.
* Eğitimde **VecNormalize** (SB3) kullanılacağı için scaler’lar kaydedilir.

**1.4 Veri Bölümleme**

* Orijinal `%60/%20/%20` korunur **ama** ek olarak:

  * **Walk-Forward Cross-Validation (WFC)**: (Train→Val→Test) pencereleri kaydırmalı.
  * **Purged & Embargoed split**: Sınır yakınındaki sızıntıları önlemek için purging ve embargo uygulanır.
* Tüm split’ler deterministik (seed) ve **config’ten** kontrol edilir.

**1.5 Çıktılar**

* Eğitim/validasyon/test DataFrame’leri.
* Split meta bilgisi (tarih aralıkları) ve feature konfigleri bir `artifacts/` klasörüne yazılır.

---

## 2) Özel Trading Çevresi — `trading_env.py`

**2.1 Sınıf**

* `class BtcUsdTradingEnv(gymnasium.Env)`

**2.2 Parametreler (`__init__`)**

* `df`: Eğitim veya test veri seti (zaman sıralı)
* `window_size`: Gözlem penceresi (örn. 50)
* `transaction_cost_percent`: İşlem maliyeti (örn. 0.001)
* `slippage_percent`: Slipaj (örn. 0.0005)
* `tplus_execution`: T+1 yürütme (True) — **varsayılan açık**
* `initial_cash`: Başlangıç nakit (örn. 10_000)
* `enable_short`: Short’a izin (bool)
* `discrete_actions`: Eğer **True** ise `Discrete(3)`; değilse **sürekli Box([-1,1])**
* `seed` vb.

**2.3 Aksiyon Alanı (`action_space`)**

* **Sürekli (varsayılan)**: `Box(low=-1.0, high=+1.0, shape=(1,))`

  * Anlamı: Hedef **net pozisyon** (−1 tam short, 0 flat, +1 tam long).
* **Ayrık (orijinal)**: `Discrete(3)` → {0: Hold, 1: Buy, 2: Sell} (tam pozisyon değişimi; prototip karşılaştırması için korunur).

**2.4 Gözlem Alanı (`observation_space`)**

* **Dict observation** (önerilir):

  * `"market"`: Son `window_size` barın OHLCV + indikatör matrisi (normalize).
  * `"inventory"`: `position` (−1…+1), `cash`, `unrealized_pnl`, opsiyonel `prev_action`.
* Alternatif olarak tek bir `Box` içinde konkatenasyon da desteklenebilir (config).

**2.5 Adım Mantığı (`step(action)`)**

* **T+1 yürütme**: Aksiyon (t)’de verilir, **uygulama (t+1)** barı fiyatından yapılır (look-ahead önlenir).
* **Maliyet modeli**: Sadece **pozisyon değişiminde** komisyon + slipaj düşülür.
* Portföy güncellenir: `cash`, `inventory`, `portfolio_value`.
* **Ödül**: Varsayılan **log-getiri**:
  [
  r_t = \log\frac{V_{t+1}}{V_t} - \text{fees} - \text{slippage} - \lambda\cdot\text{turnover}
  ]

  * `λ` ile **turnover** veya volatilite cezası (mini-Sortino) opsiyonel, config’ten set edilir.
* Dönüş: `(observation, reward, terminated, truncated, info)`

**2.6 Reset**

* Rastgele başlangıç (curriculum) veya sabit başlangıç.
* `initial_cash` ile portföy sıfırlanır.
* İlk gözlem döndürülür.

**2.7 Güvenlik / Kısıtlar**

* `max_drawdown` aşıldığında `truncated=True` ile episode kesilebilir.
* `position` sınırları (ör. kaldıraç yoksa [−1, +1]).

---

## 3) Ajan Eğitimi — `train.py`

**3.1 Çevre Kurulumu**

* `SubprocVecEnv` ile **8–32** paralel env.
* `VecNormalize(norm_obs=True, norm_reward=False)`; eğitim sonunda scaler kaydı (`vecnorm.pkl`).

**3.2 Algoritma Seçimi (config ile)**

* **Varsayılan:** `PPO("MlpPolicy", ...)` **veya** `SAC("MlpPolicy", ...)` (sürekli aksiyon).
* **Alternatif:** `DQN("MlpPolicy", ...)` (Discrete(3) denemesi için).

**3.3 Hiperparametreler**

* `gamma`, `gae_lambda` (PPO), `clip_range`, `ent_coef`, `learning_rate`, `batch_size`, `n_steps` vb. **yaml config** ile yönetilir.
* **Optuna** ile HPO (opsiyonel).

**3.4 Callbacks ve Kayıt**

* `EvalCallback` (validation split üzerinde), **early stopping** benzeri kriter (Sharpe/Calmar iyileşmiyorsa durdur).
* Her N adımda checkpoint: `./ckpts/best_model.zip`, `./ckpts/last_model.zip`.
* `mlflow`/`wandb` ile metrik ve görsel log’ları.

**3.5 Determinizm & Seed**

* Reprodüksiyon için global seed; env’ler, PyTorch ve NumPy seed’leri sabitlenir.

---

## 4) Değerlendirme & Backtest — `evaluate.py`

**4.1 Test Seti**

* Ajanın **hiç görmediği** out-of-sample test.
* `VecNormalize` scaler yüklenir (train’den).

**4.2 Benchmark’lar**

* **Buy & Hold**
* **Naive Trend** (ör. EMA crossover)
* **RSI 30/70** (mean-reversion)

**4.3 Metrikler**

* **Sharpe, Sortino, Calmar**, **Max Drawdown**, **Hit Ratio**, **Turnover**, **Avg Trade P&L**, **Exposure**
* İşlem sayısı, maliyet sonrası P&L

**4.4 Görselleştirme ve Blotter**

* Fiyat grafiği üzerinde alım/satım işaretleri
* Portföy eğrisi + **drawdown paneli**
* **Trade blotter CSV**: `timestamp, side, size, price, fee, pnl`

---

## 5) Paper Trading → Live

**5.1 Paper Trading**

* `ccxt` testnet/sandbox ile başlat (Binance testnet vb.)
* **Risk guardrails**: max günlük zarar, max pozisyon, order throttling, kill-switch.

**5.2 Live (opsiyonel)**

* API anahtarları **.env** / **secrets** üzerinden.
* **Asenkron** sinyal→emir hattı (latency düşük).
* Sağlık kontrolleri, uyarı/alarm (slack/email).

---

## 6) Proje Yapısı (dosyalar)

```
project/
  data_manager.py
  trading_env.py
  train.py
  evaluate.py
  configs/
    training.yaml
    features.yaml
    env.yaml
  artifacts/
    vecnorm.pkl
    ckpts/
  requirements.txt
  Dockerfile
  Makefile (opsiyonel)
  README.md
```

* **`configs/*.yaml`**: Tüm hiperparametreler, env ayarları, feature setleri burada.
* **`Dockerfile`**: CUDA’lı veya CPU/MPS’li image seçenekleri.
* **`README.md`**: Çalıştırma komutları, veri formatı, deney reprodüksiyon adımları.

---

## 7) `requirements.txt` (öneri)

```
gymnasium==0.29.*
stable-baselines3[extra]==2.3.*
torch>=2.2
pandas>=2.2
numpy>=1.26
ta>=0.11
matplotlib>=3.8
optuna>=3.6
mlflow>=2.14
wandb>=0.17
ccxt>=4.3
# optional
polars>=1.5
vectorbt>=0.26
```

---

## 8) Örnek Config Parçaları

**`env.yaml`**

```yaml
window_size: 50
transaction_cost_percent: 0.001
slippage_percent: 0.0005
tplus_execution: true
initial_cash: 10000
enable_short: true
discrete_actions: false   # true -> Discrete(3)
```

**`training.yaml`**

```yaml
algo: "PPO"          # "SAC" veya "DQN"
total_timesteps: 2000000
n_envs: 16
vecnormalize:
  norm_obs: true
  norm_reward: false
  clip_obs: 5.0
ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
eval:
  eval_freq: 50000
  n_eval_episodes: 5
  save_best: true
seed: 42
```

**`features.yaml`**

```yaml
ohlcv: ["open","high","low","close","volume"]
ta:
  rsi: { period: 14 }
  macd: { fast: 12, slow: 26, signal: 9 }
  bbands: { period: 20, std: 2 }
  atr: { period: 14 }
regime:
  vola_window: 20
  trend_ema_fast: 20
  trend_ema_slow: 50
normalize: { method: "zscore", window: 200 }
```

---

## 9) Başarı Kriterleri (Prototip)

1. **Out-of-sample** testte (tüm maliyetler dahil) **Calmar ≥ Buy&Hold** ve **Max DD ≤ Buy&Hold**.
2. **Sharpe/Sortino** iyileşmesi ve **Turnover**’ın sürdürülebilir seviyede kalması.
3. Yürüyen pencerelerde metriklerin **stabilitesi** (varyans düşük).
4. Öğrenilen davranışın açıklanabilir bir pattern sergilemesi (örn. volatilite rejiminde pozisyon boyutunu azaltma).

---

## 10) Beklenen Çıktılar

* **Kod**:

  * `data_manager.py`, `trading_env.py`, `train.py`, `evaluate.py`, `requirements.txt`
  * `configs/*.yaml`, `Dockerfile`, `README.md`
* **Artefaktlar**:

  * `ckpts/best_model.zip`, `ckpts/last_model.zip`, `artifacts/vecnorm.pkl`
  * Trade blotter CSV, metrik raporu (MLflow/W&B)
  * Grafikler: fiyat+işlem noktaları, portföy eğrisi, drawdown

---

Bu sürüm, senin orijinal omurganı **bozmadan**, üretim-kalitesine giden bütün “kritik rayları” döşüyor: **sürekli aksiyon (PPO/SAC)**, **T+1 yürütme**, **gerçekçi maliyet**, **walk-forward** ve **çoklu risk metriği**.
İstersen bunu **iskelet repo** yapısında (boş fonksiyon imzaları + çalışır `train/evaluate`) tek seferde çıkarıp, doğrudan kod asistanına verilebilir hâle getirebilirim.
