# Training Guide - BTC/USD RL Trading Bot

**Epic 3.3 (PBI-025): İlk Eğitim Rehberi**

Bu döküman, BTC/USD RL trading botunun ilk eğitimini çalıştırmak için gerekli adımları içerir.

---

## 🎯 Ön Gereksinimler

### 1. Veri Hazırlığı

BTC/USD 1-saatlik OHLCV verisini hazırlayın:

```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,30000.0,30100.0,29900.0,30050.0,1000000
2023-01-01 01:00:00,30050.0,30150.0,30000.0,30100.0,1100000
...
```

**Önerilen Veri Kaynakları:**
- Binance API
- CoinGecko API
- Kraken API
- Historical data CSV'leri

**Minimum Veri Miktarı:**
- En az 2000 saat (83 gün) veri
- Önerilen: 8000+ saat (333+ gün)

Veriyi `data/btcusd_1h.csv` olarak kaydedin.

---

## 🚀 İlk Eğitimi Başlatma

### Adım 1: Dependencies Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 2: Konfigürasyonları Doğrulayın

```bash
python config_validator.py
```

**Beklenen Çıktı:**
```
✓ All configurations are valid!
```

### Adım 3: Test Verisiyle Dry Run

Küçük bir test verisiyle sistemi test edin:

```bash
# Test data oluştur (örnek)
python -c "
import pandas as pd
import numpy as np

n = 2000
dates = pd.date_range('2023-01-01', periods=n, freq='1H')
close = 30000 + np.cumsum(np.random.randn(n) * 100)
df = pd.DataFrame({
    'timestamp': dates,
    'open': close * 0.999,
    'high': close * 1.01,
    'low': close * 0.99,
    'close': close,
    'volume': np.random.lognormal(10, 1, n)
})
df.to_csv('data/btcusd_1h.csv', index=False)
print('Test data created!')
"
```

### Adım 4: Eğitimi Başlatın

```bash
python train.py \
  --training-config configs/training.yaml \
  --env-config configs/env.yaml \
  --features-config configs/features.yaml
```

---

## 📊 Eğitim Parametreleri

### Varsayılan Parametreler (MVP)

**Training:**
- Total timesteps: 200,000
- Checkpoint frequency: 10,000 steps
- Seed: 42

**PPO Hyperparameters:**
- Learning rate: 0.0003
- n_steps: 2048
- Batch size: 64
- Gamma: 0.99
- Entropy coefficient: 0.01

**Environment:**
- Window size: 24 bars
- Initial cash: $10,000
- Transaction cost: 0.1%
- Slippage: 0.05%

---

## 📈 Eğitim İlerlemesini İzleme

### TensorBoard ile Monitoring

```bash
tensorboard --logdir artifacts/tensorboard/
```

Tarayıcınızda açın: http://localhost:6006

**İzlenecek Metrikler:**
- `rollout/ep_rew_mean`: Ortalama episode reward
- `rollout/ep_len_mean`: Ortalama episode uzunluğu
- `train/policy_loss`: Policy loss
- `train/value_loss`: Value function loss
- `train/entropy_loss`: Entropy loss

### Console Output

Training sırasında görecekleriniz:

```
======================================================================
BTC/USD RL TRADING AGENT TRAINING
======================================================================

[1/7] Loading and validating configurations...
✓ Environment config validation passed
✓ Training config validation passed
✓ Features config validation passed

[2/7] Preparing data...
Data split:
  Total samples: 1974
  Train: 1184 samples (60.0%)
  Val:   395 samples (20.0%)
  Test:  395 samples (20.0%)

[3/7] Creating training environment...
Using DummyVecEnv with 1 environment

[4/7] Initializing PPO agent...
PPO agent initialized with hyperparameters:
  Learning rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  ent_coef: 0.01

[5/7] Setting up callbacks...
Checkpoints will be saved to: ckpts/
Save frequency: every 10000 steps

[6/7] Starting training...
======================================================================
Training for 200,000 timesteps...

---------------------------------
| rollout/           |          |
|    ep_len_mean     | 150      |
|    ep_rew_mean     | 0.023    |
| time/              |          |
|    fps             | 245      |
|    iterations      | 10       |
|    time_elapsed    | 83       |
|    total_timesteps | 20480    |
| train/             |          |
|    entropy_loss    | -0.98    |
|    policy_loss     | -0.012   |
|    value_loss      | 0.045    |
---------------------------------
```

---

## 🎯 İlk Eğitim Başarı Kriterleri

### Episode Reward Artışı

- **Başlangıç**: ~0 (rastgele policy)
- **İlk 50k steps**: Pozitif reward'a ulaşmalı
- **100k steps**: Kararlı öğrenme görülmeli
- **200k steps**: Buy & Hold'dan daha iyi performans (hedef)

### Convergence Kontrolleri

✅ **İyi İşaretler:**
- Ortalama reward artıyor
- Episode length stabil
- Value loss azalıyor
- Policy loss stabil

⚠️ **Sorunlu İşaretler:**
- Reward sürekli düşüyor
- Episode length çok kısa (erken termination)
- Value loss exploding
- NaN values

---

## 🔧 Sorun Giderme

### Problem: "FileNotFoundError: data/btcusd_1h.csv"

**Çözüm:** Veri dosyasını doğru yere koyduğunuzdan emin olun.

```bash
ls -la data/btcusd_1h.csv
```

### Problem: "ValueError: DataFrame must have at least X rows"

**Çözüm:** Daha fazla veri ekleyin. Minimum 2000 satır gerekli.

### Problem: Training çok yavaş

**Çözüm 1:** GPU kullanın (PyTorch CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Çözüm 2:** `total_timesteps` azaltın (test için)
```yaml
# configs/training.yaml
training:
  total_timesteps: 50000  # 200000 yerine
```

### Problem: Agent öğrenmiyor

**Kontroller:**
1. **Reward signal:** Çok küçük veya çok büyük mü?
2. **Learning rate:** Çok yüksek (divergence) veya çok düşük (yavaş)?
3. **Environment:** Episode'lar çok kısa mı?

**Çözümler:**
- Entropy coefficient artır: `ent_coef: 0.01 → 0.05`
- Learning rate ayarla: `learning_rate: 0.0001 veya 0.001`
- Max steps artır: `max_steps: 500 → 1000`

---

## 📁 Eğitim Çıktıları

### Checkpoint'ler

```
ckpts/
├── rl_model_10000_steps.zip
├── rl_model_20000_steps.zip
├── ...
├── rl_model_200000_steps.zip
├── final_model.zip
└── best_model/
    └── best_model.zip  # En iyi validation performance
```

### TensorBoard Logları

```
artifacts/tensorboard/
└── PPO_1/
    └── events.out.tfevents...
```

### Evaluation Logs

```
ckpts/eval_logs/
└── evaluations.npz
```

---

## 🎓 İlk Eğitim Sonrası

### 1. Best Model'i Değerlendirin

```bash
python evaluate.py \
  --model ckpts/best_model/best_model.zip \
  --config configs/env.yaml
```

### 2. TensorBoard'da Analiz Edin

Önemli sorular:
- Agent convergence gösterdi mi?
- Validation performance training'e benzer mi (overfitting yok)?
- Buy & Hold'dan daha iyi mi?

### 3. Hyperparameter Tuning

İlk sonuçlara göre ayarlamalar:
- Learning rate
- Entropy coefficient
- Network architecture (sonraki iterasyon)

### 4. Daha Uzun Training

İlk eğitim başarılıysa:
```yaml
training:
  total_timesteps: 500000  # veya 1M
```

---

## 📋 Checklist

İlk eğitim öncesi:
- [ ] Data hazırlandı ve `data/btcusd_1h.csv` konumunda
- [ ] Config'ler validate edildi
- [ ] Dependencies yüklendi
- [ ] `ckpts/` ve `artifacts/` klasörleri oluşturuldu

Eğitim sırasında:
- [ ] TensorBoard açık ve monitoring yapılıyor
- [ ] Console output düzenli gözlemleniyor
- [ ] Checkpoint'ler kaydediliyor

Eğitim sonrası:
- [ ] Training tamamlandı (hatasız)
- [ ] Best model kaydedildi
- [ ] TensorBoard logları mevcut
- [ ] İlk evaluation yapıldı

---

## 🎯 Sonraki Adımlar

Başarılı bir ilk eğitimden sonra:

1. **Faz 4: Evaluation & Backtest**
   - Comprehensive metrics
   - Benchmark karşılaştırmaları
   - Visualization

2. **Hyperparameter Optimization**
   - Grid search
   - Optuna

3. **Advanced Features**
   - T+1 execution
   - Continuous action space
   - Regime detection

4. **Paper Trading**
   - Real-time testing
   - Risk management

---

## 📞 Yardım

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. Config validation çalıştırın
3. Test suite'i çalıştırın: `pytest -v`

**Training süresi (tahmini):**
- CPU: ~2-4 saat (200k timesteps)
- GPU: ~30-60 dakika (200k timesteps)

---

**Good luck with your first training! 🚀**
