# Product Backlog - RL Tabanlı BTC/USD Trading Botu MVP

## 🎯 MVP Hedefi
BTC/USD için temel bir Reinforcement Learning trading botunun çalışır bir prototipi. İlk aşamada basit, anlaşılabilir ve test edilebilir bir yapı.

---

## Faz 1: Temel Altyapı ve Proje İskeleti (Sprint 1)
**Hedef:** Projenin temel yapısını kurmak ve veri pipeline'ını hazırlamak

### Epic 1.1: Proje Yapısının Oluşturulması
- [ ] **PBI-001**: Proje klasör yapısını oluştur
  - `data/`, `configs/`, `artifacts/`, `ckpts/` klasörleri
  - Boş Python dosyaları: `data_manager.py`, `trading_env.py`, `train.py`, `evaluate.py`
  - Tahmini süre: 1 saat

- [ ] **PBI-002**: `requirements.txt` oluştur ve temel kütüphaneleri ekle
  - gymnasium, stable-baselines3, pandas, numpy, ta, matplotlib
  - Test et: tüm bağımlılıkların yüklendiğini doğrula
  - Tahmini süre: 1 saat

- [ ] **PBI-003**: Temel config dosyaları oluştur (basitleştirilmiş)
  - `configs/env.yaml`: Sadece temel parametreler
  - `configs/training.yaml`: Sadece PPO ve basit hiperparametreler
  - `configs/features.yaml`: Sadece temel TA indikatörleri (RSI, MACD)
  - Tahmini süre: 2 saat

### Epic 1.2: Veri Yönetimi (data_manager.py)
- [ ] **PBI-004**: BTC/USD verisi için CSV yükleme fonksiyonu
  - `load_data(file_path)` fonksiyonu
  - OHLCV kolonlarını doğrulama
  - UTC timezone kontrolü
  - Tahmini süre: 3 saat

- [ ] **PBI-005**: Temel feature engineering (sadece MVP için gerekli olanlar)
  - RSI(14)
  - MACD(12,26,9)
  - ATR(14)
  - Bollinger Bands (20)
  - Forward-looking leakage kontrolü ekle
  - Tahmini süre: 4 saat

- [ ] **PBI-006**: Basit normalizasyon fonksiyonu
  - Rolling z-score implementasyonu
  - Min-max scaling opsiyonu
  - Tahmini süre: 2 saat

- [ ] **PBI-007**: Veri bölümleme (basit versiyon)
  - 60/20/20 split (train/val/test)
  - Temporal ordering korunmalı
  - Split bilgilerini kaydet
  - Tahmini süre: 3 saat

- [ ] **PBI-008**: Data manager test ve validasyon
  - Örnek veri ile end-to-end test
  - Çıktı DataFrame'lerini kontrol et
  - Tahmini süre: 2 saat

**Faz 1 Toplam Tahmini Süre:** ~18 saat (2-3 gün)

---

## Faz 2: Trading Ortamı (Sprint 2)
**Hedef:** Gymnasium uyumlu temel trading environment'ı geliştirmek

### Epic 2.1: BtcUsdTradingEnv - Temel Yapı
- [ ] **PBI-009**: Gymnasium.Env sınıfı iskeletini oluştur
  - `BtcUsdTradingEnv` class tanımı
  - `__init__` parametreleri: df, window_size, initial_cash, transaction_cost
  - Tahmini süre: 2 saat

- [ ] **PBI-010**: Action space tanımla (basit versiyon)
  - İlk MVP için: `Discrete(3)` - {0: Hold, 1: Buy, 2: Sell}
  - (Sürekli aksiyonu Faz 3'e ertele)
  - Tahmini süre: 1 saat

- [ ] **PBI-011**: Observation space tanımla (basit versiyon)
  - Box space: son window_size barın normalize OHLCV + indikatörleri
  - İlk versiyon için Dict obs yok, sadece flat array
  - Tahmini süre: 3 saat

### Epic 2.2: Step ve Reset Mantığı
- [ ] **PBI-012**: `reset()` fonksiyonunu implement et
  - Portföyü başlangıç durumuna al
  - İlk gözlemi döndür
  - Deterministik seed desteği
  - Tahmini süre: 2 saat

- [ ] **PBI-013**: Basit `step()` mantığı (T+0 yürütme ile başla)
  - Aksiyonu al ve pozisyonu güncelle
  - Portföy değerini hesapla
  - Temel reward: log-return
  - T+1 yürütmeyi Faz 3'e ertele
  - Tahmini süre: 5 saat

- [ ] **PBI-014**: Transaction cost ve slippage ekle
  - Sadece pozisyon değişiminde maliyet
  - Basit sabit yüzde modeli
  - Tahmini süre: 2 saat

- [ ] **PBI-015**: Episode sonlandırma mantığı
  - Max step kontrolü
  - Portfolio value <= 0 kontrolü
  - Tahmini süre: 2 saat

### Epic 2.3: Test ve Validasyon
- [ ] **PBI-016**: Environment unit testleri
  - Reset/step çalışıyor mu?
  - Action space doğru mu?
  - Observation shape doğru mu?
  - Tahmini süre: 3 saat

- [ ] **PBI-017**: Manuel trading simülasyonu
  - Basit bir agent ile (random) birkaç episode çalıştır
  - Çıktıları console'da görüntüle
  - Tahmini süre: 2 saat

**Faz 2 Toplam Tahmini Süre:** ~22 saat (3-4 gün)

---

## Faz 3: Ajan Eğitimi (Sprint 3)
**Hedef:** İlk RL ajanını eğitmek ve checkpoint'leri kaydetmek

### Epic 3.1: Training Pipeline
- [ ] **PBI-018**: Basit train.py script oluştur
  - Config'den parametreleri oku
  - Environment instance'ı oluştur
  - Tahmini süre: 2 saat

- [ ] **PBI-019**: PPO ajanı kurulumu (basitleştirilmiş)
  - stable-baselines3 PPO
  - MlpPolicy
  - Temel hiperparametreler (learning_rate, n_steps, batch_size)
  - Tahmini süre: 3 saat

- [ ] **PBI-020**: VecEnv kurulumu (basit versiyon)
  - İlk MVP için sadece DummyVecEnv (1-2 env)
  - SubprocVecEnv'i sonraya ertele
  - Tahmini süre: 2 saat

- [ ] **PBI-021**: Temel training loop
  - `model.learn(total_timesteps=...)`
  - Progress bar ve basit logging
  - Tahmini süre: 2 saat

### Epic 3.2: Checkpointing ve Monitoring
- [ ] **PBI-022**: Model checkpoint kaydetme
  - Her N adımda model kaydet
  - Best model seçimi (basit metrik: episode reward)
  - `ckpts/` klasörüne kaydet
  - Tahmini süre: 3 saat

- [ ] **PBI-023**: Basit tensorboard logging
  - Episode rewards
  - Episode lengths
  - Mean reward tracking
  - Tahmini süre: 2 saat

- [ ] **PBI-024**: Training seed ve reproducibility
  - Global seed ayarları
  - Deterministic PyTorch/NumPy
  - Tahmini süre: 2 saat

### Epic 3.3: İlk Eğitim
- [ ] **PBI-025**: İlk eğitimi çalıştır
  - 100k-200k timesteps ile test
  - Convergence kontrolü
  - Problemleri belirle ve düzelt
  - Tahmini süre: 4 saat

**Faz 3 Toplam Tahmini Süre:** ~20 saat (3 gün)

---

## Faz 4: Değerlendirme ve Backtest (Sprint 4)
**Hedef:** Eğitilen modeli değerlendirmek ve görselleştirmek

### Epic 4.1: Evaluate Script
- [ ] **PBI-026**: evaluate.py iskeletini oluştur
  - Trained model yükleme
  - Test dataseti hazırlama
  - Tahmini süre: 2 saat

- [ ] **PBI-027**: Episode rollout fonksiyonu
  - Test env'de episode çalıştır
  - Tüm aksiyonları ve state'leri kaydet
  - Trade blotter oluştur (timestamp, action, price, pnl)
  - Tahmini süre: 4 saat

### Epic 4.2: Benchmark'lar
- [ ] **PBI-028**: Buy & Hold benchmark
  - Test setinde basit B&H stratejisi
  - Aynı metrikleri hesapla
  - Tahmini süre: 2 saat

- [ ] **PBI-029**: Simple RSI baseline (opsiyonel MVP için)
  - RSI 30/70 stratejisi
  - Comparison için hazırla
  - Tahmini süre: 3 saat

### Epic 4.3: Metrikler ve Raporlama
- [ ] **PBI-030**: Temel performans metrikleri
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Tahmini süre: 4 saat

- [ ] **PBI-031**: Gelişmiş metrikler
  - Sortino Ratio
  - Calmar Ratio
  - Average Trade P&L
  - Turnover
  - Tahmini süre: 3 saat

- [ ] **PBI-032**: Metrik karşılaştırma tablosu
  - RL Agent vs Benchmarks
  - Markdown veya CSV formatında
  - Tahmini süre: 2 saat

### Epic 4.4: Görselleştirme
- [ ] **PBI-033**: Fiyat ve trade noktaları grafiği
  - BTC fiyat çizgisi
  - Buy/Sell işaretleri
  - Matplotlib ile
  - Tahmini süre: 3 saat

- [ ] **PBI-034**: Portfolio değeri eğrisi
  - Agent portfolio value over time
  - Benchmark ile karşılaştırma
  - Tahmini süre: 2 saat

- [ ] **PBI-035**: Drawdown grafiği
  - Underwater plot
  - Max drawdown işaretleme
  - Tahmini süre: 2 saat

- [ ] **PBI-036**: Trade blotter CSV export
  - Detaylı trade history
  - Analiz için dışa aktar
  - Tahmini süre: 2 saat

**Faz 4 Toplam Tahmini Süre:** ~29 saat (4-5 gün)

---

## Faz 5: İyileştirme ve Genişletme (Post-MVP)
**Not:** Bu faz MVP tamamlandıktan sonra değerlendirilecek

### Epic 5.1: Gelişmiş Özellikler
- [ ] **PBI-037**: Sürekli action space (Box)
  - Trading_env'e sürekli aksiyon desteği
  - SAC algoritması entegrasyonu
  - Tahmini süre: 6 saat

- [ ] **PBI-038**: T+1 execution mantığı
  - Look-ahead bias önleme
  - Env'i güncelle
  - Tahmini süre: 4 saat

- [ ] **PBI-039**: Dict observation space
  - Market + Inventory ayrımı
  - Daha zengin state representation
  - Tahmini süre: 4 saat

### Epic 5.2: Hyperparameter Optimization
- [ ] **PBI-040**: Optuna entegrasyonu
  - HPO pipeline kurulumu
  - Search space tanımı
  - Tahmini süre: 6 saat

- [ ] **PBI-041**: MLflow tracking
  - Experiment tracking
  - Metric comparison
  - Tahmini süre: 4 saat

### Epic 5.3: Advanced Data ve Regime
- [ ] **PBI-042**: Walk-Forward Cross-Validation
  - Kaydırmalı pencere validasyonu
  - Robust performans testi
  - Tahmini süre: 6 saat

- [ ] **PBI-043**: Regime özellikleri
  - Volatilite rejimi
  - Trend strength
  - Calendar features
  - Tahmini süre: 5 saat

### Epic 5.4: Paper Trading
- [ ] **PBI-044**: CCXT entegrasyonu
  - Testnet bağlantısı
  - Real-time veri akışı
  - Tahmini süre: 8 saat

- [ ] **PBI-045**: Risk guardrails
  - Max drawdown limiti
  - Position size limiti
  - Kill-switch mekanizması
  - Tahmini süre: 6 saat

**Faz 5 Toplam Tahmini Süre:** ~49 saat (6-7 gün)

---

## 📊 MVP Özeti (Faz 1-4)

### Toplam Tahmini Süre: ~89 saat (11-15 iş günü)

### Kritik Yol (Must-Have için MVP):
1. **Faz 1**: Veri pipeline hazır
2. **Faz 2**: Trading environment çalışıyor
3. **Faz 3**: PPO agent eğitiliyor
4. **Faz 4**: Sonuçlar değerlendiriliyor ve görselleştiriliyor

### Başarı Kriterleri (MVP):
- ✅ Environment Gymnasium ile uyumlu ve test geçiyor
- ✅ Agent eğitimi convergence gösteriyor
- ✅ Test setinde evaluation çalışıyor
- ✅ Basit benchmark (B&H) ile karşılaştırma mevcut
- ✅ Temel metrikler (Sharpe, DD, Return) hesaplanıyor
- ✅ Grafikler ve trade blotter oluşuyor

### MVP Sonrası Karar Noktaları:
- Model performance tatmin edici mi? → Faz 5'e geç
- Daha fazla feature engineering gerekli mi? → Epic 5.3'e odaklan
- Paper trading'e hazır mı? → Epic 5.4'e geç

---

## 🔧 Teknik Debt ve İyileştirmeler (Backlog)
Bu itemlar MVP'de basitleştirildi, sonradan eklenecek:

- [ ] Purged & Embargoed split (data leakage önleme)
- [ ] SubprocVecEnv (paralel environments)
- [ ] VecNormalize scaler kaydetme/yükleme
- [ ] Early stopping callback
- [ ] Wandb entegrasyonu
- [ ] Dockerfile oluşturma
- [ ] Comprehensive README.md
- [ ] Unit test coverage artırma
- [ ] Logging standardizasyonu

---

## 📝 Notlar
- Her PBI için acceptance criteria ayrıca tanımlanmalı
- Kod review her epic sonunda yapılmalı
- Her faz sonunda retrospective toplantısı
- MVP tamamlandıktan sonra kullanıcı feedback toplanmalı
