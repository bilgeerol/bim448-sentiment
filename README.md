# BİM 448 – Duygu Analizi: VADER, BERT ve LLM Karşılaştırması

Bu projede İngilizce (Amazon Reviews 2023) ve Türkçe (TRSAv1) ürün yorumları üç sınıfa (Negatif, Nötr, Pozitif) ayrılmış ve üç yöntem aynı test seti üzerinde karşılaştırılmıştır:

- **VADER:** sözlük tabanlı yöntem (Türkçede doğrudan ve İngilizceye çeviri sonrası iki varyant)
- **BERT:** İngilizce için `FacebookAI/roberta-base`, Türkçe için `dbmdz/bert-base-turkish-cased` ince ayarı
- **LLM:** Groq API üzerinden `openai/gpt-oss-120b` (sıfır örnekli istem)

Her dilde test seti 3.000 yorumdan (sınıf başına 1.000) oluşur. Ölçütler ortak değerlendirme kodu `src/evaluate.py` ile hesaplanır.

## Üyeler ve roller

| Üye | Rol | Sorumluluklar |
|---|---|---|
| Zehra | A – Veri, VADER, Değerlendirme | Veri setleri, çeviri, VADER, ortak değerlendirme kodu; koordinatör |
| Bilge | B – BERT | GitHub deposu, BERT ince ayarı, karşılaştırma grafikleri; ortam sorumlusu (GitHub, Colab, README) |
| Sude | C – LLM | Groq API, istem tasarımı, toplu tahmin; video ve kaynakça |

## Klasör yapısı

~~~
bim448-sentiment/
├── README.md
├── requirements.txt
├── data/processed/        # eğitim/doğrulama/test bölünmeleri, çeviri, manuel kontrol
├── notebooks/             # 01–06 notebook'ları
├── src/evaluate.py        # ortak değerlendirme kodu
├── predictions/           # her yöntemin test seti tahminleri
├── results/
│   ├── tables/            # model başına metrikler ve karşılaştırma tablosu
│   ├── figures/           # karışıklık matrisleri, kayıp eğrileri, karşılaştırma grafikleri
│   ├── logs/              # BERT eğitim logları, LLM ham çıktıları
│   └── hata_ornekleri/    # her yöntemin yanlış sınıflandırma örnekleri
└── report/
~~~

Ham veri (`data/raw/`) ve eğitilen modeller boyutları nedeniyle depoya eklenmemiştir; modeller Google Drive'da tutulmuştur.

## Kurulum

1. Notebook'lar Google Colab'da çalıştırılır. BERT eğitimi için **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçilir.
2. Colab'da sol menüdeki anahtar simgesinden (**Secrets**) şu iki değer eklenir:
   - `GITHUB_TOKEN`: depoya erişim için GitHub kişisel erişim anahtarı
   - `GROQ_API_KEY`: yalnızca `04_llm.ipynb` için Groq API anahtarı
3. Her notebook'un ilk hücresi (kurulum hücresi) Drive'ı bağlar, depoyu `/content/bim448-sentiment` klasörüne indirir ve günceller.
4. Kütüphane sürümleri `requirements.txt` dosyasındadır: `pip install -r requirements.txt`

## Çalıştırma sırası

`01 → 02 → 03 → 04 → 05 → 06`

03 ve 04, 01'in çıktılarına bağlıdır; 05, 02–04'ün tahmin dosyalarını; 06 ise 05'in tablolarını kullanır.

| Notebook | Sahibi | Ürettiği dosyalar |
|---|---|---|
| `01_veri_hazirlama.ipynb` | Zehra | `data/processed/{en,tr}_{train,val,test}.csv`, `data/processed/{en,tr}_manuel_kontrol.csv` |
| `02_vader.ipynb` | Zehra | `data/processed/tr_test_translated.csv`, `predictions/vader_en.csv`, `predictions/vader_tr_direct.csv`, `predictions/vader_tr_mt.csv`, `results/hata_ornekleri/vader.csv` |
| `03_bert_finetune.ipynb` | Bilge | `predictions/bert_{en,tr}.csv`, `results/logs/bert_{en,tr}_log.csv`, `results/figures/bert_{en,tr}_loss.png`, `results/tables/bert_parametreler.csv`, `results/hata_ornekleri/bert.csv` |
| `04_llm.ipynb` | Sude | `predictions/llm_{en,tr}.csv` (ara kayıtlar: `llm_{en,tr}_partial.csv`), `results/logs/llm_*_raw.jsonl`, `results/logs/llm_calisma_ozeti.csv`, `results/tables/yontem_uyumu.csv`, `results/hata_ornekleri/llm.csv` |
| `05_degerlendirme.ipynb` | Zehra | `results/tables/{model}_metrikler.csv`, `{model}_hatalar.csv`, `{model}_karisiklik.csv`, `results/tables/ozet_metrikler.csv`, `results/figures/cm_{model}.png` |
| `06_karsilastirma.ipynb` | Bilge | `results/tables/karsilastirma_tablosu.csv`, `results/figures/karsilastirma_macro_f1.png`, `karsilastirma_sinif_hatalari.png`, `karsilastirma_sinif_f1.png` |

Not: BERT eğitimi T4 GPU'da dil başına yaklaşık 11 dakika sürer. Kodu hızlıca denemek için `03_bert_finetune.ipynb` içindeki `DEBUG = True` ayarı küçük bir örneklemle çalışır.
