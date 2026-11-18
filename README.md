# BİM 448 Sentiment Analysis Project

Bu proje, müşteri yorumları üzerinde duygu analizi (sentiment analysis) gerçekleştirmek amacıyla BERT, VADER ve LLM yaklaşımlarını karşılaştırmalı olarak incelemektedir.

## 📁 Tahmin Dosyaları (Predictions)
Proje kapsamında üretilen ve `predictions/` klasöründe yer alan modeller:
- **BERT Modelleri**: `bert_en.csv`, `bert_tr.csv`
- **VADER Modelleri**: `vader_en.csv`, `vader_tr_direct.csv`, `vader_tr_mt.csv`
- **LLM Yaklaşımları**: `llm_en.csv`, `llm_tr_partial.csv` vb.

## 🚀 Kullanılan Teknolojiler
- Python, Pandas, NumPy
- Hugging Face Transformers (`roberta-base`, `bert-base-turkish-cased`)
- Scikit-learn (Metrik değerlendirmeleri)
- Google Colab & GitHub
