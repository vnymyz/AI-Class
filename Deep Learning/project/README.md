# Proyek: Analisis Sentimen Deep Learning

Review film IMDb — klasifikasi biner (positif / negatif).

## Struktur File

```
project/
├── sentiment_eda_preprocessing.ipynb   ← Langkah 1: EDA + pembersihan teks
├── utils.py                            ← Fungsi pembersihan yang dipakai bersama
├── app.py                              ← Langkah 2: Aplikasi interaktif Streamlit
└── README.md
```

## Urutan yang Disarankan

```
1. Jalankan notebook  → pahami data, hasilkan CSV bersih + artefak
2. Jalankan aplikasi  → coba model yang sudah dilatih, bandingkan arsitekturnya
```

---

## Langkah 1 — Notebook: EDA & Preprocessing

```bash
jupyter notebook sentiment_eda_preprocessing.ipynb
```

**Apa yang dipelajari:**
| Bagian | Yang Kamu Pelajari |
|--------|-------------------|
| Keseimbangan kelas | Apakah dataset tidak seimbang? Perlu class weights? |
| Panjang review | Berapa `max_len` yang harus dipakai untuk model? |
| Contoh teks mentah | Apa saja noise yang ada? HTML, angka, tanda baca |
| Kata terbanyak (sebelum dibersihkan) | Stopwords mendominasi — tidak berguna untuk model |
| Pembersihan teks langkah demi langkah | Tiap fungsi pembersih dan alasannya |
| Kata terbanyak (sesudah dibersihkan) | Sekarang kita lihat sinyal sentimen: good/great vs bad/awful |
| WordCloud | Peta frekuensi visual per kelas |
| Simpan artefak | tfidf_vectorizer.pkl, keras_tokenizer.json untuk app.py |

**Pipeline pembersihan di utils.py:**
```
huruf kecil → hapus HTML → hapus URL → hapus tanda baca
           → hapus angka → hapus stopword → rapikan spasi
```

---

## Langkah 2 — Aplikasi Streamlit

```bash
pip install streamlit tensorflow scikit-learn transformers datasets matplotlib seaborn wordcloud
streamlit run app.py
```

**Fitur aplikasi:**

| Tab | Isi |
|-----|-----|
| 🔮 Predict | Masukkan review apa saja, bandingkan 3 model secara berdampingan |
| 📊 Explore | EDA interaktif — keseimbangan kelas, frekuensi kata, contoh data |
| 📈 Training Curves | Kurva loss/akurasi yang menunjukkan bagaimana model belajar |
| 📚 Theory | Penjelasan arsitektur tiap model |

**Model yang tersedia:**

| Tahap | Model | Konsep Utama |
|-------|-------|-------------|
| 1 | MLP + TF-IDF | Neural net, backprop, dropout |
| 2 | BiLSTM + Embeddings | Word embedding, LSTM gate, pemodelan sekuens |
| 3 | DistilBERT (pre-trained) | Transformer, attention, transfer learning |

---

## Akurasi yang Diharapkan (2000 data train / 500 test)

| Model | Akurasi | Waktu Training |
|-------|---------|----------------|
| MLP + TF-IDF | ~85–88% | ~1 menit |
| BiLSTM | ~87–90% | ~3 menit |
| DistilBERT | ~92–94% | 0 (sudah pre-trained) |

---

## Latihan

1. Ubah `n_train` di sidebar — amati bagaimana akurasi berubah seiring bertambahnya data
2. Coba review yang "ambigu" — apakah model berbeda pendapat?
3. Di `app.py`: ubah `EMBED_DIM = 128` jadi `64` di LSTM — apakah lebih cepat? Akurasi turun?
4. Di `app.py`: hapus wrapper `Bidirectional()` dari LSTM — bandingkan akurasinya
5. Di notebook: coba TIDAK hapus stopwords — bagaimana kata terbanyak berubah?
