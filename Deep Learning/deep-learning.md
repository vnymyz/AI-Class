# Deep Learning — Teori & Peta Belajar

> Prasyarat: kamu sudah paham ML supervised/unsupervised, NLP dasar, Python + NumPy/Pandas/Sklearn.

---

## 1. Apa Itu Deep Learning?

Machine learning biasa: kita buat fitur sendiri → model belajar batas keputusan.
Deep learning **belajar fiturnya sendiri** lewat banyak lapisan transformasi.

```
ML Biasa:      Data Mentah → [Fitur Manual] → Model → Output
Deep Learning: Data Mentah → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
                                ↑ fitur dipelajari otomatis
```

"Deep" = banyak lapisan. Makin banyak lapisan = representasi makin abstrak.

---

## 2. Neuron Buatan

Satu neuron menghitung ini:

```
output = aktivasi( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
       = aktivasi( W·x + b )
```

- `x` = input
- `W` = bobot (dipelajari model)
- `b` = bias (dipelajari model)
- `aktivasi` = fungsi non-linear

Tanpa fungsi aktivasi non-linear, menumpuk banyak layer tidak ada gunanya (linear × linear = tetap linear).

---

## 3. Fungsi Aktivasi

| Fungsi | Rumus | Kapan Dipakai |
|--------|-------|---------------|
| **ReLU** | `max(0, x)` | Default untuk hidden layer |
| **Sigmoid** | `1 / (1 + e^-x)` | Output biner (0–1) |
| **Softmax** | `e^xᵢ / Σe^xⱼ` | Output multi-kelas (probabilitas) |
| **Tanh** | `(eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)` | RNN, rentang nilai (-1, 1) |
| **Leaky ReLU** | `max(0.01x, x)` | Mengatasi masalah "ReLU mati" |

**Kenapa ReLU populer di hidden layer:** mudah dihitung, tidak jenuh untuk nilai positif, gradien mengalir dengan baik.

---

## 4. Arsitektur Neural Network

```
Input Layer → Hidden Layer(s) → Output Layer

[x₁]          [h₁]  [h₁]       [ŷ₁]
[x₂]    →    [h₂]  [h₂]  →   [ŷ₂]
[x₃]          [h₃]  [h₃]       
```

- **Input layer**: satu node per fitur
- **Hidden layer**: tempat model belajar
- **Output layer**: tergantung tugasnya
  - Regresi: 1 node, tanpa aktivasi (atau linear)
  - Klasifikasi biner: 1 node, sigmoid
  - Multi-kelas: N node, softmax

---

## 5. Loss Function

Loss mengukur seberapa salah prediksi model. Training = memperkecil loss.

| Tugas | Loss Function | Rumus |
|-------|--------------|-------|
| Regresi | MSE | `mean((ŷ - y)²)` |
| Klasifikasi Biner | Binary Cross-Entropy | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` |
| Klasifikasi Multi-kelas | Categorical Cross-Entropy | `-Σ yᵢ·log(ŷᵢ)` |

---

## 6. Backpropagation & Gradient Descent

### Masalah Utama

Neural network mulai dengan **bobot acak**. Bobot acak → prediksi salah.
Training = mengatur semua bobot supaya loss mengecil.

Tapi satu jaringan bisa punya **jutaan bobot**. Bagaimana tahu arah yang benar untuk tiap bobot?
Itulah yang diselesaikan backpropagation + gradient descent — bersama-sama.

---

### Gradient Descent — menentukan "arah yang benar"

**Gradient** memberi tahu kemiringan loss terhadap suatu bobot:
- Gradient positif → menaikkan bobot ini membuat loss **naik** → turunkan bobotnya
- Gradient negatif → menaikkan bobot ini membuat loss **turun** → naikkan bobotnya

Aturan update selalu: gerakkan bobot **berlawanan** dengan gradientnya:

```
W = W - lr × ∂Loss/∂W
```

Bayangkan loss sebagai lanskap berbukit. Model adalah bola yang duduk di suatu tempat.
Gradient descent memberi tahu bola: "menggelindinglahlah ke bawah." Lakukan cukup sering, bola mencapai lembah (loss rendah).

```
Loss
 │     *
 │   *   *
 │ *       *        ← gradient: kemiringan kurva ini di W sekarang
 │           *    *
 └──────────────────── W
               ↑
           kita di sini → gradient negatif → naikkan W → gerak kanan menuju lembah
```

**Learning rate `lr`** mengontrol ukuran langkah:
```
lr terlalu besar  → melompati lembah, mental-mental atau divergen
lr terlalu kecil  → butuh waktu sangat lama untuk sampai lembah
lr pas            → konvergen dengan stabil
```

---

### Backpropagation — mesin penghitung gradient

Gradient descent butuh `∂Loss/∂W` untuk **setiap bobot** di jaringan.
Menghitung ini satu per satu akan sangat mahal.

Backpropagation menyelesaikan ini dengan **chain rule** dari kalkulus:
jika Loss bergantung pada W lewat beberapa langkah, kalikan gradientnya sepanjang jalur tersebut.

```
Jaringan:  Input → [Layer 1: W₁] → [Layer 2: W₂] → Prediksi → Loss
```

Untuk tahu bagaimana W₁ mempengaruhi Loss, telusuri rantainya:

```
∂Loss/∂W₁  =  ∂Loss/∂ŷ  ×  ∂ŷ/∂h  ×  ∂h/∂W₁
                  ↑              ↑           ↑
           (output→loss)  (layer2→output)  (W₁→layer1)
```

Backprop melakukan ini **secara efisien dengan menggunakan ulang perhitungan**: gradient dihitung dari layer output mundur ke layer input, menggunakan ulang tiap hasil antara.

---

### Siklus Training Lengkap

```
untuk setiap batch data:

    1. FORWARD PASS
       ┌─────────────────────────────────────────────────┐
       │  X → Layer1 → Layer2 → ... → ŷ → Loss(ŷ, y)   │
       └─────────────────────────────────────────────────┘
       Tujuan: dapat prediksi dan ukur seberapa salah

    2. BACKWARD PASS (Backpropagation)
       ┌──────────────────────────────────────────────────────────┐
       │  Loss → ...∂L/∂W_last → ∂L/∂W_second → ∂L/∂W_first    │
       └──────────────────────────────────────────────────────────┘
       Tujuan: hitung gradient loss terhadap setiap bobot

    3. UPDATE BOBOT (Gradient Descent)
       ┌──────────────────────────────────┐
       │  W = W - lr × gradient           │
       └──────────────────────────────────┘
       Tujuan: geser setiap bobot ke arah yang mengurangi loss

    Ulangi ribuan batch → loss mengecil → model membaik
```

---

### Kenapa Backprop Butuh Fungsi Aktivasi

Ingat: tanpa aktivasi non-linear, banyak layer tidak ada bedanya dengan satu persamaan linear.
Backprop juga butuh aktivasi punya **gradient tidak nol** supaya informasi bisa mengalir balik.

Itulah kenapa **ReLU mati** jadi masalah: kalau neuron selalu output 0, gradientnya 0,
dan backprop tidak bisa update bobot yang masuk ke neuron itu (gradient "mati").

---

### Analogi dengan ML yang Sudah Kamu Tahu

Di scikit-learn `LogisticRegression`, gradient descent berjalan di dalam — kamu hanya tidak melihatnya.
Deep learning membuatnya eksplisit dan menjalankannya pada jutaan parameter, bukan beberapa lusin.

```
Logistic Regression:  1 layer, ~N bobot  → gradient descent update semuanya
Neural Network:       K layer, jutaan bobot → backprop hitung semua gradient,
                                              gradient descent update semuanya
```

Backpropagation bukan "algoritma belajar" tersendiri — ini hanya cara efisien menghitung
gradient di sistem multi-layer agar gradient descent bisa bekerja.

---

## 7. Optimizer

| Optimizer | Ide Utama | Kapan Dipakai |
|-----------|----------|---------------|
| **SGD** | gradient descent biasa | baseline, dengan momentum |
| **Adam** | lr adaptif per parameter | pilihan default, kebanyakan tugas |
| **RMSprop** | lr adaptif, tanpa momentum | RNN |
| **AdamW** | Adam + weight decay | transformer, NLP modern |

**Rumus Adam (disederhanakan)**:
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ       ← momentum
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²      ← skala adaptif
W = W - lr × mₜ / (√vₜ + ε)
```

---

## 8. Overfitting & Regularisasi

Overfitting = model hafal data training, gagal di data baru.

### Teknik Mengatasinya:

**Dropout**
```python
# Saat training, matikan secara acak p% neuron
x = Dropout(0.3)(x)  # 30% dimatikan
```
Memaksa jaringan belajar representasi yang redundan (tidak bergantung satu neuron saja).

**Batch Normalization**
```python
x = BatchNormalization()(x)
```
Normalisasi input tiap layer → training lebih stabil, bisa pakai learning rate lebih besar.

**L2 Regularization (Weight Decay)**
```python
Dense(64, kernel_regularizer=l2(0.001))
```
Menghukum bobot yang terlalu besar di loss function.

**Early Stopping**
Hentikan training saat validation loss berhenti membaik.

---

## 9. Convolutional Neural Network (CNN)

Dibuat untuk **data berbentuk grid** (gambar, spektrogram).

### Operasi Utama:

**Konvolusi**: filter bergeser di atas input, mendeteksi pola lokal
```
Filter (3×3) mendeteksi tepi, sudut, tekstur, dll.
Ukuran output = (Input - Filter + 2×Padding) / Stride + 1
```

**Pooling**: memperkecil dimensi spasial
```
MaxPool(2×2): ambil nilai max tiap region 2×2 → lebar & tinggi jadi setengah
```

**Arsitektur CNN tipikal:**
```
Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output
         ↑ ekstraksi fitur ↑            ↑ klasifikasi ↑
```

Tiap blok conv belajar fitur yang makin abstrak:
- Layer 1: tepi, warna
- Layer 2: bentuk, tekstur
- Layer 3+: objek, konsep tingkat tinggi

---

## 10. Recurrent Neural Network (RNN/LSTM/GRU)

Dibuat untuk **data sekuensial** (teks, time series, audio).

### Masalah RNN Biasa

```
hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
```

Masalah: **vanishing gradient** — gradient mengecil saat backpropagasi mundur, sehingga dependensi jarak jauh terlupakan.

### LSTM (Long Short-Term Memory)

Mengatasi vanishing gradient dengan **gate**:

```
Forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)    ← apa yang dilupakan
Input gate:  iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)    ← apa yang ditambahkan
Output gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)    ← apa yang dikeluarkan
Cell state:  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ])
Hidden:      hₜ = oₜ⊙tanh(Cₜ)
```

`⊙` = perkalian elemen per elemen. Cell state `Cₜ` = jalur memori jangka panjang.

### GRU (Gated Recurrent Unit)

LSTM yang disederhanakan dengan 2 gate, bukan 3. Lebih cepat, sering sama bagusnya.

---

## 11. Transformer & Attention

Menggantikan RNN untuk sebagian besar tugas NLP (BERT, GPT, dll.).

### Self-Attention

Setiap token memperhatikan semua token lain secara bersamaan (tidak perlu diproses satu per satu).

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

Q = query (apa yang aku cari?)
K = key   (apa yang aku punya?)
V = value (apa yang aku kembalikan?)
```

### Kenapa Transformer Lebih Unggul dari LSTM

| | LSTM | Transformer |
|--|------|-------------|
| Dependensi jarak jauh | Susah (gate membantu) | Mudah (attention langsung) |
| Bisa diparalelkan | Tidak (sekuensial) | Ya |
| Skalabilitas | Terbatas | Sangat besar |
| Panjang konteks | ~100-500 token | Ribuan token |

### Komponen Utama Transformer
```
Input → Embedding → Positional Encoding
      → [Multi-Head Attention → Add&Norm → FFN → Add&Norm] × N
      → Output
```

---

## 12. Transfer Learning

Training dari nol butuh jutaan contoh data. Transfer learning menggunakan ulang model yang sudah dilatih.

```
Model Pre-trained (dilatih dengan data sangat besar)
        ↓
  Hapus layer terakhir
        ↓
  Tambahkan head baru untuk tugasmu
        ↓
  Fine-tune dengan dataset kecilmu
```

**Strategi:**
- **Feature extraction**: bekukan semua bobot pre-trained, hanya latih head baru
- **Fine-tuning**: buka kunci sebagian/semua layer, latih dengan learning rate kecil

**Model pre-trained populer:**
- Gambar: ResNet, EfficientNet, VGG (ImageNet)
- Teks: BERT, GPT-2, RoBERTa, DistilBERT (HuggingFace)

---

## 13. Tips Praktis Training

### Preprocessing Data
```python
# Gambar: normalisasi ke [0,1] atau [-1,1]
x = x / 255.0
# atau normalisasi pakai mean/std spesifik model

# Teks: tokenisasi, padding sekuens
```

### Batch Size
- Batch besar → training lebih cepat, butuh memori lebih besar, bisa kurang stabil
- Umum: 32–256 untuk gambar, 8–64 untuk transformer NLP

### Learning Rate Schedule
```python
# Warm-up lalu decay (umum untuk transformer)
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

# Atau pakai ReduceLROnPlateau: kurangi lr saat val_loss stagnan
```

### Memantau Training
Selalu plot:
- Training loss vs Validation loss (gap = overfitting)
- Training accuracy vs Validation accuracy

---

## 14. Framework yang Tersedia

| Framework | Terbaik Untuk |
|-----------|--------------|
| **Keras / TensorFlow** | Pemula, API produksi |
| **PyTorch** | Riset, fleksibel, standar industri |
| **HuggingFace Transformers** | NLP, model pre-trained |
| **FastAI** | Prototyping cepat di atas PyTorch |

---

## 15. Peta Belajar

```
Fase 1 — Fondasi (kamu ada di sini)
  ├── Perceptron → MLP → Backprop
  ├── Bangun neural net dari nol (NumPy)
  └── Dasar Keras/TensorFlow

Fase 2 — Arsitektur Inti
  ├── CNN untuk klasifikasi gambar
  ├── LSTM untuk sekuens/teks
  └── Transfer learning (fine-tune ResNet / BERT)

Fase 3 — DL Modern
  ├── Transformer & mekanisme Attention
  ├── Fine-tuning BERT untuk tugas NLP
  └── Diffusion model / GAN (generatif)

Fase 4 — Produksi
  ├── Simpan & deploy model (Flask, FastAPI)
  ├── ONNX / TensorFlow Lite (mobile/edge)
  └── MLflow / Weights & Biases (experiment tracking)
```

---

## 16. Proyek yang Direkomendasikan (Pemula → Menengah)

Lihat folder `project/` untuk implementasinya.

### Proyek: Analisis Sentimen Deep Learning + Jalur Upgrade

**Tahap 1** — MLP dengan fitur TF-IDF (menghubungkan pengetahuan ML yang sudah kamu miliki)
**Tahap 2** — LSTM dengan word embedding (pemodelan sekuens inti)
**Tahap 3** — Fine-tune DistilBERT (transformer + transfer learning)

Bandingkan ketiganya: akurasi, waktu training, kompleksitas.

Proyek ini mengajarkan progression MLP → RNN → Transformer pada tugas yang sudah kamu kenal.

---

## Cheatsheet Cepat

```python
# Keras MLP
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stop])

# Keras LSTM
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

# Keras CNN (gambar)
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(H, W, C)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
```
