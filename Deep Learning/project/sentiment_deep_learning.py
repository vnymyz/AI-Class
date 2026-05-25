"""
Deep Learning Sentiment Analysis — 3-Stage Progression
Stage 1: MLP (your existing ML knowledge + neural net)
Stage 2: LSTM (sequence modeling)
Stage 3: DistilBERT fine-tuning (transformer + transfer learning)

Dataset: IMDb movie reviews (binary: positive / negative)
Install: pip install tensorflow scikit-learn transformers datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# SHARED UTILS
# ─────────────────────────────────────────────

def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_training.png')
    plt.show()


# ─────────────────────────────────────────────
# STAGE 1: MLP with TF-IDF
# Connects your existing sklearn knowledge to neural nets
# ─────────────────────────────────────────────

def stage1_mlp():
    print("\n" + "="*50)
    print("STAGE 1: MLP + TF-IDF")
    print("="*50)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts  = dataset['test']['text']
    test_labels  = dataset['test']['label']

    # TF-IDF (you already know this from ML)
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts).toarray().astype('float32')
    X_test  = vectorizer.transform(test_texts).toarray().astype('float32')
    y_train = np.array(train_labels)
    y_test  = np.array(test_labels)

    # MLP model — same input shape as TF-IDF vector
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(10_000,)),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='MLP_TFIDF')

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    plot_history(history, "Stage 1 MLP")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['negative', 'positive']))

    return acc


# ─────────────────────────────────────────────
# STAGE 2: LSTM with word embeddings
# Learns word representations + sequence order
# ─────────────────────────────────────────────

def stage2_lstm():
    print("\n" + "="*50)
    print("STAGE 2: LSTM + Word Embeddings")
    print("="*50)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    train_texts = dataset['train']['text']
    train_labels = np.array(dataset['train']['label'])
    test_texts  = dataset['test']['text']
    test_labels  = np.array(dataset['test']['label'])

    VOCAB_SIZE = 20_000
    MAX_LEN    = 256
    EMBED_DIM  = 128

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, truncating='post')
    X_test  = pad_sequences(tokenizer.texts_to_sequences(test_texts),  maxlen=MAX_LEN, truncating='post')

    # LSTM model
    # Embedding: integer token → dense vector (learned during training)
    # LSTM: processes sequence, maintains hidden state
    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='LSTM_Embedding')

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    reduce_lr  = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)

    history = model.fit(
        X_train, train_labels,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    plot_history(history, "Stage 2 LSTM")

    loss, acc = model.evaluate(X_test, test_labels, verbose=0)
    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(test_labels, preds, target_names=['negative', 'positive']))

    return acc


# ─────────────────────────────────────────────
# STAGE 3: DistilBERT fine-tuning
# Transfer learning: pre-trained transformer + your task
# ─────────────────────────────────────────────

def stage3_distilbert():
    print("\n" + "="*50)
    print("STAGE 3: DistilBERT Fine-tuning")
    print("="*50)

    import tensorflow as tf
    from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
    from datasets import load_dataset

    dataset = load_dataset("imdb")

    # Use subset for speed (remove [:2000] for full training)
    train_data = dataset['train'].shuffle(seed=42).select(range(2000))
    test_data  = dataset['test'].shuffle(seed=42).select(range(500))

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    train_enc = train_data.map(tokenize, batched=True)
    test_enc  = test_data.map(tokenize, batched=True)

    # Convert to TF datasets
    def to_tf_dataset(hf_dataset, batch_size=16, shuffle=False):
        tf_data = hf_dataset.to_tf_dataset(
            columns=['input_ids', 'attention_mask'],
            label_cols=['label'],
            batch_size=batch_size,
            shuffle=shuffle
        )
        return tf_data

    tf_train = to_tf_dataset(train_enc, shuffle=True)
    tf_test  = to_tf_dataset(test_enc)

    # Load pre-trained DistilBERT with classification head
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Fine-tune with small learning rate (important for transfer learning)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    history = model.fit(tf_train, epochs=3, validation_data=tf_test)

    results = model.evaluate(tf_test)
    print(f"\nDistilBERT Test Accuracy: {results[1]:.4f}")

    return results[1]


# ─────────────────────────────────────────────
# COMPARE ALL THREE STAGES
# ─────────────────────────────────────────────

def compare_all():
    results = {}
    results['Stage 1 — MLP + TF-IDF']        = stage1_mlp()
    results['Stage 2 — LSTM + Embeddings']    = stage2_lstm()
    results['Stage 3 — DistilBERT Finetune']  = stage3_distilbert()

    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    for name, acc in results.items():
        bar = "█" * int(acc * 40)
        print(f"{name:<35} {acc:.4f}  {bar}")


if __name__ == '__main__':
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if stage == '1':
        stage1_mlp()
    elif stage == '2':
        stage2_lstm()
    elif stage == '3':
        stage3_distilbert()
    else:
        compare_all()
