"""
Streamlit app — Deep Learning Sentiment Analysis
Stages: MLP → LSTM → DistilBERT

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, pickle, os, time

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Deep Learning Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy imports to keep startup fast ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tensorflow():
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras

@st.cache_resource(show_spinner=False)
def load_transformers():
    from transformers import pipeline
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading IMDb dataset...")
def load_data(n_train=5000, n_test=1000):
    """
    Load IMDb from HuggingFace and apply text cleaning.
    Cached so this only runs once per session.
    n_train/n_test: subset size to keep the app fast on CPU.
    """
    from datasets import load_dataset
    from utils import clean_text

    raw = load_dataset("imdb")

    def to_df(split, n):
        df = pd.DataFrame(raw[split]).sample(n, random_state=42).reset_index(drop=True)
        df['clean_text'] = df['text'].apply(clean_text)
        df['sentiment']  = df['label'].map({0: 'Negative', 1: 'Positive'})
        return df

    return to_df('train', n_train), to_df('test', n_test)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: MLP + TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training MLP model...")
def build_and_train_mlp(_df_train, _df_test):
    """
    Stage 1: Multi-Layer Perceptron with TF-IDF features.

    Architecture:
      TF-IDF (10k features) → Dense(256, ReLU) → Dropout(0.4)
                            → Dense(128, ReLU) → Dropout(0.3)
                            → Dense(64, ReLU)  → Dense(1, Sigmoid)

    Input:  TF-IDF sparse vector (bag-of-words with tf-idf weights)
    Output: probability that review is positive [0, 1]
    """
    tf, keras = load_tensorflow()
    from tensorflow.keras import layers
    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF converts text → numeric vector based on word frequency
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(_df_train['clean_text']).toarray().astype('float32')
    X_test  = vectorizer.transform(_df_test['clean_text']).toarray().astype('float32')
    y_train = _df_train['label'].values
    y_test  = _df_test['label'].values

    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(10_000,)),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='MLP_TF-IDF')

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=15, batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, vectorizer, history, acc


def predict_mlp(text, model, vectorizer):
    """
    Predict using MLP:
    1. Clean text
    2. TF-IDF transform
    3. Forward pass through MLP
    """
    from utils import clean_text
    tf, keras = load_tensorflow()

    clean = clean_text(text)
    vec   = vectorizer.transform([clean]).toarray().astype('float32')
    prob  = float(model.predict(vec, verbose=0)[0][0])
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LSTM + Word Embeddings
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training LSTM model...")
def build_and_train_lstm(_df_train, _df_test):
    """
    Stage 2: Bidirectional LSTM with learned word embeddings.

    Architecture:
      Token IDs → Embedding(20k vocab, 128 dim)
               → SpatialDropout(0.2)
               → Bidirectional LSTM(64)    ← reads sequence forward AND backward
               → Dropout(0.3)
               → Dense(1, Sigmoid)

    Key concepts:
    - Embedding: each word gets a 128-dim vector, trained end-to-end
    - Bidirectional: LSTM processes left→right AND right→left, doubles context
    - LSTM gates: forget/input/output gates prevent vanishing gradient
    """
    tf, keras = load_tensorflow()
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    VOCAB_SIZE = 20_000
    MAX_LEN    = 256
    EMBED_DIM  = 128

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(_df_train['clean_text'])

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, truncating='post', padding='post')

    X_train = encode(_df_train['clean_text'])
    X_test  = encode(_df_test['clean_text'])
    y_train = _df_train['label'].values
    y_test  = _df_test['label'].values

    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        layers.SpatialDropout1D(0.2),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM_Embedding')

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    reduce_lr  = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, verbose=0)

    history = model.fit(
        X_train, y_train,
        epochs=10, batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, tokenizer, MAX_LEN, history, acc


def predict_lstm(text, model, tokenizer, max_len):
    """
    Predict using LSTM:
    1. Clean text
    2. Tokenize → integer sequence
    3. Pad to max_len
    4. Forward pass through Embedding + LSTM
    """
    from utils import clean_text
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    clean = clean_text(text)
    seq   = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, truncating='post', padding='post')
    prob  = float(model.predict(padded, verbose=0)[0][0])
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: DistilBERT (HuggingFace pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DistilBERT (pre-trained)...")
def load_distilbert():
    """
    Stage 3: Pre-trained DistilBERT from HuggingFace.

    Using 'distilbert-base-uncased-finetuned-sst-2-english':
    - Pre-trained on BookCorpus + Wikipedia (110M parameters)
    - Already fine-tuned on SST-2 sentiment task
    - 97MB download, cached after first run

    No training needed — this model is already production-quality.
    This demonstrates why transfer learning is powerful:
    we get a better model with zero training effort.
    """
    pipeline = load_transformers()
    return pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',
        truncation=True,
        max_length=512
    )


def predict_distilbert(text, pipe):
    """
    Predict using DistilBERT pipeline.
    Returns probability of positive class.
    """
    result = pipe(text[:512])[0]   # truncate to 512 chars for speed
    label  = result['label']       # 'POSITIVE' or 'NEGATIVE'
    score  = result['score']       # confidence of predicted label

    # Normalize: return P(positive)
    return score if label == 'POSITIVE' else 1.0 - score


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_badge(prob):
    """Return emoji + label + color based on probability."""
    if prob >= 0.65:
        return "POSITIVE", "#2ecc71", "😊"
    elif prob <= 0.35:
        return "NEGATIVE", "#e74c3c", "😞"
    else:
        return "NEUTRAL", "#f39c12", "😐"


def generate_reply(prob):
    """Return an automated reply text based on predicted sentiment."""
    if prob >= 0.65:
        return (
            "POSITIVE",
            "😊 Thank you so much for your wonderful feedback! "
            "We're truly glad you had such a great experience. "
            "Your kind words mean the world to us — we hope to see you again soon!"
        )
    elif prob <= 0.35:
        return (
            "NEGATIVE",
            "😔 We're truly sorry to hear that your experience didn't meet your expectations. "
            "Your feedback is valuable and helps us improve. "
            "Please don't hesitate to reach out to us directly — we'd love the chance to make things right."
        )
    else:
        return (
            "NEUTRAL",
            "🙏 Thank you for taking the time to share your thoughts with us. "
            "We appreciate your honest feedback and will use it to keep improving your experience."
        )


def render_prediction_card(stage_name, prob, train_acc, inference_time, description):
    """Render a model result card with probability bar."""
    label, color, emoji = sentiment_badge(prob)

    st.markdown(f"### {stage_name}")
    st.caption(description)

    # Probability bar
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(prob)
        neg_pct = f"{(1-prob)*100:.1f}% Negative"
        pos_pct = f"{prob*100:.1f}% Positive"
        st.caption(f"{neg_pct} ◄──► {pos_pct}")
    with col2:
        st.markdown(
            f"<div style='text-align:center; background:{color}; color:white; "
            f"padding:8px; border-radius:8px; font-weight:bold'>"
            f"{emoji} {label}</div>",
            unsafe_allow_html=True
        )

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")
    m2.metric("Test Accuracy", f"{train_acc*100:.1f}%" if train_acc else "N/A")
    m3.metric("Inference Time", f"{inference_time:.2f}s")


def plot_training_history(history, title):
    """Plot training and validation loss/accuracy curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(history.history['loss'],     label='Train Loss',  color='#3498db')
    ax1.plot(history.history['val_loss'], label='Val Loss',    color='#e74c3c', linestyle='--')
    ax1.set_title(f'{title} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'],     label='Train Acc',  color='#3498db')
    ax2.plot(history.history['val_accuracy'], label='Val Acc',    color='#e74c3c', linestyle='--')
    ax2.set_title(f'{title} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("Model Selection")
        use_mlp   = st.checkbox("Stage 1 — MLP + TF-IDF", value=True)
        use_lstm  = st.checkbox("Stage 2 — BiLSTM + Embeddings", value=True)
        use_bert  = st.checkbox("Stage 3 — DistilBERT", value=False,
                                help="~97MB download. Slower but most accurate.")

        st.divider()
        st.subheader("Dataset Size")
        n_train = st.select_slider("Training samples", [1000, 2000, 5000, 10000], value=2000,
                                   help="More = better models but slower training")
        n_test  = st.select_slider("Test samples",     [200, 500, 1000], value=500)

        st.divider()
        st.markdown("""
        **How to use:**
        1. Select models above
        2. Paste/type a movie review
        3. Click **Analyze**
        4. Compare predictions

        **Learning path:**
        - Stage 1 → understand neural nets
        - Stage 2 → understand sequences
        - Stage 3 → understand transfer learning
        """)

    # ── Main header ─────────────────────────────────────────────────────────
    st.title("🎬 Deep Learning Sentiment Analysis")
    st.markdown("""
    Compare three deep learning approaches on the same review.
    Each stage adds a key concept from the deep learning roadmap.
    """)

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_predict, tab_explore, tab_training, tab_theory = st.tabs([
        "🔮 Predict", "📊 Explore Dataset", "📈 Training Curves", "📚 Theory"
    ])

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset..."):
        df_train, df_test = load_data(n_train, n_test)

    # ── Load / train models based on sidebar selection ───────────────────────
    mlp_model = lstm_model = bert_pipe = None
    mlp_acc   = lstm_acc   = None
    mlp_hist  = lstm_hist  = None

    if use_mlp:
        mlp_model, mlp_vectorizer, mlp_hist, mlp_acc = build_and_train_mlp(df_train, df_test)

    if use_lstm:
        lstm_model, lstm_tokenizer, lstm_maxlen, lstm_hist, lstm_acc = build_and_train_lstm(df_train, df_test)

    if use_bert:
        bert_pipe = load_distilbert()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: PREDICT
    # ════════════════════════════════════════════════════════════════════════
    with tab_predict:
        st.subheader("Enter a Movie Review")

        # Example reviews
        examples = {
            "😊 Clear Positive": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. One of the best films I've seen this year.",
            "😞 Clear Negative": "Terrible waste of time. The script was awful, acting was wooden, and the story made no sense. I walked out after 30 minutes.",
            "😐 Ambiguous":       "The movie had some good moments but also felt slow at times. The lead performance was decent, though supporting characters were underdeveloped.",
        }

        col_input, col_example = st.columns([2, 1])

        with col_example:
            st.markdown("**Try an example:**")
            for label, text in examples.items():
                if st.button(label, use_container_width=True):
                    st.session_state['review_text'] = text

        with col_input:
            review = st.text_area(
                "Movie review",
                value=st.session_state.get('review_text', ''),
                height=160,
                placeholder="Paste or type a movie review here...",
                label_visibility="collapsed"
            )
            word_count = len(review.split()) if review.strip() else 0
            st.caption(f"{word_count} words")

        analyze = st.button("🔍 Analyze Sentiment", type="primary",
                            disabled=not review.strip())

        if analyze and review.strip():
            st.divider()
            st.subheader("Results")

            if not any([use_mlp, use_lstm, use_bert]):
                st.warning("Select at least one model in the sidebar.")
            else:
                cols = st.columns(sum([use_mlp, use_lstm, use_bert]))
                col_idx = 0

                if use_mlp and mlp_model:
                    with cols[col_idx]:
                        t0 = time.time()
                        prob = predict_mlp(review, mlp_model, mlp_vectorizer)
                        elapsed = time.time() - t0
                        render_prediction_card(
                            "Stage 1 — MLP + TF-IDF", prob, mlp_acc, elapsed,
                            "Bag-of-words features → dense neural net"
                        )
                    col_idx += 1

                if use_lstm and lstm_model:
                    with cols[col_idx]:
                        t0 = time.time()
                        prob = predict_lstm(review, lstm_model, lstm_tokenizer, lstm_maxlen)
                        elapsed = time.time() - t0
                        render_prediction_card(
                            "Stage 2 — BiLSTM", prob, lstm_acc, elapsed,
                            "Learned embeddings → bidirectional LSTM"
                        )
                    col_idx += 1

                if use_bert and bert_pipe:
                    with cols[col_idx]:
                        t0 = time.time()
                        prob = predict_distilbert(review, bert_pipe)
                        elapsed = time.time() - t0
                        render_prediction_card(
                            "Stage 3 — DistilBERT", prob, None, elapsed,
                            "Pre-trained transformer, fine-tuned on SST-2"
                        )

                # ── Auto-Reply Section ────────────────────────────────────
                st.divider()
                st.subheader("💬 Automated Reply")

                # Use the best available model's probability (priority: BERT > LSTM > MLP)
                reply_prob = None
                reply_source = ""
                if use_bert and bert_pipe:
                    reply_prob   = predict_distilbert(review, bert_pipe)
                    reply_source = "DistilBERT"
                elif use_lstm and lstm_model:
                    reply_prob   = predict_lstm(review, lstm_model, lstm_tokenizer, lstm_maxlen)
                    reply_source = "BiLSTM"
                elif use_mlp and mlp_model:
                    reply_prob   = predict_mlp(review, mlp_model, mlp_vectorizer)
                    reply_source = "MLP"

                if reply_prob is not None:
                    sentiment_label, reply_text = generate_reply(reply_prob)
                    _, badge_color, badge_emoji = sentiment_badge(reply_prob)

                    st.caption(f"Generated based on **{reply_source}** prediction — {badge_emoji} {sentiment_label}")
                    st.markdown(
                        f"<div style='"
                        f"border-left: 4px solid {badge_color}; "
                        f"background: {badge_color}18; "
                        f"padding: 14px 18px; "
                        f"border-radius: 6px; "
                        f"font-size: 1.05rem; "
                        f"color: inherit;'>"
                        f"{reply_text}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Allow user to copy the reply text
                    with st.expander("📋 Copy reply text"):
                        st.code(reply_text, language=None)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: EXPLORE DATASET
    # ════════════════════════════════════════════════════════════════════════
    with tab_explore:
        import matplotlib.pyplot as plt

        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Train samples", f"{len(df_train):,}")
        col2.metric("Test samples",  f"{len(df_test):,}")
        col3.metric("Classes", "2 (Positive / Negative)")

        st.subheader("Class Balance")
        counts = df_train['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='black', rot=0)
        ax.set_ylabel("Count")
        ax.set_title("Training Set — Class Distribution")
        for bar, count in zip(ax.patches, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count:,}', ha='center')
        st.pyplot(fig)

        st.subheader("Review Length Distribution")
        df_train['word_count'] = df_train['clean_text'].str.split().str.len()
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        for label, color in [('Positive', '#2ecc71'), ('Negative', '#e74c3c')]:
            subset = df_train[df_train['sentiment'] == label]['word_count']
            ax2.hist(subset.clip(upper=500), bins=40, alpha=0.6, label=label, color=color)
        ax2.axvline(256, color='navy', linestyle='--', label='max_len=256')
        ax2.set_xlabel("Word Count (after cleaning)")
        ax2.set_ylabel("Reviews")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Top Words After Cleaning")
        from collections import Counter
        col_pos, col_neg = st.columns(2)

        for col, label, color, title in [
            (col_pos, 1, '#2ecc71', "Top 15 Positive"),
            (col_neg, 0, '#e74c3c', "Top 15 Negative")
        ]:
            words = ' '.join(df_train[df_train['label'] == label]['clean_text']).split()
            top   = Counter(words).most_common(15)
            w, c  = zip(*top)
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.barh(w[::-1], c[::-1], color=color, alpha=0.8, edgecolor='black')
            ax3.set_title(title, fontweight='bold')
            col.pyplot(fig3)

        st.subheader("Sample Reviews")
        n_show = st.slider("Number of samples to show", 3, 15, 5)
        st.dataframe(
            df_train[['sentiment', 'clean_text', 'word_count']].sample(n_show, random_state=42),
            use_container_width=True
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: TRAINING CURVES
    # ════════════════════════════════════════════════════════════════════════
    with tab_training:
        st.subheader("Training History")
        st.markdown("""
        These curves show how the model improved during training.
        - **Gap between train and val**: overfitting — model memorizes training data
        - **Val loss going up while train goes down**: stop training here (early stopping)
        - **Both decreasing**: healthy learning
        """)

        if mlp_hist:
            st.markdown("#### Stage 1 — MLP")
            st.pyplot(plot_training_history(mlp_hist, "MLP"))
            st.caption(f"Final test accuracy: **{mlp_acc*100:.1f}%**")

        if lstm_hist:
            st.markdown("#### Stage 2 — BiLSTM")
            st.pyplot(plot_training_history(lstm_hist, "BiLSTM"))
            st.caption(f"Final test accuracy: **{lstm_acc*100:.1f}%**")

        if use_bert:
            st.info("DistilBERT is loaded pre-trained — no training curves (already trained on 3.3B words).")

        if not mlp_hist and not lstm_hist:
            st.info("Enable models in the sidebar to see training curves.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4: THEORY
    # ════════════════════════════════════════════════════════════════════════
    with tab_theory:
        st.subheader("Model Architecture Explanations")

        with st.expander("Stage 1 — MLP + TF-IDF", expanded=True):
            st.markdown("""
            **TF-IDF** converts text to a fixed-size vector where each dimension = a word's importance score.
            - TF (Term Frequency): how often the word appears in this review
            - IDF (Inverse Document Frequency): penalizes words that appear in every review

            **MLP** (Multi-Layer Perceptron):
            - Stack of Dense layers, each learning a linear transformation + non-linearity
            - Forward pass: `output = ReLU(W·x + b)`
            - Backprop: gradient of loss flows back through every layer to update weights

            **Limitation:** TF-IDF loses word order.
            `"not good"` and `"good not"` produce the same vector.
            """)

        with st.expander("Stage 2 — Bidirectional LSTM"):
            st.markdown("""
            **Word Embeddings:** each word → learned 128-dimensional vector.
            Similar words cluster together (king ≈ queen, good ≈ great).

            **LSTM Gates:**
            - Forget gate: decides what to erase from memory
            - Input gate: decides what new info to store
            - Output gate: decides what to output as hidden state

            **Bidirectional:** runs LSTM left→right AND right→left, then concatenates.
            Captures both past and future context for each word.

            **Improvement over MLP:** understands that word order matters.
            `"not good"` ≠ `"good not"` because LSTM processes them sequentially.
            """)

        with st.expander("Stage 3 — DistilBERT"):
            st.markdown("""
            **Transformer self-attention:** every word attends to every other word simultaneously.
            ```
            Attention(Q, K, V) = softmax(QKᵀ / √d) × V
            ```
            `"bank"` in `"river bank"` vs `"bank account"` → different attention patterns.

            **Pre-training:** DistilBERT was trained on 3.3 billion words using:
            - Masked Language Modeling: predict hidden words from context
            - Knowledge Distillation from BERT (40% smaller, 60% faster, 97% accuracy)

            **Transfer learning:** we reuse all learned knowledge, just swap the output head
            for sentiment classification. Fine-tuning only takes minutes.

            **Why it wins:** it already "understands" language before seeing your data.
            """)

        st.divider()
        st.markdown("""
        **Full theory reference:** see `deep-learning.md` in the parent folder.

        **Next steps after this project:**
        - Try a CNN-based text classifier (1D convolutions over word embeddings)
        - Fine-tune DistilBERT with your own training loop (not pipeline)
        - Build an image classifier with CNN (different domain, same concepts)
        """)


if __name__ == '__main__':
    main()
