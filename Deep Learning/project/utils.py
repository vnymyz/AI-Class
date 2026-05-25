"""
Shared text cleaning utilities used by both the notebook and app.py
"""

import re
import string

# --- Optional: uncomment if using NLTK stopwords ---
# import nltk
# nltk.download('stopwords', quiet=True)
# from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words('english'))

# Minimal stopword list (no NLTK dependency)
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'he','him','his','she','her','hers','it','its','they','them','their',
    'what','which','who','this','that','these','those','am','is','are','was',
    'were','be','been','being','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','of','at','by','for','with','about',
    'to','from','in','out','on','off','so','no','nor','not','just','than',
    'too','very','can','will','should','now','s','t','ll','m','re','ve','d'
}


def remove_html(text: str) -> str:
    """Strip HTML tags — IMDb reviews often contain <br /> tags."""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Remove URLs."""
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers (keep alphanumeric words like '3D')."""
    return re.sub(r'\b\d+\b', '', text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces into one and strip edges."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text: str) -> str:
    """Remove common stopwords."""
    return ' '.join(w for w in text.split() if w not in STOPWORDS)


def clean_text(text: str, remove_stops: bool = True) -> str:
    """
    Full cleaning pipeline applied in order:
    1. Lowercase
    2. Remove HTML
    3. Remove URLs
    4. Remove punctuation
    5. Remove standalone numbers
    6. Remove stopwords (optional)
    7. Collapse extra whitespace
    """
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    if remove_stops:
        text = remove_stopwords(text)
    text = remove_extra_spaces(text)
    return text
