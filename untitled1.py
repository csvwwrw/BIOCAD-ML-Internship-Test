# test_imports.py
import sys
import time

def test_import(module_name, import_statement):
    print(f'Testing: {module_name}...', end=' ', flush=True)
    start = time.time()
    try:
        exec(import_statement)
        elapsed = time.time() - start
        print(f'✓ OK ({elapsed:.2f}s)')
        return True
    except Exception as e:
        print(f'✗ ERROR: {e}')
        return False

# Test each import
imports = [
    ('logging', 'import logging'),
    ('pandas', 'import pandas as pd'),
    ('pathlib', 'from pathlib import Path'),
    ('matplotlib', 'import matplotlib.pyplot as plt'),
    ('textstat', 'import textstat'),
    ('numpy', 'import numpy as np'),
    ('nltk', 'import nltk'),
    ('nltk.tokenize', 'from nltk.tokenize import word_tokenize, sent_tokenize'),
    ('nltk.corpus', 'from nltk.corpus import stopwords'),
    ('collections', 'from collections import Counter'),
    ('sklearn.tfidf', 'from sklearn.feature_extraction.text import TfidfVectorizer'),
    ('nltk.ngrams', 'from nltk.util import ngrams'),
    ('wordcloud', 'from wordcloud import WordCloud'),
    ('sentence_transformers', 'from sentence_transformers import SentenceTransformer'),  # ← Может быть медленный
    ('sklearn.cosine', 'from sklearn.metrics.pairwise import cosine_similarity'),
    ('sklearn.TSNE', 'from sklearn.manifold import TSNE'),
    ('bertopic', 'from bertopic import BERTopic'),  # ← ВЕРОЯТНЫЙ ВИНОВНИК!
    ('seaborn', 'import seaborn as sns'),
    ('re', 'import re'),
]

print('='*60)
print('IMPORT TEST')
print('='*60)

for name, statement in imports:
    success = test_import(name, statement)
    if not success:
        print(f'\n⚠️ Import failed at: {name}')
        break
    # Если импорт занимает > 10 сек - вероятно фризит
    
print('\nIf any import takes >30 seconds, press Ctrl+C and report which one!')
