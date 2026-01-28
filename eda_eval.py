# -*- coding: utf-8 -*-
'''
Exploratory Data Analysis (EDA) for Alzheimer's Disease Corpus

This script performs comprehensive EDA including:
- Data quality analysis
- Descriptive statistics
- Lexical analysis (TF-IDF, n-grams, word clouds)
- Semantic analysis (embeddings, clustering, topic modeling)
- Biomedical NER (genes, proteins, drugs, diseases)
- RAG readiness assessment
- MeSH terms analysis
- Citation density analysis

Created on Tue Jan 27 16:49:44 2026

@author: csvww
'''

import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import textstat
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import umap
from sklearn.manifold import TSNE
from bertopic import BERTopic
import seaborn as sns
import re
import requests
from bertopic.representation import KeyBERTInspired
from rank_bm25 import BM25Okapi 
from typing import List, Counter as CounterType, Dict, Tuple
import networkx as nx
from matplotlib.patches import Patch
import json

# Setting up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('eda_report')
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 100
FILEPATH = 'alzheimer_corpus.csv'

# Scispacey check for bio ner (large package, thus optional, but I used in this EDA)
try:
    import spacy
    import scispacy
    logger.info('Loading scispacy models...')
    nlp_bio = spacy.load("en_ner_bionlp13cg_md")
    nlp_med = spacy.load("en_ner_bc5cdr_md")  
    SCISPACY_AVAILABLE = True
except:
    SCISPACY_AVAILABLE = False
    logger.warning('''Scispacy not available. Install:
                   pip install spacy'
                   pip install scispacy
                   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bio
                   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5
                   ''')
                   
ENTITY_STOPLIST = {
    'AD', 'ad', 'Alzheimer', 'Alzheimer\'s', 'Alzheimer\'s disease',
    'Alzheimer disease', 'alzheimer', 'ALZHEIMER', 'Alzheimers',
    'dementia', 'Dementia'
    }
         
# LOADING DATASET
def load_dataset(filepath=FILEPATH):
    '''
    Load the Alzheimer's research corpus from CSV file.
    
    Args
        filepath (str): Path to the corpus CSV file
        
    Returns
        df (pd.DataFrame): DataFrame containing the research articles with
            columns: PMID, PMC_ID, Title, Authors, Journal, Year, DOI,
            MeSH_Terms, PubMed_URL, Fetched, Abstract, Intro, Conclusion,
            Full_text
    '''
    
    logger.info(f'Loading dataset from {filepath}...')
    df = pd.read_csv(filepath)
    logger.info(f'Loaded dataset of {len(df)} articles')
    logger.info(f'Data avaliable: {df.columns.tolist()}')
    return df

# DATA QUALITY REPORT
# technically repeats most filters during articles retrieval
def data_quality_analysis(df):
    '''
    Perform comprehensive data quality analysis and generate report.
    
    Checks for:
    - Missing values across all columns
    - Empty strings in text fields
    - Duplicate PMIDs and titles
    - Non-English content (heuristic: Latin character ratio < 90%)
    
    Args:
        df (pd.DataFrame): Input DataFrame with research articles
        
    Returns
        None: Saves report to 'eda_report/data_quality_report.txt'
        
    '''
    
    logger.info('Performing data quality analysis')
    report_lines = []
    
    # missing values
    logger.info('Missing values check')
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_percent})
    logger.info(f'\tMissing values report:\n{missing_df}')
    report_lines.append('Missing values report:')
    report_lines.append(missing_df.to_string())
    
    # empty strings
    logger.info('Empty strings check')
    text_cols = ['Abstract', 'Intro', 'Conclusion', 'Full_text']
    empty_total = []
    for col in text_cols:
        if col in df.columns:
            empty = (df[col].fillna('').str.strip() == '').sum()
            empty_total.append(f'\t{col:15s}: {empty:4d} empty ({empty/len(df)*100:.1f}%)')
            logger.info(f'\t{col:15s}: {empty:4d} empty ({empty/len(df)*100:.1f}%)')
    report_lines.append('\nEmpty strings report:')
    report_lines.append('\n'.join(empty_total))
    
    # duplicates
    logger.info('Duplicates check')
    dup_pmids = df['PMID'].duplicated().sum()
    dup_titles = df['Title'].duplicated().sum()
    logger.info(f'\tDuplicate PMIDs: {dup_pmids}')
    logger.info(f'\tDuplicate Titles: {dup_titles}')
    report_lines.append('\nDuplicates report:')
    report_lines.append(f'\tDuplicate PMIDs: {dup_pmids}')
    report_lines.append(f'\tDuplicate Titles: {dup_titles}')
    
    # heuristic language
    logger.info('Language check (latin words > 90% in abstract)')
    def is_english(text):
        '''Check if text is likely English based on Latin character ratio.'''
        if pd.isna(text) or text.strip() == '':
            return True
        latin = sum(c.isalpha() and ord(c) < 128 for c in text)
        total = sum(c.isalpha() for c in text)
        return (latin / total > 0.9) if total > 0 else True
    
    non_english = sum(~df['Abstract'].apply(is_english))
    logger.info(f'Potentially non-englush abstracts: {non_english}')
    report_lines.append(f'\nPotentially non-englush abstracts: {non_english}')
    
    
    # saving report
    report_path = OUTPUT_DIR / 'data_quality_report.txt'

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/data_quality_report.txt')

# DESCRIPTIVE STATISTICS
def descriptive_statistics(df):
    '''
    Generate descriptive statistics for the corpus.
    
    Analyzes:
        - Text length statistics for each section
        - Section ratios (intro/full_text, conclusion/full_text)
        - Publication year distribution
        - Readability scores (Flesch Reading Ease)
    
    Args
        df (pd.DataFrame): Input DataFrame with research articles
        
    Returns
        None: Saves text report to 'descriptive_statistics.txt'
            Saves visualizations to 'descriptive_stats.png'
            
    '''
    
    logger.info('Performing descriptive statistics analysis')
    report_lines = []
    
    # text length and ratios
    logger.info('Text Length Statistics (characters)')
    report_lines.append('Text Length Statistics (characters)')
    text_cols = ['Abstract', 'Intro', 'Conclusion', 'Full_text']
    length_stats = {}
    for col in text_cols:
        if col in df.columns:
            lengths = df[col].fillna('').str.len()
            length_stats[col] = {
                'mean': lengths.mean(),
                'median': lengths.median(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max()
            }
    length_df = pd.DataFrame(length_stats).T
    logger.info(f'\n{length_df.round(0)}')
    report_lines.append(length_df.to_string())

    logger.info('Section Ratios (Intro/Full_text, Conclusion/Full_text)')
    report_lines.append('\nSection Ratios (Intro/Full_text, Conclusion/Full_text)')
    df_temp = df[df['Full_text'].str.len() > 100].copy()
    df_temp['intro_ratio'] = df_temp['Intro'].str.len() / df_temp['Full_text'].str.len()
    df_temp['concl_ratio'] = df_temp['Conclusion'].str.len() / df_temp['Full_text'].str.len()
    logger.info(f"\t Intro ratio: {df_temp['intro_ratio'].mean():.2%} ± {df_temp['intro_ratio'].std():.2%}")
    logger.info(f"\t Conclusion ratio: {df_temp['concl_ratio'].mean():.2%} ± {df_temp['concl_ratio'].std():.2%}")
    report_lines.append(f"\tIntro ratio: {df_temp['intro_ratio'].mean():.2%} ± {df_temp['intro_ratio'].std():.2%}")
    report_lines.append(f"\tConclusion ratio: {df_temp['concl_ratio'].mean():.2%} ± {df_temp['concl_ratio'].std():.2%}")
    
    # years distribution
    logger.info('Year Distribution')
    report_lines.append('\nYear Distribution')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    year_counts = df['Year'].value_counts().sort_index()
    logger.info(f'\n{year_counts.tail(10)}')
    report_lines.append(year_counts.to_string())
    
    # top journals
    logger.info('Top 10 Journals')
    report_lines.append('\nTop 10 Journals')
    top_journals = df['Journal'].value_counts().head(10)
    logger.info(f'\n{top_journals}')
    report_lines.append(top_journals.to_string())
    
    # readability scores
    logger.info('Calculating readability scores...')
    sample_abstracts = df['Full_text'].dropna().sample(min(100, len(df)), random_state=42)
    flesch_scores = [textstat.flesch_reading_ease(text) for text in sample_abstracts if len(text) > 50]
    mean_flesch = np.mean(flesch_scores)
    logger.info(f'\tMean Flesch Reading Ease: {mean_flesch:.1f}')
    report_lines.append(f'\nMean Flesch Reading Ease: {mean_flesch:.1f}')

    # visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    year_counts.plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Publications by Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    top_journals.head(10).plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Top 10 Journals')
    axes[0, 1].set_xlabel('Count')
    
    length_data = []
    length_labels = []
    for col in ['Abstract', 'Intro', 'Conclusion', 'Full_text']:
        if col in df.columns:
            lengths = df[col].fillna('').str.len()
            lengths = lengths[lengths > 0]
            length_data.append(lengths)
            length_labels.append(col)
    axes[1, 0].boxplot(length_data, tick_labels=length_labels)
    axes[1, 0].set_title('Text Length Distribution (characters)')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].hist(flesch_scores, bins=20, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Flesch Reading Ease')
    axes[1, 1].set_xlabel('Score (lower = harder)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(flesch_scores), color='red', linestyle='--', label=f'Mean: {np.mean(flesch_scores):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'descriptive_stats.png', dpi=300, bbox_inches='tight')
    logger.info(f'Plots saved to {OUTPUT_DIR}/descriptive_stats.png')
    
    with open(OUTPUT_DIR / 'descriptive_statistics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/descriptive_stats.txt')
    
    return

# LEXICAL ANALYSIS
def lexical_analysis(df):
    '''
    Perform lexical analysis of the corpus.
    
    Analyzes:
    - Vocabulary size and Type-Token Ratio (TTR)
    - Top terms after stopword removal
    - TF-IDF scores for key terms
    - N-grams (bigrams, trigrams, fourgrams)
    - Word cloud visualization
    
    Args
        df (pd.DataFrame): Input DataFrame with research articles
        
    Returns
        None: Saves text report to 'lexical_analysis.txt'
            Saves visualizations to 'lexical_analysis.png'
    '''
    
    logger.info('Performing lexical analysis')
    report_lines = []
    
    texts = df['Abstract'].fillna('') + ' ' + df['Full_text'].fillna('')
    corpus = ' '.join(texts).lower()
    
    # tokenizing
    logger.info('Tokenizing corpus')
    tokens = word_tokenize(corpus)
    tokens = [t for t in tokens if (t.isalnum() or '-' in t) and len(t) > 2]
    
    stop_words = set(stopwords.words('english'))
    url = "https://raw.githubusercontent.com/seinecle/umigon-stopwords/master/src/main/resources/net/clementlevallois/stopwords/scientific_discourse_stopwords_en.txt"
    # Levallois, C., Clithero, J. A., Wouters, P., Smidts, A., & Huettel, S. A. (2012). 
    # Translating upwards: linking the neural and social sciences via neuroeconomics. 
    # Nature Reviews Neuroscience, 13(11), 789-797.
    # GitHub repository: https://github.com/seinecle/umigon-stopwords
    response = requests.get(url)
    domain_stopwords = set(response.text.strip().split('\n'))
    stop_words.update(domain_stopwords)
    
    tokens_filtered = [t for t in tokens if t not in stop_words]
    
    # vocabulary
    logger.info('Vocabulary statistics')
    report_lines.append('Vocabulary statistics')
    logger.info(f'\tTotal tokens: {len(tokens):,}')
    report_lines.append(f'\tTotal tokens: {len(tokens):,}')
    logger.info(f'\tUnique tokens: {len(set(tokens)):,}')
    report_lines.append(f'\tUnique tokens: {len(set(tokens)):,}')
    logger.info(f'\tType-Token Ratio (TTR): {len(set(tokens))/len(tokens):.3f}')
    report_lines.append(f'\tType-Token Ratio (TTR): {len(set(tokens))/len(tokens):.3f}')
    logger.info(f'\tAfter stopword removal: {len(tokens_filtered):,}')
    report_lines.append(f'\tAfter stopword removal: {len(tokens_filtered):,}')
    
    logger.info('Top 30 Terms (after stopwords)')
    report_lines.append('\nTop 30 Terms (after stopwords)')
    term_freq = Counter(tokens_filtered)
    top_terms = term_freq.most_common(30)
    for term, count in top_terms:
        logger.info(f'\t{term:20s}: {count:5d}')
        report_lines.append(f'\t{term:20s}: {count:5d}')
        
    logger.info('TF-IDF Top Terms')
    report_lines.append('\nTF-IDF Top Terms')
    tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 1))
    texts_clean = [t for t in texts if len(t) > 50]
    tfidf_matrix = tfidf.fit_transform(texts_clean)
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    tfidf_df = pd.DataFrame({
        'term': feature_names,
        'tfidf': tfidf_scores
    }).sort_values('tfidf', ascending=False).head(30)
    logger.info(f'\n{tfidf_df.to_string(index=False)}')
    report_lines.append(tfidf_df.to_string(index=False))
    
    # ngrams
    logger.info('Top Bigrams')
    report_lines.append('\nTop Bigrams')
    bigrams = list(ngrams(tokens_filtered, 2))
    bigram_freq = Counter([' '.join(bg) for bg in bigrams])
    top_bigrams = bigram_freq.most_common(20)
    for bg, count in top_bigrams:
        logger.info(f'\t{bg:30s}: {count:4d}')
        report_lines.append(f'\t{bg:30s}: {count:4d}')
    
    logger.info('Top Trigrams')
    report_lines.append('\nTop Trigrams')
    trigrams = list(ngrams(tokens_filtered, 3))
    trigram_freq = Counter([' '.join(tg) for tg in trigrams])
    top_trigrams = trigram_freq.most_common(20)
    for tg, count in top_trigrams:
        logger.info(f'\t{tg:30s}: {count:4d}')
        report_lines.append(f'\t{tg:30s}: {count:4d}')
        
    logger.info('Top Fourgrams')
    report_lines.append('\nTop Fourgrams')
    fourgrams = list(ngrams(tokens_filtered, 4))
    fourgram_freq = Counter([' '.join(fg) for fg in fourgrams])
    top_fourgrams = fourgram_freq.most_common(20)
    for fg, count in top_fourgrams:
        logger.info(f'\t{fg:30s}: {count:4d}')
        report_lines.append(f'\t{fg:30s}: {count:4d}')
        
    # visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100
                          ).generate(corpus)
    axes[0, 0].imshow(wordcloud, interpolation='bilinear')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Word Cloud (Top 100 Terms)', fontsize=14)
    
    terms, counts = zip(*top_terms[:20])
    axes[0, 1].barh(range(len(terms)), counts, color='teal')
    axes[0, 1].set_yticks(range(len(terms)))
    axes[0, 1].set_yticklabels(terms)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title('Top 20 Terms (Frequency)', fontsize=14)
    axes[0, 1].set_xlabel('Count')
    
    bgs, bg_counts = zip(*top_bigrams[:15])
    axes[1, 0].barh(range(len(bgs)), bg_counts, color='salmon')
    axes[1, 0].set_yticks(range(len(bgs)))
    axes[1, 0].set_yticklabels(bgs)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('Top 15 Bigrams', fontsize=14)
    axes[1, 0].set_xlabel('Count')
    
    top_tfidf = tfidf_df.head(20)
    axes[1, 1].barh(range(len(top_tfidf)), top_tfidf['tfidf'], color='gold')
    axes[1, 1].set_yticks(range(len(top_tfidf)))
    axes[1, 1].set_yticklabels(top_tfidf['term'])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title('Top 20 TF-IDF Terms', fontsize=14)
    axes[1, 1].set_xlabel('Average TF-IDF')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lexical_analysis.png', dpi=300, bbox_inches='tight')
    print(f'Plots saved to {OUTPUT_DIR}/lexical_analysis.png')
        
    with open(OUTPUT_DIR / 'lexical_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/lexical_analysis.png')
    
    return

# SEMANTICS ANALYSIS
def semantic_analysis(df):
    '''
    Perform semantic analysis using sentence embeddings and topic modeling.
    
    Analyzes:
        - Document embeddings using SPECTER model
        - Pairwise cosine similarity between abstracts
        - UMAP and t-SNE projections for visualization
        - Topic modeling using BERTopic
    
    Args
        df (pd.DataFrame) Input DataFrame with research articles
    
    Returns
        None: Saves text report to 'semantic_analysis.txt'
            Saves visualizations to 'semantic_analysis.png'
            Saves BERTopic model to 'bertopic_model' directory
    '''
    
    # getting rid of anniying BERTopic errors
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # ...idk why didn't it work

    logger.info('Performing semantic analysis for intro + conclusion')
    report_lines = []
    
    texts_sample = df['Intro'].fillna('') + ' ' + df['Conclusion'].fillna('')
    years_sample = df['Year'].values 
    
    # generate embeddings
    logger.info(f'Generating embeddings for {len(texts_sample)} texts...')
    logger.info('Using: sentence-transformers/allenai-specter')
    model = SentenceTransformer('sentence-transformers/allenai-specter')
    embeddings = model.encode(texts_sample, show_progress_bar=True, batch_size=32)
    
    #cosine similarity
    logger.info('\nComputing pairwise similarities...')
    report_lines.append('Pairwise cosine similarities')
    
    similarities = cosine_similarity(embeddings)
    avg_similarity = (similarities.sum() - len(similarities)) / (len(similarities) * (len(similarities) - 1))
    aps = f'\tAverage pairwise similarity: {avg_similarity:.3f}'
    logger.info(aps)
    report_lines.append(aps)
    std = f'\tStd of similarities: {similarities[np.triu_indices_from(similarities, k=1)].std():.3f}'
    logger.info(std)
    report_lines.append(std)
    
    # umap dimensionality reduction
    logger.info('UMAP projection...')
    reducer_umap = umap.UMAP(
        n_components=2,
        random_state=RANDOM_STATE,
        n_neighbors=15,
        min_dist=0.1
        )
    embeddings_2d_umap = reducer_umap.fit_transform(embeddings)
    
    # t-sne dimensionality reduction
    logger.info("t-SNE projection...")
    reducer_tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30
        )
    embeddings_2d_tsne = reducer_tsne.fit_transform(embeddings)
    
    # topic modelling
    logger.info('BERTopic modeling...')
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), 
        stop_words="english",
        min_df=2
        )
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(
        nr_topics=8,
        verbose=False,
        embedding_model=model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model
        )

    topics, _ = topic_model.fit_transform(texts_sample, embeddings)
    
    tf = f'BERTopics found: {len(set(topics)) - 1}'  # -1 for outlier topic
    logger.info(tf)
    report_lines.append(f'\n{tf}')
    topic_info = topic_model.get_topic_info()
    tfs = topic_info[['Topic', 'Count', 'Name']]
    logger.info(tfs.head(10))
    report_lines.append(tfs.to_string())
    
    # visualisation
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], 
                         c=years_sample, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax1, label='Year')
    ax1.set_title('UMAP Projection (colored by Year)')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], 
                          c=topics, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter2, ax=ax2, label='Topic')
    ax2.set_title('UMAP Projection (colored by Topic)')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    ax3 = plt.subplot(2, 3, 3)
    scatter3 = ax3.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                          c=topics, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter3, ax=ax3, label='Topic')
    ax3.set_title('t-SNE Projection (colored by Topic)')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    ax4 = plt.subplot(2, 3, 4)
    sample_idx = np.random.choice(len(similarities), min(50, len(similarities)), replace=False)
    sim_sample = similarities[np.ix_(sample_idx, sample_idx)]
    sns.heatmap(sim_sample, cmap='coolwarm', ax=ax4, cbar_kws={'label': 'Cosine Similarity'})
    ax4.set_title(f'Similarity Matrix (Sample of {len(sample_idx)})')
    
    ax5 = plt.subplot(2, 3, 5)
    topic_counts = topic_info[topic_info['Topic'] != -1]['Count'].values
    topic_names = topic_info[topic_info['Topic'] != -1]['Topic'].values
    ax5.barh(topic_names, topic_counts, color='skyblue')
    ax5.set_xlabel('Number of Articles')
    ax5.set_ylabel('Topic ID')
    ax5.set_title('Topic Distribution')
    ax5.invert_yaxis()
    
    ax6 = plt.subplot(2, 3, 6)
    sim_upper = similarities[np.triu_indices_from(similarities, k=1)]
    ax6.hist(sim_upper, bins=50, color='orchid', edgecolor='black', alpha=0.7)
    ax6.axvline(avg_similarity, color='red', linestyle='--', label=f'Mean: {avg_similarity:.3f}')
    ax6.set_xlabel('Cosine Similarity')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Pairwise Similarities')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semantic_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f'Plots saved to {OUTPUT_DIR}/semantic_analysis.png')
    
    with open(OUTPUT_DIR / 'semantic_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/semantic_analysis.png')
    
    topic_model.save(str(OUTPUT_DIR / 'bertopic_model'))
    logger.info(f'BERTopic model saved to {OUTPUT_DIR}/semantic_analysis.png')

# PRE RAG
def pre_rag(df):
    '''
    Assess RAG (Retrieval-Augmented Generation) readiness of the corpus.
    
    Analyzes:
        - Optimal chunk size based on paragraph/sentence statistics
        - Simulated chunking (500 tokens ~2000 chars, 200 char overlap)
        - BM25 retrieval performance on test queries
    
    Args
        df (pd.DataFrame):  Input DataFrame with research articles
    
    Returns
        None: Saves text report to 'pre_rag.txt'
            Saves visualizations to 'pre_rag.png'
    '''
    
    logger.info('RAG readiness check')
    report_lines = []
    
    logger.info('Chunk Size Estimation')
    texts = df['Full_text'].fillna('')
    
    paragraph_lengths = []
    sentence_counts = []
    
    for text in texts:
        # XXX too simple?
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if paragraphs:
            paragraph_lengths.extend([len(p) for p in paragraphs])
        
        sentences = sent_tokenize(text)
        sentence_counts.append(len(sentences))
    
    apl = f'Average paragraph length: {np.mean(paragraph_lengths):.0f} chars (±{np.std(paragraph_lengths):.0f})'
    logger.info(apl)
    report_lines.append(apl)
    mpl = f'Median paragraph length: {np.median(paragraph_lengths):.0f} chars'
    logger.info(mpl)
    report_lines.append(mpl)
    rcs = 'Recommended chunk size: 500-1000 tokens (~2000-4000 chars)'
    logger.info(rcs)
    report_lines.append(rcs)
    aspa = f'Average sentences per article: {np.mean(sentence_counts):.0f}'
    logger.info(aspa)
    report_lines.append(aspa)
    
    # chunking
    logger.info('Simulating Chunking (500 tokens ~2000 chars, overlap 200 chars)')
    chunk_counts = []
    
    for text in texts:
        chunks = []
        chunk_size = 2000
        overlap = 200
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        chunk_counts.append(len(chunks))
    
    acpa = f'Average chunks per article: {np.mean(chunk_counts):.1f}'
    logger.info(acpa)
    report_lines.append(acpa)
    tcic = f'Total chunks in corpus: {sum(chunk_counts)}'
    logger.info(tcic)
    report_lines.append(tcic)
    mcpa = f'Median chunks per article: {np.median(chunk_counts):.0f}'
    logger.info(mcpa)
    report_lines.append(mcpa)
    
    # query-document matching
    logger.info("\nQuery-Document Matching (BM25)")
    
    test_queries = [
        'What are potential targets for Alzheimer\'s disease treatment?',
        'Are the targets druggable with small molecules, biologics, or other modalities?',
        'What additional studies are needed to advance these targets?'
        ]
    
    texts = df['Full_text'].tolist()
    tokenized_corpus = [text.lower().split() for text in texts]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    for query in test_queries:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_k = 5
        top_indices = np.argsort(scores)[-top_k:][::-1]
        avg_score = np.mean(scores[top_indices])
        qe = f'Query: {query}'
        logger.info(qe)
        report_lines.append(qe)
        qe1 = f'\tTop-5 avg BM25 score: {avg_score:.2f}'
        logger.info(qe1)
        report_lines.append(qe1)
        qe2 = f'\tBest match score: {scores[top_indices[0]]:.2f}'
        logger.info(qe2)
        report_lines.append(qe2)
    
    # visualisation
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].hist(paragraph_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(2000, color='red', linestyle='--', label='Suggested chunk size')
    axes[0].set_xlabel('Paragraph Length (chars)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Paragraph Length Distribution')
    axes[0].legend()
    axes[0].set_xlim(0, min(10000, max(paragraph_lengths)))
    
    axes[1].hist(chunk_counts, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Chunks')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Chunks per Article Distribution')
    axes[1].axvline(np.mean(chunk_counts), color='red', linestyle='--', label=f'Mean: {np.mean(chunk_counts):.1f}')
    axes[1].legend()
    
    all_query_scores = []
    for query in test_queries:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        all_query_scores.append(scores)
    
    axes[2].boxplot(all_query_scores, tick_labels=[f'Q{i+1}' for i in range(len(test_queries))])
    axes[2].set_ylabel('BM25 Score')
    axes[2].set_xlabel('Query')
    axes[2].set_title('BM25 Score Distribution by Query')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pre_rag.png', dpi=300, bbox_inches='tight')
    logger.info(f'Plots saved to {OUTPUT_DIR}/pre_rag.png')
    
    with open(OUTPUT_DIR / 'pre_rag.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/pre_rag.png')

# BIO NER
def extract_entities_bio(texts, nlp_bio):
    '''
    Extract genes/proteins and cell types using biomedical NER model.
    
    Args:
        texts (List[str]): List of document texts to process
        nlp_bio: scispaCy biomedical NER pipeline
        
    Returns:
        Tuple of (genes_proteins_counter, cells_counter)
    '''
    
    genes_proteins = Counter()
    cells = Counter()
    
    logger.info(f'Processing {len(texts)} texts with bio model...')
    
    processed_docs = []
    for idx, doc in enumerate(nlp_bio.pipe(texts, batch_size=25, n_process=1), 1):
        processed_docs.append(doc)
        if idx % 25 == 0:
            logger.info(f'\tProcessed {idx}/{len(texts)}')
        
        for ent in doc.ents:
            text = ent.text.strip()
            if text in ENTITY_STOPLIST:
                continue
            if ent.label_ == 'GENE_OR_GENE_PRODUCT':
                genes_proteins[text] += 1
            elif ent.label_ == 'CELL':
                cells[text] += 1
    
    return genes_proteins, cells, processed_docs

def extract_entities_med(texts, nlp_med):
    '''
    Extract diseases and chemicals/drugs using medical NER model.
    
    Args:
        texts (List[str]): List of document texts to process
        nlp_med: scispaCy medical NER pipeline
        
    Returns:
        Tuple of (diseases_counter, chemicals_drugs_counter)
    '''
    
    diseases = Counter()
    chemicals_drugs = Counter()
    
    logger.info(f'Processing {len(texts)} texts with medical model...')
    
    for idx, doc in enumerate(nlp_med.pipe(texts, batch_size=25, n_process=1), 1):
        if idx % 25 == 0:
            logger.info(f'\tProcessed {idx}/{len(texts)}')
        
        for ent in doc.ents:
            text = ent.text.strip()
            if ent.label_ == 'DISEASE':
                diseases[text] += 1
            elif ent.label_ == 'CHEMICAL':
                if text in ENTITY_STOPLIST:
                    continue
                chemicals_drugs[text] += 1
    
    return diseases, chemicals_drugs


def visualize_network_interactive(G: nx.Graph, output_path):
    """
    Create interactive HTML network visualization using PyVis.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error('PyVis not installed. Run: pip install pyvis')
        return
    
    net = Network(height='800px', width='100%', bgcolor='#ffffff', 
                  font_color='black', notebook=False)
    
    # Physics
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, 
                           spring_length=100, spring_strength=0.08)
    
    color_map = {'gene': '#4A90E2', 'drug': '#E8743B', 'disease': '#19A974'}
    
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        size = G.nodes[node].get('size', 10)
        
        net.add_node(
            node,
            label=node,
            title=f"{node} ({node_type}): {size} mentions",
            color=color_map.get(node_type, '#CCCCCC'),
            size=min(size * 3, 50),
            font={'size': 14}
        )
    
    # EDGES С CAP НА ТОЛЩИНУ
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if weight > 1:
            # Cap толщины на 10-15 для PyVis
            display_width = min(weight * 2, 15)
            net.add_edge(u, v, value=display_width, 
                        title=f"Weight: {weight:.1f}")
    
    net.save_graph(str(output_path))
    logger.info(f'Interactive network saved to {output_path}')


def visualize_network_static(G: nx.Graph, output_path):
    """
    Create publication-quality static network visualization.
    """
    
    # Фильтр слабых рёбер
    G_filtered = G.copy()
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1) <= 1.5]
    G_filtered.remove_edges_from(weak_edges)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    try:
        pos = nx.kamada_kawai_layout(G_filtered)
    except:
        pos = nx.spring_layout(G_filtered, k=1.5, iterations=100, seed=42)
    
    # Node styling
    node_colors = []
    node_sizes = []
    color_map = {'gene': '#4A90E2', 'drug': '#E8743B', 'disease': '#19A974'}
    
    for node in G_filtered.nodes():
        node_type = G_filtered.nodes[node].get('type', 'unknown')
        size = G_filtered.nodes[node].get('size', 10)
        node_colors.append(color_map.get(node_type, '#CCCCCC'))
        node_sizes.append(min(size * 100, 2000))
    
    # EDGE WIDTHS С ЛОГАРИФМОМ + CAP
    edge_widths = []
    for u, v in G_filtered.edges():
        weight = G_filtered[u][v].get('weight', 1)
        width = min(np.log1p(weight) * 1.2, 6.0)  # 1→0, 10→3.5, макс 6px
        edge_widths.append(width)
    
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, 
                          alpha=0.4, edge_color='#666666', ax=ax)
    nx.draw_networkx_labels(G_filtered, pos, font_size=10, 
                           font_weight='bold', ax=ax)
    
    legend = [
        Patch(facecolor='#4A90E2', label='Genes/Proteins'),
        Patch(facecolor='#E8743B', label='Drugs/Chemicals'),
        Patch(facecolor='#19A974', label='Diseases')
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=12)
    ax.set_title('Biomedical Entity Network\n(Edge width = log(co-occurrence), filtered)', 
                fontweight='bold', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    logger.info(f'Static network saved to {output_path}')
    plt.close()



def compute_cooccurrence_matrix(processed_docs: List, top_genes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute gene/protein co-occurrence matrix from documents.
    
    Args:
        processed_docs: List of scispaCy Doc objects with NER annotations
        top_genes: List of gene names to analyze
        
    Returns:
        Tuple of (raw_matrix, normalized_matrix):
            - raw_matrix: Original counts for heatmap/reports
            - normalized_matrix: Log-normalized for graph edges
    """
    matrix = np.zeros((len(top_genes), len(top_genes)))
    
    for doc in processed_docs:
        found = [ent.text for ent in doc.ents
                if ent.label_ == 'GENE_OR_GENE_PRODUCT' and ent.text in top_genes]
        
        for i, gene1 in enumerate(top_genes):
            for j, gene2 in enumerate(top_genes):
                if gene1 in found and gene2 in found:
                    matrix[i, j] += 1
    
    # RAW матрица (для отчётов и heatmap)
    df_raw = pd.DataFrame(matrix, index=top_genes, columns=top_genes).astype(int)
    
    # НОРМАЛИЗОВАННАЯ матрица (только для графа)
    # Убираем диагональ (self-loops)
    matrix_no_diag = matrix.copy()
    np.fill_diagonal(matrix_no_diag, 0)
    
    # Логарифмическая нормализация
    df_normalized = pd.DataFrame(
        np.log1p(matrix_no_diag),  # log(1+x) сжимает 8000→9
        index=top_genes, 
        columns=top_genes
    )
    
    logger.info(f"Co-occurrence stats:")
    logger.info(f"  Raw max: {df_raw.max().max()} (diagonal)")
    logger.info(f"  Raw max off-diagonal: {matrix_no_diag.max():.0f}")
    logger.info(f"  Normalized max: {df_normalized.max().max():.2f}")
    
    return df_raw, df_normalized


def build_entity_network(
    genes_proteins: Counter,
    chemicals_drugs: Counter,
    diseases: Counter,
    cooccur_df_normalized: pd.DataFrame,  # ИСПОЛЬЗУЕМ НОРМАЛИЗОВАННУЮ
    processed_docs: List,
    texts: List[str],
    nlp_med,
    top_n_genes: int = 10,
    top_n_drugs: int = 8,
    top_n_diseases: int = 5,
    cooccur_threshold: float = 1.0  # для log-нормализованных (log(3)≈1.1)
) -> nx.Graph:
    """
    Build NetworkX graph from entity co-occurrences.
    
    Args:
        cooccur_df_normalized: LOG-NORMALIZED co-occurrence matrix (не raw!)
    """
    G = nx.Graph()
    
    # Add nodes
    top_genes = [g for g, _ in genes_proteins.most_common(top_n_genes)]
    top_drugs = [d for d, _ in chemicals_drugs.most_common(top_n_drugs)]
    top_diseases_list = [d for d, _ in diseases.most_common(top_n_diseases)]
    
    for gene in top_genes:
        G.add_node(gene, type='gene', size=genes_proteins[gene])
    for drug in top_drugs:
        G.add_node(drug, type='drug', size=chemicals_drugs[drug])
    for disease in top_diseases_list:
        G.add_node(disease, type='disease', size=diseases[disease])
    
    # GENE-GENE EDGES — нормализованные веса
    logger.info('Adding gene-gene edges...')
    edges_added = 0
    for i, gene1 in enumerate(top_genes):
        for j, gene2 in enumerate(top_genes):
            if i < j:  # no self-loops, только верхний треугольник
                weight = cooccur_df_normalized.iloc[i, j]
                if weight > cooccur_threshold:
                    G.add_edge(gene1, gene2, weight=float(weight), edge_type='gene-gene')
                    edges_added += 1
    
    logger.info(f'  Added {edges_added} gene-gene edges')
    
    # CROSS-TYPE EDGES (gene-drug, gene-disease)
    sample_size = min(50, len(texts))
    logger.info(f'Computing cross-entity edges from {sample_size} documents...')
    
    for idx, doc_med in enumerate(nlp_med.pipe(texts[:sample_size], batch_size=10)):
        bio_doc = processed_docs[idx] if idx < len(processed_docs) else None
        
        doc_genes = [ent.text for ent in bio_doc.ents 
                    if ent.label_ == 'GENE_OR_GENE_PRODUCT' and ent.text in top_genes] if bio_doc else []
        doc_drugs = [ent.text for ent in doc_med.ents 
                    if ent.label_ == 'CHEMICAL' and ent.text in top_drugs]
        doc_diseases = [ent.text for ent in doc_med.ents 
                       if ent.label_ == 'DISEASE' and ent.text in top_diseases_list]
        
        for gene in doc_genes:
            for drug in doc_drugs:
                if G.has_edge(gene, drug):
                    # Накапливаем, но с cap
                    G[gene][drug]['weight'] = min(G[gene][drug]['weight'] + 1, 10.0)
                else:
                    G.add_edge(gene, drug, weight=1.0, edge_type='gene-drug')
            
            for disease in doc_diseases:
                if G.has_edge(gene, disease):
                    G[gene][disease]['weight'] = min(G[gene][disease]['weight'] + 1, 10.0)
                else:
                    G.add_edge(gene, disease, weight=1.0, edge_type='gene-disease')
    
    logger.info(f'Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    return G


def bio_ner(df: pd.DataFrame):
    """
    Perform biomedical NER and network analysis on research articles.
    """
    
    logger.info('Starting biomedical NER analysis')
    texts = df['Full_text'].fillna('').tolist()
    
    # 1. Extract entities
    genes_proteins, cells, processed_docs = extract_entities_bio(texts, nlp_bio)
    diseases, chemicals_drugs = extract_entities_med(texts, nlp_med)
    
    # 2. Co-occurrence — ПОЛУЧАЕМ ДВЕ ВЕРСИИ
    top_genes = [g for g, _ in genes_proteins.most_common(10)]
    cooccur_df_raw, cooccur_df_normalized = compute_cooccurrence_matrix(processed_docs, top_genes)
    
    # 3. Build network (используем нормализованную)
    G = build_entity_network(
        genes_proteins, chemicals_drugs, diseases, 
        cooccur_df_normalized,  # <- НОРМАЛИЗОВАННАЯ для графа
        processed_docs, texts, nlp_med,
        top_n_genes=10, top_n_drugs=8, top_n_diseases=5
    )
    
    # 4. Visualizations
    visualize_network_interactive(G, OUTPUT_DIR / 'entity_network.html')
    visualize_network_static(G, OUTPUT_DIR / 'entity_network.png')
    
    # 5. 2x2 MATPLOTLIB PLOTS (используем RAW для heatmap!)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top genes
    top_genes_list = genes_proteins.most_common(15)
    genes_names, genes_counts = zip(*top_genes_list)
    axes[0, 0].barh(range(len(genes_names)), genes_counts, color='steelblue')
    axes[0, 0].set_yticks(range(len(genes_names)))
    axes[0, 0].set_yticklabels(genes_names)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title('Top 15 Genes/Proteins', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Articles')
    
    # Plot 2: Top drugs
    top_drugs_list = chemicals_drugs.most_common(15)
    if top_drugs_list:
        drugs_names, drugs_counts = zip(*top_drugs_list)
        axes[0, 1].barh(range(len(drugs_names)), drugs_counts, color='coral')
        axes[0, 1].set_yticks(range(len(drugs_names)))
        axes[0, 1].set_yticklabels(drugs_names)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_title('Top 15 Chemicals/Drugs', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Articles')
    
    # Plot 3: Top diseases
    top_diseases_list = diseases.most_common(12)
    diseases_names, diseases_counts = zip(*top_diseases_list)
    axes[1, 0].barh(range(len(diseases_names)), diseases_counts, color='lightgreen')
    axes[1, 0].set_yticks(range(len(diseases_names)))
    axes[1, 0].set_yticklabels(diseases_names)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('Top 12 Diseases', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Articles')
    
    # Plot 4: HEATMAP С RAW МАТРИЦЕЙ (без диагонали для читаемости)
    import seaborn as sns
    cooccur_display = cooccur_df_raw.copy()
    np.fill_diagonal(cooccur_display.values, 0)  # убрать диагональ для масштаба
    
    sns.heatmap(cooccur_display, annot=True, fmt='d', cmap='YlOrRd', 
                ax=axes[1, 1], cbar_kws={'label': 'Co-occurrences'},
                vmin=0, vmax=cooccur_display.max().max())
    axes[1, 1].set_title('Gene Co-occurrence (Raw Counts)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'bio_ner.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {OUTPUT_DIR / 'bio_ner.png'}")
    plt.close()
    
    # 6. SAVE RESULTS
    
    results = {
        'genes_proteins': dict(genes_proteins.most_common(50)),
        'diseases': dict(diseases.most_common(30)),
        'chemicals_drugs': dict(chemicals_drugs.most_common(30)),
        'cells': dict(cells.most_common(20)),
        'cooccurrence_raw': cooccur_df_raw.to_dict(),  # <- RAW для отчёта
        'network_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        }
    }
    
    with open(str(OUTPUT_DIR / 'bio_ner.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON saved to {OUTPUT_DIR / 'bio_ner.json'}")
    
    # GEXF
    nx.write_gexf(G, str(OUTPUT_DIR / 'entity_network.gexf'))
    logger.info(f'GEXF saved to {OUTPUT_DIR}/entity_network.gexf')
    
    # TXT REPORT (с RAW статистикой)
    report_lines = []
    report_lines.append('GENE/PROTEIN CO-OCCURRENCE ANALYSIS')
    report_lines.append(f'\nTop {len(top_genes)} genes analyzed')
    report_lines.append(f'Total co-occurrences: {cooccur_df_raw.sum().sum():.0f}')
    report_lines.append(f'Max co-occurrence: {cooccur_display.max().max():.0f}')
    report_lines.append('\nCo-occurrence matrix (raw counts):')
    report_lines.append(cooccur_df_raw.to_string())
    
    with open(str(OUTPUT_DIR / 'bio_ner_cooccurrence.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Report saved to {OUTPUT_DIR / 'bio_ner_cooccurrence.txt'}")


def mesh_terms_analysis(df):
    '''
    Analyze MeSH (Medical Subject Headings) terms from PubMed metadata.
    
    Analyzes:
        - Frequency of MeSH terms across corpus
        - Categorization into research aspects (Diagnosis, Therapy, Genetics, etc.)
        - Distribution visualization
        
    Args
        df (pd.DataFrame): Input DataFrame with 'MeSH_Terms' column (semicolon-separated)

    Returns
        None: Saves text report to 'mesh_terms.txt' 
    '''
    logger.info('MeSH terms analysis')
    report_lines = []
    
    all_mesh = []
    
    for mesh_str in df['MeSH_Terms'].fillna(''):
        terms = [t.strip() for t in mesh_str.split(';') if t.strip()]
        all_mesh.extend(terms)
    
    mesh_freq = Counter(all_mesh)
    
    logger.info(f'Total unique MeSH terms: {len(mesh_freq)}')
    logger.info(f'Total MeSH annotations: {sum(mesh_freq.values())}')
    logger.info('Top 30 MeSH Terms:')
    
    for term, count in mesh_freq.most_common(30):
        pct = count / len(df) * 100
        logger.info(f'{term[:50]:50s}: {count:3d} ({pct:5.1f}%)')
    
    # heuristic categorization
    categories = {
        'Diagnosis': ['diagnosis', 'diagnostic', 'biomarkers', 'imaging', 'mri', 'pet'],
        'Drug Therapy': ['drug', 'therapy', 'treatment', 'therapeutic', 'pharmacolog'],
        'Genetics': ['genetic', 'gene', 'mutation', 'polymorphism', 'apoe'],
        'Pathology': ['pathology', 'neuropathology', 'amyloid', 'tau', 'plaque'],
        'Clinical': ['clinical', 'patient', 'trial', 'cohort'],
        'Molecular': ['molecular', 'protein', 'peptide', 'enzyme'],
        'Risk Factors': ['risk factor', 'age', 'sex', 'education'],
    }
    
    category_counts = {cat: 0 for cat in categories}
    
    for term, count in mesh_freq.items():
        term_lower = term.lower()
        for category, keywords in categories.items():
            if any(kw in term_lower for kw in keywords):
                category_counts[category] += count
                break
    
    logger.info('\nMeSH Terms by Research Aspect:')
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            logger.info(f'{cat:20s}: {count:3d} terms')
    
    # visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    top_mesh = mesh_freq.most_common(15)
    mesh_names = [m[:35] for m, _ in top_mesh]
    mesh_counts = [c for _, c in top_mesh]
    
    axes[0].barh(range(len(mesh_names)), mesh_counts, color='mediumseagreen')
    axes[0].set_yticks(range(len(mesh_names)))
    axes[0].set_yticklabels(mesh_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency')
    axes[0].set_title('Top 15 MeSH Terms', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    cat_names = [c for c, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True) if v > 0]
    cat_values = [v for c, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True) if v > 0]
    
    if cat_names:
        axes[1].pie(cat_values, labels=cat_names, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('MeSH Terms by Research Aspect', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mesh_terms.png', dpi=300, bbox_inches='tight')
    logger.info(f'Plot saved to {OUTPUT_DIR}/mesh_terms.png')
    
    mesh_data = {
        'top_mesh': dict(mesh_freq.most_common(50)),
        'categories': category_counts
        }
    
    report_lines = []
    
    report_lines.append('Top 50 MeSH Terms')
    for term, count in list(mesh_data['top_mesh'].items()): 
        pct = count / len(df) * 100 if 'df' in globals() else 0
        report_lines.append(f'{term:<35}: {count:4d} ({pct:4.1f}%)')
    
    report_lines.append('\nMeSH Categories')
    for cat, count in mesh_data['categories'].items():
        report_lines.append(f'{cat:<20}: {count:4d}')
        
    with open(OUTPUT_DIR / 'mesh_terms.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Report saved to {OUTPUT_DIR}/mesh_terms.txt')
    
def citation_density_analysis(df):
    '''
    Analyze citation patterns in full texts.
    
    Detects citations in multiple formats:
        - Bracketed numbers: [^27_1], [1-5]
        - Author-year: (Smith et al., 2020)
        
    Args
        df (pd.DataFrame): Input DataFrame with 'Full_text' column
    
    Returns
        None: Saves text report to 'citation_density.txt' 
    '''
    logger.info('Citation density analysis')
    
    texts = df['Full_text'].fillna('')
    
    citation_patterns = [
        r'\[\d+\]',  # [1], [23]
        r'\[\d+[,\-]\d+\]',  # [1-5], [1,2,3]
        r'\(\w+\s+et\s+al\.,?\s+\d{4}\)',  # (Smith et al., 2020)
        r'\(\w+\s+and\s+\w+,?\s+\d{4}\)',  # (Smith and Jones, 2020)
        r'\(\w+,?\s+\d{4}\)',  # (Smith, 2020)
    ]
    
    citation_counts = []
    for text in texts:
        if len(text) < 100:
            citation_counts.append(0)
            continue
        
        total_citations = 0
        for pattern in citation_patterns:
            total_citations += len(re.findall(pattern, text))
        
        citation_counts.append(total_citations)
    
    df_temp = df.copy()
    df_temp['citation_count'] = citation_counts
    df_temp['text_len'] = texts.str.len()
    
    df_valid = df_temp[df_temp['text_len'] > 100]
    
    df_valid['citations_per_1k_chars'] = (df_valid['citation_count'] / df_valid['text_len'] * 1000).round(2)
    
    avg_citations = df_valid['citation_count'].mean()
    median_citations = df_valid['citation_count'].median()
    avg_density = df_valid['citations_per_1k_chars'].mean()
    
    logger.info('Citation Statistics:')
    logger.info(f'\tAverage citations per article: {avg_citations:.1f}')
    logger.info(f'\tMedian citations per article: {median_citations:.0f}')
    logger.info(f'\tAverage citation density: {avg_density:.2f} per 1000 chars')
    logger.info(f'\tMax citations in single article: {df_valid["citation_count"].max():.0f}')
    
    logger.info('Most Heavily Cited Articles:')
    top_cited = df_valid.nlargest(5, 'citation_count')[['Title', 'citation_count', 'citations_per_1k_chars']]
    for idx, row in top_cited.iterrows():
        logger.info(f'\t{row["Title"][:60]:60s} - {row["citation_count"]:.0f} citations ({row["citations_per_1k_chars"]:.1f}/1k)')
    
    # visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(df_valid['citation_count'], bins=30, color='lightcoral', edgecolor='black')
    axes[0].axvline(avg_citations, color='red', linestyle='--', label=f'Mean: {avg_citations:.1f}')
    axes[0].axvline(median_citations, color='green', linestyle='--', label=f'Median: {median_citations:.0f}')
    axes[0].set_xlabel('Number of Citations')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Citation Count Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].scatter(df_valid['text_len'], df_valid['citation_count'], alpha=0.6, color='steelblue')
    axes[1].set_xlabel('Text Length (characters)')
    axes[1].set_ylabel('Citation Count')
    axes[1].set_title('Citations vs Text Length', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    z = np.polyfit(df_valid['text_len'], df_valid['citation_count'], 1)
    p = np.poly1d(z)
    axes[1].plot(df_valid['text_len'], p(df_valid['text_len']), "r--", alpha=0.8, label='Trend')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'citation_density.png', dpi=300, bbox_inches='tight')
    logger.info(f'Plot saved to {OUTPUT_DIR}/citation_density.png')
    
    metrics_data = {
        'avg_citations': avg_citations,
        'avg_density': avg_density
        }

    report_lines = []
    report_lines.append(f"Average citations per article: {metrics_data['avg_citations']:.2f}")
    report_lines.append(f"Average co-author density:    {metrics_data['avg_density']:.2f}")
    
    with open(OUTPUT_DIR / 'publication_metrics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f'Metrics TXT saved to {OUTPUT_DIR}/publication_metrics.txt')

    

if __name__ == '__main__':
    df = load_dataset()
    data_quality_analysis(df)
    descriptive_statistics(df)
    lexical_analysis(df)
    semantic_analysis(df)
    pre_rag(df)
    if SCISPACY_AVAILABLE:
        bio_ner(df)
    mesh_terms_analysis(df)
    citation_density_analysis(df)