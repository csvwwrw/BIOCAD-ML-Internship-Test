# -*- coding: utf-8 -*-
'''
Global configuration for Alzheimer's RAG system.

All paths, hyperparameters, and API keys centralized here.
Import with: from config.settings import Config

Created on Thu Jan 29 19:14:39 2026

@author: csvww
'''

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    '''
    Centralized configuration class for RAG system.
    
    Contains paths, model settings, retrieval parameters, and API keys.
    All modules import from this single source of truth.
    '''
    
    PROJECT_ROOT = Path(__file__).parent.parent
    EDA_DIR = PROJECT_ROOT / 'eda_report'
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    CORPUS_CSV = RAW_DATA_DIR / 'alzheimer_corpus.csv'
    CHUNKS_PKL = PROCESSED_DATA_DIR / 'chunks.pkl'
    VECTOR_STORE_PREFIX = PROCESSED_DATA_DIR / 'alzheimer_vector_store'
    
    RANDOM_STATE = 100
    
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MIN_CHUNK_LENGTH = 100

    EMBEDDING_MODEL = 'sentence-transformers/allenai-specter'
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_DIM = 768
    
    RETRIEVAL_TOP_K = 10
    HYBRID_ALPHA = 0.6
    YEAR_MIN_DEFAULT = 2020
    
    BM25_K1 = 1.5
    BM25_B = 0.75
    
    PERPLEXITY_API_KEY = os.getenv('PPLX_API_KEY', '')
    
    DEFAULT_LLM_PROVIDER = 'perplexity'
    DEFAULT_LLM_MODEL = 'sonar-pro'
    
    MAX_TOKENS = 1500
    TEMPERATURE = 0.3
    CONTEXT_MAX_PAPERS = 7
    CONTEXT_MAX_CHUNKS_PER_PAPER = 4
    
    PUBMED_EMAIL = os.getenv('PUBMED_EMAIL', 'your_email@example.com')
    PUBMED_API_KEY = os.getenv('PUBMED_API_KEY', '')
    PUBMED_TARGET_ARTICLES = 100
    
    LOG_LEVEL = 'INFO'
    LOG_FILE = LOGS_DIR / 'alzheimer_rag.log'
    LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    STREAMLIT_TITLE = '🧠 Alzheimer\'s Drug Target Discovery Assistant'
    STREAMLIT_LAYOUT = 'wide'

def get_api_key(provider):
    '''
    Get API key for specified LLM provider.
    
    Args:
        provider (str): 'perplexity', 'openai', or 'anthropic'
    
    Returns:
        str: API key from environment or empty string
    '''
    mapping = {
        'perplexity': Config.PERPLEXITY_API_KEY,
        'openai': Config.OPENAI_API_KEY,
        'anthropic': Config.ANTHROPIC_API_KEY
    }
    return mapping.get(provider.lower(), '')

def validate_config():
    '''
    Validate critical configuration parameters.
    
    Checks paths exist, API keys formatted correctly, and parameters valid.
    Logs warnings for missing optional settings.
    
    Raises:
        ValueError: If critical settings invalid.
    '''
    import logging
    logger = logging.getLogger(__name__)
    
    if not Config.CORPUS_CSV.exists():
        logger.warning(f'Corpus file not found: {Config.CORPUS_CSV}')
    
    if not Config.PERPLEXITY_API_KEY:
        logger.warning('PPLX_API_KEY not set in .env')
    
    if not (0 <= Config.HYBRID_ALPHA <= 1):
        raise ValueError(f'HYBRID_ALPHA must be 0-1, got {Config.HYBRID_ALPHA}')
    
    if Config.CHUNK_SIZE < Config.CHUNK_OVERLAP:
        raise ValueError('CHUNK_SIZE must be >= CHUNK_OVERLAP')
    
    logger.info('Configuration validated')


if __name__ == '__main__':
    # Test configuration
    print('Alzheimer\'s RAG Configuration:')
    print(f'\tProject root: {Config.PROJECT_ROOT}')
    print(f'\tData dir: {Config.DATA_DIR}')
    print(f'\tChunk size: {Config.CHUNK_SIZE}')
    print(f'\tEmbedding model: {Config.EMBEDDING_MODEL}')
    print(f'\tDefault LLM: {Config.DEFAULT_LLM_PROVIDER}')
    print(f'\tPerplexity API: {"Set" if Config.PERPLEXITY_API_KEY else "Missing"}')
    
    validate_config()

