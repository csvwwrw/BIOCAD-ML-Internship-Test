# -*- coding: utf-8 -*-
'''
PubMed Article Retrieval System for Alzheimer's Disease Research

This module provides functionality to search, retrieve, and parse
biomedical articles from PubMed/PMC databases with guaranteed full text.

Created on Sat Jan 24 15:26:13 2026
@author: csvwwrw
'''

from Bio import Entrez, Medline
import time
from datetime import datetime
import pandas as pd
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import logging
import re
from bs4 import BeautifulSoup
import unicodedata

# Setting up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pubmed_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initializing Entrez credentials (email + API key)
load_dotenv()
ENTREZ_EMAIL = os.getenv('ENTREZ_EMAIL')
ENTREZ_API_KEY = os.getenv('ENTREZ_API_KEY')

if not ENTREZ_EMAIL:
    raise ValueError('ENTREZ_EMAIL absent in .env!')
if not ENTREZ_API_KEY:
    logger.warning('ENTREZ_API_KEY absent in .env - requests capped at 3/sec')

Entrez.email = ENTREZ_EMAIL
if ENTREZ_API_KEY:
    Entrez.api_key = ENTREZ_API_KEY
REQUEST_DELAY = 0.12 if ENTREZ_API_KEY else 0.34

def search_pubmed(query, target_results):
    '''
    Search PubMed database for articles with free full text available.
    
    Performs iterative searches with Open Access filter until target number
    of articles with PMC IDs is reached. Uses rate limiting to comply with
    NCBI guidelines (3 requests/sec).
    
    Args:
        query (str): Search term or PubMed query string
            (e.g., "Alzheimer targets").
        
        target_results (int, optional): Target number of articles with full
            text. Defaults to 100.
    
    Returns:
        list: List of PMID strings for articles with PMC full text available.
            Returns empty list if no articles found.
    
    Raises:
        Exception: If connection to PubMed fails (caught and logged).
    '''
    logger.info(f'Performing PMC search for query: "{query}", target results: {target_results}')
    
    pmc_pmids = set()
    ret_start = 0
    ret_batch = min(target_results, 100)
    
    while len(pmc_pmids) < target_results:
        try:
            esearch_handle = Entrez.esearch(
                db='pubmed',
                term=f'{query} AND "free full text"[sb] NOT preprint[pt]',
                retstart=ret_start,
                retmax=ret_batch,
                sort='relevance'
                )
            record = Entrez.read(esearch_handle)
            esearch_handle.close()
    
            new_pmids = record['IdList']
            logger.info(f'\tSearch batch {ret_start//ret_batch + 1}: {len(new_pmids)} PMIDs found')
    
            if not new_pmids:
                logger.info('No more search results')
                break
            
            pmc_count_batch = check_full_text(new_pmids)
            pmc_pmids.update(pmc_count_batch)
            
            logger.info(f'\tPMIDs with PMCs (full text avaliable): {len(pmc_count_batch)}')
            
            ret_start += ret_batch
            time.sleep(REQUEST_DELAY)
            
            if len(record['IdList']) < ret_batch:
                logger.info('No more search results')
                break
    
        except Exception as e:
            logger.error(f'\tError while batch searching for PubMed results: {e}')
            break
        
    if len(pmc_pmids) < target_results:
        logger.warning(f'Only {len(pmc_pmids)} of {target_results} eligible articles found.')
    else:
        logger.info(f'\tTotal PMCs with full text available for "{query}": {len(pmc_pmids)}')
    
    return list(pmc_pmids)

def check_full_text(pmid_batch):
    '''
    Check which PMIDs have PMC full text available.
    
    Fetches MEDLINE records for a batch of PMIDs and filters those
    with PMC ID, indicating full text availability in PubMed Central.
    
    Args:
        pmid_batch (list): List of PMID strings to check.
    
    Returns:
        list: List of PMID strings that have PMC full text available. Returns
            empty list if none found or on error.
    '''
    
    pmc_pmids = []
    
    try:
        handle = Entrez.efetch(
            db='pubmed',
            id=pmid_batch,
            rettype='medline',
            retmode='text',
            retmax=200 
        )
        records = Medline.parse(handle)
        
        total_checked = 0
        for record in records:
            total_checked += 1
            pmc_id = record.get('PMC', '')
            if pmc_id:
                pmc_pmids.append(record.get('PMID'))
            else:
                logger.debug(f"No PMC for PMID {record.get('PMID')}")
    
        handle.close()
        logger.debug(f'{len(pmc_pmids)} PMCs of {total_checked} PMIDs')
    
    except Exception as e:
        logger.error(f'Error checking full text for PMC: {e}')
    
    return pmc_pmids
    

def fetch_article_details(pmid_list, batch_size=30):
    '''
    Fetch detailed metadata for a list of PMIDs.
    
    Retrieves bibliographic information (title, authors, journal, etc.)
    for articles using MEDLINE format. Processes in batches to manage
    API rate limits.
    
    Args:
        pmid_list (list): List of PMID strings to fetch.
        batch_size (int, optional): Number of PMIDs per batch. Defaults to 30.
            Smaller batches are more reliable for large datasets.
    
    Returns:
        list: List of dictionaries containing article metadata. 
            Each dict includes:
                - PMID, PMC_ID, Title, Abstract, Authors, Journal, Year, DOI,
                    MeSH_Terms, Affiliations, PubMed_URL, Fetched timestamp
    '''
    
    articles = []
    
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i:i+batch_size]
        logger.info(f'\tProcessing batch {i//batch_size + 1}: PMID {i+1}-{min(i+batch_size, len(pmid_list))}')
        
        try:
            efetch_handle = Entrez.efetch(
                db='pubmed',
                id=batch,
                rettype='medline', #not XML, to avoid dl-ing full text for now, medline is enough to filter
                retmode='text'
                )
            
            records = Medline.parse(efetch_handle)
            
            for record in records:
                article_data = extract_metadata(record)
                articles.append(article_data)
            
            efetch_handle.close()
            
            time.sleep(0.34)
        
        except Exception as e:
            logger.warning(f'\tError during processing of the batch: {e}')
            continue
    
    logger.info(f'Total articles processed: {len(articles)}')
    return articles

def extract_metadata(record):
    '''
    Extract structured metadata from a MEDLINE record.
    
    Parses MEDLINE record fields into a standardized dictionary format
    with proper handling of missing/empty fields.
    
    Args:
        record (dict): MEDLINE record dictionary from Bio.Entrez.
    
    Returns:
        dict: Structured metadata with keys:
            - PMID (str): PubMed identifier
            - PMC_ID (str): PubMed Central identifier (empty if not available)
            - Title (str): Article title
            - Abstract (str): Article abstract text
            - Authors (str): Semicolon-separated author list
            - Journal (str): Journal title abbreviation
            - Year (str): Publication year
            - DOI (str): Digital Object Identifier
            - MeSH_Terms (str): Semicolon-separated MeSH terms
            - PubMed_URL (str): Direct URL to PubMed page
            - Fetched (str): Timestamp of data retrieval (YYYY-MM-DD HH:MM:SS)
            '''
    pmid = record.get('PMID', '')
    pmc_id = record.get('PMC', '')
    title = clean_text(record.get('TI', ''))
    abstract = clean_text(record.get('AB', ''))
    authors = record.get('AU', [])
    authors = '; '.join(authors) if authors else ''
    journal = record.get('TA', '')
    date = record.get('DP', '')
    date_year = date.split()[0] if date else ''
    
    doi_list = record.get('AID', [])
    doi = ''
    for aid in doi_list:
        if 'doi' in aid.lower():
            doi = aid.replace('[doi]', '').strip()
            break
        
    mesh_terms = record.get('MH', [])
    mesh_terms = '; '.join(mesh_terms) if mesh_terms else ''
    
    return {
        'PMID': pmid,
        'PMC_ID': pmc_id,
        'Title': title,
        'Abstract': abstract,
        'Authors': authors,
        'Journal': journal,
        'Year': date_year,
        'DOI': doi,
        'MeSH_Terms': mesh_terms,
        'PubMed_URL': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else '',
        'Fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def fetch_full_text(pmc_id):
    '''
    Fetch and parse full text from PubMed Central XML.
    
    Downloads PMC XML and extracts abstract, full text, introduction,
    and conclusion sections. Uses heuristic section detection when
    standard XML structure is not available.
    
    Args:
        pmc_id (str): PubMed Central identifier (e.g., 'PMC6510241').
    
    Returns:
        dict: Full text data with keys:
            - Full_text (str): Complete article body text
            - Intro (str): Introduction section (first 33% if no headers)
            - Conclusion (str): Discussion/Conclusion (last 33% if no headers)
            - Abstract (str): Article abstract
    '''
    
    try:
        logger.info(f'   Fetching text for {pmc_id}...')
        efetch_handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
            )
        xml_text = efetch_handle.read()
        efetch_handle.close()
        
        tree = ET.fromstring(xml_text)
        
        full_text = ''
        body_elems = tree.findall(".//body")
        for body in body_elems:
            for p in body.findall(".//p"):
                full_text += get_text_recursive(p) + '\n'
        
        all_p = []
        for p in tree.findall(".//body//p"):
            all_p.append(get_text_recursive(p))
            
        intro, conclusion = '', ''
        
        # If full text available, but titles are nonconforming,
        # introduction-body-conclusion is divided as 30%-40%-30%
        if len(all_p) > 5:
            intro = ' '.join(all_p[:len(all_p)//3])
            conclusion = ' '.join(all_p[-len(all_p)//3:])
        else:
            if len(full_text) > 500:
                split_point = len(full_text) // 3
                intro = full_text[:split_point]
                conclusion = full_text[-split_point:]
            else:
                intro = conclusion = ''
        
        full_text = clean_text(full_text)
        intro = clean_text(intro)
        conclusion = clean_text(conclusion)
        
        logger.info(f'\tSymbols, full text: {len(full_text):4d} introduction:{len(intro):4d} conclusion:{len(conclusion):4d}')
        
        return {
            'Full_text': full_text,
            'Intro': intro,
            'Conclusion': conclusion,
        }
        
    except Exception as e:
        logger.warning(f'\tError while fetching full text for {pmc_id}: {e}')
        return {
            'Full_text': '',
            'Intro': '',
            'Conclusion': '',
            }

def get_text_recursive(element):
    '''
    Recursively extract all text from an XML element and its children.
    
    Walks through element tree and collects both .text and .tail attributes,
    which together contain all text content in ElementTree.
    
    EXCLUDING metadata/math tags.
    
    Args:
        element (Element): XML element from ElementTree.
    
    Returns:
        str: Space-joined text from element and all descendants.
    '''
    
    EXCLUDE_TAGS = {
        'tex-math', 'mml:math', 'disp-formula', 'inline-formula',
        'table', 'table-wrap', 'fig', 'graphic', 'media',
        'custom-meta', 'custom-meta-group', 'kwd-group',
        'article-categories', 'article-id', 'history',
        'alternatives'
    }
    
    text = []
    
    # removing exclude tags
    for child in element.iter():
        if child.tag in EXCLUDE_TAGS:
            continue
        
        if child.text:
            text.append(child.text.strip())
        if child.tail:
            text.append(child.tail.strip())
    
    return ' '.join(text)

def clean_text(text):
    '''
    Cleans texts of HTML, special symbols and extra spaces.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Очищенный текст
    '''
    if not text or not isinstance(text, str):
        return ''
    
    # removing html
    soup = BeautifulSoup(text, "html.parser")

    for element in soup(['script', 'style', 'table', 'figure']):
        element.decompose()
    
    text = soup.get_text('\n')
    
    # removing PubMed specific notes
    text = re.sub(
        r'Copyright\s+©?\s*\d{4}.*?(Ltd|Inc|Press|Publisher|Society)\.?',
        '',
        text,
        flags=re.IGNORECASE
        )
    text = re.sub(
        r'How to cite this article:.*',
        '',
        text,
        flags=re.IGNORECASE
        )
    text = re.sub(
        r'Communicated by:.*',
        '',
        text,
        flags=re.IGNORECASE
        )
    text = re.sub(
        r'This article is licensed under.*',
        '',
        text,
        flags=re.IGNORECASE
        )
    
    # unicoding
    text = unicodedata.normalize('NFKC', text)
    
    # cleaning urls + dois
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    
    # cleaning emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # cleaning punctuation/special symbols
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # extra spaces
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'  +', ' ', line.strip()) for line in lines]
    text = '\n'.join(line for line in cleaned_lines if line)

    # underscores and short dashes
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'_{2,}', '', text)
    
    # extra elipses
    text = re.sub(r'\.{2,}', '.', text)
    
    text = text.strip()
    
    text = re.sub(r'\n\s*\n', '\n\n', text) 
    
    return text


def collect_alzheimer(queries, target_articles):
    '''
    Main pipeline to collect Alzheimer's research articles with full text.
    
    Orchestrates the complete workflow: search -> filter -> fetch metadata ->
    download full texts. Ensures balanced distribution across queries and
    deduplicates results.
    
    Args:
        queries (list): List of search query strings.
        target_articles (int, optional): Total target number of articles across
            all queries. Defaults to 100.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - PMID, PMC_ID, Title, Authors, Journal, Year, DOI
            - Abstract, intro, conclusion, full_text, PubMed_URL, id
    
    Raises:
        ValueError: If no PMC articles with text are found.
    '''
    all_pmids = set()
    articles_per_query = target_articles // len(queries)

    for query in queries:
        pmids = search_pubmed(query, articles_per_query)
        all_pmids.update(pmids)
        time.sleep(1)
    
    pmid_list = list(all_pmids)
    logger.info(f'Total unique PMIDs with PMC: {len(pmid_list)} of target {target_articles}')
    
    if not pmid_list:
        raise ValueError('No PMIDs with PMC found')
    
    if len(pmid_list) < target_articles:
        logger.warning('Total unique PMIDs less than target')
        
    logger.info('Fetching metadata...')
    articles = fetch_article_details(pmid_list)
    
    fetched_articles = []
    logger.info('Fetching texts...')
    for article in articles:
        pmc_id = article['PMC_ID']
        logger.info(f'[{len(fetched_articles)+1}/{len(articles)}] {pmc_id}')
        pmc_data = fetch_full_text(pmc_id)
        fetched_article = {**article, **pmc_data}
        fetched_articles.append(fetched_article)
        
        time.sleep(0.5)
        
    df = pd.DataFrame(fetched_articles)
    
    final_df = df[[
        'PMID',
        'PMC_ID',
        'Title',
        'Authors',
        'Journal',
        'Year',
        'DOI',
        'MeSH_Terms',
        'PubMed_URL',
        'Fetched',
        'Abstract',
        'Intro',
        'Conclusion',
        'Full_text',
    ]].copy()
    
    logger.info(f'DONE! Total: {len(final_df)}, full text: {(final_df["Full_text"].str.len() > 500).sum()}')
    
    return final_df

if __name__ == '__main__':
    queries = [
        'Alzheimer\'s disease targets',
        'Alzheimer therapeutic targets',
        'Alzheimer drug targets'
    ]
    
    try:
        df = collect_alzheimer(queries, target_articles=100)
        df = df[df['Full_text'].str.len() > 0]
        df.to_csv('alzheimer_corpus.csv', index=False)
        logger.info(f'Saved {len(df)} articles data to alzheimer_corpus.csv')
    except Exception as e:
        logger.error(f'Error: {e}')