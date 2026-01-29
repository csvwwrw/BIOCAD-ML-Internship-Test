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
import logging
import re
from bs4 import BeautifulSoup
import unicodedata
from config.config import Config
from config.logging_config import get_module_logger, setup_logging

setup_logging(log_level='INFO', log_file=Config.LOGS_DIR / 'pubmed_retrieval_log.py')
_logger = logging.getLogger(__name__)

Entrez.email = Config.PUBMED_EMAIL
REQUEST_DELAY = 0.12 if Config.PUBMED_API_KEY else 0.34

def search_pubmed(query, target_results=Config.PUBMED_TARGET_ARTICLES):
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
    _logger.info(f'Performing PMC search for query: "{query}", target results: {target_results}')
    
    pmc_pmids = set()
    ret_start = 0
    ret_batch = min(target_results, 100)
    
    while len(pmc_pmids) < target_results:
        try:
            esearch_handle = Entrez.esearch(
                db='pubmed',
                term=f'{query} AND "free full text"[sb] NOT preprint[pt] AND ("2020"[DP]:"2026"[DP])',
                retstart=ret_start,
                retmax=ret_batch,
                sort='relevance'
                )
            record = Entrez.read(esearch_handle)
            esearch_handle.close()
    
            new_pmids = record['IdList']
            _logger.info(f'\tSearch batch {ret_start//ret_batch + 1}: {len(new_pmids)} PMIDs found')
    
            if not new_pmids:
                _logger.info('No more search results')
                break
            
            pmc_count_batch = check_full_text(new_pmids)
            pmc_pmids.update(pmc_count_batch)
            
            _logger.info(f'\tPMIDs with PMCs (full text avaliable): {len(pmc_count_batch)}')
            
            ret_start += ret_batch
            time.sleep(REQUEST_DELAY)
            
            if len(record['IdList']) < ret_batch:
                _logger.info('No more search results')
                break
    
        except Exception as e:
            _logger.error(f'\tError while batch searching for PubMed results: {e}')
            break
        
    if len(pmc_pmids) < target_results:
        _logger.warning(f'Only {len(pmc_pmids)} of {target_results} eligible articles found.')
    else:
        _logger.info(f'\tTotal PMCs with full text available for "{query}": {len(pmc_pmids)}')
    
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
    
    pmc_pmids = set()
    
    try:
        handle = Entrez.efetch(
            db='pubmed',
            id=pmid_batch,
            rettype='medline',
            retmode='text',
            retmax=min(len(pmid_batch), 200)
        )
        records = Medline.parse(handle)
        
        total_checked = 0
        for record in records:
            total_checked += 1
            pmc_id = record.get('PMC', '')
            if pmc_id:
                pmc_pmids.add(record.get('PMID'))
            else:
                _logger.debug(f'No PMC for PMID {record.get("PMID")}')
    
        handle.close()
        _logger.debug(f'{len(pmc_pmids)} PMCs of {total_checked} PMIDs')
    
    except Exception as e:
        _logger.error(f'Error checking full text for PMC: {e}')
    
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
        _logger.info(f'\tProcessing batch {i//batch_size + 1}: PMID {i+1}-{min(i+batch_size, len(pmid_list))}')
        
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
            
            time.sleep(REQUEST_DELAY)
        
        except Exception as e:
            _logger.warning(f'\tError during processing of the batch: {e}')
            continue
    
    _logger.info(f'Total articles processed: {len(articles)}')
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
    Fetch and parse full text from PubMed Central XML with intelligent section
    extraction.
    
    Downloads PMC XML and extracts sections using two-stage strategy:
    1. Header-based extraction (searches for Introduction/Conclusion keywords 
                                in <sec> titles)
    2. Heuristic fallback (splits text 33%-33% if headers not found)
    
    Args:
        pmc_id (str): PubMed Central identifier
    
    Returns:
        dict: Full text data with keys:
            - Full_text (str): Complete article body text
            - Intro (str): Introduction section
            - Conclusion (str): Discussion/Conclusion section
    
    Raises:
        Exception: If PMC fetch fails (caught and logged).
    '''
    
    try:
        _logger.info(f'\tFetching text for {pmc_id}...')
        
        efetch_handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
        )
        xml_text = efetch_handle.read()
        efetch_handle.close()
        
        tree = ET.fromstring(xml_text)

        # extracting full text
        full_text = ''
        body_elems = tree.findall('.//body')
        for body in body_elems:
            for p in body.findall('.//p'):
                full_text += get_text_recursive(p) + '\n'
        
        full_text = clean_text(full_text)
        
        # header-based extraction
        intro, conclusion = '', ''
        extraction_method = 'none'
        
        for sec in tree.findall('.//sec'):
            title_elem = sec.find('title')
            if title_elem is not None:
                title = title_elem.text.lower() if title_elem.text else ''
                
                sec_text = []
                for p in sec.findall('.//p'):
                    sec_text.append(get_text_recursive(p))
                
                joined = ' '.join(sec_text)
                
                if any(kw in title for kw in ['introduction', 'background', 'intro']):
                    if not intro or len(joined) > len(intro): # longest match
                        intro = joined
                        extraction_method = 'header'
                
                elif any(kw in title for kw in ['conclusion', 'discussion', 'summary', 'concluding']):
                    if not conclusion or len(joined) > len(conclusion): # longest match
                        conclusion = joined
                        extraction_method = 'header'
        
        intro = clean_text(intro)
        conclusion = clean_text(conclusion)
        
        
        # heuristic
        min_section_length = 200
        
        if len(intro) < min_section_length or len(conclusion) < min_section_length:
            _logger.warning(f'\tHeader extraction failed for {pmc_id}, using heuristic split')
            
            all_p = []
            for p in tree.findall('.//body//p'):
                all_p.append(get_text_recursive(p))
            
            # 33 - 33 - 33 separation (if no section titles)
            if len(all_p) > 5:
                intro = ' '.join(all_p[:len(all_p)//3])
                conclusion = ' '.join(all_p[-len(all_p)//3:])
                extraction_method = 'heuristic_paragraph'
            
            # character-based split (last resort)
            elif len(full_text) > 500:
                split_point = len(full_text) // 3
                intro = full_text[:split_point]
                conclusion = full_text[-split_point:]
                extraction_method = 'heuristic_character'
            
            else:
                # article too short, no meaningful sections
                intro = conclusion = ''
                extraction_method = 'failed'
            
            intro = clean_text(intro)
            conclusion = clean_text(conclusion)
        
        _logger.info(
            f'\tSymbols: full_text={len(full_text):4d} | '
            f'intro={len(intro):4d} | conclusion={len(conclusion):4d} | '
            f'method={extraction_method}'
        )
        
        return {
            'Full_text': full_text,
            'Intro': intro,
            'Conclusion': conclusion,
        }
    
    except Exception as e:
        _logger.error(f'Error fetching {pmc_id}: {str(e)}')
        return {
            'Full_text': '',
            'Intro': '',
            'Conclusion': '',
            'Intro_Conclusion': '',
            'extraction_method': 'error'
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
        str: Cleaned text
    '''
    if not text or not isinstance(text, str):
        return ''
    
    # removing html
    soup = BeautifulSoup(text, 'html.parser')

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
    _logger.info(f'Total unique PMIDs with PMC: {len(pmid_list)} of target {target_articles}')
    
    if not pmid_list:
        raise ValueError('No PMIDs with PMC found')
    
    if len(pmid_list) < target_articles:
        _logger.warning('Total unique PMIDs less than target')
        
    _logger.info('Fetching metadata...')
    articles = fetch_article_details(pmid_list)
    
    fetched_articles = []
    _logger.info('Fetching texts...')
    for article in articles:
        pmc_id = article['PMC_ID']
        _logger.info(f'[{len(fetched_articles)+1}/{len(articles)}] {pmc_id}')
        pmc_data = fetch_full_text(pmc_id)
        if len(pmc_data['Full_text']) < 500:
            _logger.warning(f'{pmc_id}: Full text too short ({len(pmc_data["Full_text"])} chars')
            continue
        fetched_article = {**article, **pmc_data}
        fetched_articles.append(fetched_article)
        
        time.sleep(REQUEST_DELAY)
        
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
    
    _logger.info(f'DONE! Total: {len(final_df)}, full text: {(final_df["Full_text"].str.len() > 500).sum()}')
    
    return final_df

if __name__ == '__main__':
    queries = [
        'Alzheimer\'s disease targets',
        'Alzheimer therapeutic targets',
        'Alzheimer drug targets'
    ]
    
    try:
        df = collect_alzheimer(queries, target_articles=Config.PUBMED_TARGET_ARTICLES)
        df = df[df['Full_text'].str.len() > 0]
        df.to_csv(f'{Config.CORPUS_CSV}', index=False)
        _logger.info(f'Saved {len(df)} articles data to {Config.CORPUS_CSV}')
    except Exception as e:
        _logger.error(f'Error: {e}')