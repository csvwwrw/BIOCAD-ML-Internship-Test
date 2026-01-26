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

# Initializing Entrez credentials (email + API key)
Entrez.email = os.getenv('ENTREZ_EMAIL')
Entrez.api_key = os.getenv('ENTREZ_API_KEY')

def search_pubmed(query, target_results=100):
    '''
    Search PubMed database for articles with free full text available.
    
    Performs iterative searches with Open Access filter until target number
    of articles with PMC IDs is reached. Uses rate limiting to comply with
    NCBI guidelines (3 requests/sec).
    
    Args:
        query (str): Search term or PubMed query string (e.g., "Alzheimer targets").
        target_results (int, optional): Target number of articles with full text. Defaults to 100.
    
    Returns:
        list: List of PMID strings for articles with PMC full text available. Returns empty list if no articles found.
    
    Raises:
        Exception: If connection to PubMed fails (caught and logged).
    '''
    print(f'Performing PubMed search for query: "{query}", target results: {target_results}')
    
    pmc_pmids = set()
    ret_start = 0
    ret_batch = 100
    
    while len(pmc_pmids) < target_results:
        try:
            esearch_handle = Entrez.esearch(
                db='pubmed',
                term=f'{query} AND "free full text"[sb]',
                retstart=ret_start,
                retmax=ret_batch,
                )
            record = Entrez.read(esearch_handle)
            esearch_handle.close()
    
            new_pmids = record['IdList']
            print(f'   Search batch {ret_start//ret_batch + 1}: {len(new_pmids)} PMIDs found')
    
            if not new_pmids:
                print('   No more search results')
                break
            
            pmc_count_batch = check_full_text(new_pmids)
            pmc_pmids.update(pmc_count_batch)
            
            print(f'   PMCs with full text available: {len(pmc_count_batch)}')
            
            ret_start += ret_batch
            time.sleep(0.34)
            
            if len(record['IdList']) < ret_batch:
                print(' No more search results')
    
        except Exception as e:
            print (f'   Error while batch searching for PubMed results: {e}')
            break
    
    print(f'   Total PMCs with full text available for "{query}": {len(pmc_pmids)}')
    
    return list(pmc_pmids)

def check_full_text(pmid_batch):
    '''
    Check which PMIDs have PMC full text available.
    
    Fetches MEDLINE records for a batch of PMIDs and filters those
    with PMC ID, indicating full text availability in PubMed Central.
    
    Args:
        pmid_batch (list): List of PMID strings to check (max 200).
    
    Returns:
        list: List of PMID strings that have PMC full text available. Returns empty list if none found or on error.
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
        
        for record in records:
            pmc_id = record.get('PMC', '')
            if pmc_id:
                pmc_pmids.append(record.get('PMID'))
    
        handle.close()
    
    except Exception as e:
        print(f'   Error checking full text for PMC: {e}')
    
    return pmc_pmids
    

def fetch_article_details(pmid_list, batch_size=10):
    '''
    Fetch detailed metadata for a list of PMIDs.
    
    Retrieves bibliographic information (title, authors, journal, etc.)
    for articles using MEDLINE format. Processes in batches to manage
    API rate limits.
    
    Args:
        pmid_list (list): List of PMID strings to fetch.
        batch_size (int, optional): Number of PMIDs per batch. Defaults to 10. Smaller batches are more reliable for large datasets.
    
    Returns:
        list: List of dictionaries containing article metadata. Each dict includes:
            - PMID, PMC_ID, Title, Abstract, Authors, Journal, Year, DOI
            - MeSH_Terms, Affiliations, PubMed_URL, Fetched timestamp
    '''
    
    articles = []
    
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i:i+batch_size]
        print(f'   Processing batch {i//batch_size + 1}: PMID {i+1}-{min(i+batch_size, len(pmid_list))}')
        
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
            print(f'   Error during processing of the batch: {e}')
            continue
    
    print(f'Total articles processed: {len(articles)}')
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
            - Affiliations (str): Semicolon-separated affiliations
            - PubMed_URL (str): Direct URL to PubMed page
            - Fetched (str): Timestamp of data retrieval (YYYY-MM-DD HH:MM:SS)
            '''
    pmid = record.get('PMID', '')
    pmc_id = record.get("PMC", "")
    title = record.get('TI', '')
    abstract = record.get('AB', '')
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
    affiliations = record.get('AD', [])
    affiliations = '; '.join(affiliations) if affiliations else ''
    
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
        'Affiliations': affiliations,
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
            - full_text (str): Complete article body text
            - intro (str): Introduction section (first 33% if no headers)
            - conclusion (str): Discussion/Conclusion (last 33% if no headers)
            - abstract (str): Article abstract
            - has_full_text (bool): True if full_text length > 200 chars
    '''
    
    try:
        print(f'   Fetching text for {pmc_id}...')
        efetch_handle = Entrez.efetch(
            db='pmc',
            id=pmc_id,
            rettype='xml',
            retmode='xml'
            )
        xml_text = efetch_handle.read()
        efetch_handle.close()
        
        tree = ET.fromstring(xml_text)
        
        abstract = ''
        for elem in tree.findall(".//abstract"):
            abstract += get_text_recursive(elem)
        
        full_text = ''
        body_elems = tree.findall(".//body")
        for body in body_elems:
            for p in body.findall(".//p"):
                full_text += get_text_recursive(p) + '\n'
        
        all_p = []
        for p in tree.findall(".//body//p"):
            all_p.append(get_text_recursive(p))
            
        intro, conclusion = '', ''
        
        # If full text available, but titles are nonconforming, introduction-body-conclusion is divided as 30%-40%-30%
        if len(all_p) > 5:
            intro = ' '.join(all_p[:len(all_p)//3])
            conclusion = ' '.join(all_p[-len(all_p)//3:])
        
        has_text = len(full_text) > 200
        
        print(f"   Symbols, full text: {len(full_text):4d} introduction:{len(intro):4d} conclusion:{len(conclusion):4d}")
        
        return {
            'full_text': full_text,
            'intro': intro,
            'conclusion': conclusion,
            'abstract': abstract,
            'has_full_text': has_text
        }
        
    except Exception as e:
        print(f"   Error while fetching full text for {pmc_id}: {e}")
        return {'full_text': '', 'intro': '', 'conclusion': '', 
                'abstract': '', 'has_full_text': False}

def get_text_recursive(element):
    '''
    Recursively extract all text from an XML element and its children.
    
    Walks through element tree and collects both .text and .tail attributes,
    which together contain all text content in ElementTree.
    
    Args:
        element (Element): XML element from ElementTree.
    
    Returns:
        str: Space-joined text from element and all descendants.
    '''
    text = []
    for child in element.iter():
        if child.text:
            text.append(child.text.strip())
        if child.tail:
            text.append(child.tail.strip())
            
    return ' '.join(text)

def collect_alzheimer(queries, target_articles=100): #XXX переделать оптимизацию?
    '''
    Main pipeline to collect Alzheimer's research articles with full text.
    
    Orchestrates the complete workflow: search → filter → fetch metadata →
    download full texts. Ensures balanced distribution across queries and
    deduplicates results.
    
    Args:
        queries (list): List of search query strings.
        target_articles (int, optional): Total target number of articles across all queries. Defaults to 100.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - PMID, PMC_ID, Title, Authors, Journal, Year, DOI
            - Abstract, intro, conclusion, full_text, PubMed_URL, id
            - has_full_text (bool): indicates successful text retrieval
    
    Raises:
        ValueError: If no PMC articles with text are found.
    '''
    all_pmids = set()
    articles_per_query = target_articles//len(queries)

    for query in queries:
        pmids = search_pubmed(query, articles_per_query)
        all_pmids.update(pmids)
        time.sleep(1)
    
    pmid_list = list(all_pmids)[:articles_per_query * len(queries)]
    print(f'\nTotal unique PMIDs with text: {len(pmid_list)} of {len(all_pmids)}')
    
    if not pmid_list:
        raise ValueError('No PMC articles with text found')
        
    print('Fetching metadata and texts for PMC articles...')
    articles = fetch_article_details(pmid_list)
    
    fetched_articles = []
    for article in articles:
        pmc_id = article['PMC_ID']
        print(f'[{len(fetched_articles)+1}/{len(articles)}] {pmc_id}')
        pmc_data = fetch_full_text(pmc_id)
        fetched_article = {**article, **pmc_data}
        fetched_articles.append(fetched_article)
        
        time.sleep(0.5)
        
    df = pd.DataFrame(fetched_articles)
    print('DONE!')
    print(f'Total articles: {len(df)}')
    print(f'Full text articles: {df['has_full_text'].sum()} ({df['has_full_text'].mean():.1%})')  
    
    final_df = df[[
        'PMID',
        'PMC_ID',
        'Title',
        'Authors',
        'Journal',
        'Year',
        'DOI',
        'Abstract',
        'intro',
        'conclusion',
        'full_text',
        'PubMed_URL'
    ]].copy()
    
    final_df['id'] = final_df['PMID']
    
    return final_df

if __name__ == '__main__':
    queries = [
        'Alzheimer\'s disease targets',
        'Alzheimer therapeutic targets',
        'Alzheimer drug targets'
    ]
    
    df = collect_alzheimer(
        queries, 
        target_articles=120
    )
    
    df.to_csv('alzheimer_corpus.csv', index=False)