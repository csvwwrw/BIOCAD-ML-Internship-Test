# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:04:05 2026

@author: csvww
"""

from config.logging_config import get_module_logger, setup_logging
from config.config import Config
import re
import pandas as pd
import hashlib
import pickle

setup_logging(log_level='INFO')
_logger = get_module_logger(__name__)

def chunking_text(text, chunk_size=None, overlap=None):
    '''
    Split text into overlapping semantic chunks preserving paragraph boundaries.
    
    Args:
        text (str): Input text to chunk.
        chunksize (int): Target chunk size in characters. Defaults to 800.
        overlap (int): Character overlap between chunks. Defaults to 150.
        
    Returns:
        List[str]: List of chunked text segments (min 100 chars each).
    '''
    
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:

        if len(para) > chunk_size * 1.5: # if paragraph too long
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(sent) > 100 and current_length + len(sent) > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        next_chunk = current_chunk[-2:] + [sent] # overlap
                        current_chunk = next_chunk
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        chunks.append(sent)
                        current_chunk = [sent]
                        current_length = len(sent)
                else:
                    current_chunk.append(sent)
                    current_length += len(sent)
        else:
            if current_length + len(para) > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-1:] + [para] if current_chunk else [para]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(para)
                current_length += len(para)
                
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # just in case:
    chunks = [c.strip() for c in chunks if len(c.strip()) > Config.MIN_CHUNK_LENGTH]
    
    return chunks

# is class better?
class DocumentProcessor:
    def __init__(self, chunk_size=800, overlap=150):
        '''
        Initialize document chunking processor.
        
        Args:
            chunksize (int): Target chunk size in chars. Defaults to 800.
            overlap (int): Chunk overlap in chars. Defaults to 150.
        '''
        self.chunk_size = Config.CHUNK_SIZE
        self.overlap = Config.CHUNK_OVERLAP
        
    
    def process_corpus(self, df):
        '''
        Process entire corpus into searchable chunks with metadata.
        
        Args:
            df (pd.DataFrame): Corpus DataFrame with PMID, Title, Year,
                Abstract, etc.
            
        Returns:
            List[Dict]: Chunks with metadata: {chunk_id, text, pmid, title,
                                               year, section, ...}
            
        Raises:
            ValueError: If required columns missing.
        '''
        _logger.info(f'Processing {len(df)} articles...')
    
        all_chunks = []
        
        for idx, row in df.iterrows():
            pmid = row['PMID']
            title = row.get('Title')
            year = int(row.get('Year'))
            journal = row.get('Journal')
            
            mesh_terms = [t.strip() for t in str(row['MeSH_Terms']).split(';') if t.strip()]
            

            if pd.notna(row.get('Abstract')) and len(row['Abstract']) > 100:
                abstract_chunks = chunking_text(row['Abstract'])
                for i, chunk_text in enumerate(abstract_chunks):
                    chunk_id = hashlib.md5(f'{pmid}_abstract_{i}'.encode()).hexdigest()[:16]
                    all_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'pmid': pmid,
                        'title': title,
                        'year': year,
                        'section': 'abstract',
                        'journal': journal,
                        'mesh_terms': mesh_terms,
                        'chunk_index': i
                    })
            
            key_text = ''
            key_text += row['Intro'] + '\n\n'
            key_text += row['Conclusion']
            
            if len(key_text) > 200:
                key_chunks = chunking_text(key_text)
                for i, chunk_text in enumerate(key_chunks):
                    chunk_id = hashlib.md5(f"{pmid}_key_{i}".encode()).hexdigest()[:16]
                    all_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'pmid': pmid,
                        'title': title,
                        'year': year,
                        'section': 'key_sections',
                        'journal': journal,
                        'mesh_terms': mesh_terms,
                        'chunk_index': i
                    })
            
            if pd.notna(row.get('Full_text')) and len(row['Full_text']) < 100000:
                full_chunks = chunking_text(row['Full_text'])
                # XXX every 3rd: redundancy
                for i, chunk_text in enumerate(full_chunks[::2]):
                    chunk_id = hashlib.md5(f'{pmid}_full_{i}'.encode()).hexdigest()[:16]
                    all_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'pmid': pmid,
                        'title': title,
                        'year': year,
                        'section': 'full_text',
                        'journal': journal,
                        'mesh_terms': mesh_terms,
                        'chunk_index': i
                    })
            
            if (idx + 1) % 10 == 0:
                _logger.info(f'Processed {idx+1}/{len(df)} articles')
        
        _logger.info(f'Created {len(all_chunks)} chunks')
        _logger.info(f'\tAbstract: {sum(1 for c in all_chunks if c["section"]=="abstract")}')
        _logger.info(f'\tKey sections: {sum(1 for c in all_chunks if c["section"]=="key_sections")}')
        _logger.info(f'\tFull text: {sum(1 for c in all_chunks if c["section"]=="full_text")}')
        
        return all_chunks

if __name__ == '__main__':
    df = df = pd.read_csv(Config.CORPUS_CSV)
    
    processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP)
    chunks = processor.process_corpus(df)
    
    with open(Config.CHUNKS_PKL, 'wb') as f:
        pickle.dump(chunks, f)
    
    avg_length = sum(len(c['text']) for c in chunks) / len(chunks)
    _logger.info(f'Saved {len(chunks)} chunks, avg {avg_length:.0f} chars')
