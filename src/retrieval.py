# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:51:11 2026

@author: csvww
"""

from rank_bm25 import BM25Okapi
from src.vectorizing import VectorStore
from config.logging_config import get_module_logger, setup_logging
from config.config import Config

setup_logging(log_level='INFO')
_logger = get_module_logger(__name__)

class HybridRetriever:
    def __init__(self, vector_store, alpha=None):
        '''
        Initialize hybrid dense+sparse retriever.
        
        Args:
            vectorstore (VectorStore): FAISS vector store.
            alpha (float): Dense weight (1-alpha = BM25 sparse). Defaults to 0.5.
        '''
        self.vector_store = vector_store
        self.alpha = alpha or Config.HYBRID_ALPHA    
        
        _logger.info('Building BM25 index...')
        tokenized_corpus = [
            chunk['text'].lower().split() 
            for chunk in vector_store.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f'BM25 index ready: {len(tokenized_corpus)} documents')
    
    def retrieve(self, query, k=None):
        '''
        Retrieve top-K chunks using hybrid dense+sparse scoring.
        
        Args:
            query (str): Search query.
            k (int): Top-K results. Defaults to 10.
            
        Returns:
            Tuple[List[Dict], List[float]]: (top chunks, fused scores).
        '''
        # bm25
        k = k or Config.RETRIEVAL_TOP_K
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        if bm25_scores.max() > 0:
            bm25_scores_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_norm = bm25_scores
        
        # dense
        dense_chunks, dense_scores = self.vector_store.search(query, k=50)
        
        dense_score_map = {
            chunk['chunk_id']: score 
            for chunk, score in zip(dense_chunks, dense_scores)
        }
        
        # fusing
        fused_scores = []
        for idx, chunk in enumerate(self.vector_store.chunks):
            chunk_id = chunk['chunk_id']
            
            bm25_score = bm25_scores_norm[idx]
            dense_score = dense_score_map.get(chunk_id, 0.0)
            
            fused_score = (1 - self.alpha) * bm25_score + self.alpha * dense_score
            fused_scores.append((idx, fused_score, chunk))
        
        # sort, top k
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_chunks = [item[2] for item in fused_scores[:k]]
        top_scores = [item[1] for item in fused_scores[:k]]
        
        return top_chunks, top_scores
    
    def retrieve_filter(self, query, k=None, year_min=None, section=None):
        '''
        Retrieve with metadata filters (year, section).
        
        Args:
            query (str): Search query.
            k (int): Top-K results.
            year_min (int, optional): Minimum publication year.
            section (str, optional): Section filter ('abstract', 'fulltext').
            
        Returns:
            Tuple[List[Dict], List[float]]: Filtered chunks and scores.
        '''
        k = k or Config.RETRIEVAL_TOP_K
        candidates, scores = self.retrieve(query, k=k*3)
        
        filtered = []
        filtered_scores = []
        
        for chunk, score in zip(candidates, scores):
            if year_min and chunk['year'] < year_min:
                continue
            
            if section and chunk['section'] != section:
                continue
            
            filtered.append(chunk)
            filtered_scores.append(score)
            
            if len(filtered) >= k:
                break
        
        return filtered, filtered_scores

if __name__ == '__main__':
    store = VectorStore()
    store.load(Config.VECTOR_STORE_PREFIX)
    
    retriever = HybridRetriever(store, alpha=Config.HYBRID_ALPHA)  # XXX 60% dense, 40% BM25
    
    # Test
    queries = [
        'What are potential targets for Alzheimer\'s disease treatment?',
        'Are targets druggable with small molecules or biologics?',
        'What additional studies are needed?'
    ]
    
    for query in queries:
        _logger.info(f'Query: {query}')
        
        chunks, scores = retriever.retrieve(query, k=5)
        
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            print(f'[{i}] Score: {score:.4f} | PMID: {chunk["pmid"]} | Year: {chunk["year"]}')
            print(f'\tSection: {chunk["section"]}')
            print(f'\t{chunk["text"][:300]}...')












