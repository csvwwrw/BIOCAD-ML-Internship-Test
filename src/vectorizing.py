# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:41:07 2026

@author: csvww
"""

from sentence_transformers import SentenceTransformer
import faiss
import pickle
from config.logging_config import get_module_logger, setup_logging
from config.config import Config

setup_logging(log_level='INFO')
_logger = get_module_logger(__name__)


class VectorStore:
    def __init__(self, model_name=None):
        '''
        Initialize FAISS vector store with embedding model.
        
        Args:
            model_name (str): SentenceTransformer model. Defaults to SPECTER.
        '''
        
        model_name = model_name or Config.EMBEDDING_MODEL
        _logger.info(f'Loading embedding model: {model_name}')
        self.model = SentenceTransformer(model_name)
        self.dimension = Config.EMBEDDING_DIM
        self.index = None
        self.chunks = None
    
    def build_index(self, chunks, batch_size=None):
        '''
        Build FAISS index from chunk embeddings.
        
        Args:
            chunks (List[Dict]): Chunk metadata from DocumentProcessor.
            batch_size (int): Embedding batch size. Defaults to 32.
            
        Returns:
            np.ndarray: Embeddings array.
        '''
        batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        self.chunks = chunks
        
        _logger.info(f'Generating embeddings for {len(chunks)} chunks...')
        texts = [c['text'] for c in chunks]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
            )
        
        _logger.info('Building FAISS index...')
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        _logger.info(f'Index built: {self.index.ntotal} vectors')
        
        return embeddings

    def search(self, query, k=10):
        '''
        Search vector store for similar chunks.
        
        Args:
            query (str): Search query.
            k (int): Top-K results. Defaults to 10.
            
        Returns:
            Tuple[List[Dict], List[float]]: (matching chunks, similarity scores).
        '''
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        retrieved = [self.chunks[idx] for idx in indices[0]]
        similarities = scores[0].tolist()
        
        return retrieved, similarities
    
    def save(self, path_prefix=None):
        '''
        Saves vectors to path_prefix
        '''
        path_prefix = path_prefix or Config.VECTOR_STORE_PREFIX
        faiss.write_index(self.index, f'{path_prefix}_faiss.index')
        with open(f'{path_prefix}_chunks.pkl', 'wb') as f:
            pickle.dump(self.chunks, f)
        _logger.info(f'Saved to {path_prefix}_*')
    
    def load(self, path_prefix=None):
        '''
        Loads vectors from path_prefix
        '''
        path_prefix = path_prefix or Config.VECTOR_STORE_PREFIX
        self.index = faiss.read_index(f'{path_prefix}_faiss.index')
        with open(f'{path_prefix}_chunks.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        print(f'Loaded {self.index.ntotal} vectors')

if __name__ == "__main__":
    with open(Config.CHUNKS_PKL, 'rb') as f:
        chunks = pickle.load(f)
    
    store = VectorStore(model_name=Config.EMBEDDING_MODEL)
    embeddings = store.build_index(chunks, batch_size=Config.EMBEDDING_BATCH_SIZE)
    store.save(str(Config.VECTOR_STORE_PREFIX))
    
    test_query = 'What are potential targets for Alzheimer\'s disease treatment?'
    results, scores = store.search(test_query, k=5)
    
    _logger.info(f'Test query: {test_query}')
    for i, (chunk, score) in enumerate(zip(results, scores), 1):
        _logger.info(f'[{i}] Score: {score:.3f} | PMID: {chunk["pmid"]}')
        _logger.info(f'\t{chunk["text"][:200]}...')


























