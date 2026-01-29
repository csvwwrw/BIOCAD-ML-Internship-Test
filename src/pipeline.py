# -*- coding: utf-8 -*-
'''
Created on Sat Jan 24 15:26:13 2026
@author: csvwwrw
'''

from typing import List, Dict, Optional
import os
import requests
from abc import ABC, abstractmethod
from config.logging_config import get_module_logger, setup_logging
from config.config import Config

setup_logging(log_level='INFO')
_logger = get_module_logger(__name__)


class BaseLLM(ABC):
    
    @abstractmethod
    def generate(self, prompt, max_tokens=2000):
        pass
    
    @abstractmethod
    def is_configured(self):
        pass

class PerplexityLLM(BaseLLM):
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('PPLX_API_KEY')
        self.base_url = 'https://api.perplexity.ai/chat/completions'
    
    def is_configured(self):
        return bool(self.api_key and self.api_key.startswith('pplx-'))
    
    def generate(self, prompt, max_tokens=2000):
        if not self.is_configured():
            return 'Perplexity API key not configured'
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Try multiple models
        models = ["sonar-pro", "sonar-small-online", "llama-3.1-sonar-large-online"]
        
        for model in models:
            try:
                payload = {
                    'model': model,
                    'messages': self._format_messages(prompt),
                    'max_tokens': max_tokens,
                    'temperature': Config.TEMPERATURE
                }
                
                response = requests.post(
                    self.base_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=90
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                    
            except Exception:
                continue
        
        return 'All Perplexity models failed. Check API key or quota.'
    
    def _format_messages(self, prompt):
        return [
            {
                'role': 'system',
                'content': 'You are an expert in Alzheimer\'s drug discovery. '
                          'Cite sources as [PMID:XXXXX]. Answer based on provided literature.'
            },
            {'role': 'user', 'content': prompt}
        ]

class OpenAILLM(BaseLLM):
    
    def __init__(self, api_key=None, model='gpt-4o-mini'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        
        if self.is_configured():
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
    
    def is_configured(self):
        return bool(self.api_key and self.api_key.startswith('sk-'))
    
    def generate(self, prompt, max_tokens=Config.MAX_TOKENS):
        if not self.is_configured():
            return 'OpenAI API key not configured'
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert in Alzheimer\'s drug discovery. '
                                  'Answer based ONLY on provided scientific literature. '
                                  'Cite sources as [PMID:XXXXX].'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f'OpenAI API error: {str(e)}'

class AnthropicLLM(BaseLLM):
    
    def __init__(self, api_key=None, model='claude-3-5-sonnet-20241022'):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.client = None
        
        if self.is_configured():
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
    
    def is_configured(self):
        return bool(self.api_key and self.api_key.startswith('sk-ant-'))
    
    def generate(self, prompt, max_tokens=2000):
        if not self.is_configured():
            return 'Anthropic API key not configured'
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,
                system='You are an expert in Alzheimer\'s drug discovery. '
                       'Answer based ONLY on provided scientific literature. '
                       'Cite sources as [PMID:XXXXX].',
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response.content[0].text
            
        except Exception as e:
            return f'Anthropic API error: {str(e)}'

class LLMFactory:
    
    @staticmethod
    def create(provider, api_key=None, model=None):
        provider = provider.lower()
        
        if provider == 'perplexity':
            return PerplexityLLM(api_key)
        
        elif provider == 'openai':
            model = model or 'gpt-4o-mini'
            return OpenAILLM(api_key, model)
        
        elif provider == 'anthropic':
            model = model or 'claude-3-5-sonnet-20241022'
            return AnthropicLLM(api_key, model)
        
        else:
            raise ValueError(f'Unknown provider: {provider}')


class RAGPipeline:
    
    def __init__(self, retriever, provider='perplexity', api_key=None, model=None):

        self.retriever = retriever
        self.provider = provider
        self.llm = LLMFactory.create(provider, api_key, model)
    
    def generate_answer(self, 
                       query, 
                       k,
                       year_min):

        chunks, scores = self.retriever.retrieve_filter(query, k=k, year_min=year_min)
        
        context = self._build_context(chunks, scores)
        
        prompt = self._create_prompt(query, context)
        
        answer = self.llm.generate(prompt, max_tokens=1500)
        
        return {
            'query': query,
            'answer': answer,
            'sources': self._format_sources(chunks, scores),
            'context_length': len(context),
            'num_sources': len(chunks),
            'retrieval_avg_score': float(sum(scores) / len(scores)),
            'provider': self.provider,
            'api_configured': self.llm.is_configured()
        }
    
    def _build_context(self, chunks, scores):
        pmid_groups = {}
        
        for chunk, score in zip(chunks, scores):
            pmid = chunk['pmid']
            if pmid not in pmid_groups:
                pmid_groups[pmid] = {
                    'title': chunk['title'],
                    'year': chunk['year'],
                    'chunks': [],
                    'max_score': 0.0
                }
            pmid_groups[pmid]['chunks'].append(chunk['text'])
            pmid_groups[pmid]['max_score'] = max(pmid_groups[pmid]['max_score'], score)
        
        sorted_sources = sorted(pmid_groups.items(), key=lambda x: x[1]['max_score'], reverse=True)
        
        context_parts = []
        for i, (pmid, data) in enumerate(sorted_sources[:Config.CONTEXT_MAX_PAPERS]):
            chunks_text = "\n\n".join(data['chunks'][:Config.CONTEXT_MAX_CHUNKS_PER_PAPER])
            context_parts.append(
                f"""SOURCE {i+1} [PMID:{pmid}]
                Year: {data['year']} | Title: {data['title']}
                ---
                {chunks_text}
                ---"""
                )
        
        return '\n\n' + '═'*80 + '\n\n'.join(context_parts)
    
    def _create_prompt(self, query, context):
        return f"""**RESEARCH QUESTION**
    {query}
    
    **SCIENTIFIC LITERATURE** (Use ONLY these sources)
    {context}
    
    **ANALYSIS:**
    1. Identify specific protein/gene targets
    2. Discuss druggability (small molecules, biologics, etc.)
    3. Cite sources inline as [PMID:XXXXX]
    4. State "Limited evidence" if data insufficient
    
    **Answer precisely based on literature:**
    """
    
    def _format_sources(self, chunks, scores):
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                'pmid': chunk.get('pmid', 'N/A'),
                'title': chunk.get('title', 'Untitled')[:80] + '...',
                'year': chunk.get('year', 2024),
                'section': chunk.get('section', 'full_text'),
                'score': float(scores[i]) if i < len(scores) else 0.0,
                'text_preview': chunk.get('text', '')[:300] + '...'
            })
        return sources

if __name__ == '__main__':
    
    from dotenv import load_dotenv

    load_dotenv()
    PPLX_API_KEY = os.getenv('PPLX_API_KEY')
    
    print('Testing provider LLM system...')
    
        
    llm = LLMFactory.create('perplexity')
    print(f'Configured: {llm.is_configured()}')
    
    if llm.is_configured():
        answer = llm.generate('What are tau protein inhibitors? Answer in 2 sentences.')
        print(f'Answer: {answer[:200]}...')
