# -*- coding: utf-8 -*-
'''
Created on Thu Jan 29 15:28:33 2026

@author: csvww
'''

import streamlit as st
import os
from config.config import Config

# Custom CSS
st.set_page_config(
    page_title=Config.STREAMLIT_TITLE,
    page_icon='🧠',
    layout=Config.STREAMLIT_LAYOUT
)

st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .api-configured {
        background-color: #d4edda;
        color: #155724;
    }
    .api-not-configured {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
''', unsafe_allow_html=True)

# SIDEBAR SETTINGS
with st.sidebar:
    # api settings
    st.header('⚙️ API Configuration')
    
    provider = st.selectbox(
        '🤖 Select LLM Provider',
        options=['Perplexity', 'OpenAI', 'Anthropic'],
        help='Choose your preferred AI provider'
    )
    
    provider_lower = provider.lower()
    
    st.markdown(f'**{provider} API Key:**')
    
    if f'{provider_lower}_api_key' not in st.session_state:
        st.session_state[f'{provider_lower}_api_key'] = os.getenv(f'{provider_lower.upper()}_API_KEY', '')
    
    api_key_input = st.text_input(
        'Enter API Key',
        type='password',
        value=st.session_state[f'{provider_lower}_api_key'],
        key=f'{provider_lower}_key_input',
        help=f'Get your key from {provider}\'s website'
    )
    
    if api_key_input:
        st.session_state[f'{provider_lower}_api_key'] = api_key_input
    

    if provider_lower == 'openai':
        model = st.selectbox(
            'Model',
            options=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            index=1,
            help='gpt-4o-mini is fastest and cheapest'
        )
    elif provider_lower == 'anthropic':
        model = st.selectbox(
            'Model',
            options=['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
            index=0,
            help='Sonnet is best for research'
        )
    elif provider_lower == 'perplexity':
        model = st.selectbox(
            'Model',
            options=['sonar-pro',
                     'sonar-small-online',
                     'llama-3.1-sonar-large-online',
                     'llama-3.1-sonar-small-128k-online'
                     ],
            index=0,
            help='IDK just use sonar'
        )
    
    from src.pipeline import LLMFactory
    
    try:
        llm = LLMFactory.create(provider_lower, api_key_input, model)
        is_configured = llm.is_configured()
    except Exception:
        is_configured = False
    
    if is_configured:
        st.markdown(
            '<div class=\'api-status api-configured\'>✅ API Key Valid</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class=\'api-status api-not-configured\'>❌ API Key Missing/Invalid</div>',
            unsafe_allow_html=True
        )
        st.info(f'💡 Get {provider} API key:\n\n'
               f'**Perplexity:** Settings → API → Generate\n\n'
               f'**OpenAI:** platform.openai.com/api-keys\n\n'
               f'**Anthropic:** console.anthropic.com/settings/keys')
    
    st.markdown('---')
    
    # retrieval settings
    st.subheader('🔍 Retrieval Settings')
    
    k_retrieval = st.slider(
        'Number of sources',
        min_value=5,
        max_value=20,
        value=Config.RETRIEVAL_TOP_K,
        help='More sources = more comprehensive'
    )
    
    year_options = ['2020+', '2022+', '2024+']
    default_year_str = f'{Config.YEAR_MIN_DEFAULT}+'
    default_index = year_options.index(default_year_str) if default_year_str in year_options else 0
    year_filter = st.selectbox('Publication year', options=year_options, index=default_index)
    
    year_min = {
        '2020+': 2020,
        '2022+': 2022,
        '2024+': 2024
    }[year_filter]
    
    st.markdown('---')
    
    # Example questions
    st.subheader('📚 Example Questions')
    
    examples = [
        ' What are potential targets for Alzheimer\'s disease treatment?',
        'Are the targets druggable with small molecules, biologics, or other modalities?',
        'What additional studies are needed to advance these targets?'
        ]
    
    for ex in examples:
        if st.button(ex, key=f'ex_{ex[:20]}', width='stretch'):
            st.session_state.example_query = ex

# MAIN APP
st.markdown('<h1 class=\'main-header\'>🧠 Alzheimer\'s Drug Target Discovery Assistant</h1>', 
            unsafe_allow_html=True)

st.markdown(f'''
RAG-powered system analyzing **100 scientific publications** (2014-2025).  
**Current Provider:** {provider} {'✅' if is_configured else '❌ (configure in sidebar)'}
''')

if 'initialized' not in st.session_state and is_configured:
    with st.spinner('🔄 Loading RAG system...'):
        try:
            from src.vectorizing import VectorStore
            from src.retrieval import HybridRetriever
            from src.pipeline import RAGPipeline
            
            store = VectorStore()
            store.load(Config.VECTOR_STORE_PREFIX)
            retriever = HybridRetriever(store, alpha=Config.HYBRID_ALPHA)
            
            rag = RAGPipeline(
                retriever=retriever,
                provider=provider_lower,
                api_key=api_key_input,
                model=model if provider_lower != 'perplexity' else None
            )
            
            st.session_state.rag = rag
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f'❌ Failed to load RAG system: {e}')
            st.session_state.initialized = False

# Query input
query = st.text_area(
    '🔍 Enter your research question:',
    value=st.session_state.get('example_query', ''),
    height=100,
    placeholder='e.g., What are the most promising therapeutic targets for Alzheimer\'s disease?'
)

col1, col2 = st.columns([1, 5])

with col1:
    search_button = st.button('🔎 Search', type='primary', use_container_width=True)

with col2:
    if not is_configured:
        st.warning('⚠️ Configure API key in sidebar to enable search')

# Search logic
if search_button and query:
    if not is_configured:
        st.error('❌ Please configure API key in sidebar first!')
    elif not st.session_state.get('initialized'):
        st.error('❌ RAG system not initialized!')
    else:
        with st.spinner(f'🧠 Analyzing literature with {provider}...'):
            try:
                response = st.session_state.rag.generate_answer(
                    query,
                    k=k_retrieval,
                    year_min=year_min
                )
                
                # answer
                st.markdown('---')
                st.subheader('💡 Answer')
                st.markdown(response['answer'])
                
                # sources
                st.markdown('---')
                st.subheader(f"📄 Sources ({len(response['sources'])} retrieved)")
                
                tab1, tab2 = st.tabs(['📋 Source List', '📊 Metadata'])
                
                with tab1:
                    for i, source in enumerate(response['sources'], 1):
                        text_preview = source.get('text_preview', 'No preview')
                        
                        with st.expander(
                            f"[{i}] {source.get('title', 'Untitled')} ({source.get('year', 'N/A')}) "
                            f"Score: {source.get('score', 0):.3f}"
                        ):
                            st.markdown(f'''
                            **🔗 [PMID:{source.get('pmid', 'N/A')}](https://pubmed.ncbi.nlm.nih.gov/{source.get('pmid', 'N/A')}/)**
                            
                            **📅 Year:** {source.get('year', 'N/A')} | **📂 Section:** {source.get('section', 'N/A')}
                            
                            **📝 Preview:**
                            {text_preview}
                            ''')
                
                with tab2:
                    import pandas as pd
                    df_meta = pd.DataFrame([
                        {
                            'PMID': s['pmid'],
                            'Year': s['year'],
                            'Score': f"{s['score']:.3f}",
                            'Section': s['section']
                        }
                        for s in response['sources']
                    ])
                    st.dataframe(df_meta, use_container_width=True)
                
                # Stats
                st.markdown('---')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Sources', response['num_sources'])
                col2.metric('Avg Score', f"{response['retrieval_avg_score']:.3f}")
                col3.metric('Provider', response['provider'].title())
                col4.metric('Context', f"{response['context_length']} chars")
                
            except Exception as e:
                st.error(f'❌ Error: {str(e)}')
                st.exception(e)

# Footer
st.markdown('---')
st.caption(f'''
**System:** Hybrid Retrieval (BM25 + SPECTER) | {provider} Generation  
**Data:** 100 PubMed/PMC articles (2020-2025) covering tau, APP, TREM2, BACE1, and more
''')


