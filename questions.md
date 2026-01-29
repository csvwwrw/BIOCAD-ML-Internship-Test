# Question 1: What Data Modalities Can the Solution Be Expanded To?

Based on the analysis of 115 articles with 3,550 chunks, the current system focuses on scientific literature. But LLM don't work with experimental data and have little predictive power in science. The following modalities would significantly enhance drug target discovery:

### **1. Structured Biological Databases**

- **Protein structures
    - *Rationale*: Analysis lacks 3D structural context for druggability assessment
- **Drug-target databases**
    - *Rationale*: Current druggability analysis is keyword-based, but lacks experimental binding affinity data
- **Genomic databases**
    - *Rationale*: MeSH analysis shows 16.9% genetics-related terms, but missing variant-disease associations

### **2. Multi-Omics Data**

- **Transcriptomics
    - *Rationale*: The network analysis captured 576 gene-gene interactions, but lacks cell-type-specific expression patterns for microglia (1,201 mentions) - WHICH IS A VERY IMPORTANT AD TARGET
- **Proteomics**
    - *Rationale*: Identified 50+ proteins but no quantitative abundance data
- **Metabolomics**
    - *Rationale*: Cholesterol (199 mentions), glucose (163), iron (175) detected but metabolic flux missing
Additionally, this will help with pathology/norma comparisons.

### **3. Real-World Data**
*Rationale*: Need genotype-phenotype associations for identified risk genes (APOE: 326 mentions)


### **4. Chemical \& Pharmacological Data**

- **Molecular structures**
    - *Rationale*: Identified 370 chemical/drug entities but no SMILES/InChI representations for similarity search
- **Pharmacokinetics**: ADMET databases
    - *Rationale*: Drug delivery systems appear in 9 articles but BBB permeability (important for AD) data absent

### **6. Pathways \& Mechanisms**

- **Pathway databases**
    - *Rationale*: Extracted 8 interaction types (activates: 99, inhibits: 77, phosphorylates: 21) but incomplete pathway coverage

***
## Question 2: How Can This Be Implemented?

### **Architecture: Multimodal RAG System**

#### **Phase 1: Data Integration Layer**

**A. Structured Data Retrieval**

```python
class StructuredRetriever:
    def __init__(self):
        self.uniprot_api = UniProtClient()
        self.drugbank_db = DrugBankSQL()
        self.pdb_api = RCSBClient()
    
    def retrieve_protein_info(self, gene_name):
        # Fetch sequence, GO terms, domains
        protein = self.uniprot_api.query(gene_name)
        structure = self.pdb_api.get_structure(protein.pdb_id)
        druggability = assess_structure(structure)
        return {'sequence': protein.seq, 
                'domains': protein.domains,
                'druggability': druggability}
```

**Implementation approach**:
- Create separate retrievers for each modality
- Store structured data in relational DB (PostgreSQL) + graph DB (Neo4j for pathways)
- Use API wrappers for live queries to external databases

#### **Phase 2: Multimodal Embedding**

**B. Cross-Modal Encoders**

- **Text**: Continue using SPECTER (768-dim)
- **Protein sequences**: ESM-2 (evolutionary scale modeling, 1280-dim)
- **Chemical structures**: MolFormer/ChemBERTa (768-dim from SMILES)
- **Images**: ResNet-50 for brain scans (2048-dim)

**Fusion strategy**:

```python
class MultimodalEmbedder:
    def __init__(self):
        self.text_model = SentenceTransformer('allenai-specter')
        self.protein_model = ESM2Model()
        self.chem_model = ChemBERTa()
        self.fusion = CrossAttentionFusion(dims=[768, 1280, 768])
    
    def embed_target(self, gene_name):
        # Get embeddings from each modality
        text_emb = self.text_model.encode(text_mentions)
        prot_emb = self.protein_model.encode(sequence)
        chem_emb = self.chem_model.encode(known_ligands)
        
        # Late fusion with attention
        fused = self.fusion([text_emb, prot_emb, chem_emb])
        return fused
```

#### **Phase 3: Hybrid Retrieval**

**C. Modality-Specific Indexes**

- **Text**: FAISS (current: 3,550 vectors)
- **Proteins**: BLAST/MMseqs2 for sequence similarity
- **Chemicals**: FPSim2 for molecular fingerprints
- **Pathways**: Graph neural networks on Neo4j

**Retrieval pipeline**:

```python
class MultimodalRetriever:
    def retrieve(self, query, k=10):
        # Parallel retrieval
        text_results = self.text_index.search(query, k)
        protein_results = self.protein_index.search(extracted_genes, k)
        pathway_results = self.pathway_gnn.search(interactions, k)
        
        # Reciprocal rank fusion
        fused_results = reciprocal_rank_fusion([
            text_results,
            protein_results, 
            pathway_results
        ], weights=[0.5, 0.3, 0.2])
        
        return fused_results
```


#### **Phase 4: Context Enrichment**

**D. Multi-Source Context Builder**
Extend current system to:

```python
def build_multimodal_context(query, retrieved_chunks):
    context = []
    
    # Text chunks (existing)
    for chunk in retrieved_chunks[:5]:
        context.append(f"[TEXT] {chunk['text']}")
    
    # Protein data
    for gene in extracted_genes[:3]:
        protein_info = self.structured_db.get_protein(gene)
        context.append(f"[PROTEIN] {gene}: {protein_info}")
    
    # Clinical trials
    for trial in self.trials_db.search(query)[:2]:
        context.append(f"[TRIAL] {trial['nct_id']}: {trial['outcome']}")
    
    # Pathway context
    for pathway in self.pathway_db.find_relevant(extracted_genes):
        context.append(f"[PATHWAY] {pathway['name']}: {pathway['mechanism']}")
    
    return '\n\n'.join(context)
```


#### **Phase 5: LLM Integration**

**E. Modality-Aware Prompting**

```python
def create_multimodal_prompt(query, context):
    return f"""
You are analyzing drug targets for Alzheimer's from multiple data sources.

**RESEARCH QUESTION**: {query}

**LITERATURE EVIDENCE**:
{context['text_chunks']}

**STRUCTURAL DATA**:
- TREM2: Druggability score 17.4/25 (High) [file:20]
- Crystal structure: 4×4 transmembrane receptor
- Binding pocket: 12Å depth, hydrophobic

**CLINICAL VALIDATION**:
- NCT04592874: TREM2 agonist phase 2 (recruiting)
- Endpoint: CSF biomarker reduction

**PATHWAY CONTEXT**:
- TREM2 activates 99 downstream targets [file:28]
- Key interaction: TREM2 --[binds]--> DAP12 --[activates]--> SYK

**ANALYSIS**: Synthesize evidence across modalities...
"""
```


#### **Phase 6: Evaluation \& Validation**

**F. Cross-Modal Consistency Checks**

- Verify text-derived hypotheses against structural databases
- Cross-reference clinical trial outcomes with literature claims
- Use pathway databases to validate predicted interactions

***

## Question 3: What Models Did You Choose and Why?

### **Embedding Model: `allenai-specter`**

**Rationale**:
- **Domain-specific**: Trained on 1.9M scientific papers with citation-based objectives
- **Performance**: EDA semantic analysis achieved strong clustering (optimal 13 topics with positive silhouette score)
- **Dimension**: 768-dim balances expressiveness with computational efficiency
- **Evidence**: UMAP projections show clear year-based and topic-based separation

**Why not alternatives?**:
- General models (BERT, MPNet): Lack scientific domain knowledge
- SciBERT: Smaller corpus, worse on citation tasks
- GPT embeddings: Expensive, black-box, inconsistent versions


### **Vector Store: FAISS (IndexFlatIP)**

**Rationale**:
- **Speed**
- **Accuracy**: Exact search (no approximation) critical for medical applications
- **Scalability**
- **Reproducibility**: Deterministic results

**Why not alternatives?**:
- Pinecone/Weaviate: Overkill
- HNSW: Approximate search risks missing relevant medical literature


### **Sparse Retrieval: BM25 (Okapi)**

**Rationale**:
- **Complementary**: Captures exact term matches
- **Performance**: Strong EDA BM25 scores (19.91 avg for druggability query)
- **Robustness**: Handles rare terms that dense models might miss
- **Evidence**: EDA lexical analysis shows high term specificity

**Hyperparameters** (from code):
- k1=1.5: Balances term frequency saturation
- b=0.75: Medium document length normalization

### **Hybrid Fusion: α = 0.6 Dense + 0.4 Sparse**

**Rationale**:
- **Balance**: Dense captures semantics ("therapeutic targets" ≈ "drug discovery")
- **Precision**: Sparse ensures exact gene names aren't missed
- **Domain**: Scientific text benefits from both (EDA Type-Token Ratio = 0.045 shows vocabulary richness)

**Why 60/40 split?**:
- Pilot testing showed dense embeddings better for conceptual queries
- Sparse critical for entity-centric queries

### **Chunking Strategy: 800 chars, 150 overlap**

**Rationale**:
- **Context preservation**: Avg paragraph = 699 chars, so 800 chars captures 1-2 paragraphs
- **Semantic coherence**: EDA analysis shows avg 30.9 chunks/article fits within token limits
- **Overlap**: 150 chars (~25 words) prevents sentence splits at boundaries
- **Evidence**: Chunks distributed around suggested 2,000-char window

**Why not alternatives?**:
- Smaller (400 chars): Fragments sentences, loses context
- Larger (1,600 chars): Exceeds model context, reduces granularity
- Fixed sentences: Variable length (some papers have 200-word sentences)

### **Topic Modeling: BERTopic + KeyBERT**

**Rationale**:
- **Quality**: Found 13 optimal topics covering drug therapy (42.5%), pathology (22.8%), genetics (16.9%)
- **Interpretability**: KeyBERT provides human-readable topic labels
- **Clustering**: Silhouette score validates semantic groupings
- **Evidence**: Clear topic separation in UMAP visualization

**Why not alternatives?**:
- LDA: Bag-of-words, ignores semantics
- NMF: Requires manual topic count, less stable
- Top2Vec: Less control over clustering

### **Biomedical NER: scispaCy (`en_ner_bionlp13cg_md`, `en_ner_bc5cdr_md`)**

**Rationale**:
- **Precision**
- **Coverage**: Separate models for genes/proteins vs. diseases/chemicals
- **Validation**: Built co-occurrence network with 576 interactions
- **Evidence**: Entity network shows biologically plausible clusters

**Why not alternatives?**:
- BioBERT-NER: Less accurate on protein names
- PubTator API: Rate limits, offline unavailable
- General NER: Misses biomedical entities

### **LLM Integration: Multi-Provider (Perplexity, OpenAI, Anthropic)**

**Rationale**:
- **Flexibility**: Researchers can choose based on budget/speed
- **Robustness**: Fallback if one API fails
- **Specialization**:
    - Perplexity Sonar: Grounded, reduces hallucination
    - GPT-4o-mini: Fast, cost-effective (\$0.15/1M tokens)
    - Claude 3.5 Sonnet: Best reasoning for complex biomedical queries

**Generation parameters**:
- Temperature=0.3: Balance between creativity and factuality
- Max tokens=1,500: Sufficient for 3-4 paragraph answers
- Context: 7 papers × 4 chunks = ~6,000 tokens input[

### **Summary Statistics

| Metric                   | Value         |
| :----------------------- | :------------ |
| Total articles           | 115           |
| Total chunks             | 3,550         |
| Avg chunks/article       | 30.9          |
| Unique genes detected    | 697 (APP top) |
| Pathway interactions     | 576           |
| Data quality             | 0% missing    |
| BM25 score (best query)  | 19.91         |
| Druggable targets (High) | 4 of 30       |
| Topic coherence          | 13 topics     |
| Years covered            | 2020-2025     |
