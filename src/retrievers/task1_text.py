try:
    import cohere
except ImportError:
    cohere = None
    
try:
    import vertexai
except ImportError:
    vertexai = None

try:
    import voyageai
except ImportError:
    voyageai = None
    
try:
    from gritlm import GritLM
except ImportError:
    GritLM = None
    
try:
    from vllm import LLM, PoolingParams, SamplingParams
except ImportError:
    LLM = None

try:
    from InstructorEmbedding import INSTRUCTOR
except ImportError:
    INSTRUCTOR = None

import numpy as np
try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None
    
try:
    import tiktoken
except ImportError:
    tiktoken = None

from tqdm import tqdm, trange
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import os
from openai import AzureOpenAI
from pathlib import Path
import json
import torch
from src.caching import SmartCache
from src.utils import cut_text, add_instruct_concatenate, last_token_pool




def cut_text(text, tokenizer, threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text, tokenizer, threshold=6000):
    token_ids = tokenizer.encode(text, disallowed_special=())
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def add_instruct_concatenate(texts, task, instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts, task, instruction):
    return [[instruction.format(task=task), t] for t in texts]

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids, doc_ids, scores, excluded_ids):
    assert len(scores)==len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did, s in zip(doc_ids, doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(), key=lambda x:x[1], reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores


@torch.no_grad()
def retrieval_sf_qwen_e5(queries, query_ids, documents, doc_ids, task, model_id, instructions, cache_dir, excluded_ids, long_context, **kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral', device_map="auto", torch_dtype=torch.bfloat16).eval()
        max_length = kwargs.get('doc_max_length', 4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        max_length = kwargs.get('doc_max_length', 8192)
        model.config.use_cache = False
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        max_length = kwargs.get('doc_max_length', 8192)
        model.config.use_cache = False
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto", torch_dtype=torch.bfloat16).eval()
        max_length = kwargs.get('doc_max_length', 4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries, task=task, instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size', 1)

    # ==================== SMART CACHING WITH DOC ID MAPPING ====================
    cache = SmartCache(cache_dir, model_id, task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    # Encode new documents if any
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = []
        
        for start_idx in trange(0, len(docs_to_encode), batch_size, desc="Encoding"):
            batch_dict = tokenizer(
                docs_to_encode[start_idx:start_idx+batch_size], 
                max_length=max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(model.device)
            
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).float().cpu().numpy()
            new_embeddings.append(embeddings)
            
            if (start_idx + batch_size) % 1000 == 0:
                print(f"  Encoded {start_idx + batch_size}/{len(docs_to_encode)} documents")
        
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    # Build final embedding array in correct order
    print("Building final embedding array in correct order...")
    doc_emb = cache.get_embeddings_array(doc_ids)
    # ==================== END SMART CACHING ====================

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:", doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
        batch_dict = tokenizer(
            queries[start_idx:start_idx + batch_size], 
            max_length=max_length, 
            padding=True,
            truncation=True, 
            return_tensors='pt'
        ).to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).float().cpu().tolist()
        query_emb += embeddings
    
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


def retrieval_bm25(queries, query_ids, documents, doc_ids, excluded_ids, long_context, **kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        cur_scores = sorted(all_scores[str(query_id)].items(), key=lambda x:x[1], reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    return all_scores


@torch.no_grad()
def retrieval_sbert_bge(queries, query_ids, documents, doc_ids, task, instructions, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        queries = add_instruct_concatenate(texts=queries, task=task, instruction=instructions['query'])
    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    
    batch_size = kwargs.get('batch_size', 128)
    
    # ==================== SMART CACHING ====================
    cache = SmartCache(cache_dir, model_id, task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = model.encode(docs_to_encode, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # ==================== END SMART CACHING ====================
    
    query_emb = model.encode(queries, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_instructor(queries, query_ids, documents, doc_ids, task, instructions, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    if model_id=='inst-l':
        model = SentenceTransformer('hkunlp/instructor-large')
    elif model_id=='inst-xl':
        model = SentenceTransformer('hkunlp/instructor-xl')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size', 4)
    model.max_seq_length = kwargs.get('doc_max_length', 2048)
    
    # ==================== SMART CACHING ====================
    cache = SmartCache(cache_dir, model_id, task, long_context, batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = model.encode(docs_to_encode, show_progress_bar=True, batch_size=batch_size, 
                                      normalize_embeddings=True, prompt=instructions['document'].format(task=task))
        cache.update(new_embeddings, docs_to_encode_ids)
        cache.save()
    
    doc_embs = cache.get_embeddings_array(doc_ids)
    # ==================== END SMART CACHING ====================
    
    query_embs = model.encode(queries, batch_size=batch_size, show_progress_bar=True, 
                             prompt=instructions['query'].format(task=task), normalize_embeddings=True)
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_grit(queries, query_ids, documents, doc_ids, task, instructions, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    customized_checkpoint = kwargs.get('checkpoint', None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use', customized_checkpoint)
    def unwrap(m):
        while hasattr(m, "module"):
            m = m.module
        return m

    model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")

    grit = unwrap(model)
    backbone = unwrap(grit.model)

    # THIS is the important one
    backbone.config.use_cache = False

    # Optional; only if it exists
    gc = getattr(backbone, "generation_config", None)
    if gc is not None:
        gc.use_cache = False

    grit.eval()
    torch.set_grad_enabled(False)


    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length', 256)
    doc_max_length = kwargs.get('doc_max_length', 2048)
    print("doc max length:", doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size', 1)
    
    # ==================== SMART CACHING ====================
    cache = SmartCache(cache_dir, model_id, task, long_context, batch_size)
    ignore_cache = kwargs.pop('ignore_cache', False)
    cache.load(ignore_cache=ignore_cache)
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        new_embeddings = model.encode(docs_to_encode, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        if not ignore_cache:
            cache.update(new_embeddings, docs_to_encode_ids)
            cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # ==================== END SMART CACHING ====================
    
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_embedding_openai_azure_threaded(texts, azure_client, tokenizer, model="text-embedding-3-large", max_retries=3):
    """Get embeddings from Azure OpenAI with batching for Azure limits"""
    AZURE_BATCH_LIMIT = 16
    all_embeddings = []
    
    # Filter out empty strings
    texts = [text if text.strip() else " " for text in texts]
    
    for i in range(0, len(texts), AZURE_BATCH_LIMIT):
        batch = texts[i:i + AZURE_BATCH_LIMIT]
        
        for attempt in range(max_retries):
            try:
                response = azure_client.embeddings.create(
                    input=batch,
                    model=model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
    
    return all_embeddings


def encode_document_batch(batch_data):
    """Encode a single batch of documents - runs in thread"""
    batch_idx, batch_texts, azure_client, tokenizer, deployment_name = batch_data
    try:
        embeddings = get_embedding_openai_azure_threaded(
            texts=batch_texts,
            azure_client=azure_client,
            tokenizer=tokenizer,
            model=deployment_name
        )
        return batch_idx, embeddings
    except Exception as e:
        print(f"Error in batch {batch_idx}: {e}")
        return batch_idx, None


def retrieval_openai_azure(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Truncate texts
    new_queries = [cut_text_openai(text=q, tokenizer=tokenizer) for q in queries]
    queries = new_queries
    
    new_documents = [cut_text_openai(text=d, tokenizer=tokenizer) for d in documents]
    documents = new_documents
    
    # THREADING SETTINGS FOR DOCUMENTS
    doc_batch_size = 20  # Send 20 documents per thread
    max_workers = 2     # 20 concurrent threads
    
    # For cache organization
    cache_batch_size = kwargs.get('batch_size', 512)
    
    # Initialize Azure OpenAI client
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    # ==================== SMART CACHING ====================
    cache = SmartCache(cache_dir, model_id, task, long_context, cache_batch_size)
    cache.load()
    
    docs_to_encode, docs_to_encode_ids = cache.get_uncached_docs(doc_ids, documents)
    
    # ==================== THREADED DOCUMENT ENCODING ====================
    if len(docs_to_encode) > 0:
        print(f"Encoding documents with {max_workers} threads, {doc_batch_size} docs per batch...")
        
        # Prepare batches
        batch_tasks = []
        for idx in range(0, len(docs_to_encode), doc_batch_size):
            batch_texts = docs_to_encode[idx:idx + doc_batch_size]
            batch_tasks.append((
                idx,  # batch index
                batch_texts,
                azure_client,
                tokenizer,
                deployment_name
            ))
        
        print(f"Total batches to process: {len(batch_tasks)}")
        
        # Execute with ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(encode_document_batch, task_data): task_data[0] 
                for task_data in batch_tasks
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Encoding documents"):
                batch_idx = future_to_batch[future]
                try:
                    returned_idx, embeddings = future.result()
                    if embeddings is not None:
                        results[returned_idx] = embeddings
                except Exception as e:
                    print(f"Batch {batch_idx} generated an exception: {e}")
        
        # Reconstruct embeddings in correct order
        new_embeddings = []
        for idx in range(0, len(docs_to_encode), doc_batch_size):
            if idx in results:
                new_embeddings.extend(results[idx])
        
        print(f"Successfully encoded {len(new_embeddings)} documents")
        
        # Update cache (slice IDs to match successfully encoded documents)
        cache.update(new_embeddings, docs_to_encode_ids[:len(new_embeddings)])
        cache.save()
    
    doc_emb = cache.get_embeddings_array(doc_ids)
    # ==================== END SMART CACHING ====================
    
    # ==================== QUERIES (SEQUENTIAL - NO THREADING) ====================
    query_emb = []
    print("Encoding queries (sequential)...")
    query_batch_size = 100
    for idx in trange(0, len(queries), query_batch_size, desc="Encoding queries"):
        cur_emb = get_embedding_openai_azure_threaded(
            texts=queries[idx:idx + query_batch_size], 
            azure_client=azure_client,
            tokenizer=tokenizer,
            model=deployment_name
        )
        query_emb += cur_emb
    
    # Compute scores
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)

# def get_embedding_openai_azure(texts, azure_client, tokenizer, model="text-embedding-3-large", max_retries=3):
#     """Get embeddings from Azure OpenAI with batching for Azure limits"""
#     # Azure OpenAI has a limit of ~16 texts per request for embeddings
#     AZURE_BATCH_LIMIT = 16
#     all_embeddings = []
    
#     # Filter out empty strings
#     texts = [text if text.strip() else " " for text in texts]
    
#     for i in range(0, len(texts), AZURE_BATCH_LIMIT):
#         batch = texts[i:i + AZURE_BATCH_LIMIT]
        
#         for attempt in range(max_retries):
#             try:
#                 response = azure_client.embeddings.create(
#                     input=batch,
#                     model=model
#                 )
#                 batch_embeddings = [item.embedding for item in response.data]
#                 all_embeddings.extend(batch_embeddings)
#                 break
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
#                     time.sleep(2 ** attempt)  # Exponential backoff
#                 else:
#                     print(f"Failed after {max_retries} attempts: {e}")
#                     raise
    
#     return all_embeddings

# def retrieval_openai_azure(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
#     tokenizer = tiktoken.get_encoding("cl100k_base")
    
#     # Truncate texts
#     new_queries = []
#     for q in queries:
#         new_queries.append(cut_text_openai(text=q, tokenizer=tokenizer))
#     queries = new_queries
    
#     new_documents = []
#     for d in documents:
#         new_documents.append(cut_text_openai(text=d, tokenizer=tokenizer))
#     documents = new_documents
    
#     # Use smaller batch size for caching (not for API calls, which are handled internally)
#     batch_size = kwargs.get('batch_size', 512)  # Reduced from 1024
    
#     # Initialize Azure OpenAI client
#     azure_client = AzureOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("AZURE_API_VERSION"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#     )
    
#     deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    
#     # ==================== SMART CACHING (BATCH-BASED) ====================
#     cache_dir_path = Path(cache_dir) / 'doc_emb' / model_id / task / f"long_{long_context}_{batch_size}"
#     cache_dir_path.mkdir(parents=True, exist_ok=True)
    
#     doc_id_cache_path = cache_dir_path / 'doc_id_list.json'
    
#     # Load existing doc_id list if available
#     cached_doc_ids = []
#     if doc_id_cache_path.exists():
#         with open(doc_id_cache_path, 'r') as f:
#             cached_doc_ids = json.load(f)
    
#     # Build mapping of doc_id to embedding
#     cached_embeddings = {}
#     for idx, doc_id in enumerate(cached_doc_ids):
#         batch_idx = (idx // batch_size) * batch_size
#         cache_file = cache_dir_path / f'{batch_idx}.json'
#         if cache_file.exists() and doc_id not in cached_embeddings:
#             with open(cache_file) as f:
#                 batch_embs = json.load(f)
#                 local_idx = idx % batch_size
#                 if local_idx < len(batch_embs):
#                     cached_embeddings[doc_id] = batch_embs[local_idx]
    
#     print(f"Documents in corpus: {len(documents)}")
#     print(f"Already cached: {len(cached_embeddings)}")
    
#     # Collect documents to encode
#     docs_to_encode = []
#     docs_to_encode_ids = []
    
#     for doc_id, doc_text in zip(doc_ids, documents):
#         if doc_id not in cached_embeddings:
#             docs_to_encode.append(doc_text)
#             docs_to_encode_ids.append(doc_id)
    
#     print(f"Need to encode: {len(docs_to_encode)}")
    
#     # Encode new documents
#     if len(docs_to_encode) > 0:
#         print("Encoding new documents...")
#         for idx in trange(0, len(docs_to_encode), batch_size, desc="Encoding documents"):
#             batch_texts = docs_to_encode[idx:idx + batch_size]
#             batch_doc_ids = docs_to_encode_ids[idx:idx + batch_size]
            
#             # Use Azure client (handles sub-batching internally)
#             cur_emb = get_embedding_openai_azure(
#                 texts=batch_texts, 
#                 azure_client=azure_client, 
#                 tokenizer=tokenizer,
#                 model=deployment_name
#             )
            
#             for doc_id, emb in zip(batch_doc_ids, cur_emb):
#                 cached_embeddings[doc_id] = emb
        
#         # Save updated cache
#         print("Saving updated cache...")
#         all_doc_ids = list(cached_embeddings.keys())
#         with open(doc_id_cache_path, 'w') as f:
#             json.dump(all_doc_ids, f, indent=2)
        
#         # Save embeddings in batch files
#         for batch_start in range(0, len(all_doc_ids), batch_size):
#             batch_doc_ids = all_doc_ids[batch_start:batch_start + batch_size]
#             batch_embs = [cached_embeddings[doc_id] for doc_id in batch_doc_ids]
#             cache_file = cache_dir_path / f'{batch_start}.json'
#             with open(cache_file, 'w') as f:
#                 json.dump(batch_embs, f, indent=2)
    
#     # Build final embedding list in correct order
#     doc_emb = [cached_embeddings[doc_id] for doc_id in doc_ids]
#     # ==================== END SMART CACHING ====================
    
#     # Encode queries
#     query_emb = []
#     print("Encoding queries...")
#     for idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
#         cur_emb = get_embedding_openai_azure(
#             texts=queries[idx:idx + batch_size], 
#             azure_client=azure_client,
#             tokenizer=tokenizer,
#             model=deployment_name
#         )
#         query_emb += cur_emb
    
#     scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
#     scores = scores.tolist()
#     return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)
def retrieval_openai(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q, tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d, tokenizer=tokenizer))
    documents = new_documents
    
    batch_size = kwargs.get('batch_size', 1024)
    openai_client = OpenAI()
    
    # ==================== SMART CACHING (BATCH-BASED) ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / model_id / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    doc_id_cache_path = cache_dir_path / 'doc_id_list.json'
    
    # Load existing doc_id list if available
    cached_doc_ids = []
    if doc_id_cache_path.exists():
        with open(doc_id_cache_path, 'r') as f:
            cached_doc_ids = json.load(f)
    
    # Build mapping of doc_id to embedding
    cached_embeddings = {}
    for idx, doc_id in enumerate(cached_doc_ids):
        batch_idx = (idx // batch_size) * batch_size
        cache_file = cache_dir_path / f'{batch_idx}.json'
        if cache_file.exists() and doc_id not in cached_embeddings:
            with open(cache_file) as f:
                batch_embs = json.load(f)
                local_idx = idx % batch_size
                if local_idx < len(batch_embs):
                    cached_embeddings[doc_id] = batch_embs[local_idx]
    
    print(f"Documents in corpus: {len(documents)}")
    print(f"Already cached: {len(cached_embeddings)}")
    
    # Collect documents to encode
    docs_to_encode = []
    docs_to_encode_ids = []
    
    for doc_id, doc_text in zip(doc_ids, documents):
        if doc_id not in cached_embeddings:
            docs_to_encode.append(doc_text)
            docs_to_encode_ids.append(doc_id)
    
    print(f"Need to encode: {len(docs_to_encode)}")
    
    # Encode new documents
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        for idx in trange(0, len(docs_to_encode), batch_size):
            batch_texts = docs_to_encode[idx:idx + batch_size]
            batch_doc_ids = docs_to_encode_ids[idx:idx + batch_size]
            
            cur_emb = get_embedding_openai(texts=batch_texts, openai_client=openai_client, tokenizer=tokenizer)
            
            for doc_id, emb in zip(batch_doc_ids, cur_emb):
                cached_embeddings[doc_id] = emb
        
        # Save updated cache
        print("Saving updated cache...")
        all_doc_ids = list(cached_embeddings.keys())
        with open(doc_id_cache_path, 'w') as f:
            json.dump(all_doc_ids, f, indent=2)
        
        # Save embeddings in batch files
        for batch_start in range(0, len(all_doc_ids), batch_size):
            batch_doc_ids = all_doc_ids[batch_start:batch_start + batch_size]
            batch_embs = [cached_embeddings[doc_id] for doc_id in batch_doc_ids]
            cache_file = cache_dir_path / f'{batch_start}.json'
            with open(cache_file, 'w') as f:
                json.dump(batch_embs, f, indent=2)
    
    # Build final embedding list in correct order
    doc_emb = [cached_embeddings[doc_id] for doc_id in doc_ids]
    # ==================== END SMART CACHING ====================
    
    query_emb = []
    for idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


def retrieval_cohere(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    batch_size = kwargs.get('batch_size', 8192)
    cohere_client = cohere.Client()
    
    # ==================== SMART CACHING (BATCH-BASED) ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / model_id / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    doc_id_cache_path = cache_dir_path / 'doc_id_list.json'
    
    cached_doc_ids = []
    if doc_id_cache_path.exists():
        with open(doc_id_cache_path, 'r') as f:
            cached_doc_ids = json.load(f)
    
    cached_embeddings = {}
    for idx, doc_id in enumerate(cached_doc_ids):
        batch_idx = (idx // batch_size) * batch_size
        cache_file = cache_dir_path / f'{batch_idx}.json'
        if cache_file.exists() and doc_id not in cached_embeddings:
            with open(cache_file) as f:
                batch_embs = json.load(f)
                local_idx = idx % batch_size
                if local_idx < len(batch_embs):
                    cached_embeddings[doc_id] = batch_embs[local_idx]
    
    print(f"Documents in corpus: {len(documents)}")
    print(f"Already cached: {len(cached_embeddings)}")
    
    docs_to_encode = []
    docs_to_encode_ids = []
    
    for doc_id, doc_text in zip(doc_ids, documents):
        if doc_id not in cached_embeddings:
            docs_to_encode.append(doc_text)
            docs_to_encode_ids.append(doc_id)
    
    print(f"Need to encode: {len(docs_to_encode)}")
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        for idx in trange(0, len(docs_to_encode), batch_size):
            batch_texts = docs_to_encode[idx:idx + batch_size]
            batch_doc_ids = docs_to_encode_ids[idx:idx + batch_size]
            
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(texts=batch_texts, input_type="search_document",
                                                  model="embed-english-v3.0").embeddings
                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            
            for doc_id, emb in zip(batch_doc_ids, cur_emb):
                cached_embeddings[doc_id] = emb
        
        print("Saving updated cache...")
        all_doc_ids = list(cached_embeddings.keys())
        with open(doc_id_cache_path, 'w') as f:
            json.dump(all_doc_ids, f, indent=2)
        
        for batch_start in range(0, len(all_doc_ids), batch_size):
            batch_doc_ids = all_doc_ids[batch_start:batch_start + batch_size]
            batch_embs = [cached_embeddings[doc_id] for doc_id in batch_doc_ids]
            cache_file = cache_dir_path / f'{batch_start}.json'
            with open(cache_file, 'w') as f:
                json.dump(batch_embs, f, indent=2)
    
    doc_emb = [cached_embeddings[doc_id] for doc_id in doc_ids]
    # ==================== END SMART CACHING ====================
    
    query_emb = []
    for idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


def retrieval_voyage(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q, tokenizer=tokenizer, threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents, desc='preprocess documents'):
        new_documents.append(cut_text(text=d, tokenizer=tokenizer, threshold=16000))
    documents = new_documents

    batch_size = kwargs.get('batch_size', 1)
    voyage_client = voyageai.Client()
    
    # ==================== SMART CACHING ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / model_id / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading existing cache...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        
        print(f"Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = []
    docs_to_encode_ids = []
    
    for doc_id, doc_text in zip(doc_ids, documents):
        if doc_id not in cached_embeddings:
            docs_to_encode.append(doc_text)
            docs_to_encode_ids.append(doc_id)
    
    print(f"Documents in corpus: {len(documents)}")
    print(f"Already cached: {len(cached_embeddings)}")
    print(f"Need to encode: {len(docs_to_encode)}")
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        for i in trange(0, len(docs_to_encode), batch_size):
            success = False
            threshold = 16000
            cur_texts = docs_to_encode[i:i+batch_size]
            cur_doc_ids = docs_to_encode_ids[i:i+batch_size]
            count_over = 0
            exec_count = 0
            while not success:
                exec_count += 1
                if exec_count > 5:
                    print('voyage document too many times')
                    exit(0)
                try:
                    cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                    
                    next_index = len(cached_embeddings)
                    for j, doc_id in enumerate(cur_doc_ids):
                        cached_embeddings[doc_id] = cur_emb[j]
                        doc_id_to_index[doc_id] = next_index + j
                    
                    if (i + batch_size) % 1000 == 0:
                        all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
                        np.save(embeddings_cache_path, all_embeddings)
                        with open(mapping_cache_path, 'w') as f:
                            json.dump(doc_id_to_index, f, indent=2)
                    
                    success = True
                except Exception as e:
                    print(e)
                    count_over += 1
                    threshold = threshold-500
                    if count_over>4:
                        print('voyage:', count_over)
                    new_texts = []
                    for t in cur_texts:
                        new_texts.append(cut_text(text=t, tokenizer=tokenizer, threshold=threshold))
                    cur_texts = new_texts
                    time.sleep(5)
        
        print("Saving final cache...")
        all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
        np.save(embeddings_cache_path, all_embeddings)
        with open(mapping_cache_path, 'w') as f:
            json.dump(doc_id_to_index, f, indent=2)
        print(f"Cache updated: {len(cached_embeddings)} total embeddings")
    
    doc_emb = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])
    # ==================== END SMART CACHING ====================

    query_emb = []
    for i in trange(0, len(queries), batch_size, desc="Encoding queries"):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:', count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t, tokenizer=tokenizer, threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


def retrieval_google(queries, query_ids, documents, doc_ids, task, model_id, cache_dir, excluded_ids, long_context, **kwargs):
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    batch_size = kwargs.get('batch_size', 8)
    
    # ==================== SMART CACHING ====================
    cache_dir_path = Path(cache_dir) / 'doc_emb' / model_id / task / f"long_{long_context}_{batch_size}"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_cache_path = cache_dir_path / 'embeddings.npy'
    mapping_cache_path = cache_dir_path / 'doc_id_mapping.json'
    
    cached_embeddings = {}
    doc_id_to_index = {}
    
    if embeddings_cache_path.exists() and mapping_cache_path.exists():
        print("Loading existing cache...")
        cached_emb_array = np.load(embeddings_cache_path, allow_pickle=True)
        with open(mapping_cache_path, 'r') as f:
            doc_id_to_index = json.load(f)
        
        for doc_id, idx in doc_id_to_index.items():
            cached_embeddings[doc_id] = cached_emb_array[idx]
        
        print(f"Loaded {len(cached_embeddings)} cached embeddings")
    
    docs_to_encode = []
    docs_to_encode_ids = []
    
    for doc_id, doc_text in zip(doc_ids, documents):
        if doc_id not in cached_embeddings:
            docs_to_encode.append(doc_text)
            docs_to_encode_ids.append(doc_id)
    
    print(f"Documents in corpus: {len(documents)}")
    print(f"Already cached: {len(cached_embeddings)}")
    print(f"Need to encode: {len(docs_to_encode)}")
    
    if len(docs_to_encode) > 0:
        print("Encoding new documents...")
        for start_idx in tqdm(range(0, len(docs_to_encode), batch_size), desc='embedding'):
            batch_texts = docs_to_encode[start_idx:start_idx + batch_size]
            batch_doc_ids = docs_to_encode_ids[start_idx:start_idx + batch_size]
            
            cur_emb = get_embedding_google(
                texts=batch_texts, task='RETRIEVAL_DOCUMENT',
                model=model
            )
            
            next_index = len(cached_embeddings)
            for i, doc_id in enumerate(batch_doc_ids):
                cached_embeddings[doc_id] = cur_emb[i]
                doc_id_to_index[doc_id] = next_index + i
            
            if (start_idx + batch_size) % 1000 == 0:
                all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
                np.save(embeddings_cache_path, all_embeddings)
                with open(mapping_cache_path, 'w') as f:
                    json.dump(doc_id_to_index, f, indent=2)
        
        print("Saving final cache...")
        all_embeddings = np.array([cached_embeddings[doc_id] for doc_id in sorted(doc_id_to_index.keys(), key=lambda x: doc_id_to_index[x])])
        np.save(embeddings_cache_path, all_embeddings)
        with open(mapping_cache_path, 'w') as f:
            json.dump(doc_id_to_index, f, indent=2)
        print(f"Cache updated: {len(cached_embeddings)} total embeddings")
    
    doc_emb = np.array([cached_embeddings[doc_id] for doc_id in doc_ids])
    # ==================== END SMART CACHING ====================
        
    query_emb = []
    for start_idx in tqdm(range(0, len(queries), batch_size), desc='embedding queries'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size], task='RETRIEVAL_QUERY', model=model)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


def get_embedding_google(texts, task, model, dimensionality=768):
    from vertexai.language_models import TextEmbeddingInput
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai_azure,
    'google': retrieval_google
}


def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    """
    Calculate retrieval metrics including oracle reranker evaluation
    
    Args:
        results: {qid: {pid: float (retriever score)}}
        qrels: {qid: {pid: [0/1] (relevance label)}}
        k_values: list of cutoff values
    
    Returns:
        dict with NDCG, MAP, Recall, Precision, MRR, and Oracle NDCG metrics
    """
    if pytrec_eval is None:
        print("Warning: pytrec_eval not installed. Skipping metrics calculation.")
        # Return zeros or basic dict to prevent crash
        output = {}
        for k in k_values:
            output[f"NDCG@{k}"] = 0.0
            output[f"MAP@{k}"] = 0.0
            output[f"Recall@{k}"] = 0.0
            output[f"P@{k}"] = 0.0
        output["MRR"] = 0.0
        return output

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, 
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"}
    )
    scores = evaluator.evaluate(results)
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)
    
    # ==================== Oracle Reranker Evaluation ====================
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), 
                                      key=lambda x: results[query_id][x], 
                                      reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if (doc_id in top_100_ids[query_id] and 
                query_id in qrels and 
                doc_id in qrels[query_id]):
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id]
            else:
                oracle_results[query_id][doc_id] = 0
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, 
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"}
    )
    oracle_scores = evaluator.evaluate(oracle_results)
    
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)
    
    output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg}
    print(output)
    return output



from vllm.transformers_utils.tokenizer import get_tokenizer as get_vllm_tokenizer
class VQwen3EmbeddingModel:
    def __init__(self, model_path, max_length=16384, device="auto"):
        self.model = LLM(model=model_path, task="embed", gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count())
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.max_length = max_length 
        self.tokenizer = get_vllm_tokenizer(model_path, trust_remote_code=False)

    def truncate_text(self, text):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self.tokenizer.decode(text_ids)
        return text

    def embed_query(self, query):
        outputs = self.model.embed(query)
        return outputs[0].outputs.embedding

    def embed_queries(self, query):
        input_queries = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(x) for x in query]
        input_queries = [self.truncate_text(x) for x in query]
        outputs = self.model.embed(input_queries)
        return [x.outputs.embedding for x in outputs]

    def embed_doc(self, doc):
        outputs = self.model.embed("Represent this text:{}".format(doc))
        return outputs[0].outputs.embedding

    def embed_docs(self, docs):
        docs = ["Represent this text:{}".format(doc) for doc in docs]
        docs = [self.truncate_text(doc, ) for doc in docs]
        outputs = self.model.embed(docs)
        return [x.outputs.embedding for x in outputs]
'''transformers version - no vLLM - FIXED bfloat16 issue'''
import torch
from transformers import AutoModel, AutoTokenizer

class Qwen3EmbeddingModel:
    def __init__(self, model_path, max_length=16384, device="auto"):
        print(f"Loading model from {model_path} using transformers...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.bfloat16
        ).eval()
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.max_length = max_length
        print(f"✓ Model loaded successfully")

    def truncate_text(self, text):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[:self.max_length]
            text = self.tokenizer.decode(text_ids)
        return text

    def last_token_pool(self, last_hidden_states, attention_mask):
        """Extract the last token embedding (EOS token)"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def embed_query(self, query):
        """Embed a single query"""
        input_query = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(query)
        input_query = self.truncate_text(input_query)
        
        inputs = self.tokenizer(
            input_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        embedding = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        # FIX: Convert bfloat16 to float32 before numpy conversion
        return embedding[0].float().cpu().numpy()

    @torch.no_grad()
    def embed_queries(self, queries, batch_size=8):
        """Embed multiple queries in batches"""
        input_queries = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}'.format(x) for x in queries]
        input_queries = [self.truncate_text(x) for x in input_queries]
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(input_queries), batch_size), desc="embded queries"):
            batch = input_queries[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # FIX: Convert bfloat16 to float32 before numpy conversion
            all_embeddings.extend(embeddings.float().cpu().numpy())
        
        return all_embeddings

    @torch.no_grad()
    def embed_doc(self, doc):
        """Embed a single document"""
        input_doc = "Represent this text:{}".format(doc)
        input_doc = self.truncate_text(input_doc)
        
        inputs = self.tokenizer(
            input_doc,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        embedding = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        # FIX: Convert bfloat16 to float32 before numpy conversion
        return embedding[0].float().cpu().numpy()

    @torch.no_grad()
    def embed_docs(self, docs, batch_size=8):
        """Embed multiple documents in batches"""
        input_docs = ["Represent this text:{}".format(doc) for doc in docs]
        input_docs = [self.truncate_text(doc) for doc in input_docs]
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(input_docs), batch_size), desc="embedd documents"):
            batch = input_docs[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # FIX: Convert bfloat16 to float32 before numpy conversion
            all_embeddings.extend(embeddings.float().cpu().numpy())
        
        return all_embeddings

@torch.no_grad()
def retrieval_qwen3_ft_diver(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    cache_model_name = kwargs.get('model_name', 'diver')
    batch_size = kwargs.get('encode_batch_size',1)
    model_path =kwargs.get('model_path', False)  
    #model_path = #"/scratch/fs201059/aa17626/ms-swift/output/original_query_instruct_20_epochs_labels/v0-20251128-152212/checkpoint-480"
    #"/leonardo_scratch/fast/L-AUT_024/ms-swift/output/embedding_model_full_no_negative_32768_instruct/v1-20251027-024252/checkpoint-1005" #'/leonardo_work/L-AUT_024/hf_cache/local_models/Diver-Retriever-4B'
    model = VQwen3EmbeddingModel("AQ-MedAI/Diver-Retriever-4B", max_length=8192) #8192

    # Check if documents are already encoded 
    # document_postfix = '_'+kwargs['document_postfix'] if len(kwargs['document_postfix']) > 0 else ''
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', cache_model_name, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents)
        torch.cuda.empty_cache() 
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    with torch.inference_mode():
        query_emb = model.embed_queries(queries)
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()

    # if len(kwargs['document_postfix']) > 0:  # rechunk setting
    #     dedup_doc_ids = set(doc_ids)
    #     dedup_scores = []  # shape:[len(scores), len(dedup_doc_ids)], save only the best score for each query-doc pair
    #     for query_idx in range(len(query_emb)):
    #         best_scores = {}  # for each query, save the best score for each doc_id
    #         for idx, score in enumerate(scores[query_idx]):
    #             doc_id = doc_ids[idx]
    #             if doc_id not in best_scores or score > best_scores[doc_id]:
    #                 best_scores[doc_id] = score
    #         q_doc_scores = []
    #         for doc_id in dedup_doc_ids:
    #             q_doc_scores.append(best_scores.get(doc_id))
    #         dedup_scores.append(q_doc_scores)

    #     doc_ids, scores = dedup_doc_ids, dedup_scores
    #     print("Dedup Scores shape:", len(scores[0]))
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



def retrieval_contriever(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    # model = AutoModel.from_pretrained('facebook/contriever')
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')
    model = model.to('cuda')
    model.eval()

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(model, texts, show_progress_bar=True,batch_size=1, normalize_embeddings=True): # encode a batch of documents into the embeddings
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            # move inputs to cuda
            for k, v in inputs.items():
                inputs[k] = v.to('cuda')
            with torch.inference_mode():
                outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        all_embeddings = all_embeddings.cpu().numpy()
        return all_embeddings

    batch_size = kwargs.get('batch_size', 1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = encode(model, documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = encode(model, queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_reasonir(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # NOTE: HF version does not come with pooling function, need to add it manually.
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'reasonir/ReasonIR-8B'
        # customized_checkpoint = '../model/reasonir__ReasonIR-8B'  # reasonir检索
    else:
        print('use',customized_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model = AutoModel.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    model.to(device)
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',32768)
    doc_max_length = kwargs.get('doc_max_length',32768)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)

    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    # if not os.path.isdir(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}")):
    #     os.makedirs(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
        elif ignore_cache:
            inputs = tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)
            doc_emb = model(**inputs)[0]
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)
    # cur_cache_file = os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    # save query embedding
    # np.save(cur_cache_file, query_emb)
    # if os.path.isfile(cur_cache_file):
    #     query_emb = np.load(cur_cache_file, allow_pickle=True)
    # elif ignore_cache:
    #     query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    # else:
        # query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
        # np.save(cur_cache_file, query_emb)
    if skip_doc_emb:
        exit()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



@torch.no_grad()
def retrieval_rader(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_name = 'Raderspace/RaDeR_Qwen_25_7B_instruct_MATH_LLMq_CoT_lexical'  # rader检索
    batch_size = kwargs.get('encode_batch_size',1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()
    # print("Model type", model)
    # max_length = kwargs.get('doc_max_length',4096)

    # Append instructions before queries 
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])

    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_name, task, f"long_{long_context}_{batch_size}", f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []

        for i in tqdm(range(0, len(documents))): #len(documents)
            text = documents[i]
            inputs = tokenizer(f"document: {text[:8192]}{tokenizer.eos_token}", return_tensors='pt', padding=True, truncation=True) 

            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if i==0:
                    print("Doc outputs shape", outputs.last_hidden_state.shape)
                
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
            torch.cuda.empty_cache()  # 清理缓存
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    for i in tqdm(range(0, len(queries))):
        text = queries[i]
        inputs = tokenizer(f"query: {text}{tokenizer.eos_token}", return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # if model_name == "qwen2.5_7b_instruct":
            #     last_hidden = outputs.hidden_states[-1]  # Last layer
            #     embeddings = last_hidden.mean(dim=1)  #Mean pooling
            # else:
            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)
     
    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)
    print("First doc embedding:", doc_emb[0,:])

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

def retrieval_nomic(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    customized_checkpoint = kwargs.get('checkpoint', None)
    if customized_checkpoint is not None:
        model = SentenceTransformer(customized_checkpoint, trust_remote_code=True)
    else:
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    batch_size = kwargs.get('batch_size', 1)
    model_id = customized_checkpoint  # making sure the doc cache is reused
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_m2(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        "togethercomputer/m2-bert-80M-32k-retrieval",
        trust_remote_code=True
    )
    max_length = 32768

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        model_max_length=max_length
    )

    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    doc_emb = []
    batch_size = kwargs.get('encode_batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for start_idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{start_idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                embeddings = json.load(f)
        else:
            batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_dict)
            embeddings = outputs['sentence_embedding'].cpu().tolist()
            with open(cur_cache_file,'w') as f:
                json.dump(embeddings,f,indent=2)
        doc_emb += embeddings
    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = outputs['sentence_embedding'].cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


RETRIEVAL_FUNCS['diver-retriever'] = retrieval_qwen3_ft_diver
RETRIEVAL_FUNCS['contriever'] = retrieval_contriever
RETRIEVAL_FUNCS['reasonir'] = retrieval_reasonir
RETRIEVAL_FUNCS['m2'] = retrieval_m2
RETRIEVAL_FUNCS['rader'] = retrieval_rader
RETRIEVAL_FUNCS['nomic'] = retrieval_nomic
