
import os
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

from .data import DataLoader
from .caching import SmartCache

# We will need to import calculate_retrieval_metrics from somewhere. 
# Ideally we move it to src/metrics.py or utils.py, but for now we can import from retrievers if we don't move it.
# Or better, let's just implement a simple one here or assume it's available.
# The original code imported from retrievers.py
# FIX: specific import later, for now we assume it exists in retrievers package or we implement it.

class EvaluationRunner:
    def __init__(self, description, retriever_funcs_map, task_type="text_text"):
        """
        task_type: 
           'text_text' (Task 1)
           'multimodal_text' (Task 2)
           'text_image' (Task 3)
           'text_pair' (Task 4)
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.retriever_funcs = retriever_funcs_map
        self.task_type = task_type
        self._setup_args()

    def _setup_args(self):
        self.parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory (for images)')
        self.parser.add_argument('--domains', type=str, nargs='+', default=['academia'], help='List of domains')
        self.parser.add_argument('--model', type=str, required=True, choices=list(self.retriever_funcs.keys()), help='Model to use')
        self.parser.add_argument('--model_name', type=str, default=None, help='Specific HF model name')
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--encode_batch_size', type=int, default=-1) # alias for compatibility
        self.parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
        self.parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
        self.parser.add_argument('--config_dir', type=str, default='configs', help='Config directory')
        self.parser.add_argument('--debug', action='store_true', help='Debug mode')
        self.parser.add_argument('--force_rerun', action='store_true', help='Force rerun')
        
        # Original script args preservation
        self.parser.add_argument('--query_max_length', type=int, default=-1)
        self.parser.add_argument('--doc_max_length', type=int, default=-1)
        self.parser.add_argument('--checkpoint', type=str, default=None)
        self.parser.add_argument('--key', type=str, default=None)
        self.parser.add_argument('--ignore_cache', action='store_true')
        self.parser.add_argument('--chunk_size', type=int, default=50000)
        self.parser.add_argument('--topk', type=int, default=1000)


    def run(self):
        args = self.parser.parse_args()
        
        # Setup directories
        task_suffix = getattr(self, "task_suffix", args.model) # Allow custom suffix
        if self.task_type == "text_image":
            subdir = f"multimodal_it2i_{args.model}"
        elif self.task_type == "text_pair":
            subdir = f"it2it_{args.model}"
        else:
            subdir = f"run_{args.model}" # Generic fallback
            if "multimodal" in sys.argv[0]: # Hacky check if it's task 2
                subdir = f"multimodal_{args.model}"
            
        final_output_dir = os.path.join(args.output_dir, subdir)
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Load config
        config = self._load_config(args)
        
        # Initialize DataLoader
        loader = DataLoader() # uses default mm-bright/MM-BRIGHT
        
        all_domain_results = {}
        
        for domain in args.domains:
            try:
                result = self._evaluate_domain(domain, loader, args, config, final_output_dir)
                if result:
                    all_domain_results[domain] = result
            except Exception as e:
                print(f"❌ Error evaluating {domain}: {e}")
                import traceback
                traceback.print_exc()
        
        self._save_summary(all_domain_results, args, final_output_dir)

    def _load_config(self, args):
        config = {'instructions': {'query': '', 'document': ''}}
        config_file = os.path.join(args.config_dir, args.model, "default.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
        return config

    def _evaluate_domain(self, domain, loader, args, config, final_output_dir):
        print(f"\n{'='*80}\nEVALUATING DOMAIN: {domain.upper()} ({self.task_type})\n{'='*80}")
        
        # 1. Load Data based on Task Type
        if self.task_type == 'text_text':
            # Task 1 & 2
            # Load Corpus
            doc_ids, documents = loader.load_corpus_texts(domain)
            if not documents: return None
            
            # Load Queries (Task 1: examples, Task 2: multimodal examples?)
            # The user's original scripts distinguished them. task 1 used 'examples', task 2 used 'examples'??
            # Let's check original run_task1 vs run_task2.
            # run_task1: "filtered_queries" -> examples
            # run_task2: "filtered_queries" -> examples
            # Wait, Task 2 is Multimodal-to-Text. Queries should have images.
            # Ah, prepre.py mapped "filtered_queries" -> examples
            # And "filtered_queries_with_filtered_images" -> examples_multimodal
            # So Task 1 & 2 use diff query sets?
            # run_task2.py loaded from 'filtered_queries' just like task 1? That seems wrong for multimodal-to-text if it needs images.
            # actually run_task2 imports MULTIMODAL_RETRIEVAL_FUNCS but loads from "filtered_queries".
            # If the user script does that, I should stick to it OR fix it.
            # But the 'examples' config in README has 'image_paths'. So 'examples' IS multimodal?
            # README says: Task 1: Text-to-Text. Task 2: Multimodal-to-Text.
            # 'examples' config has 'image_paths'. So it can support both.
            # Okay, I will load 'examples' for text_text/text_multimodal.
            
            queries, query_ids, gold_ids_map, excluded_ids, pos_imgs = loader.load_queries(domain, "examples")
            
            # If Task 2 (Multimodal), we might need those images?
            # But run_task2.py passed 'queries' (text) and 'documents' (text). 
            # It seems run_task2 might be just using text part of M-search? Or maybe it handles images internally?
            # Let's enable images if available.
            
        elif self.task_type == 'multimodal_text':
            # Task 2
            # Load Corpus (Text)
            doc_ids, documents = loader.load_corpus_texts(domain)
            if not documents: return None
            
            # Load Queries (Multimodal - has text and image_paths)
            queries, query_ids, gold_ids_map, excluded_ids, query_images_map = loader.load_queries(domain, "examples_multimodal")

        elif self.task_type == 'text_image':
            # Task 3
            # Load Queries with images
            queries, query_ids, gold_ids_map, excluded_ids, query_images_map = loader.load_queries(domain, "examples_multimodal")
            
            # Load Image Corpus
            img_ids, img_paths = loader.load_image_corpus_paths(domain, Path(args.dataset_dir) / "passage_images")
            if not img_paths: return None
            
            # doc_ids/documents become image paths for the retriever
            doc_ids = img_ids
            documents = img_paths

        elif self.task_type == 'text_pair':
             # Task 4
             # Load Corpus & Images & Build Pairs
             p_doc_ids, p_docs = loader.load_corpus_texts(domain)
             doc_ids, documents, doc_images, _, _ = loader.build_it_it_pairs(p_doc_ids, p_docs, Path(args.dataset_dir) / "passage_images", domain)
             
             queries, query_ids, gold_ids_map, excluded_ids, query_images_map = loader.load_queries(domain, "examples_multimodal")

        if not queries: return None
        
        # 2. Debug Slicing
        if args.debug:
            print("[DEBUG] Slicing data...")
            documents = documents[:30]
            doc_ids = doc_ids[:30]
            queries = queries[:10]
            query_ids = query_ids[:10]
            if self.task_type == 'text_pair':
                 doc_images = doc_images[:30]

        # 3. Validation / Qrel building
        # ...

        # 4. Run Retrieval
        domain_out = os.path.join(final_output_dir, domain)
        os.makedirs(domain_out, exist_ok=True)
        score_file = os.path.join(domain_out, 'scores.json')
        
        if not os.path.exists(score_file) or args.force_rerun:
            print(f"Running retrieval...")
            
            # Prepare kwargs
            run_kwargs = {
                'queries': queries,
                'query_ids': query_ids,
                'documents': documents,
                'doc_ids': doc_ids,
                'task': domain,
                'instructions': config['instructions'],
                'excluded_ids': excluded_ids,
                'cache_dir': args.cache_dir,
                'long_context': False,
                'model_id': args.model,
                'batch_size': args.batch_size if args.batch_size > 0 else 8
            }
            # Add extras
            if args.model_name: run_kwargs['model_name'] = args.model_name
            if self.task_type == 'text_pair': run_kwargs['doc_images'] = doc_images
            if 'query_images_map' in locals(): run_kwargs['query_images_map'] = query_images_map
            
            scores = self.retriever_funcs[args.model](**run_kwargs)
            
            with open(score_file, 'w') as f:
                json.dump(scores, f, indent=2)
        else:
            print("Loading existing scores...")
            with open(score_file) as f:
                scores = json.load(f)

        # 5. Metrics
        # Need to reconstruct qrels (ground_truth)
        # This is strictly dependent on the map (gold_ids for T1/2/4, pos_imgs for T3)
        # Simple generic qrel builder:
        qrels = {}
        for qid in query_ids:
            qrels[str(qid)] = {}
            # Prefer gold_ids if T1/2
            if self.task_type in ['text_text', 'multimodal_text', 'text_pair']:
                for gid in gold_ids_map.get(qid, []):
                    qrels[str(qid)][str(gid)] = 1 
            else: # T3 - uses positive images (query_images_map contains query images, not relevant here)
                 # For T3, gold_ids_map should still contain relevant document/image IDs
                 for gid in gold_ids_map.get(qid, []):
                     qrels[str(qid)][str(gid)] = 1

        # Calculate metrics
        from src.retrievers.task1_text import calculate_retrieval_metrics
        metrics = calculate_retrieval_metrics(scores, qrels)
        
        # Save metrics
        metrics_file = os.path.join(domain_out, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_file}")
        print(f"  NDCG@10: {metrics.get('NDCG@10', 'N/A')}, Recall@100: {metrics.get('Recall@100', 'N/A')}")
        
        # Result dict
        return {'metrics': metrics, 'domain': domain, 'num_queries': len(query_ids), 'num_docs': len(doc_ids)}

    def _save_summary(self, results, args, out_dir):
        """Save aggregated summary of all domain results."""
        if not results:
            print("No results to summarize.")
            return
            
        summary = {
            'model': args.model,
            'task_type': self.task_type,
            'domains': {},
            'aggregated': {}
        }
        
        # Aggregate metrics across domains
        all_metrics = {}
        for domain, result in results.items():
            summary['domains'][domain] = result.get('metrics', {})
            for k, v in result.get('metrics', {}).items():
                if isinstance(v, (int, float)):
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
        
        # Calculate averages
        for k, values in all_metrics.items():
            if values:
                summary['aggregated'][k] = round(sum(values) / len(values), 5)
        
        summary_file = os.path.join(out_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*80}")
        print(f"SUMMARY saved to {summary_file}")
        print(f"Aggregated metrics: {summary['aggregated']}")
        print(f"{'='*80}")
