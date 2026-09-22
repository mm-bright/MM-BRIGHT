"""
Validate the fixed Task 4 evaluator against the runs that produced Table 6.

The December 2025 runs stored, per domain, the raw retriever scores
(`scores.json`) and the resulting metrics (`results.json`). Those scores are a
fixed input: replaying them through freshly built qrels isolates the qrel and
candidate-pool logic from the model, so the whole port can be checked without a
GPU.

  --protocol legacy  must reproduce results.json exactly, in all 29 domains.
  --protocol paper   must reproduce it in the 28 domains whose passage IDs are
                     unchunked; Biology is expected to differ, because that is
                     precisely the bug being fixed.

Usage:
    python validate_reproduction.py --runs_dir <stored-runs> --protocol legacy --model clip
"""

import argparse
import json
import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import DataLoader
from src.eval_runner import EvaluationRunner
from src.retrievers.task1_text import calculate_retrieval_metrics

REVISION = "97702ca9ea81cd0a25288e74a9402439550d6bd4"
REPO = "mm-bright/MM-BRIGHT"

DOMAINS = [
    'academia', 'apple', 'askubuntu', 'aviation', 'bioacoustics', 'bioinformatics',
    'biology', 'bitcoin', 'chemistry', 'christianity', 'crypto', 'earthscience',
    'economics', 'gaming', 'gis', 'islam', 'law', 'math', 'medicalsciences',
    'philosophy', 'physics', 'pm', 'psychology', 'quant', 'quantumcomputing',
    'robotics', 'salesforce', 'sustainability', 'travel'
]


def _parquet(config, domain):
    return pd.read_parquet(hf_hub_download(
        REPO, f"{config}/{domain}.parquet", repo_type="dataset", revision=REVISION))


def _image_paths(val):
    """positive_images / negative_images -> list of forward-slashed paths."""
    if val is None:
        return []
    out = []
    for x in val:
        p = x.get("image_path") if isinstance(x, dict) else x
        if p:
            out.append(str(p).replace("\\", "/"))
    return out


def build_qrels(domain, protocol):
    """Rebuild the Task 4 candidate pool and qrels exactly as eval_runner does."""
    docs = _parquet("documents", domain)
    p_doc_ids = [str(x) for x in docs["id"]]
    p_docs = [str(x) for x in docs["content"]]

    # Only the image *keys* matter for qrels and pair IDs, so avoid decoding
    # 2 GB of image bytes: map every path to None.
    img_paths = _parquet("document_images", domain)["path"].astype(str).tolist()
    corpus_images_map = {p: None for p in img_paths}

    q = _parquet("examples_multimodal", domain)
    query_ids = list(q["id"])
    gold_ids_map, positive_images_map, negative_images_map = {}, {}, {}
    for _, r in q.iterrows():
        qid = r["id"]
        gold_ids_map[qid] = [str(g) for g in (r["gold_ids"] if r["gold_ids"] is not None else [])]
        positive_images_map[qid] = _image_paths(r["positive_images"])
        negative_images_map[qid] = _image_paths(r["negative_images"])

    loader = DataLoader()
    legacy = (protocol == "legacy")
    doc_ids, documents, doc_images, _, base_to_passage_ids = loader.build_it_it_pairs(
        p_doc_ids, p_docs, corpus_images_map, domain, legacy_keys=legacy)

    runner = EvaluationRunner("validate", {}, task_type="text_pair")
    qrels, doc_ids, documents, doc_images, excluded = runner._build_task4_qrels(
        query_ids=query_ids,
        gold_ids_map=gold_ids_map,
        positive_images_map=positive_images_map,
        negative_images_map=negative_images_map,
        doc_ids=doc_ids,
        documents=documents,
        doc_images=doc_images,
        passage_id_to_text=dict(zip(p_doc_ids, p_docs)),
        base_to_passage_ids=base_to_passage_ids,
        corpus_images_map=corpus_images_map,
        domain=domain,
        legacy_keys=legacy,
    )
    return qrels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True,
                    help="Directory of stored runs, containing it2it_<model>/<domain>/"
                         "{scores.json,results.json}")
    ap.add_argument("--protocol", choices=["legacy", "paper"], default="legacy")
    ap.add_argument("--model", default="clip")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--metric", default="NDCG@10")
    args = ap.parse_args()

    print(f"Protocol: {args.protocol}   model: {args.model}   metric: {args.metric}\n")
    print(f"{'domain':<18}{'published':>11}{'replayed':>11}{'delta':>9}   status")
    print("-" * 62)

    match = mismatch = skipped = 0
    for domain in args.domains:
        run_dir = os.path.join(args.runs_dir, f"it2it_{args.model}", domain)
        score_f, res_f = os.path.join(run_dir, "scores.json"), os.path.join(run_dir, "results.json")
        if not (os.path.isfile(score_f) and os.path.isfile(res_f)):
            print(f"{domain:<18}{'-':>11}{'-':>11}{'-':>9}   no stored run")
            skipped += 1
            continue

        with open(score_f) as f:
            scores = json.load(f)
        with open(res_f) as f:
            published = json.load(f)[args.metric]

        qrels = build_qrels(domain, args.protocol)
        qrels = {q: v for q, v in qrels.items() if v}          # pytrec_eval rejects empty
        scores = {q: v for q, v in scores.items() if q in qrels}

        # A replayed score is only meaningful if the ranking being replayed could
        # in principle contain the judged candidates. Where a protocol change
        # introduces qrel entries that the stored run never scored, the replayed
        # metric is an artifact of the missing candidates, not a corrected result.
        judged = [pid for g in qrels.values() for pid in g]
        unrankable = sum(1 for q, g in qrels.items() for pid in g if pid not in scores.get(q, {}))

        replayed = calculate_retrieval_metrics(scores, qrels)[args.metric]
        delta = replayed - published
        ok = abs(delta) < 5e-5
        status = "MATCH" if ok else "DIFFERS"
        if not ok and unrankable:
            status = "DIFFERS (not comparable)"
        print(f"{domain:<18}{published:>11.5f}{replayed:>11.5f}{delta:>9.5f}   {status}")
        if not ok and unrankable:
            print(f"{'':<18}  ⚠️  {unrankable}/{len(judged)} judged pairs are absent from the "
                  f"stored ranking; this metric is NOT a corrected score.")
            print(f"{'':<18}      Re-run the model under --protocol paper to obtain one.")
        match += ok
        mismatch += (not ok)

    print("-" * 62)
    print(f"{match} match, {mismatch} differ, {skipped} skipped")


if __name__ == "__main__":
    main()
