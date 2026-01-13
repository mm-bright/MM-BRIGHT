import os
import subprocess
import argparse

# ==================== CONFIGURATION ====================

DOMAINS = [
    "salesforce", 
    # "academia", "apple", "askubuntu", "aviation", "bioacoustics", 
    # "bioinformatics", "biology", "bitcoin", "chemistry", "christianity", 
    # "crypto", "earthscience", "economics", "gaming", "gis", "islam", 
    # "law", "math", "medicalsciences", "philosophy", "physics", "pm", 
    # "psychology", "quant", "quantumcomputing", "robotics", 
    # "sustainability", "travel"
]

# Task 1: Text-to-Text Retrieval
TASK1_MODELS = [
    # "sbert", "bge", 
    # "inst-l", "inst-xl", "grit", "diver-retriever", "contriever", "reasonir", "m2", 
    # "rader", "nomic"
    # "cohere", "voyage", "openai",  "bm25","sf", "qwen", "qwen2", "e5",
    #"google",
]

# Task 2: Multimodal-to-Text Retrieval
# (Queries have images, documents are text)
#
TASK2_MODELS = [
    
    # "bge-vl-large", "bge-vl-base", "nomic-vision",  "clip" , "siglip",    "jina-clip" , "gme-qwen2-vl-2b", "gme-qwen2-vl-7b", 

]

# Task 3: Text-to-Image Retrieval
# (Query is text+image (or just text), target is image)
TASK3_MODELS = [ #"clip",  "siglip",
    # "jina-clip", "nomic-vision", 
    "bge-vl-large", "gme-qwen2-vl-2b", "gme-qwen2-vl-7b"
]

# Task 4: Multimodal-to-Multimodal (IT -> IT)
TASK4_MODELS = [
    # "clip", "siglip", "jina-clip", "bge-vl-large", 
    # "gme-qwen2-vl-2b", "gme-qwen2-vl-7b", "nomic-vision"
]

# ==================== SCRIPT ====================

def main():
    parser = argparse.ArgumentParser(description="Run all MM-BRIGHT experiments")
    parser.add_argument("--dataset_dir", type=str, default=".", help="Path to dataset directory (for images)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands instead of running them")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4], help="Tasks to run (1-4)")
    args = parser.parse_args()

    domains_str = " ".join(DOMAINS)
    
    cmds = []

    # --- Task 1 ---
    if 1 in args.tasks:
        print(f"\n Generating commands for Task 1 ({len(TASK1_MODELS)} models)...")
        for model in TASK1_MODELS:
            cmd = f"python run_task1.py --dataset_dir {args.dataset_dir} --model {model} --domains {domains_str}"
            cmds.append(cmd)

    # --- Task 2 ---
    if 2 in args.tasks:
        print(f"\n Generating commands for Task 2 ({len(TASK2_MODELS)} models)...")
        for model in TASK2_MODELS:
            cmd = f"python run_task2.py --dataset_dir {args.dataset_dir} --model {model} --domains {domains_str}"
            cmds.append(cmd)

    # --- Task 3 ---
    if 3 in args.tasks:
        print(f"\n Generating commands for Task 3 ({len(TASK3_MODELS)} models)...")
        for model in TASK3_MODELS:
            cmd = f"python run_task3.py --dataset_dir {args.dataset_dir} --model {model} --domains {domains_str}"
            cmds.append(cmd)

    # --- Task 4 ---
    if 4 in args.tasks:
        print(f"\n Generating commands for Task 4 ({len(TASK4_MODELS)} models)...")
        for model in TASK4_MODELS:
            cmd = f"python run_task4.py --dataset_dir {args.dataset_dir} --model {model} --domains {domains_str}"
            cmds.append(cmd)

    print(f"\nTotal commands to run: {len(cmds)}")
    
    if args.dry_run:
        print("\n--- COMMANDS ---")
        for cmd in cmds:
            print(cmd)
        print("\n(Use --dry_run to see this list again, or omit it to execute)")
    else:
        print("\n--- EXECUTING ---")
        for i, cmd in enumerate(cmds):
            print(f"[{i+1}/{len(cmds)}] Running: {cmd}")
            try:
                ret = subprocess.call(cmd, shell=True)
                if ret != 0:
                    print(f"❌ Command failed with return code {ret}: {cmd}")
                    # Decide if you want to stop or continue. Continuing allows bulk runs to proceed.
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"❌ Execution error: {e}")

if __name__ == "__main__":
    main()
