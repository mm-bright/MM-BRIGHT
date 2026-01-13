
from src.eval_runner import EvaluationRunner
from src.retrievers import MULTIMODAL_PAIR_RETRIEVAL_FUNCS

if __name__ == "__main__":
    runner = EvaluationRunner(
        description="Task 4: Multimodal Pair Retrieval (Image+Text -> Image+Text)",
        retriever_funcs_map=MULTIMODAL_PAIR_RETRIEVAL_FUNCS,
        task_type="text_pair"
    )
    runner.run()
