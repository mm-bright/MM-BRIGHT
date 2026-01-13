
from src.eval_runner import EvaluationRunner
from src.retrievers import TEXT_RETRIEVAL_FUNCS

if __name__ == "__main__":
    runner = EvaluationRunner(
        description="Task 1: Text-to-Text Retrieval",
        retriever_funcs_map=TEXT_RETRIEVAL_FUNCS,
        task_type="text_text"
    )
    runner.run()