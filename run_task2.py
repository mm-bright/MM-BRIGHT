
from src.eval_runner import EvaluationRunner
from src.retrievers import MULTIMODAL_RETRIEVAL_FUNCS

if __name__ == "__main__":
    runner = EvaluationRunner(
        description="Task 2: Multimodal Retrieval (Image+Text -> Text)",
        retriever_funcs_map=MULTIMODAL_RETRIEVAL_FUNCS,
        task_type="multimodal_text"
    )
    runner.run()
