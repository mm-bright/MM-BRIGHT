
from src.eval_runner import EvaluationRunner
from src.retrievers import MULTIMODAL_IMAGE_RETRIEVAL_FUNCS

if __name__ == "__main__":
    runner = EvaluationRunner(
        description="Task 3: Multimodal Image Retrieval (Text+Image -> Image)",
        retriever_funcs_map=MULTIMODAL_IMAGE_RETRIEVAL_FUNCS,
        task_type="text_image"
    )
    runner.run()
