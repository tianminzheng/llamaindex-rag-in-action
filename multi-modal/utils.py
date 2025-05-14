import uuid
from pathlib import Path

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from constants import STORAGE_DIR


def get_multi_modal_llm(
    llm_name: str, model_temperature: int, openai_api_key: str, max_new_tokens: int = 1000
) -> OpenAIMultiModal:
    """基于OpenAI创建一个多模态LLM
    """

    llm = OpenAIMultiModal(
        model=llm_name,
        temperature=model_temperature,
        api_key=openai_api_key,
        max_new_tokens=max_new_tokens,
    )
    return llm


def generate_unique_path(file_name: str) -> Path:
    """根据文件名生成一个唯一的文件路径。
    """

    file_path = Path(STORAGE_DIR, str(uuid.uuid4()), file_name)
    return file_path
