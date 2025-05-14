from typing import Dict, Sequence

from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.schema import ImageDocument

from models import Restaurant
from utils import get_multi_modal_llm

def extract_data(
    image_documents: Sequence[ImageDocument],
    data_extract_str: str,
    llm_name: str,
    model_temperature: int,
    api_key: str,
) -> Dict:
    """从图像文档中提取数据
    """

    llm = get_multi_modal_llm(llm_name, model_temperature, api_key, max_new_tokens=1000)
    openai_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(Restaurant),
        image_documents=image_documents,
        prompt_template_str=data_extract_str,
        multi_modal_llm=llm,
        verbose=True,
    )
    response = openai_program()
    return response
