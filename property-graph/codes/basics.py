import os

from llama_index.core import (
    SimpleDirectoryReader, PropertyGraphIndex,
    StorageContext, load_index_from_storage, set_global_handler
)

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)


from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import phoenix as px

px.launch_app()
set_global_handler("arize_phoenix")


# 初始化LLM和嵌入模型
llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding()

# 加载数据到索引
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader('./data/sushi/').load_data()

    index = PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=[
            ImplicitPathExtractor(),  # Creates previous-next relation
            SimpleLLMPathExtractor(  # Creates more complex or semantic relationship
                llm=llm,
                num_workers=4,
                max_paths_per_chunk=10,
            ),
        ],
        show_progress=True,
    )
    # 持久化
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # 加载现有索引
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./storage")
    )

index.property_graph_store.save_networkx_graph(name="./pg.html")

# 检索器
retriever = index.as_retriever(
    include_text=False,  # 包含源文本, 默认True
)

nodes = retriever.retrieve("苏轼的履历是怎么样的？")

for node in nodes:
    print(node.text)


# 查询
query_engine = index.as_query_engine(
    include_text=True
)

response = query_engine.query("苏轼的履历是怎么样的？")

print(response.response)
input("Press <ENTER> to exit...")