import os
from typing import Any

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
# from app_settings import settings

import nest_asyncio
nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "请输入自己的APIKey"

# 初始化LLM和嵌入模型
llm = MistralAI(model="open-mixtral-8x22b", api_key="请输入自己的APIKey")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载数据到索引
documents = SimpleDirectoryReader('./data/paul_graham/').load_data()
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="neo4j@123456",
    url="bolt://localhost:7687",
)
graph_store.refresh_schema()

# 检查图存储库中是否有使用自定义Cypher查询的节点
exists = True if graph_store.get() else False

if exists:
    index = PropertyGraphIndex.from_existing(
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True,
    )
else:
    index = PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True,
    )

query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What did author do at Interleaf?")
print(response.response)

