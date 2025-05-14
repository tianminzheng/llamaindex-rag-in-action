from llama_index.core import VectorStoreIndex, StorageContext, SQLDatabase
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import NLSQLTableQueryEngine, RetrieverQueryEngine
from llama_index.core.query_engine.sql_vector_query_engine import SQLAutoVectorQueryEngine 
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
import gradio as gr
import chromadb, sys, logging

from llama_index.readers.wikipedia import WikipediaReader

load_dotenv()

# 设置INFO日志级别
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#创建chroma客户端和集合
db = chromadb.PersistentClient(path="chroma_database")
chroma_collection = db.get_or_create_collection(
    "wiki_cities"
)

#定义节点解析器
node_parser = SentenceSplitter(chunk_size=1024)

#创建空向量索引
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

#创建SQL Engine
engine = create_engine("sqlite:///wiki_cities.db", future=True)
metadata_obj = MetaData()
metadata_obj.drop_all(engine)

#创建数据库表
table_name = "wiki_cities"
wiki_cities_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

metadata_obj.tables.keys()

#插入数据
from sqlalchemy import insert
rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(wiki_cities_table).values(**row)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM wiki_cities")
    print(cursor.fetchall())
    

def data_ingestion_indexing():

    cities = ["Toronto", "Berlin", "Tokyo"]
    wiki_docs = WikipediaReader().load_data(pages=cities)

    #构建SQL索引
    sql_database = SQLDatabase(engine, include_tables=["wiki_cities"])
    
    #将文档插入向量索引中，每个文档都附有元数据
    for city, wiki_doc in zip(cities, wiki_docs):
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        #为每个节点添加元数据
        for node in nodes:
            node.metadata = {"title": city}
        vector_index.insert_nodes(nodes)

    #创建NLSQLTableQueryEngine
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["wiki_cities"],
        verbose=True

    )
    
    #创建vector query engine
    vector_store_info = VectorStoreInfo(
        content_info="articles about different cities",
        metadata_info=[
            MetadataInfo(
                name="title",
                type="str",
                description="The name of the city"
            ),
        ],
    )
    vector_auto_retriever = VectorIndexAutoRetriever(vector_index, vector_store_info=vector_store_info)
    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever, 
        llm=OpenAI(temperature=0, model_name="gpt-4", streaming=True)
    )
    
    #创建SQL query tool
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )
    
    #创建vector query tool
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description=(
            f"Useful for answering semantic questions about different cities"
        ),
    )

    #定义SQLAutoVectorQueryEngine
    query_engine = SQLAutoVectorQueryEngine(
        sql_tool,
        vector_tool,
        llm=OpenAI(temperature=0, model_name="gpt-4", streaming=True),
        verbose=True
    )

    # SQLJoinQueryEngine使用方法
    # from llama_index.core.query_engine import SQLJoinQueryEngine
    #
    # query_engine = SQLJoinQueryEngine(
    #     sql_tool, vector_tool, llm=OpenAI(model="gpt-4")
    # )
    # #
    # response = query_engine.query(
    #     "Tell me about the arts and culture of the city with the highest population"
    # )
    #
    # response = query_engine.query(
    #     "Compare and contrast the demographics of Berlin and Toronto"
    # )

    return query_engine


query_engine = data_ingestion_indexing()
response = query_engine.query(
    "Tell me about the history of the city with the highest population")
print(str(response))

response = query_engine.query("Tell me about the history of Berlin")
print(str(response))

response = query_engine.query(
    "Can you give me the country corresponding to each city?"
)
print(str(response))











def data_querying(input_text):
    #queries the engine with the input text
    query_engine = data_ingestion_indexing()
    response = query_engine.query(input_text)
    return response.response


iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Things to do in wiki cities")

#data ingestion and indexing
iface.launch(share=False)
