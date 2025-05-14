from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # env_prefix = "app_"

    # OpenAI
    openai_api_key: str

    # Anthropic
    anthropic_api_key: str

    # Cohere
    cohere_api_key: str

    # Neo4j
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # Additional settings
    llm_model_name: str = "claude-3-5-sonnet-20240620"
    embed_model_name: str = "text-embedding-3-small"


settings = AppSettings()  # type: ignore[call-arg]
