import datetime
import json
import os

import pandas as pd
import psycopg

DB_CONNECTION_STRING = "dbname=postgres user=postgres host=localhost password=postgres"


def store_result(payload: json, image_path: str) -> None:
    """将提取的图像数据和图像路径保存到数据库中。
    """

    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO neatapp (payload, image_path, created_timestamp) VALUES (%s, %s, %s)",
                (
                    payload,
                    image_path,
                    datetime.datetime.now(),
                ),
            )


def load_data_as_dataframe() -> pd.DataFrame:
    """从数据库中加载所有已提取的数据。
    """

    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        df = pd.read_sql_query("SELECT created_timestamp, image_path, payload FROM neatapp", conn)
    return df
