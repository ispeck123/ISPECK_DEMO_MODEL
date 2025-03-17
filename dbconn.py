from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
# from classes.utilities import Config
from urllib.parse import quote as urlquote
import pandas as pd
import json
from Config import Config as cfg
from sqlalchemy import text
from datetime import datetime
from pytz import timezone
# Init the Configuration
# config = Config()
# database = config.getDatabaseConfig()
# print(database.host)

# Create a sqlite engine instance
engine = create_engine("mysql+pymysql://" + cfg.DB_USER + ":" +
                       urlquote(cfg.DB_PASSWORD) + "@" + cfg.DB_HOST + ":" + str(cfg.DB_PORT) + "/" + cfg.DB_SCHEMA)

# Create a DeclarativeMeta instance
Base = declarative_base()

# Create the database
Base.metadata.create_all(engine)


# Region : For Raw Query
def raw_connection():
    raw_connection = engine.connect()
    return raw_connection


def raw_query(my_query: str):
    try:
        data_list = pd.read_sql(text(my_query), raw_connection())
        results = json.loads(data_list.to_json(orient="table", index=False))["data"]
    except Exception as ex:
        results = ex
        print(ex)
        # raise Exception(str(ex))
    return results


def insert_into_event_transaction(_data):
    rc = raw_connection()
    now_sg = datetime.now(timezone('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    ins_query = f"""insert into {cfg.TABLE_OBJECT_INSIGHT} 
                (SITE_ID, AREA_ID, CAMERA_ID, PIPELINE_ID, STREAM_ID, MODEL_ID, TIMESTAMP, EVENT_TYPE, BBOX, SCORE, FILE_NAME) values 
                ({_data.get('site_id')}, {_data.get('area_id')}, {_data.get('cam_id')}, {_data.get('pipeline_id')}, {_data.get('class_id')}, 
                {_data.get('model_id')}, '{now_sg}', '{_data.get('event_type')}', '{_data.get('bbox')}', {_data.get('score')}, '{_data.get('filename')}')"""
    rc.execute(text(ins_query))
    rc.commit()
    rc.close()
    print(f"Insert into db success {ins_query}")
    