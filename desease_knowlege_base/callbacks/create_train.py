import json
import random
from datetime import datetime
from sqlalchemy import MetaData
import sqlalchemy
from uuid import uuid1
from typing import List, Dict


def get_all_classes(conn_settings: Dict) -> List[Dict]:
    """Получить список всех сгенерированных классов

    Returns:
        List[Dict]: Список классов
    """
    try:
        usr = conn_settings["usr"]
        pswd = conn_settings["pswd"]
        host = conn_settings["host"]
        port = conn_settings["port"]
        db = conn_settings["db"]
    except KeyError as key_error:
        raise KeyError("Bad postgres settings") from key_error
    engine = sqlalchemy.create_engine(
        f"postgresql://{usr}:{pswd}@{host}:{port}/{db}")

    metadata = MetaData(bind=engine)
    metadata.reflect()
    metadata.create_all(checkfirst=True)

    with engine.connect() as conn:
        data = conn.execute("select * from desease_classes")
    data = [row._data for row in data]
    return data


def prepare_data_for_classes_tbl() -> List[Dict]:
    classes_data = get_all_classes({
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    })
    result_list = []
    for deseases_class in classes_data:
        name_class = deseases_class[1]
        symptoms = deseases_class[2]
        for feature in symptoms:
            name_feature = feature["feature"]
            feature_periods = feature["periods"]
            for num, period in enumerate(feature_periods):
                duration_lower = period["duration_lower"]
                duration_upper = period["duration_upper"]
                now_dict = {
                    "name-id": name_class,
                    "feature-id": name_feature,
                    "period-id": num,
                    "lower-id": duration_lower,
                    "upper-id": duration_upper
                }
                result_list.append(now_dict)
    return result_list
