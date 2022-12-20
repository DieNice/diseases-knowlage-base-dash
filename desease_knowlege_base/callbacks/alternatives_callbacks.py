import copy
import json
import random
from datetime import datetime
from random import randint, sample
from typing import Dict, List, Tuple
from uuid import uuid1

import sqlalchemy
from app import app
from callbacks.create_train import get_all_classes, get_all_train
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State
from sqlalchemy import MetaData


def get_all_desease_histories_by_desease(name_deasease: str) -> List:
    """Получение историй болезеней по имени класса болезни

    Args:
        name_deasease (str): Название болезни

    Returns:
        List: Список историй болезеней
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }

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
        data = conn.execute(
            f"select distinct number_history from desease_train dt where dt.name_class='{name_deasease}'")
    data = [row._data for row in data]
    result = sum([[*row] for row in data], [])
    del data
    return result


def get_all_deseases() -> List:
    """Получение всех болезеней

    Returns:
        List:
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }

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
        data = conn.execute(
            f"select distinct name_class from desease_train dt")
    data = [row._data for row in data]

    result = sum([[*row] for row in data], [])
    del data
    return result


def get_features_by_desease(name_desease: str) -> List:
    """Получение списка признаков по названию болезни

    Args:
        name_desease (str): Название болезни

    Returns:
        List: Список признаков
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }

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
        data = conn.execute(
            f"select symptoms from desease_classes dc where dc.name='{name_desease}'")
    data = [row._data for row in data]
    result = sum([[*row] for row in data], [])
    result = result.pop()
    del data
    return result


@app.callback(
    output={
        "alert": Output("alert-alternatives-id", "children")
    },
    inputs={
        "generate": Input("generate-alternatives-id", "n_clicks"),
    },
    prevent_initial_call=True
)
def generate_alternatives(generate: int) -> Dict:
    NUM_HOSTORIES = 2
    NUM_FEATURES = 2
    all_deseases = get_all_deseases()

    configs = []

    for desease in all_deseases:
        config = {}
        config['desease'] = desease
        histories = get_all_desease_histories_by_desease(desease)
        features = get_features_by_desease(desease)

        selected_histories = sample(histories, NUM_HOSTORIES)
        config['histories'] = selected_histories

        selected_features = sample(features, NUM_FEATURES)
        names_features = [feature["name"] for feature in selected_features]
        del selected_features
        config["features"] = names_features
        configs.append(config)

    return {"alert": "Generation successfull!"}
