import copy
import json
import random
from datetime import datetime
from random import randint, sample
from typing import Dict, List
from uuid import uuid1

import sqlalchemy
from app import app
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State
from sqlalchemy import MetaData


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
            name_feature = feature["name"]
            feature_periods = feature["periods"]
            for period in feature_periods:
                duration_lower = period["lower_duration"]
                duration_upper = period["upper_duration"]
                num_period = period["num_period"]
                values = period["values"]
                values = ",".join([str(i) for i in values])
                now_dict = {
                    "name-id": name_class,
                    "feature-id": name_feature,
                    "period-id": num_period,
                    "values-id": values,
                    "lower-id": duration_lower,
                    "upper-id": duration_upper
                }
                result_list.append(now_dict)
    return result_list


def generate_train(classes_data: List[Dict], generation_seed: int, num_observations: int) -> List[Dict]:
    """Генерация обучающей выборки на основе модели

    Args:
        classes_data (List[Dict]): Данные классов

    Returns:
        List[Dict]: Обучающая выборка
    """
    max_observetions = num_observations

    generation_train = []

    for record in classes_data:
        name_class = record[1]
        features = record[2]

        for number_history in range(generation_seed):
            for feature in features:
                random.seed(datetime.now().timestamp())
                name_feature = feature["name"]
                periods = feature['periods']
                for period in periods:
                    num_period = period["num_period"]
                    period_duration = random.randint(period["lower_duration"],
                                                     period["upper_duration"])
                    values = period["values"]

                    num_observations = randint(1, max_observetions)

                    if period_duration < num_observations:
                        period_duration += num_observations

                    observetions = sample(
                        [i for i in range(1, period_duration+1)], num_observations)

                    observetions.sort()

                    for num_observation in range(num_observations):

                        random.seed(datetime.now().timestamp())
                        random_index = randint(0, len(values)-1)
                        observation_value = values[random_index]

                        class_instance = {
                            "id": str(uuid1()),
                            "number_history": f"История болезни номер:{number_history+1}",
                            "name_class": name_class,
                            "name_feature": name_feature,
                            "num_period": num_period,
                            "moment_observation": observetions[num_observation],
                            "value": observation_value,
                            "duration": period_duration
                        }
                        generation_train.append(class_instance)
    return generation_train


def save_train_to_db(data: List[Dict], conn_settings: Dict) -> None:
    """Сохранение обучающей выборки в базу данных

    Args:
        data (List[Dict]): Список с моментами наблюдений
    """
    try:
        usr = conn_settings["usr"]
        pswd = conn_settings["pswd"]
        host = conn_settings["host"]
        port = conn_settings["port"]
        db = conn_settings["db"]
    except KeyError as key_error:
        raise KeyError(f"Bad postgres settings") from key_error

    engine = sqlalchemy.create_engine(
        f"postgresql://{usr}:{pswd}@{host}:{port}/{db}")

    metadata = MetaData(bind=engine)
    metadata.reflect()
    deseases_train_tbl = metadata.tables['desease_train']
    metadata.create_all(checkfirst=True)
    with engine.connect() as conn:
        conn.execute("truncate table desease_train")
        for record in data:
            conn.execute(deseases_train_tbl.insert(),
                         record)


@app.callback(
    output={
        "alert": Output("alert-id", "children"),
        "train-tbl": Output("train-tbl-id", "data")
    },
    inputs={
        "generate": Input("generate-train-id", "n_clicks"),
        "seed": State("num-instance-id", "value"),
        "num_observations": State("num-observetions-id", "value"),
    },
    prevent_initial_call=True
)
def generate_train_dataset(generate: int, seed: int, num_observations: int) -> Dict:
    """Генерация истории болезни

    Returns:
        Dict: _description_
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }
    if generate is None:
        raise PreventUpdate
    if seed is None:
        raise PreventUpdate

    classes_data = get_all_classes(conn_settings)
    data_train = generate_train(classes_data, seed, num_observations)
    save_train_to_db(data_train, conn_settings)
    return {"alert": "Генерация модельной выборки прошла успешно!",
            "train-tbl": data_train}


@app.callback(
    output={
        "data": Output("classes-tbl-id", "data")
    },
    inputs={
        "update": Input("update-classes-tbl-id", "n_clicks")
    },
    prevent_initial_call=True
)
def update_classes_tbl(update: int) -> Dict:
    """Обновление classes-tbl-id

    Returns:
        List[Dict]: _description_
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }
    if update is None:
        raise PreventUpdate
    result = prepare_data_for_classes_tbl()
    return {"data": result}


def get_all_train(conn_settings: Dict) -> List[Dict]:
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
        data = conn.execute("select * from desease_train")
    data = [row._data for row in data]
    return data


@app.callback(
    output={
        "data": Output("train-tbl-id", "data")
    },
    inputs={
        "update": Input("update-train-tbl-id", "n_clicks")
    },
    prevent_initial_call=True
)
def update_train_tbl(update: int) -> Dict:
    """Обновление classes-tbl-id

    Returns:
        List[Dict]: _description_
    """
    conn_settings = {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    }
    if update is None:
        raise PreventUpdate
    raw_data = get_all_train(conn_settings)

    result = []
    for row in raw_data:
        now_data = {"name_class": row[2],
                    "number_history": row[1],
                    "name_feature": row[3],
                    "num_period": row[4],
                    "moment_observation": row[5],
                    "value": row[6],
                    "duration": row[7]}
        result.append(now_data)
    return {"data": result}
