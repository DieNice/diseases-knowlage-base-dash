import json
import random
from datetime import datetime
from sqlalchemy import MetaData
import sqlalchemy
from uuid import uuid1
from typing import List, Dict
import copy
from dash_extensions.enrich import Input, Output, State
from app import app
from dash.exceptions import PreventUpdate


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
                    "period-id": num + 1,
                    "lower-id": duration_lower,
                    "upper-id": duration_upper
                }
                result_list.append(now_dict)
    return result_list


def generate_train(classes_data: List[Dict], generation_seed: int, num_observetions: int) -> List[Dict]:
    """Генерация обучающей выборки на основе модели

    Args:
        classes_data (List[Dict]): Данные классов

    Returns:
        List[Dict]: Обучающая выборка
    """
    max_obervetions = num_observetions

    generation_train = []

    for record in classes_data:
        name_class = record[1]
        features = record[2]

        for number_history in range(generation_seed):
            for feature in features:
                random.seed(datetime.now())
                name_feature = feature['feature']
                periods = feature['periods']
                for num_period, period in enumerate(periods):
                    period_duration = random.randint(period["duration_lower"],
                                                     period["duration_upper"])
                    values = period["values"]
                    actual_values = values[random.randint(0, len(values)-1)]
                    if len(actual_values) > max_obervetions:
                        observations = random.sample(
                            actual_values, max_obervetions)
                    else:
                        observations = copy.copy(actual_values)
                        for num_observetion, observation in enumerate(observations):
                            class_instance = {
                                "id": str(uuid1()),
                                "number_history": f"История болезни номер:{number_history+1}",
                                "name_class": name_class,
                                "name_feature": name_feature,
                                "num_period": num_period + 1,
                                "num_observetion": num_observetion + 1,
                                "value": observation[name_feature],
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
        "num_observetions": State("num-observetions-id", "value"),
    },
    prevent_initial_call=True
)
def generate_train_dataset(generate: int, seed: int, num_observetions: int) -> Dict:
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
    data_train = generate_train(classes_data, seed, num_observetions)
    save_train_to_db(data_train, conn_settings)
    return {"alert": "Генерация модельной выборки прошла успешно!",
            "train-tbl": data_train}


@app.callback(
    output={
        "data": Output("classes-tbl-id", "data")
    },
    inputs={
        "generate": Input("generate-btn-id", "n_clicks")
    },
    prevent_initial_call=True
)
def update_classes_tbl(n_clicks: int) -> Dict:
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
    result = get_all_classes(conn_settings)
    return {"data": result}
