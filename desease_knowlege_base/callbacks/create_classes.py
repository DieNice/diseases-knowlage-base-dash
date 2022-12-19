import json
import random
from datetime import datetime
from functools import partial
from random import randint, sample
from typing import Dict, List, TypeAlias
from uuid import uuid1

import sqlalchemy
from app import app
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State
from sqlalchemy import MetaData


def generate_features(min_features: int,
                      max_features: int,
                      min_values: int,
                      max_values: int,
                      min_num_periods: int,
                      max_num_periods: int,
                      min_period_duration: int,
                      max_period_duration: int,
                      ) -> List:
    """Generation features for classes by parameters
    """

    result = []

    features_num = randint(min_features, max_features)

    for i in range(1, features_num+1):

        possible_values = [i for i in range(min_values, max_values+1)]

        size_normal_values = randint(1, len(possible_values))
        normal_values = sample(possible_values, size_normal_values)

        num_periods = randint(min_num_periods, max_num_periods)

        periods = generate_periods(
            num_periods, possible_values, min_period_duration, max_period_duration)

        result.append(
            {
                "name": f"feature_{i}",
                "possible_values": possible_values,
                "normal_values": normal_values,
                "num_periods":  num_periods,
                "periods": periods
            }

        )

    return result


def generate_periods(num_periods: int,
                     possible_values: List,
                     min_periods_duration: int,
                     max_periods_duration: int
                     ) -> List:
    """Generation periods
    """
    result_periods = []

    last_values = set()

    for i in range(1, num_periods+1):
        random.seed(datetime.now().timestamp())
        middle_duration = int(
            (max_periods_duration + min_periods_duration) // 2)

        lower_duration = randint(min_periods_duration, middle_duration)
        upper_duration = randint(middle_duration, max_periods_duration)

        size_of_sample = randint(1, len(possible_values)-1)

        values = sample(possible_values, size_of_sample)

        values = set(values)
        values -= last_values

        if len(values) == 0:
            alternative_values = set(possible_values) - last_values
            size_alternative = randint(1, len(alternative_values))
            values = sample(alternative_values, size_alternative)
            values = set(values)

        now_period = {
            "num_period": i,
            "lower_duration": lower_duration,
            "upper_duration": upper_duration,
            "values": list(values)
        }

        last_values = values.copy()

        result_periods.append(now_period)

    return result_periods


def save_classes_to_database(data: Dict, conn_settings: Dict) -> None:
    """Сохранение данных классов в базу данных

    Args:
        data (Dict): Словарь классов
    """
    try:
        usr = conn_settings["usr"]
        pswd = conn_settings["pswd"]
        host = conn_settings["host"]
        port = conn_settings["port"]
        db = conn_settings["db"]
    except KeyError as key_error:
        raise KeyError(f"Bad postgres settings") from key_error

    prepared_data = [{"id": str(uuid1()),
                      "name": record["name"],
                      "symptoms": record["symptoms"]
                      } for record in data]

    engine = sqlalchemy.create_engine(
        f"postgresql://{usr}:{pswd}@{host}:{port}/{db}")

    metadata = MetaData(bind=engine)
    metadata.reflect()
    deseases_tbl = metadata.tables['desease_classes']
    metadata.create_all(checkfirst=True)
    with engine.connect() as conn:
        for record in prepared_data:
            conn.execute(deseases_tbl.insert(),
                         record)


@app.callback(
    output={
        "alert": Output("alert-id", "children")
    },
    inputs={
        "generate": Input("generate-btn-id", "n_clicks"),
        "min_values_num": State("min-values-num-id", "value"),
        "max_values_num": State("max-values-num-id", "value"),
        "min_periods_num": State("min-periods-num-id", "value"),
        "max_periods_num": State("max-periods-num-id", "value"),
        "min_period_duration": State("min-period-duration-id", "value"),
        "max_period_duration": State("max-period-duration-id", "value"),
        "min_values_by_period": State("min-values-by-period-id", "value"),
        "max_values_by_period": State("max-values-by-period-id", "value"),
        "min_classes_num": State("min-classes-num-id", "value"),
        "max_classes_num": State("max-classes-num-id", "value"),
        "min_features_in_classes": State("min-features-in-class-id", "value"),
        "max_features_in_classes": State("max-features-in-class-id", "value"),
        "name_class_pattern": State("name-class-pattern-id", "value"),
    },
    prevent_initial_call=True
)
def generate(generate: int,
             min_values_num: int,
             max_values_num: int,
             min_periods_num: int,
             max_periods_num: int,
             min_period_duration: int,
             max_period_duration: int,
             min_values_by_period: int,
             max_values_by_period: int,
             min_classes_num: int,
             max_classes_num: int,
             min_features_in_classes: int,
             max_features_in_classes: int,
             name_class_pattern: str,
             ):
    if generate is None:
        raise PreventUpdate
    if (
            (min_values_num is None) or
            (max_values_num is None) or
            (min_periods_num is None) or
            (max_periods_num is None) or
            (min_period_duration is None) or
            (max_period_duration is None) or
            (min_values_by_period is None) or
            (max_values_by_period is None) or
            (min_classes_num is None) or
            (max_classes_num is None) or
            (min_features_in_classes is None) or
            (max_features_in_classes is None)):
        raise PreventUpdate
    if name_class_pattern is None:
        raise PreventUpdate
    if name_class_pattern == "":
        raise PreventUpdate

    min_features_num = min_features_in_classes
    max_features_num = max_features_in_classes
    knowledge_database_model = []

    num_classes = randint(min_classes_num, max_classes_num)

    for i in range(1, num_classes+1):
        now_class = {"name": f"{name_class_pattern}_{i}"}
        now_class["symptoms"] = generate_features(
            min_features_num, max_features_num, min_values_num, max_values_num,
            min_periods_num, max_periods_num, min_period_duration, max_period_duration)
        knowledge_database_model.append(now_class)

    save_classes_to_database(knowledge_database_model, {
        "usr": "user",
        "pswd": "password",
        "host": "localhost",
        "port": 5432,
        "db": "deseases"
    })

    return {"alert": "Generation successfull"}
