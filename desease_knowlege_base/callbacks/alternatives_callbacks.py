import copy
import json
import random
from datetime import datetime
from random import randint, sample
from typing import Dict, List, Tuple
from uuid import uuid1

import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import sqlalchemy
from app import app
from callbacks.create_train import get_all_classes, get_all_train
from config import dev_conn_settings
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input, Output, State
from sqlalchemy import MetaData


def sql_query(conn_settings: Dict, text_query: str) -> List:
    """Sql query

    Returns:
        List: _description_
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
        data = conn.execute(text_query)
    return [row._data for row in data]


def get_all_desease_histories_by_desease(name_deasease: str) -> List:
    """Получение историй болезеней по имени класса болезни

    Args:
        name_deasease (str): Название болезни

    Returns:
        List: Список историй болезеней
    """

    query = f"select distinct number_history from desease_train dt where dt.name_class='{name_deasease}'"
    data = sql_query(dev_conn_settings, query)
    result = sum([[*row] for row in data], [])
    del data
    return result


def get_all_deseases() -> List:
    """Получение всех болезеней

    Returns:
        List:
    """
    query = f"select distinct name_class from desease_train dt"
    data = sql_query(dev_conn_settings, query)
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
    query = f"select symptoms from desease_classes dc where dc.name='{name_desease}'"
    data = sql_query(dev_conn_settings, query)
    result = sum([[*row] for row in data], [])
    result = result.pop()
    del data
    return result


def get_batch_by_config(config: Dict) -> List:
    """Получить пачку строк обучающей выборки по параметрам config

    Args:
        config (Dict): Параметры

    Returns:
        List: Список данных
    """
    query = "select * from desease_train dt where"\
            f" dt.name_class = '{config['desease']}'"\
            f" and dt.number_history = '{config['history']}'"\
            f" and dt.name_feature ='{config['feature']}'"\
        " order by num_period"
    data = sql_query(dev_conn_settings, query)
    return data


def generate_random_configs() -> List[Dict]:
    """Генерация случайных конфигураций
    Выбор 2 любых признаков, 2-х любых историй болезни всех болезней

    Returns:
        List[Dict]: _description_
    """
    NUM_HISTORIES = 2
    NUM_FEATURES = 2
    all_deseases = get_all_deseases()

    configs = []

    for desease in all_deseases:

        histories = get_all_desease_histories_by_desease(desease)
        features = get_features_by_desease(desease)

        selected_histories = sample(histories, NUM_HISTORIES)

        selected_features = sample(features, NUM_FEATURES)
        names_features = [feature["name"] for feature in selected_features]
        del selected_features

        for history in selected_histories:
            for feature in names_features:
                configs.append(
                    {
                        "desease": desease,
                        "history": history,
                        "feature": feature
                    }
                )
    del all_deseases, histories, features, selected_histories, names_features, history
    return configs


def preparing_df_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """Нормализация моментов наблюдения и длительности по периодам

    Args:
        df_batch (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    names_columns = ['uuid', 'history', 'desease', 'feature',
                     'num_period', 'moment_observation', 'value', 'duration']
    df_batch = df_batch.set_axis(names_columns, axis=1)
    len_periods = len(list(df_batch.num_period.unique()))

    if len_periods > 1:
        num_period = 2
        for i in range(len_periods-1):
            durations = df_batch.groupby(["num_period", "duration"]).count().reset_index()[
                "duration"].tolist()
            durations.pop()
            duration = durations[i]
            condlist = [df_batch.num_period == num_period]
            choicelist_1 = [df_batch.duration + duration]
            choicelist_2 = [df_batch.moment_observation + duration]
            df_batch["duration"] = pd.np.select(
                condlist, choicelist_1, df_batch.duration)
            df_batch["moment_observation"] = pd.np.select(
                condlist, choicelist_2, df_batch.moment_observation)
            num_period += 1

    return df_batch


def transform_df_batch_to_graph(df_batch: pd.DataFrame) -> dcc.Graph:
    """Преобразование пачки данных в объект графа

    Args:
        df_batch (pd.DataFrame): _description_

    Returns:
        dcc.Graph: _description_
    """
    history = list(df_batch.history)[0]
    desease = list(df_batch.desease)[0]
    feature = list(df_batch.feature)[0]
    graph_title = f"{desease} \n {history} \n {feature}"

    fig = px.scatter(df_batch, x="moment_observation",
                     y="value", title=graph_title, width=200)
    x_periods = list(df_batch.duration.unique())
    for x in x_periods:
        fig.add_vline(x=x, line_width=3, line_dash="dash", line_color="green")

    return dcc.Graph(figure=fig)


@ app.callback(
    output={
        "alert": Output("alert-alternatives-id", "children"),
        "graphs": Output("graphs-content", "children")
    },
    inputs={
        "n_clicks": Input("generate-alternatives-id", "n_clicks"),
    },
    prevent_initial_call=True


)
def generate_alternatives(n_clicks: int) -> Dict:
    configs = generate_random_configs()

    graphs = []
    for config in configs:
        batch = get_batch_by_config(config)
        df_batch = pd.DataFrame(batch)
        df_batch = preparing_df_batch(df_batch)
        graphs.append(transform_df_batch_to_graph(df_batch))

    return {"alert": "Generation successfull!",
            "graphs": graphs}
