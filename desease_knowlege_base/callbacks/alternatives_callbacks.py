from itertools import combinations
from random import sample
from typing import Dict, List, Tuple

import dash as dcc
import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.express as px
import sqlalchemy
from app import app
from config import dev_conn_settings
from dash import dash_table, html
from dash_extensions.enrich import Input, Output
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


def transform_df_batch_permutations_to_table(df_batch_permutations: List[pd.DataFrame]) -> dash_table.DataTable:
    """Преобразование списка пачек с перестановками в Dash таблицу

    Args:
        df_batch_permutations (List[pd.DataFrame]): Список перестановок

    Returns:
        dash_table.DataTable: _description_
    """

    concated_df = pd.DataFrame()

    for i, df in enumerate(df_batch_permutations):
        df["num_comb"] = i + 1
        concated_df = pd.concat([concated_df, df])
    concated_df.drop(["uuid"], axis=1, inplace=True)

    new_table = dash_table.DataTable(
        data=concated_df.to_dict("records"),
        columns=[{"id": "num_comb", "name": "Номер комбинации"},
                 {"id": "desease", "name": "Название класса"},
                 {"id": "history", "name": "Номер истории"},
                 {"id": "num_period", "name": "Номер периода"},
                 {"id": "moment_observation", "name": "Момент наблюдения"},
                 {"id": "value", "name": "Значение периода динамики"},
                 {"id": "duration", "name": "Длительность периода динамики"},
                 ],
        id="classes-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            },
            {'if': {'column_id': 'num_comb'},
             'width': '15%'},
            {'if': {'column_id': 'num_period'},
             'width': '15%'},
            {'if': {'column_id': 'value'},
             'width': '20%'},
            {'if': {'column_id': 'history'},
             'width': '20%'},
            {'if': {'column_id': 'duration'},
             'width': '20%'},
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        },
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable='multi',
    )
    return html.Div(new_table)


def generate_permutations(df_batch: pd.DataFrame) -> List[pd.DataFrame]:
    """Генерация расстановок

    Args:
        df_batch (pd.DataFrame): Пачка данных

    Returns:
        pd.DataFrame: Пачка данных с расстановками
    """
    def restore_durations_periods(moments: List, durations: List) -> Tuple[List, List]:
        """Маппинг моментов наблюдения и сгенерированных периодов,
        Правильная расстановка периодов, длительности периодов по комбинациям duriations

        Args:
            moments (List): Список моментов наблюдения
            durations (List): Список длительностей
        """
        result_durations = []
        result_num_periods = []
        moments_i = 0
        len_moments = len(moments)
        for num_period, duration in enumerate(durations):
            for i in range(moments_i, len_moments):
                if duration > moments[i]:
                    result_durations.append(duration)
                    result_num_periods.append(num_period+1)
                    moments_i += 1
        if len(result_durations) != len_moments:
            raise Exception(f"Error restore durations for moments {moments}")
        return result_durations, result_num_periods

    NUM_PERIODS = 5

    moments = df_batch.moment_observation.tolist()

    fake_periods = []
    for i in range(len(moments)-1):
        fake_periods.append(sum(moments[i:i+2])/2)
    last_moment = moments[-1]+1
    fake_periods.append(last_moment)

    len_fake_periods = len(fake_periods)

    all_combs = []
    for num_period in range(1, NUM_PERIODS+1):
        len_comb = len_fake_periods - num_period
        try:
            combs = [i for i in combinations(fake_periods, len_comb)]
        except ValueError as value_error:
            pass
        else:
            all_combs.extend(combs)
    new_all_combs = []
    set_periods = set(fake_periods)
    for comb in all_combs:
        new_comb = set_periods - set(comb)
        if len(new_comb) <= NUM_PERIODS:
            if last_moment in new_comb:
                new_all_combs.append(sorted(list(new_comb)))
    del all_combs

    result_duration_batch_combs = []
    for comb in new_all_combs:
        restored_comb_durations, restored_periods = restore_durations_periods(
            moments, comb)
        new_batch = df_batch.copy()
        new_batch.duration = np.array(restored_comb_durations)
        new_batch.num_period = np.array(restored_periods)
        result_duration_batch_combs.append(new_batch)

    return result_duration_batch_combs


def get_alternative(permutation: pd.DataFrame) -> pd.DataFrame:
    """ На основе каждой расстановки границ периодов динамики
    сформировать альтернативу индуктивной базы знаний (эта альтернатива
    будет относиться к конкретному признаку из конкретной истории
    болезни)
     Условия формирования:
        * для каждого периода динамики формируется значение параметра
        «Значения для периода», в него включаются все неповторяющиеся
        значения в моменты наблюдения, которые попали в этот период;
        * для каждого периода динамики формируется значение параметра
        «Нижняя граница» как разница первого момента наблюдения в
        периоде и левой границы периода;
        * для каждого периода динамики формируется значение параметра
        «Верхняя граница» как разница последнего момента наблюдения в
        периоде и левой границы периода.


    Args:
        permutations (List[pd.DataFrame]): _description_

    Returns:
        pd.DataFrame: _description_
    """

    res_df = permutation.groupby('num_period').agg(
        {'value': lambda x: list(x),
         'moment_observation': min,
         'duration': lambda x: x.iloc[0]
         }).reset_index('num_period')

    res_df.value = res_df.value.apply(lambda x: list(set(x)))

    res_df.rename(columns={"moment_observation": "lower_duration",
                  "duration": "upper_duration"}, inplace=True)
    res_df["history"] = permutation.history[0]
    res_df["desease"] = permutation.desease[0]
    res_df["feature"] = permutation.feature[0]
    res_df["amount_period"] = len(res_df.num_period.unique())

    return res_df


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

    all_alternatives = pd.DataFrame()
    graphs = []
    for config in configs:
        batch = get_batch_by_config(config)
        df_batch = pd.DataFrame(batch)
        df_batch = preparing_df_batch(df_batch)
        graphs.append(transform_df_batch_to_graph(df_batch))
        permutations = generate_permutations(df_batch)

        for permutation in permutations:
            new_alternative = get_alternative(permutation)
            all_alternatives = pd.concat([all_alternatives, new_alternative])

        table = transform_df_batch_permutations_to_table(permutations)
        graphs.append(table)

    result = all_alternatives.groupby(['desease', 'feature', 'amount_period', 'num_period']).agg(
        {'value': lambda x: set(list(sum(x, []))),
         'lower_duration': min,
         'upper_duration': max
         }).reset_index(['desease', 'feature', 'amount_period', 'num_period'])

    return {"alert": "Generation successfull!",
            "graphs": graphs}
