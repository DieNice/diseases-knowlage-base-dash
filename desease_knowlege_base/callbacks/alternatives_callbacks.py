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

    common_features = []
    for desease in all_deseases:
        common_features.extend(get_features_by_desease(desease))
    common_features = {feature['name'] for feature in common_features}

    names_features = sample(common_features, NUM_FEATURES)

    configs = []

    for desease in all_deseases:

        histories = get_all_desease_histories_by_desease(desease)
        features = get_features_by_desease(desease)

        selected_histories = sample(histories, NUM_HISTORIES)

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
            df_batch["duration"] = np.select(
                condlist, choicelist_1, df_batch.duration)
            df_batch["moment_observation"] = np.select(
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
                 {"id": "num_period", "name": "НП"},
                 {"id": "moment_observation", "name": "МН"},
                 {"id": "value", "name": "ЗПД"},
                 {"id": "duration", "name": "ДПД"},
                 ],
        id="classes-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold',
            'fontSize': '1em'
        },
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable='multi',
    )
    return html.Div(new_table)

def transform_alternative_to_table(alternatives: List[pd.DataFrame]) -> dash_table.DataTable:
    """Преобразование альтернатив болезни по определенному признакн в таблицу
    алтернатив

    Args:
        alternatives (List[pd.DataFrame]): Альтернативы

    Returns:
        dash_table.DataTable: Таблица альтернатив
    """
    concated_df = pd.DataFrame()

    for i, df in enumerate(alternatives):
        df["num_alt"] = i + 1
        concated_df = pd.concat([concated_df, df])
    concated_df.value = concated_df.value.apply(str)
    new_table = dash_table.DataTable(
        data=concated_df.to_dict("records"),
        columns=[{"id": "num_alt", "name": "Номер альтернативы"},
                 {"id": "desease", "name": "Название класса"},
                 {"id": "num_period", "name": "Номер периода"},
                 {"id": "value", "name": "ЗПД"},
                 {"id": "lower_duration", "name": "НГ"},
                 {"id": "upper_duration", "name": "ВГ"},
                 ],
        id="classes-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            },
            {'if': {'column_id': 'num_alt'},
             'width': '15%'},
            {'if': {'column_id': 'num_period'},
             'width': '15%'},
            {'if': {'column_id': 'value'},
             'width': '20%'},
            {'if': {'column_id': 'lower_duration'},
             'width': '20%'},
            {'if': {'column_id': 'upper_duration'},
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


def unite_alternatives(alternatives: List[pd.DataFrame]) -> List[Dict]:
    """Объединение альтернатив по одноимённым признакам у двух историй болезни
    с одним экзаменом.
        Условия:
            • объединяются альтернативы с одинаковым ЧПД;
            • при объединении для соответствующих периодов
            «Значения для периода» объединяются;
            • при объединении для соответствующих периодов
            «Нижняя граница» выбирается минимальной из двух;
            • при объединении для соответствующих периодов
            «Верхняя граница» выбирается максимальной из двух;
            • после объединения сохраняется только результат объединения
            (объединявшиеся альтернативы удаляются).

    Args:
        alternatives (List[pd.DataFrame]): Список с альтернативами

    Returns:
        List[Dict]: Готовые варианты алтернатив
    """
    df = pd.DataFrame(alternatives)
    agg_df = df.groupby(['desease', 'feature']).agg(
        {'alternatives': lambda x: list(x)}).reset_index()
    agg_dicts = agg_df.to_dict('records')
    del df, agg_df

    result_alts = []

    for agg_dict in agg_dicts:
        alts_1, alts_2 = tuple(agg_dict['alternatives'])
        tmp_alts = []
        for alt_1 in alts_1:
            alt_1_amount_period = alt_1.amount_period[0]
            for alt_2 in alts_2:
                alt_2_amount_period = alt_2.amount_period[0]
                if alt_1_amount_period == alt_2_amount_period:
                    sub_df = pd.concat([alt_1, alt_2])
                    sub_res = sub_df.groupby(['desease', 'feature', 'amount_period', 'num_period']).agg(
                        {'value': lambda x: list(set(list(sum(x, [])))),
                         'lower_duration': min,
                         'upper_duration': max
                         }).reset_index(['desease', 'feature', 'amount_period',
                                         'num_period'])
                    del sub_df
                    if is_corrected_alternative(sub_res):
                        tmp_alts.append(sub_res)
        result_alts.append({
            'desease': agg_dict['desease'],
            'feature': agg_dict['feature'],
            'alternatives': tmp_alts
        })

    return result_alts


def is_corrected_alternative(alternative: pd.DataFrame) -> bool:
    """Проверка на валидность алтернативы, если значения в соседних периодах
    динамики пересекаются то False

    Args:
        alternative (pd.DataFrame): Сгенерированная альтернатива

    Returns:
        bool: True - корректно, False -  некорректно
    """
    agg_alternative = alternative.groupby(['num_period']).agg(
        {'value': lambda x: sum(list(x), [])}).reset_index()
    alternative_dicts = agg_alternative.to_dict('records')
    if len(agg_alternative['num_period'].unique()) == 5:
        pass
    prev_set = set()
    for alternative_dict in alternative_dicts:
        sub_set = set(alternative_dict['value']) & prev_set
        if len(sub_set) != 0:
            return False
        prev_set = set(alternative_dict['value'])
    if len(agg_alternative['num_period'].unique()) == 5:
        pass
    return True


def choose_best_alternative(alternatives: List[Dict]) -> pd.DataFrame:
    """Выбор лучшей альтернативы

    Args:
        alternatives (List[pd.DataFrame]): Список Альтернатив конкретной болезни
    определенного признака
    Returns:
        pd.DataFrame: Лучшая альтернативая DataFrame
    """
    feautures = get_features_by_desease(alternatives['desease'])
    need_model_feature = None
    for model_feature in feautures:
        if model_feature['name'] == alternatives['feature']:
            need_model_feature = model_feature
            del feautures
            break
    if need_model_feature is None:
        raise ValueError("Model feature don't exists!")
    alts = alternatives['alternatives']

    similar_alts = []
    for need_num_periods in range(need_model_feature['num_periods'], 0, -1):
        for alt in alts:
            alt_num_period = max(alt.num_period)
            if alt_num_period == need_num_periods:
                similar_alts.append(alt)
        if similar_alts:
            break

    best_alt = None
    best_mark = 0
    model_periods = model_feature['periods']
    for similar_alt in similar_alts:
        similar_dicts = similar_alt.to_dict('records')
        sub_mark = 0
        for i, similar_dict in enumerate(similar_dicts):
            model_period_values = model_periods[i]['values']
            sub_set = set(model_period_values) & set(similar_dict['value'])
            sub_mark += len(sub_set) / len(model_period_values)
        if sub_mark >= best_mark:
            best_alt = similar_alt
            best_mark = sub_mark
            
    if best_alt is None:
        raise ValueError("Similar alternative does not exists")
    return best_alt

def generate_periods_report(best_alts:List[pd.DataFrame])->html.Div:
    """Сравнение ЧПД у одноименных признаков отдельно для каждого заболевания,
    для всех заболеваний

    Args:
        best_alts (List[pd.DataFrame]): Список лучших альтернатив

    Returns:
        html.Div: Отчёт сравнения ЧПД
    """
    concated_alts = pd.DataFrame()
    for alt in best_alts:
        concated_alts = pd.concat([concated_alts,alt])
    result_alts = concated_alts.groupby(['desease',
    'feature']).agg({'amount_period':'max'}).reset_index()
    alt_dicts = result_alts.to_dict('records')
    del result_alts, concated_alts

    result_dicts = []
    for alt_dict in alt_dicts:
        desease_features = get_features_by_desease(alt_dict['desease'])
        need_feature = None
        for feature in desease_features:
            if feature['name'] == alt_dict['feature']:
                need_feature = feature
                break
        model_amount_period = need_feature['num_periods']
        del desease_features, need_feature
        result_dicts.append({**alt_dict,
            'model_amount_period': model_amount_period
        })
    
    result_report = pd.DataFrame(result_dicts)
    del result_dicts
    result_report['eq'] = result_report.amount_period == result_report.model_amount_period
    percentage_by_desease = result_report.groupby(['desease']).agg({'amount_period':'count',
    'eq':'sum'}).reset_index()
    percentage_by_desease['eq'] = percentage_by_desease['eq'].astype(float)
    percentage_by_desease['amount_period'] = percentage_by_desease['amount_period'].astype(float)
    percentage_by_desease['percentage'] = percentage_by_desease['amount_period'] / percentage_by_desease['eq'] * 100;
    percentage_by_all = percentage_by_desease['percentage'].mean()


    new_table = dash_table.DataTable(
        data=result_report.to_dict("records"),
        columns=[{"id": "desease", "name": "Название класса"},
                 {"id": "feature", "name": "Название признака"},
                 {"id": "amount_period", "name": "ИФБЗ ЧПД"},
                 {"id": "model_amount_period", "name": "МБ ЧПД"},
                 ],
        id="periods-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
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
    description = "Процент совпадения ЧПД: \n" +"\n".join([f"Заболевание: {dct['desease']} - {dct['percentage']}%" for
     dct in percentage_by_desease.to_dict('records')])
    description += f"\n Средний процент совпадения ЧПД: {percentage_by_all}%"


    return html.Div([new_table, html.P(description)])


def generate_zdp_report(alts:List[pd.DataFrame])->html.Div:
    """Соотнесение обслатей значений признаков (ЗДП) в соответствующих периодах


    Args:
        alts (List[pd.DataFrame]): Список лучший альтернатив

    Returns:
        html.Div: Отчёт
    """
    result_report = pd.DataFrame()
    for alt in alts:
        alt_desease = alt.desease[0]
        alt_feature = alt.feature[0]
        feautures = get_features_by_desease(alt_desease)

        need_model_feature = None
        for model_feature in feautures:
            if model_feature['name'] == alt_feature:
                need_model_feature = model_feature
                del feautures
                break
        if need_model_feature is None:
            raise ValueError("Model feature don't exists!")
        if alt.amount_period.max() != need_model_feature['num_periods']:
            continue
        else:
            model_value = [period['values'] for period in need_model_feature['periods']]
            alt['model_value'] = model_value
            del model_value
            alt['percentage'] = alt.apply(lambda x: len(set(x['value']) & set(x['model_value']))/len(x['model_value'])*100 ,axis=1) 
            result_report = pd.concat([result_report,alt])
    
    result_report['value'] = result_report['value'].astype(str)
    result_report['model_value'] = result_report['model_value'].astype(str)

    new_table = dash_table.DataTable(
        data=result_report.to_dict("records"),
        columns=[{"id": "desease", "name": "Заболевание"},
                 {"id": "feature", "name": "Признак"},
                 {"id": "num_period", "name": "Номер периода"},
                 {"id": "value", "name": "ИФБЗ ЗДП"},
                 {"id": "model_value", "name": "МБ ЗДП"},
                 {"id": "percentage", "name": "Процент совпадения"},
                 ],
        id="zdp-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            },
            {'if': {'column_id': 'num_alt'},
             'width': '15%'},
            {'if': {'column_id': 'num_period'},
             'width': '15%'},
            {'if': {'column_id': 'value'},
             'width': '20%'},
            {'if': {'column_id': 'lower_duration'},
             'width': '20%'},
            {'if': {'column_id': 'upper_duration'},
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
    common_percentage = result_report['percentage'].mean()
    description = f"Средний процент соотнесения областей значений признаков ЗДП для всех заболеваний {common_percentage}%"

    return html.Div([new_table,html.P(description)])

@ app.callback(
    output={
        "alert": Output("alert-alternatives-id", "children"),
        "report": Output("report-content", "children")
    },
    inputs={
        "n_clicks": Input("generate-alternatives-id", "n_clicks"),
    },
    prevent_initial_call=True


)
def generate_alternatives(n_clicks: int) -> Dict:
    """Генерация отчёта о формирования альтернатив

    Args:
        n_clicks (int): _description_

    Returns:
        Dict: _description_
    """
    configs = generate_random_configs()

    report = []
    report.append(html.H2("Комбинации"))
    for config in configs:
        batch = get_batch_by_config(config)
        df_batch = pd.DataFrame(batch)
        df_batch = preparing_df_batch(df_batch)
        report.append(transform_df_batch_to_graph(df_batch))
        permutations = generate_permutations(df_batch)

        filtered_permutations = []
        alternatives = []
        for permutation in permutations:
            new_alternative = get_alternative(permutation)
            if is_corrected_alternative(new_alternative):
                alternatives.append(new_alternative)
                filtered_permutations.append(permutation)
        config['alternatives'] = alternatives
        table = transform_df_batch_permutations_to_table(filtered_permutations)
        report.append(table)

    alternatives = unite_alternatives(configs)
    report.append(html.H2("Альтернативы"))

    best_alts = []
    for alt in alternatives:
        best_alt = choose_best_alternative(alt)
        best_alts.append(best_alt)
        report.append(html.H3(f"{alt['desease']} {alt['feature']}"))
        report.append(transform_alternative_to_table(alt['alternatives']))
        report.append(
            html.H4(f"Лучшая альтернатива {alt['desease']} {alt['feature']}",
                    style={'background-color': 'green',
                           'color': 'white'}))
        report.append(transform_alternative_to_table([best_alt]))
    
    report.append(html.H4("Сравнение периодов",style={'background-color': 'purple',
                           'color': 'white'})) 
    report.append(generate_periods_report(best_alts))
    report.append(html.H4("Соотнесение областей значений признаков (ЗДП)",style={'background-color': 'purple',
                           'color': 'white'}))
    report.append(generate_zdp_report(best_alts))
    return {"alert": "Generation successfull!",
            "report": report}
