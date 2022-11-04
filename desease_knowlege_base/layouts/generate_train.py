import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash_extensions.enrich import html
from dash import dash_table
from callbacks.create_train import prepare_data_for_classes_tbl

generate_train_layout = dbc.Container([
    dbc.Row([dbc.Col(html.Label("Число экземпляров для класс"))]),
    dbc.Row([dbc.Col(dbc.Input(id="num-instance-id", type="number",
            min=1, placeholder="Input seed of instances"))]),
    dbc.Row(
        dash_table.DataTable(
            data=prepare_data_for_classes_tbl(),
            columns=[{"id": "name-id", "name": "Название класса"},
                     {"id": "feature-id", "name": "Название признака"},
                     {"id": "period-id", "name": "Номер периода"},
                     {"id": "lower-id", "name": "Нижняя граница"},
                     {"id": "upper-id", "name": "Верхняя граница"},
                     {"id": "value", "name": ""}
                     ]
        ), id="classes-tbl-id")
]
)
