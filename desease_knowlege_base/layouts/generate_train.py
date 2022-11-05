import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash_extensions.enrich import html
from dash import dash_table
from callbacks.create_train import prepare_data_for_classes_tbl

generate_train_layout = dbc.Container([
    dbc.Row(dbc.Alert("Generation input form",
                      color="primary", id="alert-id")),
    dash_table.DataTable(
        data=prepare_data_for_classes_tbl(),
        columns=[{"id": "name-id", "name": "Название класса"},
                 {"id": "feature-id", "name": "Название признака"},
                 {"id": "period-id", "name": "Номер периода"},
                 {"id": "lower-id", "name": "Нижняя граница"},
                 {"id": "upper-id", "name": "Верхняя граница"},
                 ],
        id="classes-tbl-id",
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['Date', 'Region']
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
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
    ),
    dbc.Row([dbc.Col(html.Label("Число экземпляров на класс"))],
            style={"margin-top": "1vh"}),
    dbc.Row([dbc.Col(dbc.Input(id="num-instance-id", type="number",
            min=1, placeholder="Input seed of instances"))], style={"margin-top": "1vh"}),
    dbc.Row([
            dbc.Button("Generate Train", color="primary",
                       className="me-1", id="generate-train-id")], style={"margin-top": "1vh",
                                                                          "margin-bottom": "2vh"}),
    dash_table.DataTable(
        data=[{}, {}, {}],
        columns=[{"id": "name_class", "name": "Название класса"},
                 {"id": "name_feature", "name": "Название признака"},
                 {"id": "num_period", "name": "Номер периода"},
                 {"id": "value", "name": "Значение"},
                 {"id": "duration", "name": "Время"},
                 ],
        id="train-tbl-id",
        style_table={'height': '500px', 'overflowY': 'auto','margin-bottom': '4vh'},
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['Date', 'Region']
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
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
    )
]
)
