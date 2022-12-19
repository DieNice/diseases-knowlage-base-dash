import dash_bootstrap_components as dbc
import dash_core_components as dcc
from callbacks.create_train import prepare_data_for_classes_tbl
from dash import dash_table
from dash_extensions.enrich import html

generate_train_layout = dbc.Container([
    dbc.Row([dbc.Alert("Generation input form",
                       color="primary", id="alert-id"),
            dbc.Button("Update generated classes table", color="success",
                       className="me-3", id="update-classes-tbl-id", size='sm')]),
    dash_table.DataTable(
        fixed_rows={'headers': True},
        data=prepare_data_for_classes_tbl(),
        columns=[{"id": "name-id", "name": "Название класса"},
                 {"id": "feature-id", "name": "Название признака"},
                 {"id": "period-id", "name": "Номер периода"},
                 {"id": "values-id", "name": "Значения периода динамики"},
                 {"id": "lower-id", "name": "Нижняя граница"},
                 {"id": "upper-id", "name": "Верхняя граница"},
                 ],
        id="classes-tbl-id",
        style_table={'height': '300px',
                     'overflowY': 'auto', 'overflowX': 'auto'},
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
            },
            {'if': {'column_id': 'name-id'},
             'width': '15%'},
            {'if': {'column_id': 'period-id'},
             'width': '15%'},
            {'if': {'column_id': 'values-id'},
             'width': '20%'},
            {'if': {'column_id': 'feature-id'},
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
    ),
    dbc.Row([dbc.Col(html.Label("Число экземпляров на класс"))],
            style={"margin-top": "1vh"}),
    dbc.Row([dbc.Col(dbc.Input(id="num-instance-id", type="number",
            min=1, placeholder="Введите число генерируемых историй болезней на один класс"))], style={"margin-top": "1vh"}),
    dbc.Row([dbc.Col(html.Label("Максимальное число моментов наблюдений в периоде"))],
            style={"margin-top": "1vh"}),
    dbc.Row([dbc.Col(dbc.Input(id="num-observetions-id", type="number",
            min=1, max=4, placeholder="Введите максимальное число моментов наблюдей в периоде"))], style={"margin-top": "1vh"}),
    dbc.Row([
            dbc.Button("Generate Train", color="primary",
                       className="me-1", id="generate-train-id"),
            dbc.Button("Update generated train table", color="success",
                       className="me-3", id="update-train-tbl-id", size='sm')], style={"margin-top": "1vh",
                                                                                       "margin-bottom": "2vh"}),

    dash_table.DataTable(
        data=[{}, {}, {}],
        columns=[{"id": "name_class", "name": "Название класса"},
                 {"id": "number_history", "name": "История болезни"},
                 {"id": "name_feature", "name": "Название признака"},
                 {"id": "num_period", "name": "Номер периода"},
                 {"id": "num_observetion", "name": "Момент наблюдения"},
                 {"id": "value", "name": "Значение"},
                 {"id": "duration", "name": "Длительность периода динамики"},
                 ],
        id="train-tbl-id",
        style_table={'height': '500px',
                     'overflowY': 'auto', 'margin-bottom': '4vh'},
        style_cell_conditional=[
        ],
        style_data={
            'color': 'black',
            'backgroundColor': 'white',
            'whiteSpace': 'normal',
            'width': 'auto'
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
        column_selectable='multi',
    )
]
)
