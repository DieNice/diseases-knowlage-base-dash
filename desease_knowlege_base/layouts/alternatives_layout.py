import dash_bootstrap_components as dbc
import dash_core_components as dcc
from callbacks.create_train import prepare_data_for_classes_tbl
from dash import dash_table
from dash_extensions.enrich import html

alternatives_layout = dbc.Container(
    [
        dbc.Row([dbc.Alert("Формирование альтернатив индуктивной базы знаний",
                           color="success", id="alert-alternatives-id")]),
        dbc.Row([dbc.Col(), dbc.Col(dbc.Button("Generation alternatives", color="primary",
                                               className="me-3", id="generate-alternatives-id", size='lg')), dbc.Col()]),
        html.Div(id="graphs-content"),
        dbc.Label("Альтернативы",style={"text-align":"center"}),
        html.Hr(),
        html.Hr(),
        html.Div(id="alternatives-content"),
        html.Hr(),
        html.Hr(),
    ]
)
