import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash_extensions.enrich import html

create_model_layout = dbc.Container(
    [
        # dcc.Download(id="download-classes-id"),
        dbc.Row(dbc.Alert("Generation input form",
                color="primary", id="alert-id")),
        # dbc.Row([html.Label("Объём генерируемой выборки"),
        #          dbc.Col([dbc.Input(id="generation-seed-id", type="number", min=1,
        #                             max=1000, placeholder="Input generation seed")])
        #          ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число признаков")),
                dbc.Col(html.Label("Максимальное число признаков"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-features-num-id", type="number", min=1,
                               max=100, placeholder="Input features min num")]),
            dbc.Col([dbc.Input(id="max-features-num-id", type="number", min=1,
                               max=100, placeholder="Input features max num")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное значение признака")),
                dbc.Col(html.Label("Максимальное значение признака"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-values-num-id", type="number", min=1,
                               max=100, placeholder="Input values min num")]),
            dbc.Col([dbc.Input(id="max-values-num-id", type="number", min=1,
                               max=100, placeholder="Input values max num")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число периодов")),
                dbc.Col(html.Label("Максимальное число периодов"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-periods-num-id", type="number", min=1,
                               max=100, placeholder="Input periods min num")]),
            dbc.Col([dbc.Input(id="max-periods-num-id", type="number", min=1,
                               max=100, placeholder="Input periods max num")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальная продолжительность периода")), dbc.Col(
            html.Label("Максимальная продолжительность периода"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-period-duration-id", type="number", min=1,
                               max=100, placeholder="Input min period duration")]),
            dbc.Col([dbc.Input(id="max-period-duration-id", type="number", min=1,
                               max=100, placeholder="Input max period duration")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное значение периода")),
                dbc.Col(html.Label("Максимальное значение периода"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-values-by-period-id", type="number", min=1,
                               max=100, placeholder="Input min values by period")]),
            dbc.Col([dbc.Input(id="max-values-by-period-id", type="number", min=1,
                               max=100, placeholder="Input max values by period")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число классов")),
                dbc.Col(html.Label("Максимальное число классов"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-classes-num-id", type="number", min=1,
                               max=100, placeholder="Input min classes num")]),
            dbc.Col([dbc.Input(id="max-classes-num-id", type="number", min=1,
                               max=100, placeholder="Input max classes num")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число признаков в классе")), dbc.Col(
            html.Label("Максимальное число признаков в классе"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-features-in-class-id", type="number", min=1,
                               max=100, placeholder="Input min classes features")]),
            dbc.Col([dbc.Input(id="max-features-in-class-id", type="number", min=1,
                               max=100, placeholder="Input max classes features")])
        ]),
        dbc.Row(dbc.Col(html.Label("Паттерн имени для классов"))),
        dbc.Row([
            dbc.Col([dbc.Input(id="name-class-pattern-id", type="text",
                               placeholder="Input name class pattern")])
        ]),
        dbc.Row([
            dbc.Button("generate", color="primary",
                       className="me-1", id="generate-btn-id"),
        ])
    ],
)
