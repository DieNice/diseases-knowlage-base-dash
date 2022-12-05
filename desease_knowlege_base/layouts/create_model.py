import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash_extensions.enrich import html

create_model_layout = dbc.Container(
    [
        dbc.Row(dbc.Alert("Generation input form",
                color="primary", id="alert-id")),
        dbc.Row(dbc.Col(html.Label("Паттерн имени для классов"))),
        dbc.Row([
            dbc.Col([dbc.Input(id="name-class-pattern-id", type="text",
                               placeholder="Введите паттерн классов")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число генерируемых классов")),
                 dbc.Col(html.Label("Максимальное число генерируемых классов"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-classes-num-id", type="number", min=1,
                               max=100, placeholder="Введите минимальное число генерируемых классов")]),
            dbc.Col([dbc.Input(id="max-classes-num-id", type="number", min=1,
                               max=100, placeholder="Введите максимальное число генерируемых классов")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число признаков в классе")), dbc.Col(
            html.Label("Максимальное число признаков в классе"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-features-in-class-id", type="number", min=1,
                               max=100, placeholder="Введите минимальное число признаков в классе")]),
            dbc.Col([dbc.Input(id="max-features-in-class-id", type="number", min=1,
                               max=100, placeholder="Введите максимальное число признаков в классе")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное значение признака")),
                dbc.Col(html.Label("Максимальное значение признака"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-values-num-id", type="number", min=1,
                               max=100, placeholder="Введите минимальное значение признака")]),
            dbc.Col([dbc.Input(id="max-values-num-id", type="number", min=1,
                               max=100, placeholder="Введите максимальное значение признака")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное число периодов")),
                dbc.Col(html.Label("Максимальное число периодов"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-periods-num-id", type="number", min=1,
                               max=5, placeholder="Введите минимальное число периодов")]),
            dbc.Col([dbc.Input(id="max-periods-num-id", type="number", min=1,
                               max=5, placeholder="Введите максимальное число периодов")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальная продолжительность периода")), dbc.Col(
            html.Label("Максимальная продолжительность периода"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-period-duration-id", type="number", min=1,
                               max=100, placeholder="Введите минимальную продолжительность периода")]),
            dbc.Col([dbc.Input(id="max-period-duration-id", type="number", min=1,
                               max=100, placeholder="Введите максимальную продолжительность периода")])
        ]),
        dbc.Row([dbc.Col(html.Label("Минимальное значение периода")),
                dbc.Col(html.Label("Максимальное значение периода"))]),
        dbc.Row([
            dbc.Col([dbc.Input(id="min-values-by-period-id", type="number", min=1,
                               max=100, placeholder="Введите минимальное значение периода")]),
            dbc.Col([dbc.Input(id="max-values-by-period-id", type="number", min=1,
                               max=100, placeholder="Введите максимальное значение периода")])
        ]),
        dbc.Row([
            dbc.Button("generate", color="primary",
                       className="me-1", id="generate-btn-id"),
        ])
    ],
)
