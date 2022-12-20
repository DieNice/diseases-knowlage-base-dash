import os
from typing import Any

import dash_bootstrap_components as dbc
import dash_core_components as dcc
from app import app
from app import srv as server
from callbacks.alternatives_callbacks import generate_alternatives
from callbacks.create_classes import generate
from callbacks.create_train import (generate_train_dataset, update_classes_tbl,
                                    update_train_tbl)
from dash_extensions.enrich import Input, Output, html
from layouts.alternatives_layout import alternatives_layout
from layouts.create_model import create_model_layout
from layouts.generate_train import generate_train_layout
from layouts.navbar import Navbar

app_name = os.getenv("DASH_APP_PATH", "/desease_knowlege_base")

nav = Navbar()


header = html.Div(
    children=[
        html.H1(
            children="Deseases knowlege base", className="header-title"
        ),
    ],
    className="header",
)

content = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])

container = dbc.Container([header, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname: str) -> Any:
    """Функция для оторисовки страниц

    Args:
        pathname (str): url

    Returns:
        Any: Layout
    """

    if pathname in [app_name, app_name + "/", '/']:
        return html.Div(
            [
                dcc.Markdown(
                    """
            Данное приложение необходимо для анализа модельной базы знаний
            
        """, className='main-content'
                ),
                dbc.Carousel(
                    items=[
                        {"key": "1", "src": "./assets/images/infectious diseases.jpeg"},
                    ],
                    controls=False,
                    indicators=False,
                    interval=2000,
                    ride="carousel",
                )

            ],
            className="home",
        )
    elif pathname.endswith("/create"):
        return create_model_layout
    elif pathname.endswith("/generate"):
        return generate_train_layout
    elif pathname.endswith("/induction"):
        return alternatives_layout
    elif pathname.endswith("/eval"):
        return html.Div("Developing, please wait")
    else:
        return "ERROR 404: Page not found!"


def index():
    return html.Div([nav, container])


app.layout = index()

if __name__ == '__main__':
    app.run_server(host='localhost', port=8050, debug=False)
