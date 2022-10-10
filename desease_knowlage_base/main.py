import os
from typing import Any

import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import html
from dash_extensions.enrich import Input, Output

from app import app
from app import srv as server
from layouts.navbar import Navbar

app_name = os.getenv("DASH_APP_PATH", "/desease_knowlage_base")

nav = Navbar()


header = html.Div(
    children=[
        html.P(children="📊", className="header-emoji"),
        html.H1(
            children="Deseases knowlage base", className="header-title"
        ),
        html.P(
            children="Анализ данных модельной базы знаний",
            className="header-description",
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
                )

            ],
            className="home",
        )
    elif pathname.endswith("/create"):
        return html.Div()
    else:
        return "ERROR 404: Page not found!"


def index():
    return html.Div([nav, container])


app.layout = index()

if __name__ == '__main__':
    app.run_server(debug=True)
