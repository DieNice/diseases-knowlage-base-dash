import os

import dash_bootstrap_components as dbc


app_name = os.getenv("DASH_APP_PATH", "/desease-knowlege-base-dash")


def Navbar():
    """Навигационная панель
    """
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Создание модельной базы знаний",
                        href=f"{app_name}/create")),
            dbc.NavItem(dbc.NavLink("Генерация модельной выборки",
                        href=f"{app_name}/generate")),
            dbc.NavItem(dbc.NavLink("Индуктивное формирование",
                        href=f"{app_name}/induction")),
            dbc.NavItem(dbc.NavLink("Оценка",
                        href=f"{app_name}/eval")),
        ],
        brand="Главная",
        brand_href="/",
        sticky="top",
        color="black",
        dark=True,
        expand="lg",
    )
    return navbar
