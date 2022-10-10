import os

import dash_bootstrap_components as dbc


app_name = os.getenv("DASH_APP_PATH", "/dns-acquiring-dash")


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
        brand_href=f"{app_name}/",
        sticky="top",
        color="black",1
        dark=False,
        expand="lg",
    )
    return navbar
