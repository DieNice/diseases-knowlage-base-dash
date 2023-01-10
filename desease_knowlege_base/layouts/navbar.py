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
            dbc.NavItem(dbc.NavLink("Индуктивное формирование/Отчёт",
                        href=f"{app_name}/induction"))
        ],
        brand="Главная",
        brand_href="/",
        sticky="top",
        color="black",
        dark=True,
        expand="lg",
    )
    return navbar
