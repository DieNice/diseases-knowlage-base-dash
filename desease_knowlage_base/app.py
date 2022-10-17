from dash_extensions.enrich import DashProxy
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, MultiplexerTransform

app = DashProxy(__name__, assets_folder='assets',
                external_stylesheets=[dbc.themes.MATERIA], transforms=[MultiplexerTransform()])

app.title = "Desease knowlage base"

srv = app.server


app.config.suppress_callback_exceptions = True
