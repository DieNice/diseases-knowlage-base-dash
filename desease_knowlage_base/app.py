from dash_extensions.enrich import DashProxy


app = DashProxy(__name__, assets_folder='/desease_knowlage_base/assets')

app.title = "Desease knowlage base"

srv = app.server


app.config.suppress_callback_exceptions = True
