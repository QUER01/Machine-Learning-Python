import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import pandas as pd


df_symbol = pd.read_csv('tickers.csv')


app = dash.Dash()

app.layout = html.Div([
    html.H1('Stock Tickers'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{'label': s[0], 'value': s[1]}
                 for s in zip(df_symbol.Company, df_symbol.Symbol)],
        value='COKE'
    ),
    dcc.Graph(id='my-graph')
])

@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(
        selected_dropdown_value, data_source='google',
        start=dt(2017, 1, 1), end=dt.now())
    return {
        'data': [{
            'x': df.index,
            'y': df.Close
        }]
    }

if __name__ == '__main__':
    app.run_server()