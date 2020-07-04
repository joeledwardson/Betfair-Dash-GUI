import plotly
import plotly.graph_objs as go
import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from myutils import generic, betting, timing, guiserver, importer
from myutils import customlogging
import importlib
from betfairlightweight.resources.streamingresources import MarketDefinition
from betfairlightweight.resources.bettingresources import RunnerBook, PriceSize
from typing import List


importlib.reload(customlogging)
mylogger = customlogging.create_dual_logger(__name__, 'log/dashlog.log', file_reset=True)
importer.reload_utils()

trading = betting.get_api_client()
trading.login()

historical_queue = betting.get_historical(trading, r'data/bfsample10')
historical_list = list(historical_queue.queue)
print(f'Data entries: {len(historical_list)}')

app = dash.Dash(__name__)

fig = px.scatter(pd.DataFrame({'x': [1,2,3,4], 'y':[10, 40, 35, 70]}))

d0 = historical_list[2][0]

record_info = {
    'Record Time':  lambda d0: d0.publish_time.isoformat(sep=' ', timespec='milliseconds'),
    'Market Time':  lambda d0:  d0.market_definition.market_time.isoformat(sep=' ', timespec='milliseconds'),
    'Event Name':   lambda d0: d0.market_definition.event_name,
    'Name':         lambda d0: d0.market_definition.name,
    'Betting Type': lambda d0: d0.market_definition.betting_type
}


def t(name, index):
    return f'{name}-{index}'


def empty_table(index):
    return [
        {
            t('atb', index):  None,
            t('odds', index): tick,
            t('atl', index):  None,
            t('tv', index):   None
        } for tick in betting.TICKS_DECODED
    ]


def update_table(tbl, price_list: List[PriceSize], key):

    for p in price_list:

        # convert "price" (odds value) to encoded integer value
        t = betting.float_encode(p.price)

        # check that price appears in ticks array
        if t in betting.LTICKS:

            # and get index where it appears
            i = betting.LTICKS.index(t)

            # update value in table
            tbl[i][key] = p.size


BOOK_ABRVS = {
    'available_to_back': 'atb',
    'available_to_lay': 'atl',
    'traded_volume': 'tv'
}


def runner_card(index, name, runner_book: RunnerBook):
    tbl = empty_table(index)

    if runner_book:
        for bk, bk_abrv in BOOK_ABRVS.items():
            update_table(tbl, getattr(runner_book.ex, bk), t(bk_abrv, index))

    return html.Div(className='runner-card', children=[
        html.Div(className="runner-component runner-title", children=[
            html.Div(name, className='title-container')
        ]),
        html.Div(className="runner-component runner-price-chart", children=[

        ]),
        html.Div(className="runner-component runner-table-container", children=[
            dash_table.DataTable(
                id=f'runner-table-{index}',
                columns=[
                    {
                        'name': bk_abrv,
                        'id': f'{bk_abrv}-{index}'
                    } for bk_abrv in ['atb', 'odds', 'atl', 'tv']
                ],
                data=tbl,
                fixed_rows={'headers': True},
                page_size=betting.TICKS.shape[0]
            )
        ])
    ])


def get_book(runners, id):
    for runner in runners:
        if id == runner.selection_id:
            return runner
    else:
        return None

runner_names = {
    runner.selection_id: runner.name
    for runner in d0.market_definition.runners
}
id_list = [runner.selection_id for runner in d0.market_definition.runners]



ltps = np.zeros([len(historical_list), len(runner_names.keys())])
timestamps = []
for i, e in enumerate(historical_list):
    timestamps.append(e[0].publish_time)
    for r in e[0].runners:
        ltps[i, id_list.index(r.selection_id)] = r.last_price_traded

df = pd.DataFrame(ltps, index=timestamps, columns=list(runner_names.values()))
fig = px.line(df)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0))

display_count = 3
selected_ids = list(runner_names.keys())[:display_count]
runner_cards = []
for i in range(display_count):
    name = None
    book = None
    if len(selected_ids) >= i:
        id = selected_ids[i]
        name = runner_names[id]
        book = get_book(d0.runners, id)
    runner_cards.append(runner_card(i, name, book))

app.layout = html.Div(className='content-container', children=[
    html.H1('Betfair GUI', className='header-container'),
    html.Div(className="info-container", children=[
        dash_table.DataTable(
            id='info-table',
            columns=[{"name": i, "id": i} for i in ['Info', 'Value']],
            data= [{'Info': k, 'Value': f(historical_list[0][0])} for k, f in record_info.items()],
        ),
    ]),
    html.Div(className='nav-container', children=[
        dcc.Slider(
                id='my-slider',
                min=0,
                max=1000,
                step=1,
                value=0,
        ),
        html.Div(id='slider-output-container'),
    ]),
    html.Div(className='runner-chart-container', children=[
        dcc.Graph(
            id='runner-graph',
            figure=fig,
            config=dict(displayModeBar=False, staticPlot=True)
        )
    ]),
    html.Div(className='card-container', children=runner_cards)
])

@app.callback(
    [
        dash.dependencies.Output('slider-output-container', 'children'),
    ],
    [
        dash.dependencies.Input('my-slider', 'value'),
    ]
)
def update_output(value):
    return [f'You have selected "{value}"']


if __name__ == '__main__':
    app.run_server(debug=True)