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
from myutils import generic, betting, timing, guiserver, importer, customlogging


from betfairlightweight.resources.bettingresources import RunnerBook, PriceSize, MarketBook
from typing import List, Dict

mylogger = customlogging.create_dual_logger('gui', 'log/dashlog.log', file_reset=True)

class RunnerCard(generic.StaticClass):

    BOOK_ABRVS = {
        'available_to_back': 'atb',
        'available_to_lay': 'atl',
        'traded_volume': 'tv'
    }

    CELL_WIDTH = '70px'

    @staticmethod
    def t(name, index):
        return f'{name}-{index}'

    @classmethod
    def empty_table(cls, index):
        return [
            {
                cls.t('atb', index): None,
                cls.t('odds', index): tick,
                cls.t('atl', index): None,
                cls.t('tv', index): None
            } for tick in betting.TICKS_DECODED
        ]

    @staticmethod
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

    @staticmethod
    def title(name):
        return html.Div(className="runner-component runner-title", children=[
            html.Div(name, className='title-container')
        ])

    @staticmethod
    def price_chart():
        return html.Div(className="runner-component runner-price-chart", children=[

        ])

    @classmethod
    def table(cls, index, tbl):
        return html.Div(className="runner-component runner-table-container", children=[
            dash_table.DataTable(
                id=f'runner-table-{index}',
                columns=[
                    {
                        'name': col,
                        'id': RunnerCard.t(col, index)
                    } for col in ['atb', 'odds', 'atl', 'tv']
                ],
                data=tbl,
                fixed_rows={
                    'headers': True
                },
                page_size=betting.TICKS.shape[0],
                style_cell={
                    # all three widths are needed
                    'minWidth': cls.CELL_WIDTH,
                    'width': cls.CELL_WIDTH,
                    'maxWidth': cls.CELL_WIDTH,
                    'textAlign': 'center'
                }
            )
        ])

    @classmethod
    def runner_card(cls, index, name, runner_book: RunnerBook):
        tbl = cls.empty_table(index)

        if runner_book:
            for bk_name, bk_abrv in cls.BOOK_ABRVS.items():
                cls.update_table(
                    tbl=tbl,
                    price_list=getattr(runner_book.ex, bk_name),
                    key=cls.t(bk_abrv, index))

        return html.Div(className='runner-card', children=[
            cls.title(name),
            cls.price_chart(),
            cls.table(index, tbl)
        ])


class DashGUI(generic.StaticClass):

    RECORD_INFO = {
        'Record Time': lambda r: r.publish_time.isoformat(sep=' ', timespec='milliseconds'),
        'Market Time': lambda r: r.market_definition.market_time.isoformat(sep=' ', timespec='milliseconds'),
        'Event Name': lambda r: r.market_definition.event_name,
        'Name': lambda r: r.market_definition.name,
        'Betting Type': lambda r: r.market_definition.betting_type
    }

    N_CARDS = 3

    app = None
    historical_list = None
    timer = timing.TimeSimulator()
    running = False
    runner_names = None
    ltps = None
    fig = None
    active_index = 0
    index_count = None

    @staticmethod
    def header():
        return html.H1(
            'Betfair GUI',
            className='header-container'
        )

    @classmethod
    def info_columns(cls, market_book: MarketBook):
        data = [
            {
                'Info': 'Simulated Time',
                'Value': cls.timer.current().isoformat(sep=' ', timespec='milliseconds')
            }
        ]
        data += [
            {
                'Info': k,
                'Value': f(market_book)
            }
            for k, f in cls.RECORD_INFO.items()
        ]
        return data

    @classmethod
    def info(cls, info_cols: List[Dict]):
        return html.Div(className="info-container", children=[
            dash_table.DataTable(
                id='info-table',
                columns=[{"name": i, "id": i} for i in ['Info', 'Value']],
                data=info_cols,
            ),
        ])

    @staticmethod
    def nav():
        return html.Div(className='nav-container', children=[
            html.Div(className="buttons-container", children=[
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Pause', id='pause-button', n_clicks=0),
                html.Div(id='running-indicator')
            ]),
            dcc.Slider(
                id='my-slider',
                min=0,
                max=1000,
                step=1,
                value=0,
            ),
            html.Div(id='slider-output-container'),
        ])

    @staticmethod
    def runner_chart(fig):
        return html.Div(className='runner-chart-container', children=[
            dcc.Graph(
                id='runner-graph',
                figure=fig,
                config=dict(displayModeBar=False, staticPlot=True)
            )
        ])

    @classmethod
    def runner_cards(cls, runner_names, runners):
        selected_ids = list(runner_names.keys())[:cls.N_CARDS]
        runner_cards = []
        for i in range(cls.N_CARDS):
            name = None
            book = None
            if len(selected_ids) >= i:
                id = selected_ids[i]
                name = runner_names[id]
                book = betting.get_book(runners, id)
            runner_cards.append(RunnerCard.runner_card(i, name, book))
        return html.Div(
            className='card-container',
            children=runner_cards)

    @classmethod
    def at_end(cls):
        return not (cls.active_index < cls.index_count)

    @classmethod
    def start_running(cls):
        if not cls.at_end() and not cls.running:
            cls.timer.start()
            cls.running = True

    @classmethod
    def stop_running(cls):
        if cls.running:
            cls.timer.stop()
        cls.running = False



    @classmethod
    def set_callbacks(cls):

        @cls.app.callback(
            [dash.dependencies.Output('slider-output-container', 'children')],
            [dash.dependencies.Input('my-slider', 'value')])
        def update_output(value):
            return [f'You have selected "{value}"']

        @cls.app.callback(
            [dash.dependencies.Output('running-indicator', 'children')],
            [
                dash.dependencies.Input('play-button', 'n_clicks'),
                dash.dependencies.Input('pause-button', 'n_clicks')
            ])
        def play_button(a, b):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            div_content = 'STOPPED'
            if 'play-button' in changed_id:
                cls.start_running()
                mylogger.info('Start button pressed')
                div_content = 'RUNNING'
            elif 'pause-button' in changed_id:
                cls.stop_running()
                mylogger.info('Stop button paused')
            else:
                mylogger.warn(f'Button pressed not recognised: {changed_id}')
            return [div_content]

        @cls.app.callback(
            [
                dash.dependencies.Output('info-table', 'data')
            ],
            [dash.dependencies.Input('interval-component', 'n_intervals')])
        def update_content(n_intervals):
            return [
                cls.info_columns(cls.historical_list[0][0])
            ]

    @classmethod
    def create(cls, name, record_list):

        assert len(historical_list)
        cls.historical_list = record_list
        cls.index_count = len(record_list)

        cls.timer.reset_start(cls.historical_list[0][0].publish_time)
        cls.app = dash.Dash(name)
        cls.runner_names = betting.get_names(historical_list[0][0].market_definition)
        cls.ltps = betting.get_ltps(cls.historical_list, cls.runner_names)
        cls.fig = px.line(cls.ltps, width=500, height=400)
        cls.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0))

        cls.app.layout = html.Div(className='content-container', children=[
            DashGUI.header(),
            DashGUI.info(cls.info_columns(historical_list[0][0])),
            DashGUI.nav(),
            DashGUI.runner_chart(cls.fig),
            DashGUI.runner_cards(cls.runner_names, historical_list[5][0].runners),
            dcc.Interval(
                id='interval-component',
                interval=1*1000, # in milliseconds
                n_intervals=0
            )
        ])

        cls.set_callbacks()


if __name__ == '__main__':
    trading = betting.get_api_client()
    trading.login()
    historical_queue = betting.get_historical(trading, r'data/bfsample10')
    historical_list = list(historical_queue.queue)
    DashGUI.create(__name__, historical_list)

    DashGUI.app.run_server(debug=True)