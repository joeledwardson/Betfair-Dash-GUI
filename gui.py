import plotly
import plotly.graph_objs as go
import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_table
import plotly.express as px
from plotly.graph_objs import Figure
import pandas as pd
import numpy as np
from myutils import generic, betting, timing, guiserver, importer, customlogging
import itertools

from betfairlightweight.resources.bettingresources import RunnerBook, PriceSize, MarketBook
from typing import List, Dict

mylogger = customlogging.create_dual_logger('gui', 'log/dashlog.log', file_reset=True)


class GUIInterface:

    historical_list: List = None
    active_record: MarketBook = None
    active_index: int = 0
    index_count: int = None

    timer: timing.TimeSimulator = None
    running = False

    runner_names: Dict[int, str] = None
    ltps: pd.DataFrame = None

    selected_ids: List = None

    def initialise(self, record_list, n_cards):
        assert len(record_list)
        self.historical_list = record_list
        self.index_count = len(record_list)
        self.active_index = 0
        self.active_record = record_list[self.active_index][0]

        self.timer = timing.TimeSimulator()
        self.timer.reset_start(self.active_record.publish_time)
        self.running = False

        self.runner_names = betting.get_names(self.active_record.market_definition)
        self.selected_ids = list(self.runner_names.keys())[:n_cards]

        self.ltps = betting.get_ltps(record_list, self.runner_names)

    def update_current_record(self):

        while 1:

            # check next record has not reach end
            if not (self.active_index + 1 < self.index_count):
                break

            # get next record timestamp
            next_time = self.historical_list[self.active_index + 1][0].publish_time

            # if next record is beyond current time, break, continue to use current record
            if next_time >= self.timer.current():
                break

            # otherwise, not at end, next record not beyond current time, so increment
            self.active_index += 1

        if self.active_index < 0 or self.active_index >= self.index_count:

            # sanity check index for out of bounds
            mylogger.error(f'index "{self.active_index}" is out of bounds, len is "{self.index_count}"')

        else:

            # update current record
            self.active_record = self.historical_list[self.active_index][0]

    def at_end(self):
        return not (self.active_index < self.index_count)

    def start_running(self):
        if not self.at_end() and not self.running:
            self.timer.start()
            self.running = True

    def stop_running(self):
        if self.running:
            self.timer.stop()
        self.running = False


class GuiComponent:
    def create(self, g: GUIInterface):
        raise Exception(f'Cannot use base "{self.__class__}" class')

    def callbacks(self, g: GUIInterface, app: dash.Dash) -> List[Dict]:
        return []


class InfoComponent(GuiComponent):

    RECORD_INFO = {
        'Record Time': lambda r: r.publish_time.isoformat(sep=' ', timespec='milliseconds'),
        'Market Time': lambda r: r.market_definition.market_time.isoformat(sep=' ', timespec='milliseconds'),
        'Event Name': lambda r: r.market_definition.event_name,
        'Name': lambda r: r.market_definition.name,
        'Betting Type': lambda r: r.market_definition.betting_type
    }

    def create(self, g: GUIInterface):
        return html.Div(className="info-container", children=[
            dash_table.DataTable(
                id='info-table',
                columns=[{"name": i, "id": i} for i in ['Info', 'Value']],
                data=self.info_columns(g),
            ),
        ])

    def info_columns(self, g: GUIInterface):
        data = [
            {
                'Info': 'Simulated Time',
                'Value': g.timer.current().isoformat(sep=' ', timespec='milliseconds')
            },
            {
                'Info': 'Record index',
                'Value': g.active_index
            }
        ]
        data += [
            {
                'Info': k,
                'Value': f(g.active_record)
            }
            for k, f in self.RECORD_INFO.items()
        ]
        return data

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        def update():
            g.update_current_record()
            return self.info_columns(g)

        return [{
            'output': dash.dependencies.Output('info-table', 'data'),
            'function': update
        }]


class NavComponent(GuiComponent):
    def create(self, g: GUIInterface):
        return html.Div(className='nav-container', children=[
            html.Div(className="buttons-container", children=[
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Pause', id='pause-button', n_clicks=0),
                html.Span(id='running-indicator')
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

    def callbacks(self, g: GUIInterface, app: dash.Dash):
        @app.callback(
            [dash.dependencies.Output('slider-output-container', 'children')],
            [dash.dependencies.Input('my-slider', 'value')])
        def update_output(value):
            return [f'You have selected "{value}"']

        @app.callback(dash.dependencies.Output('running-indicator', 'children'),
                      [dash.dependencies.Input('play-button', 'n_clicks'),
                       dash.dependencies.Input('pause-button', 'n_clicks')])
        def play_button(a, b):

            # get ID of button which was last pressed
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

            # initialise blank identifying element
            element = ''

            if 'play-button' in changed_id:

                # play button pressed - start running
                g.start_running()
                mylogger.info('Start button pressed')

                # set html element to indicate race is now running
                element = 'RUNNING'

            elif 'pause-button' in changed_id:

                # pause button pressed
                g.stop_running()
                mylogger.info('Stop button paused')

                # set html element to indicate race has stopped running
                element  = 'STOPPED'

            else:

                # id of button pressed not recognised
                mylogger.warn(f'Button pressed not recognised: {changed_id}')

            return element

        return []


class ChartComponent(GuiComponent):
    def __init__(self):
        self.chart: Figure = None

    def create(self, g: GUIInterface):
        self.fig = px.line(g.ltps, width=500, height=400)
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0))

        return html.Div(className='runner-chart-container', children=[
            dcc.Graph(
                id='runner-graph',
                figure=self.fig,
                config=dict(displayModeBar=False, staticPlot=True)
            )
        ])


class RunnerCard(GuiComponent):

    BOOK_ABRVS = {
        'available_to_back': 'atb',
        'available_to_lay': 'atl',
        'traded_volume': 'tv'
    }
    CELL_WIDTH = '70px'

    def __init__(self, runner_id, index, name):
        self.runner_id = runner_id
        self.index = index
        self.table_id = f'runner-table-{index}'
        self.name = name

    def create(self, g: GUIInterface):
        tbl_data = self.empty_table(self.index)

        return html.Div(className='runner-card', children=[
            self.title(self.name, self.runner_id),
            self.price_chart(),
            self.table(tbl_data)
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        def update():

            # get empty ladder table
            tbl_data = self.empty_table(self.index)

            # get index of runner's id in record's runner list
            list_index = generic.get_index(g.active_record.runners, lambda r: r.selection_id == self.runner_id)

            # check runner found
            if list_index:

                # get runner book object
                runner_book = g.active_record.runners[list_index]

                # loop market book attribute names
                for bk_name, bk_abrv in self.BOOK_ABRVS.items():

                    # get market book data and assign to ladder
                    self.update_data(
                        tbl=tbl_data,
                        price_list=getattr(runner_book.ex, bk_name),
                        key=self.t(bk_abrv, self.index))

            return tbl_data

        return [{
            'output': dash.dependencies.Output(self.table_id, 'data'),
            'function': update
        }]

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
    def update_data(tbl, price_list: List[PriceSize], key):

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
    def title(runner_name, runner_id):
        return html.Div(className="runner-component runner-title", children=[
            html.Div(runner_name, className='title-container'),
            html.Div(runner_id)
        ])

    @staticmethod
    def price_chart():
        return html.Div(className="runner-component runner-price-chart", children=[

        ])

    def table(self, table_data):
        return html.Div(className="runner-component runner-table-container", children=[
            dash_table.DataTable(
                id=self.table_id,
                columns=[
                    {
                        'name': col,
                        'id': self.t(col, self.index)
                    } for col in ['atb', 'odds', 'atl', 'tv']
                ],
                data=table_data,
                fixed_rows={
                    'headers': True
                },
                style_cell={
                    # all three widths are needed
                    'minWidth': self.CELL_WIDTH,
                    'width': self.CELL_WIDTH,
                    'maxWidth': self.CELL_WIDTH,
                    'textAlign': 'center'
                }
            )
        ])


class CardComponents(GuiComponent):

    def __init__(self):
        self.runner_cards = List[RunnerCard]

    def create(self, g: GUIInterface):
        self.runner_cards = [RunnerCard(runner_id, index, g.runner_names[runner_id])
                             for index, runner_id in enumerate(g.selected_ids)]

        card_elements = [r.create(g) for r in self.runner_cards]

        return html.Div(
            className='card-container',
            children=card_elements)

    def callbacks(self, g: GUIInterface, app: dash.Dash):
        return list(itertools.chain(*[r.callbacks(g, app) for r in self.runner_cards]))


class TitleComponent(GuiComponent):
    def create(self, g: GUIInterface):
        return html.H1(
            'Betfair GUI',
            className='header-container'
        )


class DashGUI(generic.StaticClass):

    N_CARDS = 3

    app = None
    g: GUIInterface = GUIInterface()

    componentList: List[GuiComponent] = [TitleComponent(),
                                         InfoComponent(),
                                         NavComponent(),
                                         ChartComponent(),
                                         CardComponents()]

    @classmethod
    def set_callbacks(cls):

        interval_callbacks = list(itertools.chain(*[
            c.callbacks(cls.g, cls.app) for c in cls.componentList
        ]))

        @cls.app.callback([c['output'] for c in interval_callbacks],
                          [dash.dependencies.Input('interval-component', 'n_intervals')])
        def interval_update(n_intervals):
            output_list = []
            for c in interval_callbacks:
                output_list.append(c['function']())
            return output_list

    @classmethod
    def create(cls, name, record_list):

        cls.app = dash.Dash(name)
        cls.g.initialise(record_list, cls.N_CARDS)

        children = [
            dcc.Interval(
                id='interval-component',
                interval=1 * 3000,  # in milliseconds
                n_intervals=0
            )
        ]
        children += [
            c.create(cls.g) for c in cls.componentList
        ]

        cls.app.layout = html.Div(className='content-container', children=children)
        cls.set_callbacks()


if __name__ == '__main__':
    trading = betting.get_api_client()
    trading.login()
    historical_queue = betting.get_historical(trading, r'data/bfsample10')
    historical_list = list(historical_queue.queue)

    DashGUI.create(__name__, historical_list)

    DashGUI.app.run_server(debug=False)