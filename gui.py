import dash_core_components as dcc
import dash
import dash_html_components as html
import dash_table
import plotly.express as px
from plotly.graph_objs import Figure
import pandas as pd
from datetime import timedelta
from myutils import generic, betting, timing, guiserver, importer, customlogging
import itertools

from betfairlightweight.resources.bettingresources import RunnerBook, PriceSize, MarketBook
from typing import List, Dict

myLogger = customlogging.create_dual_logger('gui', 'log/dashlog.log', file_reset=True)


class GUIInterface:

    def __init__(self, record_list, n_cards: int):

        assert len(record_list)
        self.historical_list = record_list
        self.index_count = len(record_list)
        self.active_index = 0
        self.active_record: MarketBook = record_list[self.active_index][0]

        self.timer = timing.TimeSimulator()
        self.timer.reset_start(self.active_record.publish_time)
        self.running = False

        self.runner_names = betting.get_names(self.active_record.market_definition)
        self.n_cards = n_cards

        self.ltps: pd.DataFrame = betting.get_ltps(record_list, self.runner_names)
        self.chart_start_index = 0
        self.chart_end_index = 0

        self.slider_val = 0

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
            myLogger.error(f'index "{self.active_index}" is out of bounds, len is "{self.index_count}"')

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
    N_STEPS = 1000

    def slider(self, html_id, start_str, end_str):
        return dcc.Slider(
            id=html_id,
            min=0,
            max=self.N_STEPS,
            step=1,
            value=0,
            marks={
                0: start_str,
                self.N_STEPS: end_str
            }
        )

    def create(self, g: GUIInterface):
        return html.Div(className='nav-container', children=[
            html.Div(className="buttons-container", children=[
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Pause', id='pause-button', n_clicks=0),
                html.Span(id='running-indicator')
            ]),
            self.slider('my-slider', 'start', 'end'),
            self.slider('my-recent-slider', '-30mins', 'end'),
            html.Div(id='slider-output-container'),
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        def get_changed_id():
            # get ID of button which was last pressed
            return [p['prop_id'] for p in dash.callback_context.triggered][0]

        @app.callback(
            dash.dependencies.Output('slider-output-container', 'children'),
            [dash.dependencies.Input('my-slider', 'value'),
             dash.dependencies.Input('my-recent-slider', 'value')])
        def update_output(slider_val, recent_slider_val):

            # get ID of button which was last pressed
            changed_id = get_changed_id()

            # get datetime start and end
            t_start = g.historical_list[0][0].publish_time
            t_end = g.historical_list[-1][0].publish_time

            value = None

            if 'recent' in changed_id:
                value = recent_slider_val
                myLogger.info(f'recent slider triggered: {value}')

                # use 30 minutes from end for recent slider
                t_start = t_end - timedelta(minutes=30)

            elif 'slider' in changed_id:
                value = slider_val
                myLogger.info(f'full slider triggered: {value}')

            else:
                myLogger.info(f'slider id "{changed_id}" not recognised')
                return ['']

            # reset chart indexing values
            g.chart_start_index = 0
            g.chart_end_index = 0

            # reset record indexing value
            g.active_index = 0

            # calculate current time based on percentage of steps taken
            t_current = t_start + ((t_end - t_start) * (value / self.N_STEPS))

            # get datetime string
            t_str = t_current.isoformat(sep=' ', timespec='milliseconds')

            # set simluation timer current datetime value
            g.timer.reset_start(t_current)

            return [f'You have selected {t_str}']

        @app.callback(dash.dependencies.Output('running-indicator', 'children'),
                      [dash.dependencies.Input('play-button', 'n_clicks'),
                       dash.dependencies.Input('pause-button', 'n_clicks')])
        def play_button(a, b):

            # get ID of button which was last pressed
            changed_id = get_changed_id()

            # initialise blank identifying element
            element = ''

            if 'play-button' in changed_id:

                # play button pressed - start running
                g.start_running()
                myLogger.info('Start button pressed')

                # set html element to indicate race is now running
                element = 'RUNNING'

            elif 'pause-button' in changed_id:

                # pause button pressed
                g.stop_running()
                myLogger.info('Stop button paused')

                # set html element to indicate race has stopped running
                element  = 'STOPPED'

            else:

                # id of button pressed not recognised
                myLogger.warning(f'Button pressed not recognised: {changed_id}')

            return element

        return []


class ChartComponent(GuiComponent):
    def __init__(self, chart_span_s):
        self.chart: Figure = None
        self.span_s = chart_span_s

    def get_fig(self, g: GUIInterface):
        start = g.timer.current()
        end = start + timedelta(seconds=self.span_s)

        while g.chart_start_index + 1 < g.index_count and \
                g.historical_list[g.chart_start_index + 1][0].publish_time < start:
            g.chart_start_index += 1

        while g.chart_end_index + 1 < g.index_count and \
                g.historical_list[g.chart_end_index][0].publish_time < end:
            g.chart_end_index += 1

        df = g.ltps[g.chart_start_index: g.chart_end_index + 1]
        self.fig = px.line(df, width=500, height=400)
        self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0),
                               yaxis_type="log",
                               xaxis=dict(range=[start, end]))
        return self.fig

    def create(self, g: GUIInterface):

        return html.Div(className='runner-chart-container', children=[
            dcc.Graph(
                id='runner-graph',
                figure=self.get_fig(g),
                config=dict(displayModeBar=False, staticPlot=True)
            )
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash) -> List[Dict]:
        def update():
            return self.get_fig(g)

        return [{
            'output': dash.dependencies.Output('runner-graph', 'figure'),
            'function': update
        }]


class RunnerCard(GuiComponent):

    BOOK_ABRVS = {
        'available_to_back': 'atb',
        'available_to_lay': 'atl',
        'traded_volume': 'tv'
    }

    def __init__(self, runner_id, index, name):
        self.runner_id = runner_id
        self.index = index
        self.table_id = f'runner-table-{index}'
        self.name = name

    def create(self, g: GUIInterface):
        tbl_data = self.empty_table(self.index)

        return html.Div(className='runner-card', children=[
            self.title(self.index, self.name, self.runner_id, g),
            self.price_chart(),
            self.table(tbl_data)
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        @app.callback(dash.dependencies.Output(self.t('selected-indicator', self.index), 'children'),
                      [dash.dependencies.Input(self.t('dropdown', self.index), 'value')])
        def update_selected(new_runner_id):

            myLogger.info(f'Index {self.index} ladder just selected id {new_runner_id}')
            selected = self.runner_id

            if new_runner_id not in g.runner_names.keys():
                myLogger.warn(f'Value selected by index {self.index} is not found in runner indexes')
                return selected

            self.runner_id = new_runner_id
            self.name = g.runner_names[new_runner_id]
            return [
                html.Div(self.name),
                html.Div(self.runner_id)
            ]


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
                tbl[i][key] = f'{p.size:.2f}'

    @classmethod
    def dropdown(cls, index, runner_id, g: GUIInterface):

        return dcc.Dropdown(
            id=cls.t('dropdown', index),
            options=[
                {'label': name, 'value': _id}
                for _id, name in g.runner_names.items()
            ],
            value = runner_id
        )

    @classmethod
    def title(cls, index, runner_name, runner_id, g: GUIInterface):
        return html.Div(className="runner-component runner-title", children=[
            cls.dropdown(index, runner_id, g),
            html.Div(id=cls.t('selected-indicator', index), children=[
                html.Div(runner_name),
                html.Div(runner_id)
            ])
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
                    'textAlign': 'center'
                }
            )
        ])


class CardComponents(GuiComponent):

    def __init__(self):
        self.runner_cards: List[RunnerCard] = None

    def create(self, g: GUIInterface):

        # sorting function - based on last traded price (if exists)
        def sorter(runner):
            return runner.last_price_traded or float('inf')

        # get pre race records
        pre_race = [h for h in g.historical_list if not h[0].inplay]

        # get last record before race starts
        last_record = pre_race[-1][0]

        # get sorted list of runners based on ltp
        sorted_runners = sorted(last_record.runners, key=sorter)[:g.n_cards]

        # get sorted list of runner ids
        selected_ids = [r.selection_id for r in sorted_runners]

        # create runner card instances
        self.runner_cards = [RunnerCard(runner_id, index, g.runner_names[runner_id])
                             for index, runner_id in enumerate(selected_ids)]

        # create html elements with runner card instances
        card_elements = [r.create(g) for r in self.runner_cards]

        # return card instances in the container
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
    CHART_SPAN_S = 60

    app = None
    g: GUIInterface = None

    # nav component must come before chart so chart positions are updated first on slider move
    componentList: List[GuiComponent] = [TitleComponent(),
                                         InfoComponent(),
                                         NavComponent(),
                                         ChartComponent(CHART_SPAN_S),
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
        cls.g = GUIInterface(record_list, cls.N_CARDS)

        children = [
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ]
        children += [
            c.create(cls.g) for c in cls.componentList
        ]

        cls.app.layout = html.Div(className='content-container', children=children)
        cls.set_callbacks()


def run(name, record_list, debug):
    DashGUI.create(name, record_list)
    DashGUI.app.run_server(debug=debug)


if __name__ == '__main__':
    trading = betting.get_api_client()
    trading.login()
    historical_queue = betting.get_historical(trading, r'data/bfsample10')
    historical_list = list(historical_queue.queue)
    run(__name__, historical_list, False)
