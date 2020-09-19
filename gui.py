import dash_core_components as dcc
import dash
import dash_html_components as html
import dash_table
import plotly.express as px
from plotly.graph_objs import Figure
import pandas as pd
from datetime import timedelta
from myutils import generic, betting, timing, customlogging, bf_strategy
import itertools
from betfairlightweight.resources.bettingresources import PriceSize, MarketBook, MarketCatalogue, RunnerBook
from typing import List, Dict
import os
import argparse
import re

# TODO - invert ladders to higher prices appear above
# TODO - make recent slider number of minutes into an input arg

# create log folder if doesnt exist
if not os.path.isdir('log'):
    os.mkdir('log')

# custom logging instance which prints to console and logs debug to file
myLogger = customlogging.create_dual_logger('gui', 'log/dashlog.log', file_reset=True)


# GUI interface class holds data which is shared between different GUI Components
# - for example, simulated race time needs to be accessible to the chart component, not just the information table
# - which displays the simulated time
class GUIInterface:

    # record_list should be list of [MarketBook], (that is, each element is a list with 1 element of type MarketBook)
    # - record_list is based off api_client.streaming.create_historical_stream() where api_client is a
    # - betfairlightweight.APIClient instance
    # n_cards is number of runner cards to display in GUI
    def __init__(self, record_list: List[List[MarketBook]], n_cards: int, catalogue: MarketCatalogue=None):

        # check the record list is not empty
        assert len(record_list)

        # assign historical list to local copy
        self.historical_list = record_list

        # quickly accessible number of records, shorthand of len(self.historical_list)
        self.index_count = len(record_list)

        # index, within record list of current (active) record
        self.active_index = 0
        # active record, shorthand of self.historical_list[self.active_index][0]
        # should be updated every time self.active_index is updated
        self.active_record: MarketBook = record_list[self.active_index][0]

        # timing simulator instance
        self.timer = timing.TimeSimulator()
        # set simulator start time to timestamp from active (first) record
        self.timer.reset_start(self.active_record.publish_time)

        # race simulator is running indicator
        self.running = False

        if catalogue:

            # if catalogue specified, use it to get runner names etc
            self.runner_names = betting.get_names(catalogue, name_attr='runner_name')
            self.event_name = catalogue.event.name
            self.market_name = catalogue.market_name

        else:

            # otherwise, assume historical file and ust first record
            self.runner_names = betting.get_names(self.active_record.market_definition)
            self.event_name = self.active_record.market_definition.event_name
            self.market_name = self.active_record.market_definition.name

        self.market_time = self.active_record.market_definition.market_time

        # number of runner cards to display in GUI
        self.n_cards = n_cards

        # get data frame of last traded prices
        self.ltps: pd.DataFrame = betting.runner_table(record_list)
        self.ltps.columns = [self.runner_names[selection_id] for selection_id in self.ltps.columns]

        # index of record at the start of the last traded prices graph
        self.chart_start_index = 0
        # index of record at the end of the last traded prices graph
        self.chart_end_index = 0

        # slider relative position
        self.slider_val = 0

    # update self.active_index and self.active_record based on which record in list has a timestamp closest to (but not
    # more than) current simulation time
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

    # check if active index is last in list
    def at_end(self):
        return not (self.active_index < self.index_count)

    # start race running GUI simulation
    def start_running(self):
        if not self.at_end() and not self.running:
            self.timer.start()
            self.running = True

    # stop race running simulation
    def stop_running(self):
        if self.running:
            self.timer.stop()
        self.running = False


# abstract class for GUI components
class GuiComponent:

    # create() method must be derived
    # takes a GUIInterface instance to interact with GUI properties and returns a html dash Component
    def create(self, g: GUIInterface) -> dash.development.base_component.Component:
        raise Exception(f'Cannot use base "{self.__class__}" class')

    # callbacks() is optional to derive
    # takes a GUIInterface instance to interact with GUI properties and returns a html dash Component
    # takes a Dash instance which can be used to declare callbacks with syntax @app.callback(...)[...]
    # must return a list of dicts, where each dict will be added to callbacks triggered on periodic updating, dict
    # values are:
    # - 'output': dash.dependencies.Output() instance
    # - 'function': callback function with 0 arguments
    def callbacks(self, g: GUIInterface, app: dash.Dash) -> List[Dict]:
        return []


# GUI Component - Race information table
class InfoComponent(GuiComponent):

    # dict of MarketBook properties, whereby key is property display name and value is function to get value from
    # MarketBook
    RECORD_INFO = {
        'Record Time': lambda r, intf:      r.publish_time.isoformat(sep=' ', timespec='milliseconds'),
        'Market Time': lambda r, intf:      intf.market_time,
        'Event Name': lambda r, intf:       intf.event_name,
        'Market Name': lambda r, intf:      intf.market_name,
        'Betting Type': lambda r, intf:     r.market_definition.betting_type,
        'Total Matched': lambda r, intf:    '£{:,.0f}'.format(sum(runner.total_matched or 0 for runner in r.runners)),
        'In Play': lambda r, intf:          r.market_definition.in_play
    }

    def create(self, g: GUIInterface):

        # encapsulate table in div with class for css styling
        return html.Div(className="info-container", children=[

            # create datatable 'Info' and 'Value' columns for property display names and values respectively
            dash_table.DataTable(
                id='info-table',
                columns=[{"name": i, "id": i} for i in ['Info', 'Value']],
                data=self.info_columns(g),
            ),

        ])

    # get list of dict items with 'Info' attribute as property display name and 'Value' as property value
    def info_columns(self, g: GUIInterface):

        # get simulated time and active record index
        data = [
            {
                'Info': 'Simulated Time',
                'Value': g.timer.current().isoformat(sep=' ', timespec='milliseconds')
            },
            {
                'Info': 'Time to off',
                'Value': re.sub(r'\..*', '', str(g.market_time - g.timer.current())), # remove milliseconds ".123" etc
            },
            {
                'Info': 'Record index',
                'Value': g.active_index
            }
        ]

        # add active record properties to property list
        data += [
            {
                'Info': k,
                'Value': f(g.active_record, g)
            }
            for k, f in self.RECORD_INFO.items()
        ]

        return data

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        # period callback function
        def update():

            # update active index and record here
            g.update_current_record()

            # generate datatable data list and return
            return self.info_columns(g)

        # period update has output of 'data' attribute of datatable
        # function updates current record and returns updated info
        return [{
            'output': dash.dependencies.Output('info-table', 'data'),
            'function': update
        }]


# GUI Component - race navigation
class NavComponent(GuiComponent):

    # number of steps in sliders
    N_STEPS = 1000

    # create html slider - requires unique {html_id}, and strings to display at start {start_str} and end {end_str}
    # of slider
    def slider(self, html_id, start_str, end_str) -> dcc.Slider:
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

            # create play and pause buttons, and 'running-indicator' which tells if simulator is running or not
            html.Div(className="buttons-container", children=[
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Pause', id='pause-button', n_clicks=0),
                html.Span(id='running-indicator')
            ]),

            # slider for start to end of all historical records
            self.slider('my-slider', 'start', 'end'),

            # slider for 30 minutes before to last historical record
            self.slider('my-recent-slider', '-30mins', 'end'),

            # div which displays last timestamp selected by either of the sliders
            html.Div(id='slider-output-container'),
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        # get ID of button which was last pressed
        def get_changed_id():
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
            t_end = g.market_time

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
                element = 'STOPPED'

            else:

                # id of button pressed not recognised
                myLogger.warning(f'Button pressed not recognised: {changed_id}')

            return element

        # no periodic callback updating required
        return []


# GUI Component - runner chart
class ChartComponent(GuiComponent):

    # pass chart span (seconds) to constructor
    def __init__(self, chart_span_s):
        self.chart: Figure = None
        self.span_s = chart_span_s

    # generate figure of last traded prices of runners from current simulation time
    def get_fig(self, g: GUIInterface) -> Figure:

        # use current simulation time as end of chart
        end = g.timer.current()

        # subtract chart span (seconds) from end to get start time of chart
        start = end - timedelta(seconds=self.span_s)

        # # increment chart start index while (next record is not last record) and (next record time is less than start
        # # time). Once next record is beyond start time, loop will exit, and index will be the last one before
        # # exceeding the start time
        # while g.chart_start_index + 1 < g.index_count and \
        #         g.historical_list[g.chart_start_index + 1][0].publish_time < start:
        #     g.chart_start_index += 1
        #
        # # increment chart end index while (next record is not last record) and (next record time is less than end time)
        # # Once next record beyond end time, loop will exit, index will be last once before exceeding end time
        # while g.chart_end_index + 1 < g.index_count and \
        #         g.historical_list[g.chart_end_index][0].publish_time < end:
        #     g.chart_end_index += 1
        #
        # # end index is last record before exceeding end time. However, need record beyond end time so that entries fill
        # # up to the end of the chart. Thus need chart_end_index + 1, but need to add another 1 because slicing does not
        # # include the upper value (i.e. slice(0, 3) is from 0 to 2 not 0 to 3)
        # df = g.ltps.iloc[g.chart_start_index: g.chart_end_index + 2]

        df = g.ltps
        df = df[df.index >= start]
        df = df[df.index <= end]

        if df.shape[0]:

            # create figure based off dataframe
            self.fig = px.line(df, width=500, height=400)

            # set to 0 margins, logarithmic type, and xaxis timestamp start and end
            self.fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0),
                                   yaxis_type="log",
                                   xaxis=dict(range=[start, end]))

        else:

            self.fig = Figure()

        return self.fig

    def create(self, g: GUIInterface):
        return html.Div(className='runner-chart-container', children=[
            dcc.Graph(
                id='runner-graph',
                figure=self.get_fig(g),
                # don't show mode bar (as too quick to interact), and set to static to speed up rendering
                config=dict(displayModeBar=False, staticPlot=True)
            )
        ])

    def callbacks(self, g: GUIInterface, app: dash.Dash) -> List[Dict]:

        # periodic update function to return updated figure object
        def update():
            return self.get_fig(g)

        # period update is 'figure; attribute of chart object
        return [{
            'output': dash.dependencies.Output('runner-graph', 'figure'),
            'function': update
        }]


# GUI Component (not to be used directly) - single runner card
class RunnerCard(GuiComponent):

    # dict mapping of book attribute names within runner book to display names in runner card
    BOOK_ABRVS = {
        'available_to_back': 'atb',
        'available_to_lay': 'atl',
        'traded_volume': 'tv'
    }

    """
    runner_id is the ID of the runner selected for the card.
    index is the index of the card displayed on the GUI - this is used for creation of unique keys in html elements
    name of the name of the runner
    """
    def __init__(self, runner_id, index, name):
        self.runner_id = runner_id
        self.index = index
        self.table_id = f'runner-table-{index}'
        self.name = name

    def callbacks(self, g: GUIInterface, app: dash.Dash):

        # callback triggered on runner dropdown selection change
        @app.callback(dash.dependencies.Output(self.t('selected-indicator', self.index), 'children'),
                      [dash.dependencies.Input(self.t('dropdown', self.index), 'value')])
        def update_selected(new_runner_id):

            # log runner change
            myLogger.info(f'Index {self.index} ladder just selected id {new_runner_id}')

            # get existing selected ID
            selected = self.runner_id

            # check ID exists and quit if not with existing ID
            if new_runner_id not in g.runner_names.keys():
                myLogger.warn(f'Value selected by index {self.index} is not found in runner indexes')
                return selected

            # assign new ID
            self.runner_id = new_runner_id

            # assign new name
            self.name = g.runner_names[new_runner_id]

            return [
                html.Div(self.name),
                html.Div(self.runner_id),
            ]

        def update():

            # get empty ladder table
            tbl_data = self.table_data(self.index)

            # get index of runner's id in record's runner list
            list_index = generic.get_index(g.active_record.runners, lambda r: r.selection_id == self.runner_id)

            # if runner not found, return empty table
            if not list_index:
                return tbl_data

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

    # create unique id based on element name and card index
    def create(self, g: GUIInterface):

        # create dataset for empty table
        tbl_data = self.table_data(self.index)

        return html.Div(className='runner-card', children=[

            # title element displays runner name and ID
            self.create_title(self.index, self.name, self.runner_id, g),

            # price chart shows short term last traded price info for runner
            self.create_price_chart(),

            # ladder
            self.create_table(tbl_data, self.table_id, self.index)
        ])

    # create html Div containing ladder table to hold ladder data
    @classmethod
    def create_table(cls, table_data, table_id, index):
        return html.Div(className="runner-component runner-table-container", children=[
            dash_table.DataTable(
                id=table_id,

                # as specified in the empty_table() cols are abbreviations of 'available to back',
                # 'available to lay', 'traded volume' with odds in the centre
                columns=[
                    {
                        'name': col,
                        'id': cls.t(col, index)
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

    # create table with odds ticks filled, but empty for everything else
    @classmethod
    def table_data(cls, index):
        return [
            {
                cls.t('atb', index): None,
                cls.t('odds', index): tick,
                cls.t('atl', index): None,
                cls.t('tv', index): None
            } for tick in betting.TICKS_DECODED
        ]

    # update table data for a given market book (list of price sizes) and its abbreviation within dict (key)
    @staticmethod
    def update_data(tbl, price_list: List[Dict], key):

        for p in price_list:

            # convert "price" (odds value) to encoded integer value
            t = betting.float_encode(p['price'])

            # check that price appears in ticks array
            if t in betting.LTICKS:

                # and get index where it appears
                i = betting.LTICKS.index(t)

                # update value in table
                tbl[i][key] = f'{p["size"]:.2f}'

    # create dropdown element for selecting a runner
    @classmethod
    def create_dropdown(cls, index, runner_id, g: GUIInterface):

        return dcc.Dropdown(
            id=cls.t('dropdown', index),
            options=[
                {'label': name, 'value': _id}
                for _id, name in g.runner_names.items()
            ],
            value = runner_id
        )

    # create title element which holds dropdown for runner select, selected runner indicators
    @classmethod
    def create_title(cls, index, runner_name, runner_id, g: GUIInterface):
        return html.Div(className="runner-component runner-title", children=[
            cls.create_dropdown(index, runner_id, g),
            html.Div(id=cls.t('selected-indicator', index), children=[
                html.Div(runner_name),
                html.Div(runner_id)
            ])
        ])

    # TODO create price chart of short time price movements for runner
    @staticmethod
    def create_price_chart():
        return html.Div(className="runner-component runner-price-chart", children=[

        ])


# GUI Component - runner cards group
class CardComponents(GuiComponent):

    def __init__(self):
        self.runner_cards: List[RunnerCard] = None

    def create(self, g: GUIInterface):

        # sorting function - based on last traded price (if exists)
        def sorter(runner):
            return runner.last_price_traded or float('inf')

        # get pre race records
        pre_race = betting.pre_off(g.historical_list, g.market_time)

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
        return list(itertools.chain(
            # run callbacks() for each runner card returning a list [{'output': ..., 'function': ...}]
            # itertools.chain takes multiple args so must unpack (*) resulting list
            # result is lists of {'output': ..., 'function'...} dicts are combined into a single list
            *[r.callbacks(g, app) for r in self.runner_cards]
        ))


# GUI Component - GUI title
class TitleComponent(GuiComponent):
    def create(self, g: GUIInterface):
        return html.H1(
            'Betfair GUI',
            className='header-container'
        )


# GUI class - must use class definition and not create instances, given there (should?) be only one GUI instance
# running at a time
class DashGUI(generic.StaticClass):

    # number of runner cards to display
    N_CARDS = 3

    # time-span of runner chart (seconds)
    CHART_SPAN_S = 60

    # Dash app instance
    app: dash.Dash = None

    # GUI interface instance
    g: GUIInterface = None

    # nav component must come before chart so chart positions are updated first on slider move
    componentList: List[GuiComponent] = [TitleComponent(),
                                         InfoComponent(),
                                         NavComponent(),
                                         ChartComponent(CHART_SPAN_S),
                                         CardComponents()]

    @classmethod
    def _set_callbacks(cls):

        # get list of interval dicts for each component
        interval_callbacks = list(itertools.chain(

            # each callbacks() returns a list, but must unpack comprehension list to pass to chain()
            *[c.callbacks(cls.g, cls.app) for c in cls.componentList]

        ))

        # create interval callback
        # dash outputs are created from 'output' attributes in each callback dict returned from component
        @cls.app.callback([c['output'] for c in interval_callbacks],
                          [dash.dependencies.Input('interval-component', 'n_intervals')])
        def interval_update(n_intervals):

            # list to store outputs to return from callback
            output_list = []

            for c in interval_callbacks:
                # 'function' attribute is the callback function to run - take its output and add to output list
                output_list.append(c['function']())

            return output_list

    # entry point - create and return dash app, having created components and callbacks
    @classmethod
    def create_app(cls, name, record_list, catalogue) -> dash.Dash:

        # get absolute path of assets folder
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        assets_path = os.path.join(dir_path, 'assets')

        # create app instance
        # specify absolute folder of assets incase called from a different dir
        cls.app = dash.Dash(name, assets_folder=assets_path)

        # create 'GUIInterface' instance
        cls.g = GUIInterface(record_list, cls.N_CARDS, catalogue)

        # create list of children with first instance as interval callback to trigger periodic updating
        children = [
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ]
        # create components from component list and to children
        children += [
            c.create(cls.g) for c in cls.componentList
        ]

        # add children to app layout
        cls.app.layout = html.Div(className='content-container', children=children)

        # set callbacks for components
        cls._set_callbacks()

        return cls.app


# create and run dash app - 'name' specifies app name, 'record_list' is historical record list and 'debug' set to True
# or False passed directly to app.run_server()
# def run(name, record_list, catalogue, debug):
#     app = DashGUI.create_app(name, record_list, catalogue)
#     app.run_server(debug=debug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('race_file',
                        type=str,
                        help='historical horse racing file to run with GUI')
    parser.add_argument('--catalogue_file',
                        type=str,
                        help='catalogue file, if unspecified, "race_file" will be taken as historical and market information taken from there')
    parser.add_argument('--dash_debug',
                        action='store_true',
                        help='set to enable dash debug mode')

    args = parser.parse_args()
    file_name = args.race_file

    myLogger.info('Logging into betfair API...')
    trading = betting.get_api_client()
    trading.login()

    myLogger.info(f'Processing historical file "{file_name}"...')
    historical_queue = betting.get_historical(trading, file_name)
    historical_list = list(historical_queue.queue)

    catalogue=None
    if args.catalogue_file:
        catalogue = bf_strategy.get_hist_cat(args.catalogue_file)
        if not catalogue:
            return

    myLogger.info(f'Launching GUI...')

    app = DashGUI.create_app(__name__, historical_list, catalogue)
    app.run_server(debug=args.dash_debug)


if __name__ == '__main__':
    main()