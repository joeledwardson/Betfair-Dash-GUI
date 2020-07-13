import dash
import dash_html_components as html
import dash_core_components as dcc
from plotly import express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

maxVal = 100000

df = pd.DataFrame({
    'x': list(range(maxVal)),
    'y': [i*2 for i in range(maxVal)]
})

window = [0, 100]

fig = px.line(df,
              x='x',
              y='y')
fig.update_layout(yaxis_type='log')
fig.update_xaxes(range=window)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Slider(
        id='my-slider',
        min=0,
        max=maxVal,
        value=0,
    ),
    html.Div(id='slider-output-container'),
    dcc.Graph(
        id='my-graph',
        figure=fig
    )
])


@app.callback(
    [
        dash.dependencies.Output('slider-output-container', 'children'),
        dash.dependencies.Output('my-graph', 'figure')
    ],
    [
        dash.dependencies.Input('my-slider', 'value'),
    ]
)
def update_output(value):
    global window
    window = [value, min(value+100, maxVal)]
    fig.update_xaxes(range=window)
    fig.update_yaxes(range=[df['y'][window[0]], df['y'][window[1]-1]])

    return [
        'You have selected "{}"'.format(value),
        fig
    ]


if __name__ == '__main__':
    app.run_server(debug=True)