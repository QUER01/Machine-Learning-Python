# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import datetime
from bokeh.models import ColumnDataSource, Panel
from bokeh.layouts import layout, Spacer
from bokeh.models.widgets import TableColumn, DataTable
from bokeh.plotting import figure
from bokeh.models.tools import  HoverTool


def table_tab_facebook(data):
    data_src = ColumnDataSource(data)
    # Columns of table
    table_columns = [
        TableColumn(field='Date',	title='Date'),
        TableColumn(field='year', title='year'),
        TableColumn(field='weekday', title='weekday'),
        TableColumn(field='FB_High', title='FB_High'),
        TableColumn(field='FB_Open', title='FB_Open'),
        TableColumn(field='FB_Low', title='FB_Low'),
        TableColumn(field='FB_Close', title='FB_Close'),
        TableColumn(field='FB_Volume', title='FB_Volume')
    ]

    data_table = DataTable(source=data_src,columns=table_columns, width=1400)


    summary_data = data.describe()
    summary_data['Stats'] = summary_data.index.values
    data_src = ColumnDataSource(summary_data.drop(['year', 'weekday'], axis=1))
    summary_table_columns = [
        TableColumn(field='Stats', title='Stats'),
        TableColumn(field='FB_High', title='FB_High'),
        TableColumn(field='FB_Open', title='FB_Open'),
        TableColumn(field='FB_Low', title='FB_Low'),
        TableColumn(field='FB_Close', title='FB_Close'),
        TableColumn(field='FB_Volume', title='FB_Volume')
    ]
    summary_table = DataTable(source=data_src, columns=summary_table_columns, width=800, height = 300)



    # Line Chart
    x = pd.to_datetime(data['Date'])
    y = data['FB_Close']

    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "($Date, $FB_Close)")
    ])

    plot_timeseries = figure(  plot_width=600, plot_height=300, tools=[hover], title ="Stock movement over time")
    # add a line renderer
    plot_timeseries.line(x,y , line_width=2)


    # Group by Week
    df = pd.DataFrame()
    # Convert that column into a datetime datatype
    df['datetime'] = pd.to_datetime(data['Date'])
    # Set the datetime column as the index
    df.index = df['datetime']
    # Create a column from the numeric score variable
    df['FB_Close'] = list(data['FB_Close']) #list(np.random.randint(low=1, high=1000, size=len(data))) #data['FB_Close']

    grouped_data = df.resample('M').mean()

    x = grouped_data.index.values
    y = grouped_data['FB_Close']
    print(grouped_data)
    #print(x)
    #print(y)


    p1 = figure(plot_width=1400, plot_height=400)
    # add a line renderer
    p1.line(x,y , line_width=2)


    # Putting it all together

    li = layout([[summary_table,Spacer(width = 50),  plot_timeseries],[p1]], sizing_mode='fixed')

    tab = Panel( child = li, title = 'Facebook Stock Market Table')

    return tab




