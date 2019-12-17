# Pandas for data management
import pandas as pd

# os methods for manipulating paths
from os.path import dirname, join

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs


# Each tab is drawn by one script
from scripts.tab_facebook import table_tab_facebook
from scripts.tab_google import table_tab_google

# Using included state data from Bokeh for map
from bokeh.sampledata.us_states import data as states

# Read data into dataframes
data = pd.read_csv(join(dirname(__file__), 'data', 'finance_dataset_clean.csv'),delimiter=';' )
data_facebook = data[['Date','year','month','weekday','FB_High','FB_Open','FB_High','FB_Low','FB_Close','FB_Volume']]
data_google = data[['Date','year','month','weekday','GOOGL_High','GOOGL_Open','GOOGL_High','GOOGL_Low','GOOGL_Close','GOOGL_Volume']]
#data_allianz = data[['Date','year','month','weekday','ALV_High','ALV_Low','ALV_Volume']]
#data_msci = data[['Date','year','month','weekday','MSCI_High','MSCI_Open','MSCI_High','MSCI_Low','MSCI_Volume']]



# Formatted Flight Delay Data for map
#map_data = pd.read_csv(join(dirname(__file__), 'data', 'flights_map.csv'),header=[0,1], index_col=0)

# Create each of the tabs
#tab1 = histogram_tab(flights)
#tab2 = density_tab(flights)

tab1 = table_tab_facebook(data_facebook)
tab2 = table_tab_google(data_google)
#tab3 = table_tab(data_allianz)
#tab4 = table_tab(data_msci)
#tab4 = map_tab(map_data, states)
#tab5 = route_tab(flights)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab1,tab2])

# Put the tabs in the current document for display
curdoc().add_root(tabs)


