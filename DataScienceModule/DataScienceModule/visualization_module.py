#!/usr/bin/env python
"""Provides several visualization functions using bokeh, matplotlip and seaborn.

Long Summary:

"""

__author__ = "Julian Quernheim"
__copyright__ = "Copyright 2018"
__credits__ = ["Julian Quernheim"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Julian Quernheim"
__email__ = "julian-quernheim@hotmail.de"
__status__ = "Development"




def bokeh_bar_plot(x, y, title,legend_title, color='yes'):
    '''
    @ Sample usage
    @ bokeh_bar_plot(x = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries'], y = [5, 3, 4, 2, 4, 6], title ="test" , legend_title = "Fruits" )
    :param x:
    :param y:
    :param title:
    :param legend_title:
    :param color:
    :return:
    '''

    from bokeh.io import show, output_file
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Spectral6
    from bokeh.plotting import figure

    output_file(title + ".html")

    source = ColumnDataSource(data=dict(fruits=x, counts=y, color=Spectral6))

    p = figure(x_range=x, y_range=(0,9), plot_height=250, title=title,
               toolbar_location=None, tools="")

    p.vbar(x='fruits', top='counts', width=0.9, color='color', legend=legend_title, source=source)

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    show(p)
    return p


def bokeh_scatter_plot(x_data, y_data, title, x_label, y_label, color='yes'):
    '''
    @ Sample usage
    @ bokeh_scatter_plot(x_data = [5, 3, 4, 2, 4, 6], y_data =[5, 3, 4, 2, 4, 6], title = "test",y_label = "y_axis", x_label = "x_axis", color='yes')
    :param x_data:
    :param y_data:
    :param title:
    :param x_label:
    :param y_label:
    :param color:
    :return:
    '''

    from bokeh.io import show, output_file
    from bokeh.plotting import figure

    output_file(title + ".html", title=title)
    p = figure(title=title, x_axis_label=x_label,
                  y_axis_label=y_label)

    p.circle(x_data, y_data, size=5, alpha=0.5)

    show(p)
    return p




def bokeh_histogram_plt(data):

    '''
    @ import numpy as np
    @ data = np.random.normal(0, 0.5, 1000)
    @ bokeh_histogram_plt(data)
    :param data:
    :return:
    '''


    import numpy as np
    import scipy.special
    from bokeh.plotting import figure, show, output_file
    from bokeh.palettes import Spectral6
    p = figure(title="Normal Distribution (μ=0, σ=0.5)", tools="save",
                background_fill_color="#ffffff")

    hist, edges = np.histogram(data, density=True, bins=50)

    mu, sigma = 0, 0.5
    x = np.linspace(-2, 2, 1000)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    cdf = (1 + scipy.special.erf((x - mu) / np.sqrt(2 * sigma ** 2))) / 2

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color=Spectral6[0], line_color="#033649")
    p.line(x, pdf, line_color=Spectral6[5], line_width=3, alpha=0.7, legend="PDF")
    p.line(x, cdf, line_color=Spectral6[4], line_width=3, alpha=0.7, legend="CDF")

    p.legend.location = "center_right"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'

    show(p)
    return p



def visualize_tree(sklearn_model_tree, feature_names):

    """Create tree png using graphviz.
    @ example for data and decision tree model can be found here: http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
    @ Sample Usage:
    @ visualize_tree(sklearn_model_tree = dt, feature_names = features)

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """

    from graphviz import Source
    from sklearn import tree
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = Source(tree.export_graphviz(sklearn_model_tree, out_file=None, feature_names=feature_names))
    graph.format = 'png'
    graph.render('dtree_render', view=True)
    print('dtree_render.png file with decision tree was created in working directory')




def pair_wise_scatter_plot_seaborn(data, title):
    '''
    @ sample usage:
    @ import numpy as np
    @ import pandas
    @ import random
    @ df = pandas.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    @ pair_wise_scatter_plot_seaborn(data = df, title = "My Title")


    :param data:
    :param title:
    :return:
    '''

    # Pair-wise Scatter Plots
    import seaborn as sns
    cols = data.columns.values
    pp = sns.pairplot(data[cols], size=1.8, aspect=1.8,
                      plot_kws=dict(edgecolor="k", linewidth=0.5),
                      diag_kind="kde", diag_kws=dict(shade=True))

    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.suptitle(title, fontsize=14)
    fig.savefig(title + ".png")





def summary_statistics(dataset1,dataset2, dataset1_title, dataset2_title):
    '''
    @ sample usage:
    @ import pandas
    @ import random
    @ import numpy as np
    @ df = pandas.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    @ df_summary_statistics = summary_statistics(dataset1 = df, dataset2 = df, dataset1_title = "dataset1", dataset2_title = "dataset2")
    @ df_summary_statistics

    :param dataset1:
    :param dataset2:
    :param dataset1_title:
    :param dataset2_title:
    :return:
    '''
    import pandas as pd

    dataset1_subset_attributes = dataset1.columns.values
    dataset2_subset_attributes = dataset2.columns.values
    rs = round(dataset1[dataset1_subset_attributes].describe(), 2)
    ws = round(dataset2[dataset2_subset_attributes].describe(), 2)

    df_summary_statistics = pd.concat([rs, ws], axis=1, keys=[dataset1_title, dataset2_title])
    return(df_summary_statistics)



def scatter_plot_regression_seaborn(data,x_colname, y_colname,hue_colname,col_colname,  title, subtitle):
    '''
    @ sample usage:
    @ import pandas
    @ import random
    @ import numpy as np
    @ df = pandas.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    @ df["C"] = np.random.choice(['red', 'white','yellow','pink'], 100, p=[0.5, 0.1, 0.1, 0.3])
    @ df["D"] = np.random.choice(['pooh', 'rabbit', 'piglet', 'Christopher'], 100, p=[0.5, 0.1, 0.1, 0.3])
    @ scatter_plot_regression_seaborn(data = df, x_colname =  'A', y_colname= 'B', hue_colname = 'C', col_colname = 'D' , title= "title", subtitle="subtitle")

    :param data:
    :param x_colname:
    :param y_colname:
    :param hue_colname:
    :param col_colname:
    :param title:
    :param subtitle:
    :return:
    '''

    import seaborn as sns; sns.set(color_codes=True)
    import matplotlib.pyplot as plt


    pp = sns.lmplot(x=x_colname, y=y_colname,  col= col_colname,
                    palette={"red": "#FF9999", "white": "#FFE888"},
                    data=data, fit_reg=True, legend=True,
                    scatter_kws=dict(edgecolor="k", linewidth=0.5))

    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.suptitle(title, fontsize=14)
    fig.savefig(title + ".png")



def correlation_plot_seaborn(data, title):

    '''
    @ sample usage:
    @ import numpy as np
    @ import pandas as pd
    @ from string import ascii_letters
    @ # Generate a large random dataset
    @ rs = np.random.RandomState(33)
    @ data = pd.DataFrame(data=rs.normal(size=(100, 26)),columns=list(ascii_letters[26:]))
    @ correlation_plot_seaborn(data, 'correlation plot')

    :param data:
    :param title:
    :return:
    '''

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")



    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    pp = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    fig = pp.get_figure()
    fig.savefig(title + ".png")

    return fig


