'''
    Example script.
'''
import sys
sys.path.insert(0, '/home/ricardo/Desktop/WineSoilRepo/whic_wine_2/tools')
import tools
import os
import pandas as pd

def preprocessing_and_data_visualization():
    """ Data analysis and visualization example

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    def interpret_input(input_str, default):
        '''Interpret the input. If the string is empty, it returns the default, otherwise return it change to a integer.'''
        if len(input_str) == 0:
            return default
        else:
            return int(input_str)
    #import the data
    print('Importing data')
    df = tools.import_data()
    input('Data imported. Type anything to describe the data:\n')
    #access the desciption of all the columns in the dataframe
    print('Describing the data:')
    tools.column_description()
    input('Type anything to describe the missing values:\n')
    #verify the missing values in the dataframe 
    print('Describing the missing records:')
    tools.nan_description()
    input('Type anything to describe the [description] feature counts:\n')
    # Select only the varieties having more than a thousand descriptions
    print('''Counting and describing the number of descriptions for each variety''')
    print(df.groupby(['variety'])['description'].count().describe())
    print('''The data presents grape's varieties descriptions counts ranging from 1 to 14482. In the following, type the number of minimum description counts you want to consider for fitting the model and further exploring the data. The higher this number, the faster the code is and more precise the model, but one may leave many varieties behind.''')
    n = input('''Type the (integer) minimum number of description to consider (type Enter to use the default n=500):\n''')
    n = interpret_input(n, 500)
    print('Selecting the data')
    df = tools.select_number_of_descriptions(df=df, n=n)
    # check the number of varieties with more than 1000 descriptions
    print('Printing a countplot of the descriptions.')
    tools.variety_counts_plot(df, n=n)
    input('''Type anything to plot a histogram of the [points] feature distribution:\n''')
    # plot a histogram of the points given to the wines
    print('''Ploting the histogram of the points distrobution of the data.''')
    tools.plot_points_hist(df=df, n=n)
    input('''Type anything to plot boxplots of [price] and [points] of the most described features:\n''')
    # plot boxplots of points and plots the most described
    m = input('''Type the (integer) number of boxes (press Enter to use the default value of 30 boxes):\n''')
    m = interpret_input(m, 500)
    print('''Ploting boxplots of the [price] and [points] distribution for the top described varieties.''')
    tools.plot_box_points_price_popular(df=df, n=m, top=True, to_png=None)
    input('Type anything to plot a [price] versus [points] regression:\n')
    # plot the regression of price and points
    print('''Ploting a regression of the [price] per [points] features.''')
    tools.plot_price_points(df=df, to_png=None)
    print('''Notice that a log expression fits well to the regression suggesting that at a expensive wine is no guarateed to be a well evaluated one.''')
    input('Type anything to plot the model cross validation boxplots:\n')
    # plot the cross validation results of the model with boxplots
    print('''I shall fit and explore the model now, in the following we will plot boxplots of the model cross validation results for different minimum description counts. To do this you need to specify the start, end and step parameters.''')
    start = input('''Type the (integer) minimal description count start (press Enter to use the default start=500):\n''')
    start = interpret_input(start, 500)
    end = input('''Type the (integer) minimal description count end (press Enter to use the default end=8000):\n''')
    end = interpret_input(end, 8000)
    step = input('''Type the (integer) minimal description count loop step (press Enter to use the default end=300):\n''')
    step = interpret_input(step, 300)
    print('Generating cross validation results boxplots.')
    tools.plot_results_cv_boxplots(to_png=None, start=start, end=end,step=step)  #nopep8  E128
    # plot the results and the number of available varieties as a function of the minimum description counts
    print('''Finally, we plot the variation of the number of distinct variaties and the model results as function of the minimum description counts.''')
    tools.plot_results_lineplots(start=start, end=end, step=step)
    return None


def print_probability_predictions(sample_size=10, min_descriptions=100):
    """Fit the model and predict the probability of a sample of the test set 
    displaying the 5 most probable varieties based on the descriptions

    Parameters
    ----------
    sample_size :   an integer, the number of elements to display
         (Default value = 10)
    min_descriptions : an integer, the number of description above which to
    filter in the dataset
         (Default value = 100)

    Returns
    None
    -------

    """
    tools.predict_and_print_probabilities(sample_size=sample_size,
     min_descriptions=min_descriptions)  #nopep8  E128
    return None


def recomendation_system(min_descriptions=1000):
    """From a vector of description inputed by the user, estimate the pro
    babilities for it to be a variety of the given dataset

    Parameters
    ----------
    min_descriptions : an integer, the number of description above which to
    filter in the dataset
         (Default value = 1000)

    Returns
    -------

    """
    tools.predict_and_print_recomendations(min_descriptions=min_descriptions)
    return None

df = tools.import_data()
print('\n\n\n')
print('''Welcome to the which_wine package example script. Use one of the options below to explore:''')
flag = input('''Type DA (data analisis) to explore the data in an verbose interface; type DPP (data processing and predictions) for a verbose model fitting and use of the datafile test set to print the predicted probabilities of the descriptions of wines, and; type RS (recomendation system) for a verbose recomendation system where you will be asked to type recomendations to the code so that the model can predict the variety of wines you have in mind. Be warned that typing any other string other than the three options, the job will end.\n''')
if flag.strip().lower() == 'da':
    preprocessing_and_data_visualization()
elif flag.strip().lower() == 'dpp':
    sample_size = input('''Enter the (integer) size of sample you want to display (press Enter to use the default of sample_size=10):\t''')
    if len(sample_size) == 0:
        sample_size = 10
    else:
        sample_size = int(sample_size)
    min_descriptions = input('''Enter the (integer) minimum description counts of each variety to consider it in the model fitting you (press Enter to use the default of sample_size=500):\t''')
    if len(min_descriptions) == 0:
        min_descriptions = 500
    else:
        min_descriptions = int(min_descriptions)
    print_probability_predictions(sample_size, min_descriptions)
elif flag.strip().lower() == 'rs':
    min_descriptions = input('''Enter the (integer) minimum description counts of each variety to consider it in the model fitting you (press Enter to use the default of sample_size=500):\t''')
    if len(min_descriptions) == 0:
        min_descriptions = 500
    else:
        min_descriptions = int(min_descriptions)
    recomendation_system(min_descriptions)
else:
    print('Exiting the example')
