'''
    Example script.
'''

import os 
import src.tools as tools
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
        '''Interpret the input. If the string is empty, it returns
        the default, otherwise return it change to a integer.'''
        if len(input_str) == 0:
            return default
        else:
            return int(input_str)
    #import the data
    print('Importing data')
    df = tools.import_data()
    input('\nData imported. Type anything to describe the data:')
    #access the desciption of all the columns in the dataframe
    print('Describing the data:')
    tools.column_description()
    input('\nType anything to describe the missing values:')
    #verify the missing values in the dataframe 
    print('Describing the missing records:')
    tools.nan_description()
    input('\nType anything to describe the description feature counts:')
    # Select only the varieties having more than a thousand descriptions
    print('''Counting and describing the number of descriptions for each
     variety''')
    print(df.groupby(['variety'])['description'].count().describe())
    print('''The data presents grape's varieties descriptions counts ranging
    from 1 to 14482. In the following, type the number of minimum description
    counts you want to consider for fitting the model and exploring 
    further the data. The higher this number, the faster the code and 
    more precise the model, but one may leave many varieties behind.''')
    n = input('''\nType the (integer) minimum number of description to consider
     (type Enter to use the default n=500):\n''')
    n = interpret_input(n, 500)
    print('Selecting the data')
    df = tools.select_number_of_descriptions(df=df, n=n)
    # check the number of varieties with more than 1000 descriptions
    print('Printing a countplot of the descriptions.')
    tools.variety_counts_plot(df, n=n)
    input('''\nType anything to plot a histogram of the points feature
     distribution:''')
    # plot a histogram of the points given to the wines
    print('''Ploting the histogram of the points distrobution of the data.''')
    tools.plot_points_hist(df=df, n=n)
    input('''\nType anything to plot boxplots of price and points of the
     most described features:''')
    # plot boxplots of points and plots the most described
    m = input('''\nType the (integer) number of boxes (press Enterto use the 
    default value of 30 boxes):\t''')
    m = interpret_input(m, 500)
    print('''Ploting boxplots of the price and points distribution for 
    the top described varieties.''')
    tools.plot_box_points_price_popular(df=df, n=m, top=True, to_png=None)
    input('\nType anything to plot a price versus points regression:')
    # plot the regression of price and points
    print('''Ploting a regression of the price per points features.''')
    tools.plot_price_points(df=df, to_png=None)
    print('''Notice that a log expression fits perfectly to the regression
    meaning that at a expensive wine is no guarateed to be a well evaluated
    wine.''')
    input('\nType anything to plot the model cross validation boxplots:')
    # plot the cross validation results of the model with boxplots
    print('''We shall fit and explore the model now, in the following
    we will plot boxplots of the model cross validation results for diferent
    minimum description counts. To do this you need to specify the start
    end and step parameters.''')
    start = input('''\nType the (integer) minimal description count start (
        press Enter to use the default start=500):\t''')
    start = interpret_input(start, 500)
    end = input('''\nType the (integer) minimal description count end (
        press Enter to use the default end=8000):\t''')
    end = interpret_input(end, 8000)
    step = input('''\nType the (integer) minimal description count loop step (
        press Enter to use the default end=300):\t''')
    step = interpret_input(step, 300)
    print('Generating results cross validation boxplots.')
    
    tools.plot_results_cv_boxplots(to_png=None, start=start, end=end,
                                     step=step)  #nopep8  E128
    input('''\nType anything to plot the variation of results and varieties
     counts as function of the he minimum description
    counts:''')
    # plot the results and the number of available varieties as a function of the minimum description counts
    print('''Finally, we plot the variation of the number of distinct
    variaties and the model results as function of the minimum description
    counts.''')
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
# move the dataset to the apropriated file 
filename = input('write down the path to the file' +
'"winemag-data-130k-v2.csv" :')
os.rename(filename, './data/winemag-data_first150k.csv')

df = tools.import_data()
print('\n\n\n')
print('''\tWelcome to the which_wine package example script. Please type one
of the options below to explore:\n''')
flag = input('''\tType DA to explore the data in an verbose interface; 
\ttype DPP for a verbose model fitting and use of the datafile test set 
to print the predicted probabilities of the descriptions of wines, and;
\ttype RS for a verbose recomenddation system where you will be asked to
feed typed recomendations to the code so that it can predict the variety
of wine you have in mind. \n \tBe warned that typing any other string 
the job will be ended.\n: ''')
if flag.strip().lower() == 'da':
    preprocessing_and_data_visualization()
elif flag.strip().lower() == 'dpp':
    sample_size = input('''Enter the (integer) size of sample you want to
     display (press Enter to use the default of sample_size=10):\t''')
    if len(sample_size) == 0:
        sample_size = 10
    else:
        sample_size = int(sample_size)
    min_descriptions = input('''Enter the (integer) minimum description counts
     of each variety to consider it in the model fitting you (press Enter to
     use the default of sample_size=500):\t''')
    if len(min_descriptions) == 0:
        min_descriptions = 500
    else:
        min_descriptions = int(min_descriptions)
    print_probability_predictions(sample_size, min_descriptions)
elif flag.strip().lower() == 'rs':
    min_descriptions = input('''Enter the (integer) minimum description counts
     of each variety to consider it in the model fitting you (press Enter to
     use the default of sample_size=500):\t''')
    if len(min_descriptions) == 0:
        min_descriptions = 500
    else:
        min_descriptions = int(min_descriptions)
    recomendation_system(min_descriptions)
else:
    print('Exiting the example')
