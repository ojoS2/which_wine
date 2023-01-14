'''
    Set of tools to process the data. The first part consists in
 preprocessing and data loading routines. The second part consists
 of model fitting and routines. The last part contain data vizualization
 functions 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
# _______________ importing tools__________________


def import_data(filename='./data/winemag-data_first150k.csv', separator=','):
    """
    Import data from a .csv file only and return a dataframe

    Parameters
    ----------
    filename  : a string indicating the file name

    separator : a string that flag the data separation in the file

    Returns
    -------
    A pandas.DataFrame
    type pandas.DataFrame object if the file exists and is a .csv
    None otherwise
    """
    aux = filename[-4:]
    if aux == '.csv':
        df = pd.read_csv(filename, sep=separator)
        df.drop(labels='Unnamed: 0', axis=1, inplace=True)
        return df
    else:
        raise TypeError('Only .cvs files are compilable in this function!\n')


def column_description(column='all'):
    """
    Describe the features of the wines dataframe

    Parameters
    ----------
    column  : a string indicating the feature name, if none is specified
        then all features are explained. If column is not a wine dataframe
        feature, then raises a value error

    Returns
    -------
        None

    type None
    """
    temp = {}
    temp['country'] = 'The country that the wine is from'
    temp['description'] = """A few sentences from a sommelier describing
     the wine's taste, smell, look, feel, etc."""
    temp['designation'] = """The vineyard within the winery where the
     grapes that made the wine are from."""
    temp['points'] = """The number of points WineEnthusiast rated the wine on
     a scale of 1-100 (though they say they only post reviews for wines
     that score >=80)"""
    temp['price'] = "The cost for a bottle of the wine"
    temp['province'] = "The province or state that the wine is from"
    temp['region_1'] = "The wine growing area in a province or state (ie Napa)"
    temp['region_2'] = """Sometimes there are more specific regions specified
     within a wine growing area (ie Rutherford inside the Napa Valley), but
      this value can sometimes be blank"""
    temp['variety'] = """The type of grapes used to make the wine
     (ie Pinot Noir)"""
    temp['winery'] = "The winery that made the wine"
    if column == 'all':
        for index, values in temp.items():
            print(f"Column [{index}] description: {values}")
    elif column in temp.keys():
        print(f"Column [{column}] description: {temp[column]}")
    else:
        raise ValueError("Unidendified Column.")
    print('\n')
    return None


def nan_description(df=pd.DataFrame({'A': []}), column='all'):
    """
    Describe the features regarding missing values

    Parameters
    ----------
    df : a dataframe or empty. If empty, it loads the raw dataframe from the
     data directory

    column  : a string indicating the feature name, if none is specified
        then all features are explained. If column is not a wine dataframe
        feature, then raises a value error

    Returns
    -------
        None

    type None
    """
    if df.empty:
        df = import_data()
    if column == 'all':
        temp = df.isnull().sum().to_frame()
        for index in range(len(temp)):
            if temp.iloc[index, 0] > 0:
                print("""The feature {0} presents
                 {1} missing values corresponding
                 to {2}%
                 of the entries""".format(temp.index[index],
                                        temp.iloc[index, 0],
                                        round(temp.iloc[index, 0]
                                        / df.shape[0] * 100, 4)))  # nopep8: E128
            else:
                print(f'No missing found in the {temp.index[index]} column.')
        print('\n')
        return None
    elif column in df.columns:
        temp = df[[column]].isnull().sum()
        if temp[0] > 0:
            print("""The feature {0} presents {1} missing values
             corresponding to {2}% of the entries
             """.format(column, temp[0], round(temp[0] / df.shape[0] * 100, 4
             )))  # nopep8: E127
        else:
            print('No missing found in feature ' + column)
        print('\n')
        return None
    else:
        raise ValueError("Unidendified Column.\n")


def drop_missing_column(column, df=pd.DataFrame({'A': []})):
    """
    Drop the missing values of a column if they exist. It raises an error if
    the colunm do not exist

    Parameters
    ----------
    df : a dataframe to be analysed or an empty dataframe. If empty it loads
    the data from the data directory

    column  : a string indicating the feature name.

    Returns
    -------
        a DataFrame

    type pandas DataFrame
    """
    if df.empty:
        df = import_data()
    if column not in df.columns:
        raise ValueError("Unidendified Column.\n")
    temp = df[[column]].isnull().sum()
    if temp[0] == 0:
        print(f'the column {column} do not have any missing to drop.\n')
    else:
        old = df.shape[0]
        df.dropna(axis=0, subset=column, inplace=True)
        print("""{0} rows, or {1}% of the data frame rows were deleted.\n
        """.format(old - df.shape[0], round((1.0 - df.shape[0] / old), 5) *
         100))   # nopep8: E127
    return df


def select_number_of_descriptions(df=pd.DataFrame({'A': []}), n=500):
    """
    Drop the all the wine varieties which number of description is lower
    than n

    Parameters
    ----------
    df : a dataframe or an empty dataframe. If empty, it loads the dataframe
    from the data directory

    n  : a integer, the lower limit of the description counts

    Returns
    -------
        df

    type pandas DataFrame
    """
    if df.empty:
        df = import_data()
    test = df.groupby(['variety'])['description'].count().reset_index(\
        name='Count').sort_values(['Count'], ascending=False)
    test = test[test.Count >= n]
    df = df[df.variety.isin(test.variety)]
    return df
# _________________________________________________
# ________________processing_and_modeling__________


def cross_val_data(start=1000, end=9000, step=500):
    '''Make the cross validation over the range of selected minimum
    number of descriptions and save it to a file in the data directory

    Parameters
    ----------
    start, end, step: integers used in the cross validation for loop

    Returns
    -------

    cv, a dataframe

    type pandas dataframe
    '''
    def get_crossval_results(cv_X, X, b_p):
        results, best_params, n_descriptions = [], [], []
        for label in list(cv_X.columns):
            if label[0:5] == 'split':
                n_descriptions.append(X)
                results.append(cv_X[label].values[0])
                best_params.append(b_p)
        return pd.DataFrame({'results': results,
                             'n_descriptions': n_descriptions,
                             'best_param': best_params})  # nopep8: E127
    df = import_data()
    temp = []
    for min_description in range(start, end, step):
        best_params, _, cv_X = model_tunning(
                    df=select_number_of_descriptions(df, n=min_description))  # nopep8: E127
        temp.append(get_crossval_results(cv_X, min_description,
                     best_params['alpha']))  # nopep8: E127
    cv = pd.concat(temp)
    cv['n_descriptions'] = cv['n_descriptions'].astype('object')
    cv.to_csv('data/cv_boxplot.csv')
    return cv


def get_lemmatized_phrases(phrases):
    """
    Lematize sentences leaving out non-alphanumeric characters and english
    stop words, transforming the tokens to lower case

    Parameters
    ----------
    phrases  : a list of strings representing the sentences

    Returns
    -------
    A list (the same size as the original) containing all the tokens from
    the phrases. It does not separate the phrases

    type list
    """
    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    new_phrases = []
    for phrase in phrases:
        test = nlp(phrase)
        lemmas = [token.lemma_ for token in test]
        aux = (' '.join([lemma.lower() for lemma in lemmas if (lemma.isalpha()
             or lemma == '-PRON-') and lemma not in stopwords]))  # nopep8: E127
        new_phrases.append(aux)
    return new_phrases


def model_fit(df, vectorizer_option=0, ngrams_limit=(1, 2)):
    """
    Fit a Multinomial Naive Bayes model to the data. First decide what type
    of preprocessing to use, then divide the population, then transform the
    data, instanciate the model (with the best params given by the
    [model_tunning] model), fit the data, calculate the model score and
    confusion matrix.

    Parameters
    ----------
    df  : the data frame

    vectorizer_option :  a integer, the default 0 uses tf-idf preprocessing,
    the option 1 use the simpler CountVectorizer preprocessing

    ngrams_limit : a tuple, the limits of ngram sizes to consider when
    vectorizer_option == 0, the default uses up to 2grams

    Returns
    -------
    The accurance of the fitted model, the confusion matrix associated with
    the process and the model

    type tuple
    """
    # preprocessing
    if vectorizer_option == 0:
        vectorizer = TfidfVectorizer(strip_accents='ascii',
                     stop_words='english', ngram_range=ngrams_limit)  # nopep8: E127
    elif vectorizer_option == 1:
        vectorizer = CountVectorizer(strip_accents='ascii',
                                     stop_words='english', lowercase=True)  # nopep8: E127
    else:
        raise ValueError('''Option not reconized by the parameter
                         vectorizer_option''')  # nopep8: E127
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df['description'],
                                     df['variety'], test_size=0.3,
                                     random_state=13)  # nopep8: E127
    # fit transform the train data, transform the test data
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    # instanciate the model and fit to the transformed test data
    clf = MultinomialNB(alpha=0.001)
    clf.fit(X_train_bow, y_train)
    # calculating the scores
    accuracy = clf.score(X_test_bow, y_test)
    c_m = confusion_matrix(y_test, clf.predict(X_test_bow))
    return accuracy, c_m, clf


def model_cv(df, vectorizer_option=0, ngrams_limit=(1, 2)):
    """
    Fit a Multinomial Naive Bayes model to the data and make a cross
    validation. First decide what type of preprocessing to use, then
    divide the population, then transform the data, instanciate the model
    (with the best params given by the [model_tunning] model), fit the data,
    calculate the model score  and confusion matrix.

    Parameters
    ----------
    df  : the data frame

    vectorizer_option :  a integer, the default 0 uses tf-idf preprocessing,
    the option 1 use the simpler CountVectorizer preprocessing

    ngrams_limit : a tuple, the limits of ngram sizes to consider when
    vectorizer_option == 0, the default uses up to 2grams.

    Returns
    -------
    The cross validation results in a list

    type list
    """
    if vectorizer_option == 0:
        vectorizer = TfidfVectorizer(strip_accents='ascii',
                                     stop_words='english',
                                     ngram_range=ngrams_limit)  # nopep8: E127
    elif vectorizer_option == 1:
        vectorizer = CountVectorizer(strip_accents='ascii',
                                     stop_words='english', lowercase=True)  # nopep8: E127
    else:
        raise ValueError('''Option not reconized by the parameter vectorizer
                        _option''')  # nopep8: E127
    kf = KFold(n_splits=10, shuffle=True, random_state=13)
    X = vectorizer.fit_transform(df['description'])
    y = df['variety']
    clf = MultinomialNB(alpha=0.01)
    cv_results = list(cross_val_score(clf, X, y, cv=kf))
    return cv_results


def model_prob_tab(df, vectorizer_option=0, ngrams_limit=(1, 2),
                     test_size=0.1):  # nopep8: E127
    """
    Fit a Multinomial Naive Bayes model to the data and make probability
    predictions on the test set

    Parameters
    ----------
    df  : the data frame

    vectorizer_option :  a integer, the default 0 uses tf-idf preprocessing,
    the option 1 use the simpler CountVectorizer preprocessing

    ngrams_limit : a tuple, the limits of ngram sizes to consider when
    vectorizer_option == 0, the default uses up to 2grams

    test_size the relative size to divide the data

    Returns
    -------
    A data frame conteining the probabilities

    type list
    """
    if vectorizer_option == 0:
        vectorizer = TfidfVectorizer(strip_accents='ascii',
                                     stop_words='english',
                                     ngram_range=ngrams_limit)  # nopep8: E127
    elif vectorizer_option == 1:
        vectorizer = CountVectorizer(strip_accents='ascii',
                                 stop_words='english', lowercase=True)  # nopep8: E127
    else:
        raise ValueError('''Option not reconized by the parameter
                             vectorizer_option''')  # nopep8: E127
    X_train, X_test, y_train, y_test = train_test_split(df['description'],
                     df['variety'], test_size=test_size, random_state=13)  # nopep8: E127
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    clf = MultinomialNB(alpha=0.001)
    clf.fit(X_train_bow, y_train)
    proba = clf.predict_proba(X_test_bow)
    prob_tab = pd.DataFrame(proba, columns=clf.classes_)
    prob_tab['Probability_Sum'] = prob_tab.sum(axis=1)
    prob_tab['Test_Labels'] = y_test.values
    return prob_tab


def model_results(start=3000, end=9000, step=200, filename=None):
    """
    Take three integers, start, and, and step to fit a Multinomial
    Naive Bayes model in a for loop where in each loop we define the
    minimum number of description to consider a wine variety and
    prints to a file a  data frame conteining the minimum number of
    descriptions, the variety, and the accurancy of the model.

    Parameters
    ----------
    start  : a integer marking the begining of the loop

    end  : a integer marking the limit of the loop

    step  : a integer marking the step in the loop

    Returns
    -------
    None

    type None
    """
    df = import_data()
    X, Y, Z = [], [], []
    for size in range(start, end, step):
        df = select_number_of_descriptions(df, n=size)
        red_wine_varieties = list(set(df.variety.unique()))
        X.append(size)
        Y.append(len(red_wine_varieties))
        acc, cm, model = model_fit(df)
        Z.append(acc)
    results = pd.DataFrame({'n_descriptions': X, 'variety': Y, 'acc': Z})
    if type(filename) == str:
        results.to_csv('./data/' + filename + '.csv')
        return None
    elif filename is None:
        return results
    else:
        raise ValueError('Unreconized [filename] parameter.')


def model_cross_validation_tab(start=3000, end=8000, step=300):
    """
    Print the cross validation results of the model fitting in a
    system with minimum description number in a loop which ranges
    and step are given variables

    Parameters
    ----------

    start  : a integer marking the begining of the loop

    end  : a integer marking the limit of the loop

    step  : a integer marking the step in the loop

    Returns
    -------

    None
    """
    df = import_data()
    dict = {'cv_scores': [], 'n_values': [], 'n_select_cat': []}
    for size in range(start, end, step):
        df = select_number_of_descriptions(df, n=size)
        wine_varieties = list(set(df.variety.unique()))
        df = df[df.variety.isin(wine_varieties)]
        aux = model_cv(df)
        for item in aux:
            dict['cv_scores'].append(item)
            dict['n_values'].append(str(size))
            dict['n_select_cat'].append(len(wine_varieties))
    cv_results = pd.DataFrame(dict)
    cv_results.to_csv('./data/cv_results.csv')
    return None


def model_tunning(df, vectorizer_option=0, n_splits=10, ngram_range=(1, 2)):
    """
    Use cross validation and hyperparameter tuning to verify the best
    parameters to use

    Parameters
    ----------
    df  : the data frame

    vectorizer_option :  the default uses tf-idf preprocessing
    (ngrams ranging from 1 to 5), the option 1 use the simpler
    CountVectorizer preprocessing

    n_splits : a integer, how we divided the cross validation

    ngram_range : a tuple of two integers, range of ngrams to consider in
    the model

    Returns
    -------
    The best avaluated parameters, the best score reached and all the
    results in a dataframe

    type tuple
    """
    if vectorizer_option == 0:
        vectorizer = TfidfVectorizer(strip_accents='ascii',
                                     stop_words='english',
                                     ngram_range=ngram_range)
    elif vectorizer_option == 1:
        vectorizer = CountVectorizer(strip_accents='ascii',
                                     stop_words='english',
                                     lowercase=True)  # nopep8: E127
    else:
        raise ValueError('''Option not reconized by the parameter
                         vectorizer_option''')      # nopep8: E127
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=13)
    X = vectorizer.fit_transform(df['description'])
    y = df['variety']
    clf = MultinomialNB()
    param_grid = {"alpha": np.arange(0.001, 1, 10)}
    clf_cv = GridSearchCV(clf, param_grid, cv=kf)
    X_train, X_test, y_train, y_test = train_test_split(
        df['description'], df['variety'], test_size=0.3, random_state=13)
    X_train_bow = vectorizer.fit_transform(X_train)
    clf_cv.fit(X_train_bow, y_train)
    return (clf_cv.best_params_, clf_cv.best_score_,
             pd.DataFrame(clf_cv.cv_results_))   # nopep8: E127


def predict_and_print_probabilities(sample_size=100, min_descriptions=100):
    """
    Fit to the data, and select some test descriptions and print
    the varieties which presents the highest probability of a
    correspondence

    Parameters
    ----------
    min_descriptions: an integer, the number of minimum descriptions in the
    dataframe to consider (The model is fitted only to varieties
    having more descriptions than this minimum)

    sample_size: an integer, the number of descriptions to consider

    Returns
    -------
    None

    type list
    """
    df = import_data()
    df = select_number_of_descriptions(df, n=min_descriptions)
    vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english',
                                ngram_range=(1, 2))  # nopep8: E127
    X_train, X_test, y_train, y_test = train_test_split(df['description'],
                             df['variety'], test_size=0.3, random_state=13)   # nopep8: E127
    X_train_bow = vectorizer.fit_transform(X_train)
    Naive_Bayes_model = MultinomialNB(alpha=0.001)
    Naive_Bayes_model.fit(X_train_bow, y_train)
    sample = pd.DataFrame({'description': X_test, 'variety': y_test})\
                        .sample(n=sample_size).reset_index()  # nopep8: E127
    X_test_bow = vectorizer.transform(sample['description'])
    proba = Naive_Bayes_model.predict_proba(X_test_bow)
    prob_tab = pd.DataFrame(proba, columns=Naive_Bayes_model.classes_)
    prob_tab['Test_Labels'] = sample['variety'].values
    for index, row in prob_tab.iterrows():
        prob_list = []
        row.drop(labels='Test_Labels', inplace=True)
        for i in np.argsort(list(row.values)[:-1]):
            prob_list.append((list(prob_tab.columns)[i], list(row.values)[i]))
        print('\n\n')
        print('The description: ')
        print('\"' + sample['description'][index] + '\"')
        print('\n')
        print('May be refeering to a:')
        for i in range(-1, -5, -1):
            print('''{0} with likelyhood of {1}%;'''.format(prob_list[i][0],\
                 round(prob_list[i][1], 6) * 100))  # nopep8: E127
        num_desc = df[df['variety'] == sample['variety'][index]].shape[0]
        print('It is a ' + sample['variety'][index] + ''', according to the da
            taset. This variety presents {0} descriptions.'''.format(num_desc)\
            )  # nopep8: E127


def predict_and_print_recomendations(min_descriptions=100):
    """
    Based on the model fitted to the data, get description typed
    by the users to estimate the probability of whitch wine is
    being described.

    Parameters
    ----------
    min_descriptions: the number of minimum descriptions in the
    dataframe to consider (The model is fitted only to varieties
    having more descriptions than this minimum)

    Returns
    -------
    None

    type list
    """
    df = import_data()
    df = select_number_of_descriptions(df, n=min_descriptions)
    print("""In the following, write the description of the desired wine. 
        The description should have a description of expected sensations from
         the wine, followed by what you want to drink it with and end with how
         long you want it to be aged, or what age you want to drink it.
         Type examples to see examples from the database.""")   # nopep8: E127
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    flag = input('''\nType example (whitout spaces) to see the examples or
     type go to print the descriptions:\t''')
    if flag == 'example':
        while flag == 'example':
            print('The description: \n')
            temp = df.sample(n=1)
            print(temp[['description']].values)
            print('is associated to the variety')
            print(temp['variety'])
            print('\n\n')
            print('''Type example (whitout spaces) to generate more examples
             or go to type the description''')   # nopep8: E127
            print('Any other words will exit the function')
            flag = input(':\t')
    if flag == 'go':
        print('''Type the phrases and press enter to go to the next phrase.
         Type exit to stop inputing and run the model''')   # nopep8: E127
        sample = []
        temp = input('Input the first description:\t')
        while temp != 'exit':
            sample.append(temp)
            print("""Print another description or exit to run the model
             with the already given descriptions.""")
            temp = input(':\t')
        print('Closing the files.')
    else:
        print('Exiting the job.')
        return None
    if len(sample) == 0:
        print('Exiting the job.')
        return None
    else:
        vectorizer = TfidfVectorizer(strip_accents='ascii',\
                                     stop_words='english', ngram_range=(1, 2))   # nopep8: E127
        X, y = df['description'], df['variety']
        X_train_bow = vectorizer.fit_transform(X)
        Naive_Bayes_model = MultinomialNB(alpha=0.001)
        Naive_Bayes_model.fit(X_train_bow, y)
        X_test_bow = vectorizer.transform(sample)
        proba = Naive_Bayes_model.predict_proba(X_test_bow)
        prob_tab = pd.DataFrame(proba, columns=Naive_Bayes_model.classes_)
        for index, row in prob_tab.iterrows():
            prob_list = []
            for i in np.argsort(list(row.values)[:-1]):
                prob_list.append((list(prob_tab.columns)[i],\
                                 list(row.values)[i]))   # nopep8: E127
            print('\n\n')
            print('The description: ')
            print('"' + sample[index] + '"')
            print('\n')
            print('May be refeering to a:')
            for i in range(-1, -5, -1):
                print('''{0} with likelyhood of {1}%;
                    '''.format(prob_list[i][0], round(prob_list[i][1], 6)\
                     * 100))   # nopep8: E127
# _________________________________________________
# ________________image_production_________________


def variety_counts_plot(df=pd.DataFrame({'A': []}), n=300, to_png=None):
    """
    Plot the counts of the varieties in the dataframe. If the dataframe is
    given, it will be used, otherwise the raw dataframe will be downloaded
    from the _data_ directory and the varieties with description number =<n
    will be filtered out. The plot is not good if n<300 a warning message
    will be displayed in this case. If the parameter 'to_png' is not a string
    or None, it raises a value error. If None, the figure is printed to the
    console, if a string, the figure will be saved in a png format on
    ./data/[to_png].png

    Parameters
    ----------
    df : a dataframe or None. If None, the raw dataframe will be loaded and
    filtered according to the value of n.

    n : an integer, if df is None, a raw dataframe will be loaded and
    filtered so tha n is the minimum counting a variety must have to be
    considered

    to_png : a string or None. If None, the plot will be displayed in the
    console. If string the plot will be saved in a .png format at
    ./data/[to_png].png, otherwise, it raises an error

    Returns
    -------
    None

    type list
    """
    if n < 300:
        print(""" n={0} may print too many varieties and unformat the plot.
        \n Consider using n > 300 or build your own plot.""".format(n))
    if df.empty:
        df = import_data()
        df = select_number_of_descriptions(df, n=n)
    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.countplot(data=df, y='variety', order=df['variety']\
                .value_counts().index, ax=ax)  # nopep8: E127
    ax.set_ylabel("Wine varieties", fontweight='bold')
    ax.set_xlabel("Count of descriptions", fontweight='bold')
    ax.set_title("Number of records per variety", fontweight='bold',
                 fontsize=15)  # nopep8: E127
    plt.yticks(fontsize=9)
    plt.subplots_adjust(left=0.4, bottom=0.1, right=0.9, top=0.9, wspace=0.1,
                         hspace=0.1)  # nopep8: E127
    if to_png is None:
        plt.show()
    else:
        if type(to_png) is str:
            plt.savefig('./data/' + to_png + '.png')
        else:
            raise ValueError('to_png must be None or a string')
    return None


def plot_points_hist(df=pd.DataFrame({'A': []}), n=1, to_png=None):
    """
    Plot a histogram with KDE of the values in the [points] feature.
    If the dataframe is given, it will be used, otherwise the raw dataframe
    will be downloaded from the _data_ directory and the varieties with
    description number =<n will be filtered out. If the parameter 'to_png'
    is not a string or None, it raises a value error. If None, the figure
    is printed to the console, if a string, the figure will be saved in a
    png format on ./data/[to_png].png

    Parameters
    ----------
    df : a dataframe or None. If None, the raw dataframe will be loaded
    and filtered according to the value of n.

    n : an integer, if df is None, a raw dataframe will be loaded and
    filtered so tha n is the minimum counting a variety must have to be
    considered

    to_png : a string or None. If None, the plot will be displayed in
    the console. If string the plot will be saved in a .png format at
    ./data/[to_png].png, otherwise, it raises an error

    Returns
    -------
    None

    type list
    """
    if df.empty:
        df = import_data()
        df = select_number_of_descriptions(df, n=n)
    df.dropna(subset='price', axis=0, inplace=True)
    df.sort_values(by='price', ascending=False, inplace=True)
    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_ylabel("Probability density", fontweight='bold')
    ax.set_xlabel("Average points given", fontweight='bold')
    ax.set_title("""Histogram of the points \n given to the products
                 \n with KDE""", fontweight='bold', fontsize=15)   # nopep8: E127
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    sns.histplot(data=df, x='points', bins=20, stat='density', kde=True,
                 color='black', fill=True)  # nopep8: E127
    if to_png is None:
        plt.show()
    else:
        if type(to_png) is str:
            plt.savefig('./data/' + to_png + '.png')
        else:
            raise ValueError('to_png must be None or a string')
    return None


def plot_box_points_price_popular(df=pd.DataFrame({'A': []}), n=30,
                                 top=True, to_png=None):  # nopep8: E127
    """
    Plot a boxplot of the price and points distribution group by the points.
    If the dataframe is given, it will be used, otherwise the raw dataframe
    will be downloaded from the _data_ directory. If [top] == True then the
    n first varieties (ordered by the points count) will be plotted. If [top]
    is a integer, than [top-n:top] entries will be plotted. If the parameter
    'to_png' is not a string or None, it raises a value error. If None, the
    figure is printed to the console, if a string, the figure will be saved
    in a png format on ./data/[to_png].png

    Parameters
    ----------
    df : a dataframe or None. If None, the raw dataframe will be loaded and
    filtered according to the value of n.

    n : an integer controling the number of boxes

    top : bool or int, controls where are the plots in the sorted dataframe.
    if True plots the top n. If False, plots the tail n. If int plots the
    [top-n:top] elements. Otherwise raises an error

    to_png : a string or None. If None, the plot will be displayed in the
    console. If string the plot will be saved in a .png format at
    ./data/[to_png].png, otherwise, it raises an error

    Returns
    -------
    None

    type list
    """
    if df.empty:
        df = import_data()
    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    if top is True:
        index = (df.groupby(['variety'])['points'].count()\
                .reset_index(name='Count')\
                .sort_values(['Count'], ascending=False))['variety']\
                .values[0:n]  # nopep8: E127
    elif top is False:
        index = (df.groupby(['variety'])['points'].count()\
                .reset_index(name='Count')\
                .sort_values(['Count'], ascending=False))['variety']\
                .values[-n:-1]  # nopep8: E127
    elif type(top) is int:
        index = (df.groupby(['variety'])['points'].count()\
        .reset_index(name='Count')\
        .sort_values(['Count'], ascending=False))['variety']\
        .values[top - n: top]  # nopep8: E502
    else:
        raise ValueError('top must be a boolean or a integer')
    temp = df[df['variety'].isin(index)]
    df.dropna(subset='price', axis=0, inplace=True)
    f, ax = plt.subplots(2, 1, figsize=(6, 18), sharex=True)
    g1 = sns.boxplot(data=temp, y='price', x='variety', ax=ax[0])
    ax[1].set_xticklabels(ax[1].get_xticks(), rotation=90)
    g2 = sns.boxplot(data=temp, y='points', x='variety', ax=ax[1])
    ax[0].set(yscale="log", xlabel=None)
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel("Price (log scale)", fontweight='bold', fontsize=10)
    ax[0].set_title("Price distribution for the most popular varieties",
                     fontweight='bold', fontsize=10)  # nopep8: E127
    ax[1].set_ylabel("Points", fontweight='bold', fontsize=10)
    ax[1].set_title("Points distribution for the most popular varieties",
                     fontweight='bold', fontsize=10)  # nopep8: E127
    plt.subplots_adjust(left=0.15, bottom=0.3, right=0.9, top=.95,
                         wspace=0.1, hspace=0.15)  # nopep8: E127
    if to_png is None:
        plt.show()
    else:
        if type(to_png) is str:
            plt.savefig('./data/' + to_png + '.png')
        else:
            raise ValueError('to_png must be None or a string')
    return None


def plot_price_points(df=pd.DataFrame({'A': []}), to_png=None):
    """
    Plot a plot and linear regression of the price per point using the
    dataframe. If the dataframe is given, it will be used, otherwise
    the raw dataframe will be downloaded from the _data_ directory. If
    the parameter 'to_png' is not a string or None, it raises a value
    error. If None, the figure is printed to the console, if a string, the
    figure will be saved in a png format on ./data/[to_png].png

    Parameters
    ----------
    df : a dataframe or None. If None, the raw dataframe will be loaded and
    filtered according to the value of n.

    to_png : a string or None. If None, the plot will be displayed in the
    console. If string the plot will be saved in a .png format at
    ./data/[to_png].png, otherwise, it raises an error

    Returns
    -------
    None

    type list
    """
    if df.empty:
        df = import_data()
    temp = df
    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    f, ax = plt.subplots(2, 1, figsize=(7, 7))
    ax[0].set(xscale="log")
    temp['y'] = 78.5 + 2.85 * np.log(temp['price'])
    sns.scatterplot(data=temp, x='price', y='points', color='black',
                     ax=ax[0], s=5, alpha=0.5, label='data')  # nopep8: E127
    sns.lineplot(data=temp, x='price', y='y', color='r', ax=ax[0],
                 label='linear fit')  # nopep8: E127
    sns.scatterplot(data=temp, x='price', y='points', color='black',
                     ax=ax[1], s=5, alpha=0.5, label='data')  # nopep8: E127
    sns.lineplot(data=temp, x='price', y='y', color='r', ax=ax[1],
                 label='log fit')  # nopep8: E127
    ax[0].set_title("Points versus logarithm of the price plot",
                     fontweight='bold', fontsize=12)  # nopep8: E127
    ax[0].set_ylabel("Points", fontweight='bold', fontsize=10)
    ax[0].set_xlabel("Price (log scale)", fontweight='bold', fontsize=10)
    ax[1].set_title("Points versus price plot", fontweight='bold',
                     fontsize=12)  # nopep8: E127
    ax[1].set_ylabel("Points", fontweight='bold', fontsize=10)
    ax[1].set_xlabel("Price", fontweight='bold', fontsize=10)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=.9, wspace=0.1,
                         hspace=0.4)  # nopep8: E127
    plt.legend()
    if to_png is None:
        plt.show()
    else:
        if type(to_png) is str:
            plt.savefig('./data/' + to_png + '.png')
        else:
            raise ValueError('to_png must be None or a string')
    return None


def plot_results_cv_boxplots(df=pd.DataFrame({'A': []}), to_png=None,
                             start=2000, end=8000, step=100,
                             vectorizer_option=0, ngrams_limit=(1, 2)):
    """
    Plot boxplots of the crossvalidation results. If the dataframe is given,
    it will be used, otherwise if is a string indicating a file name, this
    file will be loaded and querid about the right sintaxe. It will raise
    errors if the df is not None, a string ending with .csv or a dataframe.
    If df=None, the raw dataframe will be downloaded from the
    _data_ directory and preporcessed to get the cross validation results
    across diferent minimum number of descriptions. The star, end and the
    step arguments are called in this sitiuation and copose the loop to get
    the data from. If the parameter 'to_png' is not a string or None, it
    raises a value error. If None, the figure is printed to the console,
    if a string, the figure will be saved in a png format on
    ./data/[to_png].png. vectorize_option and ngrams_limit are parameters
    related to the cross validation function.

    Parameters
    ----------
    df : a dataframe or None. If None, the raw dataframe will be loaded
    and filtered according to the value of n.

    to_png : a string or None. If None, the plot will be displayed in the
    console. If string the plot will be saved in a .png format at
    ./data/[to_png].png, otherwise, it raises an error

    Returns
    -------
    None

    type list
    """

    if df.empty:
        df = import_data()
        X, Y, Z = [], [], []
        for size in range(start, end, step):
            temp = select_number_of_descriptions(df, n=size)
            cv_results = model_cv(temp, vectorizer_option=vectorizer_option,
                                 ngrams_limit=ngrams_limit)  # nopep8: E127
            aux = len(list(temp.variety.unique()))
            for i in range(len(cv_results)):
                X.append(size)
                Y.append(cv_results[i])
                Z.append(aux)
        CV = pd.DataFrame({'n_values': X, 'cv_scores': Y, 'n_select_cat': Z})
    elif type(df) == str and df[-4:] == '.csv':
        CV = pd.read_csv(df)
    elif type(df) == 'pandas.core.frame.DataFrame':
        CV = df
    else:
        raise ValueError('unreconized dataframe features.')

    test = list(CV.columns)
    if 'n_values' not in test or 'cv_scores' not in test\
        or 'n_select_cat' not in test:   # nopep8: E127
        raise ValueError('Some of the necessary colunms are missing')

    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=.9, wspace=0.1,
                         hspace=0.2)  # nopep8: E127
    sns.boxplot(data=CV, x='n_values', y='cv_scores', hue='n_select_cat',
                dodge=False, flierprops={"marker": "x"}, ax=ax)
    ax.set_title("Boxplots of the cross \n validation scores",
                 fontweight='bold', fontsize=15)  # nopep8: E127
    ax.set_ylabel("Cross validation scores", fontweight='bold', fontsize=15)
    ax.set_xlabel("Minimum count of descriptions", fontweight='bold',
                 fontsize=15)  # nopep8: E127
    ax.set_xticklabels(list(CV['n_values'].unique()), rotation=45)
    plt.legend(title='Variety')
    if to_png is None:
        plt.show()
    else:
        plt.savefig('./data/' + to_png + '.png')
    return None


def plot_results_lineplots(df=pd.DataFrame({'A': []}), start=500, end=8000,
                            step=300, to_png=None):  # nopep8: E127
    """
    Plot a lineplots of the model results and variety number in function of
     the minimum description number minimum of descriptions. If df is a
     dataframe, it will be used. If it is a string ending in .csv it will
     be loaded. In both cases the dataframe will be tested if present the
     right features. If df == None, then we download the raw data and use it
     to build another dataframe in a loop limited by the parameters start,
     end and step were the results are calculated. Otherwise, it will raise
     an error. if the to_png feature is None, the graph will be ploted in
     the console, if is a string the figure will be saved in a png format
     on ./data/[to_png].png

    Parameters
    ----------
    df : a dataframe a string or None. If None, the raw dataframe will be
     loaded, if is a string ending in .csv representing a file path, the
      file will be loaded and tested for the right format.

    start, end step : integers used in the preprocessing step keyword for.
     It specifies the details
    of the data plotted in the x-axis


    to_png : a string or None. If None, the plot will be displayed in
     the console. If string
    the plot will be saved in a .png format at ./data/[to_png].png,
     otherwise, it raises an error

    Returns
    -------
    None

    type list
    """
    if df.empty:
        df = import_data()
        df = model_results(start=start, end=end, step=step)
    elif type(df) == str and df[-4:] == '.csv':
        df = pd.read_csv(df)
    elif type(df) != 'pandas.core.frame.DataFrame':
        raise ValueError('unreconized dataframe feature')
    sns.set_style('darkgrid')
    sns.set_context(context='talk', font_scale=0.7)
    f, ax = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=.8,
                        wspace=0.3, hspace=0.2)
    sns.lineplot(data=df, x='n_descriptions', y='variety', ax=ax[0])
    sns.lineplot(data=df, x='n_descriptions', y='acc', ax=ax[1], color='r')
    ax[0].set_title("""number of varieties \n per minimum number \n of d
                    escriptions""", fontweight='bold', fontsize=12)
    ax[1].set_title("model score \n per minimum number \n of descriptions",
                     fontweight='bold', fontsize=12)  # nopep8: E127
    ax[0].set_ylabel("Number of different varieties", fontweight='bold',
                     fontsize=12)
    ax[0].set_xlabel("Minimum number of descriptions", fontweight='bold',
                     fontsize=12)
    ax[1].set_ylabel("model accurance's score", fontweight='bold',
                     fontsize=12)
    ax[1].set_xlabel("Minimum number of descriptions", fontweight='bold',
                     fontsize=12)
    if to_png is None:
        plt.show()
    else:
        plt.savefig('./data/' + to_png + '.png')
# _________________________________________________
# _________________________________________________
