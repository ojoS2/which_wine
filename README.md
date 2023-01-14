# which_wine
A portfolio code to display skills related to data-processing, natural language processing, data analysis and machine learning model implementations to build a code that is able predict the type of wine depending on the description and that can predict the desired wine-type based on descriptions inputted by the user.

## How it came to be

The main file for this project was found in a kaggle repository [here](https://www.kaggle.com/datasets/zynicide/wine-reviews) which is dataset about wine containing over 150K descriptions of over 630 grapes varieties (though the majority of these present only one description and therefore not of much use). In the of the producer of such dataset: "*I plan to use deep learning to predict the wine variety using words in the description/review. The model still won't be able to taste the wine, but theoretically it could identify the wine based on a description that a sommelier could give.*" I undertook such task and build a multinomial Naive Bayes to predict the wine-type bassed on the given wine description and expand it a little to also recomendate wine varieties based on the user description of what he wants to taste including sensations, tastes aging, etc, the most information given the more precise the prediction. 

The prediction is possible and precise (the model can have over 95% accurancy in some specific cases) because although each description is written in a somewhat free structure and poetics, some specific patterns always emerge. For example, look at the two description below taken from the record 111640 and 98966 respectivelly:

*Saturated Malbec, with an opaque color and dense, sweet aromas that are slightly floral and pastry-like. Big and rich in the mouth, with bold cassis, blackberry and plum flavors floating on a soft, almost jammy structure. Finishes full and oily in texture. Rich and ready to drink now.*, and;

*The Barolo Rocche dell'Annunziata exhibits bright berry notes followed by delicate tones of violet, slate, wet earth, spice and smoke. In the mouth, this beautiful wine delivers density and concentration capped by bright acidity and drying, firm tannins. Drink after 2015.*

In these description the sensations presents a rich vocabulary and expressions which are terrible for machine learning algorithms. Nevertheless, the overall structure is workable as long as we have plenty descriptions as the same varieties presents similar characteristics such as *an opaque color and dense* or *density and concentration capped by bright acidity and drying*. What I mean is that for the somellier writing down the description, there are some specific topics to describe and the same variety presents similar descriptions. As te descriptions of those characteristics can be presented in a finite number of ways, the more data over a specific variety, the more precise a natural prossessing language model build to predict this variety will be. This also mean that the majority of the varieties will be useless as few of them prsents more than 100 descriptions.


## The examples in the main.py file

To use the code and the examples you must have the file 'inemag-data_first150k.csv' somewhere in your computer and the first thing you will be asked once you run the main.py is the path to this file. You can download it from [here](https://www.kaggle.com/datasets/zynicide/wine-reviews) or use the kaggle API [here](https://www.geeksforgeeks.org/how-to-download-kaggle-datasets-into-jupyter-notebook/). After you input the file path it will be transfered to ./data and from there we beging the analisis. Notice that you should remove or coment the filename entry once the transfer was done. This is the main file of the code and presents three intrinsic functions that uses he code in ./src/tools.py to perform different types of analysis. running the main.py file prints to the console instructions to choose among three options DA (data analysis), DPP (data processing and predictions) or rs (recomendation system). Typing ny other thing will kill the job. The option DA will display a verbose interactive process which will load, preprocess and analyse the data showing the results in graphs. The option dpp will also present a verbose interface where you will be asked to enter some parameters manually or just press enter to use the default values. This function will divide the dataset in train and test sets, fit the model get a sample from the test dataset and print the full description of each record of the sample, the top five varieties probabilities calculated by the model, the variety it belongs according to the dataset and the count of description this variety has for comparison. Finally, the RS option does the same thing except it asks for you to entry descriptions which will be appended to a vector and when you finish inputing descriptions, the model will return each sentence with the varieties with the top five probabilities as recomendations.


##  src folder 
In the *src* folder you can find the main code of this package. In the file tools.py you will find 22 functions divided into three groups based on the nature of their tasks:



### the importing tools group
are a group of functions dedicated to loading and preprocessing the data from the directory /data which is of the same generation as the /src directory.
#### The import_data function
loads the data into a pandas dataframe. It has to arguments which default are 
filename='./data/winemag-data_first150k.csv', and
separator=','. 
#### The column_description function
describes the funtion features. It has one argument which default is 
column='all', which prints the description of all features. 
Typing the columns name it prints only its description
#### The nan_description function
describes the missing values of the data. It has two arguments which defaults are: 
df=emptyDataframe ,and
column='all'
emptyDataframe is an empty pandas dataframe wich if is selected, the data is loaded directely from the \data file, otherwise, the function will use the given dataframe. The colunm feature selects the column to display the information. The 'all' default prints all features 
#### The drop_missing_column function
drops the missing records of a column. It has a mandatory argument:
column, which is the feature to drop the missing; and, it has a optional argument, 
df=emptyDataframe, wich if is selected, the data is loaded directely from the \data file, otherwise, the function will use the given dataframe.
#### The select_number_of_descriptions function
filter the dataframe according to the minimum number of descriptions. It has two arguments which defaults are : 
df=emptyDataFrame, and n=500
If is passes an empty dataframe then the data will be loaded directly from /data directory, otherwise the given dataframe will be used. The parameter n is the minimum number of descriptions the wine variety must have to not be filterd out in this function.


### the processing and modeling group
present functions related to tokenize, fit a machine learn model and predict its results 


#### The model_fit function
fits a Multinomial Naive Bayes model to the data. It takes a mandatory argument:
df, a dataframe where we get the data. And two other arguments which defaults are:
    vectorizer_option=0, which decides the vectorization to use (0 for TfidfVectorizer and 1 for CountVectorizer), and  ngrams_limit=(1, 2), which controls the range of ngrams the TfidfVectorizer will consider
#### The model_cv function
perform a cross validation of the model fitting, it takes the same arguments as the model_fit function
#### The model_tunning function 
perform a grid search to find the best parameters to fit the Multinomial Naive Bayes. In the set tested, I find that alpha=0.001 was the best across many minimum descriptions counts and implemented it in all the function that requires model building. It takes the same arguments as the model_cv function and additionaly, and argument with default n_splits=10 that controls the number of splits in the grid search.
#### The cross_val_data function
produces the cross validation results of the model in a loop varying the minimum number of descriptions to filter the dataframe. It takes three arguments: 
start=1000, end=9000, step=500, which are integers used in the loop
#### The get_lemmatized_phrases function
produces lemmatized versions of a vector of phrases it takes as argument. It takes one mandatory argument:
phrases, which is a vector of sentences to be lemmatized. In the lemmatization, all phrases are transformed in lower case and english stop words are ignored.
#### The predict_and_print_probabilities function
filters the dataset according to the parameter min_descriptions=100, which controls the minimum number of descriptions to consider the variety to be valid, build a model with this data, separates sample_size=100 descriptions from the test set and print out the description, its top five probabilities to belong to a variety togheter with the variety 
#### The predict_and_print_recomendations function
filters the dataset according to the parameter min_descriptions=100 and fits the model to the whole filtered dataset and ask the user to fill in a vector of descriptions. After the user end filling the vector, the code use it to predict the wine variety the user was refeering to in each sentence of the vector and display the top five probabilities.


### the image production group
Are functions to plot image aids of the data processing and presentation of results steps 
#### The variety_counts_plot function
Plot the varieties' counts in the dataframe. It takes three optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
n=300: the minimum number of descriptions to consider the variety valid
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
It returns None

#### The plot_points_hist function
Plot the a histogram with KDE of the points feature distribution. It takes three optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
n=1: the minimum number of descriptions to consider the variety valid
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
It returns None


#### The plot_box_points_price_popular function
Plot boxplots of both the price and the points of each variety. It takes four optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
n=30: the number of boxes to plot
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
top=True: if True, the function plots the top n plots  
It returns None

#### The plot_price_points function
Plot a regression of the price versus points features in two frames, the first displays the points versus log price to show the linera character of this relashionship and the second frame the axis are the usual but the regression is logarithmic. It takes two optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
It returns None

#### The plot_results_cv_boxplots function
Plot boxplots of the cross validations results taken from a loop over the minimum description counts. It takes seven optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
vectorizer_option=0: it decides the vectorization to use (0 for TfidfVectorizer and 1 for CountVectorizer)
ngrams_limit=(1, 2): controls the range of ngrams the TfidfVectorizer will consider
start=2000: a integer controling where the loop over the minimum description number begins
end=8000: a integer controling where the loop over the minimum description number ends
step=100: the step of the loop
It returns None

#### The plot_results_lineplots function
Plot lineplots of the results of the model and the number of available varieties as function of the minimum number of descriptions. It takes five optional arguments:
df = emptyDataFrame: The dataframe. It is downloaded from the \data deirectory if its empty but used otherwise
to_png=None: if None, plot the figure to the console, else, if it is a string, save the figure in the local ./data/to_png.png where to_png is the string given
start=2000: a integer controling where the loop over the minimum description number begins
end=8000: a integer controling where the loop over the minimum description number ends
step=100: the step of the loop
It returns None


##### Credits : [Kaggle: Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)



