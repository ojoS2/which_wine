# which_wine
A portfolio code to display skills related to data-processing, natural language processing, data analysis and machine learning model implementations to build a code that is able predict the type of wine depending on the description and that can predict the desired wine-type based on descriptions inputted by the user.

## How it came to be

The main file for this project was found in a kaggle repository [here](https://www.kaggle.com/datasets/zynicide/wine-reviews) which is dataset about wine containing over 150K descriptions of over 630 grapes varieties (though the majority of these present only one description and therefore not of much use). In the of the producer of such dataset: "*I plan to use deep learning to predict the wine variety using words in the description/review. The model still won't be able to taste the wine, but theoretically it could identify the wine based on a description that a sommelier could give.*" I undertook such task and build a multinomial Naive Bayes to predict the wine-type bassed on the given wine description and expand it a little to also recomendate wine varieties based on the user description of what he wants to taste including sensations, tastes aging, etc, the most information given the more precise the prediction. 

The prediction is possible and precise (the model can have over 95% accurancy in some specific cases) because although each description is written in a somewhat free structure and poetics, some specific patterns always emerge. For example, look at the two description below taken from the record 111640 and 98966 respectivelly:

*Saturated Malbec, with an opaque color and dense, sweet aromas that are slightly floral and pastry-like. Big and rich in the mouth, with bold cassis, blackberry and plum flavors floating on a soft, almost jammy structure. Finishes full and oily in texture. Rich and ready to drink now.*, and;

*The Barolo Rocche dell'Annunziata exhibits bright berry notes followed by delicate tones of violet, slate, wet earth, spice and smoke. In the mouth, this beautiful wine delivers density and concentration capped by bright acidity and drying, firm tannins. Drink after 2015.*

In these description the sensations presents a rich vocabulary and expressions which are terrible for machine learning algorithms. Nevertheless, the overall structure is workable as long as we have plenty descriptions as the same varieties presents similar characteristics such as *an opaque color and dense* or *density and concentration capped by bright acidity and drying*. What I mean is that for the somellier writing down the description, there are some specific topics to describe and the same variety presents similar descriptions. As te descriptions of those characteristics can be presented in a finite number of ways, the more data over a specific variety, the more precise a natural prossessing language model build to predict this variety will be. This also mean that the majority of the varieties will be useless as few of them prsents more than 100 descriptions.


## The examples in the main.py file




##### Credits : [Kaggle: Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)



