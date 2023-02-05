import unittest
import io
import contextlib
import pandas as pd 
# importing sys
import sys
sys.path.insert(0, '/home/ricardo/Desktop/WineSoilRepo/whic_wine_2/tools')
import tools


class TestTools(unittest.TestCase):
  
  
  def test_import_data(self):
      df = tools.import_data()
      self.assertEqual(df.empty, False)
      self.assertEqual(list(df.columns), ['country', 'description',
          'designation','points', 'price', 'province', 'region_1',
            'region_2', 'variety', 'winery'])
      self.assertEqual(df.shape, (150930, 10))


  def test_column_description(self):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
          tools.column_description(column = 'country')
      temp = buf.getvalue()
      self.assertEqual(temp, 'Column [country] description: The country that the wine is from\n\n\n')
      buf.close()


  def test_nan_description(self):
    df = tools.import_data()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tools.nan_description(df, column = 'country')
    temp_0 = buf.getvalue()
    temp_1 = 'The feature country presents '+str(df[['country']].isnull().sum()[0])+' missing values corresponding to '+str(round(df[['country']].isnull().sum()[0]/df.shape[0]*100,4))+'% of the entries\n\n\n'
    self.assertEqual(temp_0,temp_1)
    df=df[['country']].dropna()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tools.nan_description(df, column = 'country')
    temp_0 = buf.getvalue()
    temp_1 = 'No missing found in feature country\n'
    self.assertEqual(buf.getvalue(),'No missing found in feature country\n\n\n')
    buf.close()
    buf = io.StringIO()


  def test_drop_missing_column(self):
    df = tools.import_data()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      tools.drop_missing_column(column = 'country', df=df)
    self.assertEqual(buf.getvalue(),'5 rows, or 0.003% of the data frame rows were deleted.\n')
    df=df[['country']].dropna()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tools.drop_missing_column(column = 'country', df=df)
    self.assertEqual(buf.getvalue(),'The column country do not have any missing to drop.\n')
    buf.close()
    buf = io.StringIO()


  def test_select_number_of_descriptions(self):
      df = tools.import_data()
      self.assertEqual(tools.select_number_of_descriptions(df, n=-1).shape,(150930, 10))
      self.assertEqual(tools.select_number_of_descriptions(df, n=0).shape,(150930, 10))
      self.assertEqual(tools.select_number_of_descriptions(df, n=50).shape,(146491, 10))
      self.assertEqual(tools.select_number_of_descriptions(df, n=10000).shape,(51635, 10))
      self.assertEqual(tools.select_number_of_descriptions(df, n=20000).shape,(0, 10))


  def test_get_lemmatized_phrases(self):
    phrases = ['Hello! Howdy p#artner test this, but why?',
                'Goo%ddy spaces are alowed?',
                  '100 pushups brings 100% d_own belly pain', '']
    temp = tools.get_lemmatized_phrases(phrases=phrases)
    assert temp == ['hello howdy test', 'space alowe', 'pushup bring belly pain', '']
    assert len(temp) == len(phrases)


  def test_model_fit(self):
    df = tools.import_data()
    df = tools.select_number_of_descriptions(df, n=500)
    accurancy, Confusion_matrix, model = tools.model_fit(df)
    self.assertTrue(abs(accurancy-0.7708160338189779) <= 0.001 )
    self.assertEqual(Confusion_matrix.shape,(42, 42))
    self.assertEqual(str(model),'MultinomialNB(alpha=0.001)')
    df = tools.select_number_of_descriptions(df, n = 5000)
    accurancy, Confusion_matrix, model = tools.model_fit(df)
    self.assertTrue(abs(accurancy-0.842395072806624) <= 0.001) 
    self.assertEqual(Confusion_matrix.shape,(9, 9))
    df = tools.select_number_of_descriptions(df, n = 10000)
    accurancy, Confusion_matrix, model = tools.model_fit(df)
    self.assertTrue(abs(accurancy-0.9209863791879156) <= 0.001)
    self.assertEqual(Confusion_matrix.shape,(4, 4))
  
  
  def test_model_cv(self):
    df = tools.import_data()
    df = tools.select_number_of_descriptions(df, n = 500)
    cv_r = tools.model_cv(df, vectorizer_option = 0, ngrams_limit = (1,2))
    self.assertEqual(len(cv_r),10)
    self.assertEqual(cv_r,[0.8201857024231901, 0.8102966709443647,
     0.8082584736166679, 0.8096172718351324, 0.8146750207594172,
       0.8138446440703555, 0.8195063033139579, 0.8062202762889711,
         0.8171523478786049, 0.8075645477880115])
    df = tools.select_number_of_descriptions(df, n = 5000)
    cv_r = tools.model_cv(df, vectorizer_option = 0, ngrams_limit = (1,2))
    self.assertEqual(len(cv_r),10) 
    self.assertEqual(cv_r,[0.8736082221950324, 0.8656387665198237,
     0.8638032305433186, 0.8704111600587372, 0.8753059226627509, 
     0.8734703866862458, 0.8672295643661282, 0.8666177190406266, 
     0.8691874694077337, 0.872858541360744])
    df = tools.select_number_of_descriptions(df, n = 10000)
    cv_r = tools.model_cv(df, vectorizer_option = 0, ngrams_limit = (1,2))
    self.assertEqual(len(cv_r),10)
    self.assertEqual(cv_r,[0.9399690162664601, 0.9440356312935708,
     0.9386134779240899, 0.9370642912470952, 0.936289697908598,
       0.9432500484214604, 0.9418942475305055, 0.9447995351539803,
         0.937052101491381, 0.9424753050552005])


  def test_model_prob_tab(self):
    df = tools.import_data()
    df = tools.select_number_of_descriptions(df, n = 8000) 
    prob_tab = tools.model_prob_tab(df, vectorizer_option = 0, ngrams_limit = (1,2))
    for _, row in prob_tab.iterrows(): 
      self.assertTrue((row['Probability_Sum'] - 1.0) < 0.0001)
    df = tools.import_data()
    df = tools.select_number_of_descriptions(df, n = 4000) 
    prob_tab = tools.model_prob_tab(df, vectorizer_option = 0, ngrams_limit = (1,2))
    for _, row in prob_tab.iterrows(): 
      self.assertTrue((row['Probability_Sum'] - 1.0) < 0.0001)

  
  def test_model_results(self):
    tools.model_results(start = 3000, end = 4000, step = 200,filename='results')
    df = pd.read_csv('./data/results.csv')
    n_descriptions = [3000, 3200, 3400, 3600, 3800]
    variety = [12, 12, 10, 10, 9]
    acc = [0.8224603576859025, 0.8224603576859025, 0.8333333333333334, 0.8333333333333334, 0.8423950728066249]
    self.assertEqual(list(df['n_descriptions'].values), n_descriptions)
    self.assertEqual(list(df['variety'].values), variety)
    #print(abs(list(df['acc'].values) - acc))
    #self.assertTrue(abs(list(df['acc'].values) - acc) <0.001)


  def test_model_cross_validation_tab(self):
    tools.model_cross_validation_tab(start = 3000, end = 4000, step = 300)
    df=pd.read_csv('./data/cv_results.csv')
    cv_score = [0.8473066898349262, 0.8546915725456126, 0.8489357080799305,
                 0.8556533072662105, 0.8530465949820788, 0.8539154990767894,
                   0.8556533072662105, 0.8543499511241447, 0.8560877593135657,
                     0.8531552079939176, 0.8570946326094295, 0.8540564870034882,
                       0.8591200630133904, 0.8629458759986497, 0.8621582086193316,
                         0.8639432815665091, 0.8582039162727887, 0.8638307449921224,
                           0.8620301598019356, 0.8611298672068423, 0.8615528531337698,
                             0.8619036482694107, 0.862488306828812, 0.8652946679139383,
                               0.866931711880262, 0.8647100093545369, 0.8710243217960711,
                                 0.8657623947614593, 0.8652946679139383, 0.8650608044901777,
                                   0.8736082221950324, 0.8656387665198237, 0.8638032305433186,
                                     0.8704111600587372, 0.8753059226627509, 0.8734703866862458,
                                       0.8672295643661282, 0.8666177190406266, 0.8691874694077337,
                                         0.872858541360744]
    n_values = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                 3000, 3000, 3300, 3300, 3300, 3300, 3300, 3300,
                   3300, 3300, 3300, 3300, 3600, 3600, 3600, 3600,
                     3600, 3600, 3600, 3600, 3600, 3600, 3900, 3900,
                       3900, 3900, 3900, 3900, 3900, 3900, 3900, 3900]
    n_select_cat = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11,
                     11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10,
                       10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    self.assertEqual(list(df.cv_scores), cv_score)
    self.assertEqual(list(df.n_values), n_values)
    self.assertEqual(list(df.n_select_cat), n_select_cat)

  
  def test_model_tunning(self):
    df = tools.import_data()
    df = tools.select_number_of_descriptions(df, n=5000)
    best_params, best_score, df_cv_results = tools.model_tunning(df, vectorizer_option=0)
    self.assertTrue(abs(best_params['alpha'] - 0.001) < 0.0001)
    self.assertTrue(abs(best_score - 0.8271625241564937) < 0.0001)












if __name__ == '__main__':
    unittest.main()