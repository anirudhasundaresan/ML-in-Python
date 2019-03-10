# A set of guidelines for working through any dataset - I'll keep adding to these once in a while
'''
Links for studying:
- https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420
- https://towardsdatascience.com/a-complete-machine-learning-project-walk-through-in-python-part-two-300f1f8147e2
- https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-three-388834e8804b
- https://www.accelebrate.com/blog/pandas-monium-analyzing-data-with-the-python-pandas-package/
- https://www.accelebrate.com/blog/fraud-detection-using-python/
- https://www.accelebrate.com/blog/using-defaultdict-python/
- https://hackernoon.com/overview-of-exploratory-data-analysis-with-python-6213e105b00b
- https://www.kdnuggets.com/2017/07/exploratory-data-analysis-python.html
- https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e
- https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python (detailed and prob most informative)
- https://www.kaggle.com/pavansanagapati/a-simple-tutorial-on-exploratory-data-analysis (good)
- https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68
- https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
- https://www.commonlounge.com/discussion/ba07d359df5b4323a1219a69b6e0b827
- https://spandan-madan.github.io/DeepLearningProject/ (do this for sure)
- https://www.saedsayad.com/data_mining_map.htm (good resource - like a book)
- https://cdn.oreillystatic.com/en/assets/1/event/278/Bighead_%20Airbnb_s%20end-to-end%20machine%20learning%20platform%20Presentation.pdf (message people in this pdf)
- https://towardsdatascience.com/a-data-science-for-good-machine-learning-project-walk-through-in-python-part-two-2773bd52daf0
- https://towardsdatascience.com/master-python-through-building-real-world-applications-part-1-b040b2b7faad (implementing a dictionary)
- https://towardsdatascience.com/cat-or-dog-introduction-to-naive-bayes-c507f1a6d1a8 (Naive Bayes understand)
- https://www.dataquest.io/blog/pandas-big-data/ (how to work with big data)
- https://www.kaggle.com/frankherfert/tips-tricks-for-working-with-large-datasets (‘’)

Main steps:
- Data cleaning, formatting, care of missing values (imputation, ffill, bfill, removing)
- EDA
- feature engineering/ selection
- establish baseline; compare ML models with performance metrics
- hyperparameter tuning on best models
- evaluate best model on test set
- interpret the model results
- draw conclusions and document work
'''

# DATA CLEANING
# reading in a dataframe
data = pd.read_csv("filename/ path.csv")

# display the top 5 rows of the dataframe - to get a feel of the data
data.head()
# taking 5 random samples
ufo_df.sample(5)

# when you are focusing on interpretability, you need to know what the variables are in order to make sense of your model
# thus, start EDA by understanding what your variables mean - wikipedia, blogs, pdfs, reddit

# see column data types and non-missing values
data.info()
# analyze which are Python objects and which would need to be converted to floats - id categorical variables
ufo_df.dtypes
ufo_df.describe()

# check out some eye-catching stuff
ufo_df[ufo_df['year'] < 1900].count()  # 24

# remove some cols which you do not want?
ufo_df.drop(index=bad_dates.index, inplace=True)

# incorrect datatypes - check if all numbers are actually numbers and not strings
data[col] = data[col].astype(float)

# replace all 'not available' to np.nan
data = data.replace({'Not Available': np.nan})

# remove duplicate rows, if any
movies_df = movies_df.drop_duplicates(keep='first')

# total number of nans in the whole dataframe
df.isnull().sum().sum()
# https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe/39734251#39734251
# if you want number of missing values per column
df.isna().sum()

# if col has high percentage of missing values, not a useful col - remove it (depends on the problem)
# for each column, see if there are outliers - remove it if outliers pose a risk to model building process
# sometimes, outliers are valuable data points - and maybe sometimes, they are due to typos
# https://www.theanalysisfactor.com/outliers-to-drop-or-not-to-drop/
# https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
# Imputation - do it with domain knowledge


# EDA - open ended process
# detect trends, anomalies, patterns and relationships within the data
expensive_movies_df = movies_df.sort_values(by ='budget', ascending=False).head()

# check distribution of all the columns - all features, and especially the target variable
# see if anything suspicious pops out - see what units the target variable is in - reason it out
ufo_df['month'].hist(bins=12)

# linear interpolation
df.interpolate()

# main goal of EDA is to look for relationships between the features; you must be aware of how multi-collinearity might arise
# density plot for categorical variables vs. target variables using seaborn

# for relationships between variables, plot the Pearson Correlation Coefficient
correlations_data = data.corr()['<target_variable>'].sort_values()
# try making sense of these and note their ranks

# use scatter plots to see 2-variable plots - use color of categories to better understand
# use pairs plot from seaborn - to see all variables against each other
grid = sns.PairGrid(data = plot_data, size = 3)

# FEATURE ENGINEERING AND SELECTION
# feature engg - developing transformations of the data - natural log & transformations
# also includes converting categorical variables to one-hot-encoded - and adding in new cols
movies_df['profit'] = movies_df['revenue'] - movies_df['budget']

# ask questions - what all movies rated above 7.0?
movies_df[movies_df['vote_average'] >= 7.0]

# which year did we have most profitable movies?
profits_year = movies_df.groupby('release_year')['profit'].sum()

# to get correlations between multiple features:
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

# contingency tables:
pd.crosstab(df['Churn'], df['International plan'])

# is data in a col normally distributed?
# https://www.youtube.com/watch?v=okjYjClSjOg

# feature selection - choosing only the most relevant features from the data
# helps to generalize to new data and create a mpre interpretable model
# http://blog.minitab.com/blog/understanding-statistics/handling-multicollinearity-in-regression-analysis
categorical_subset = pd.get_dummies(categorical_subset)
# we need to avoid multicollinearity when it is between features only!
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on/43104383#43104383
# https://www.statisticshowto.datasciencecentral.com/variance-inflation-factor/

# BASELINE
# guess the median of the target - in the case of regressions
# choose the error metric - MAE, RMSE, etc
# Try out many - see which ones can give you a predict_proba and which can give you feature importances
# https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime
