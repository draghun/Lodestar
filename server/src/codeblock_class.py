class CodeBlock:

    def load_dataset(self, filename):
        import os

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/../../data/', filename + ".csv")
        df = pd.read_csv(data_path)
        return df

    def save_bytes_image(self, image_list):
        import io
        import matplotlib.pyplot as plt
        import base64

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        image_list.append(base64.b64encode(bytes_image.getvalue()))
        bytes_image.seek(0)

    """ mandatory """
    def performance_metric(self, y_true, y_predict):
        from sklearn.metrics import r2_score

        """ Calculates and returns the performance score between
            true and predicted values based on the metric chosen. """
        """ TODO: Calculate the performance score between 'y_true' and 'y_predict' """
        score = r2_score(y_true,y_predict)

        """ Return the score """
        return score

    """ mandatory """
    def fit_model(self, X, y):
        """ Performs grid search over the 'max_depth' parameter for a
            decision tree regressor trained on the input data [X, y]. """

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import ShuffleSplit

        """ Create cross-validation sets from the training data """
        cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

        """ TODO: Create a decision tree regressor object """
        regressor = DecisionTreeRegressor()

        """ TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10 """
        params = {'max_depth': range(1,11)}

        """ TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' """
        scoring_fnc = make_scorer(self.performance_metric)

        """ TODO: Create the grid search object """
        grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)

        """ Fit the grid search object to the data to compute the optimal model """
        grid = grid.fit(X, y)

        """ Return the optimal model after fitting the data """
        return grid.best_estimator_

    """ mandatory """
    def initializeClustersForKmeans(self, df):
        '''Use k-means++ to initialize a good set of centroids'''
        from pandas.api.types import is_numeric_dtype
        from sklearn.metrics import pairwise_distances
        import numpy as np
        import pandas as pd
        import sys

        k = 50
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        centroids = np.zeros((k, len(quantitativeColumns)))
        data = df[quantitativeColumns].values

        """ Randomly choose the first centroid. """
        """ Since we have no prior knowledge, choose uniformly at random """
        idx = np.random.randint(data.shape[0])
        centroids[0] = data[idx,:]
        """ Compute distances from the first centroid chosen to all the other data points """
        squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2

        for i in xrange(1, k):
            """ Choose the next centroid randomly, so that the probability for each data point to be chosen """
            """ is directly proportional to its squared distance from the nearest centroid. """
            """ Roughtly speaking, a new centroid should be as far as from ohter centroids as possible. """
            idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
            centroids[i] = data[idx,:]
            """ Now compute distances from the centroids to all data points """
            squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)

        final = {}
        for i,c in enumerate(quantitativeColumns):
            final[c] = centroids[:,i]
        return pd.DataFrame(final)

    """ mandatory """
    def calcConditionalFreqDist(self, df):
        import nltk
        import pandas as pd        
        
        nltk.download('brown')

        """ words = ['can', 'could', 'may', 'might', 'must', 'will']"""
        words = (df.select_dtypes(include='object').values).ravel()
        genres = ['adventure', 'romance', 'science_fiction']

        cfdist = nltk.ConditionalFreqDist(
                    (genre, word)
                    for genre in genres
                    for word in nltk.corpus.brown.words(categories=genre)
                    if word in words)

        data = {'conditionalDist': cfdist}
        return pd.DataFrame(data)

    """ mandatory"""
    def calcWordVec(self, df):
        from keras.preprocessing.text import Tokenizer
        from gensim.models import word2vec
        import numpy as np
        import pandas as pd

        texts = df.select_dtypes(include='object')

        MAX_NB_WORDS = 5000
        EMBEDDING_DIM = 100

        tokenize = Tokenizer(nb_words=MAX_NB_WORDS)
        """ sequences = tokenize.texts_to_sequences(texts)"""
        word_index = tokenize.word_index

        nb_words = min(MAX_NB_WORDS, len(word_index))+1

        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
        """ print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))"""

        data = {"wordvec":[]}
        data['wordvec'].append(np.sum(np.sum(embedding_matrix, axis=1) == 0))
        return pd.DataFrame(data)

    def anova_variance(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        from pandas.api.types import is_numeric_dtype
        from pandas.api.types import is_string_dtype
        from scipy import stats
        
        current_df = loaded_dataset

        category_cols = current_df.select_dtypes(include='object').columns
        numerical_cols = current_df.select_dtypes(include='number').columns

        res_df = pd.DataFrame(columns=category_cols, index=numerical_cols)

        if len(category_cols) == 0 or len(numerical_cols) == 0:

                res = {
                    'output': "Dataframe contained incorrect values",
                    'result' : "Dataframe contained incorrect values",
                    'description' : "Dataframe contained incorrect values",
                    'type' : 'error'
                }
                return res

        for num_col in numerical_cols:
            """check to make sure num_col has all numeric values:"""
            if is_numeric_dtype(current_df[num_col]) != True:
                res = {
                    'output': "Illegal dataframe value num_col",
                    'result' : "Illegal dataframe value",
                    'description' : "Illegal dataframe value",
                    'type' : 'error'
                }
                return res

            for cat_col in category_cols:
                """assuming this is checking for strings"""
                if is_string_dtype(current_df[cat_col]) != True:
                    res = {
                    'output': "Illegal dataframe value cat_col",
                    'result' : "Illegal dataframe value",
                    'description' : "Illegal dataframe value",
                    'type' : 'error'
                    }

                    """print (res['output'])"""
                    return res

                if current_df[cat_col].value_counts().count() <= 10:
                    groups = current_df.groupby(cat_col).groups.keys()
                    """print groups"""
                    """print current_df[current_df[cat_col] == groups[0]][num_col]"""
                    if len(groups) >= 3:
                        f_val, p_val = stats.f_oneway(current_df[current_df[cat_col] == groups[0]][num_col], current_df[current_df[cat_col] == groups[1]][num_col], current_df[current_df[cat_col] == groups[2]][num_col])
                        res_df[cat_col][num_col] = p_val

        res = {
            'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
            'result' : res_df.round(3).to_json(orient='table'),
            'description' : description,
            'type' : method
        }

        intermediate_df.append(res_df.round(3))

        return res

    def bootstrap(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        from sklearn.utils import resample
        import numpy as np

        """get columns that have numerical values"""
        """check dtype of all values in dataframe"""

        current_df = loaded_dataset.select_dtypes('number')

        if current_df.empty == True:
            res = {
                'output': "Dataframe has no numeric values",
                'result' : "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : 'error'
            }
            return res

        mean = []
        statistics = pd.DataFrame()

        for i in range(0, 1000):
            boot = resample(current_df, replace=True, n_samples=int(0.5 * len(current_df.index)))
            mean.append(boot.mean().to_dict())
        for key in mean[0]:
            curr_list = [item[key] for item in mean]
            alpha = 0.95
            p = (1.0-alpha) * 100
            lower = np.percentile(curr_list, p)
            p = alpha * 100
            upper = np.percentile(curr_list, p)
            statistics[key] = [str(round(lower, 3)) + '-' + str(round(upper, 3))]
        res = {
            'output' : current_df.head(10).round(3).to_json(orient='table'),
            'result' : statistics.to_json(orient='table'),
            'description' : description,
            'type' : method
        }
        intermediate_df.append(current_df.head(10).round(3))
        return res

    def cat_boxplot(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        import seaborn as seaborn

        df = loaded_dataset

        numerical_cols = df.select_dtypes(include='number').columns
        """print(numerical_cols)"""
        category_cols = df.select_dtypes(include='object').columns
        """print(category_cols)"""

        if len(category_cols) == 0 or len(numerical_cols) == 0:
            res = {
                'output': "Dataframe contained incorrect values",
                'result' : "Dataframe contained incorrect values",
                'description' : "Dataframe contained incorrect values",
                'type' : 'error'
            }
            return res

        image_list = []
        for cat_var in category_cols:
            if df[cat_var].value_counts().count() <= 5:
                for num_var in numerical_cols:
                    plt.clf()
                    ax = seaborn.boxplot(x=cat_var, y=num_var, data=df)
                    plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
                    plt.xticks(rotation=45)
                    self.save_bytes_image(image_list)
                    if len(image_list) >= 5:
                        break
            if len(image_list) >= 5:
                break
        res = {
            'output' : df.head(10).round(3).to_json(orient='table'),
            'result' : image_list,
            'description' : description,
            'type' : method
        }
        intermediate_df.append(df.head(10).round(3))
        return res

    def cat_count(self, loaded_dataset, intermediate_df, description, method):
        from pandas.api.types import is_string_dtype
        import seaborn

        df = loaded_dataset
        category_df = df.select_dtypes(include='object')
        """print(category_df)"""

        if category_df.empty == True:
            res = {
                'output': "Dataframe contained incorrect values",
                'result' : "Dataframe contained incorrect values",
                'description' : "Dataframe contained incorrect values",
                'type': "error"
            }
            return res

        image_list = []

        category_df = category_df.dropna(axis='columns')
        """print("new DF \n", category_df)"""
        for col in category_df:
            """check to make sure 'object' type is actually a string - assuming this is what is needed"""
            if is_string_dtype(category_df[col]) != True:
                    res = {
                    'output': "Illegal dataframe value",
                    'result' : "Illegal dataframe value",
                    'description' : "Illegal dataframe value",
                    'type': 'error'
                    }
                    return res

            if category_df[col].value_counts().count() <= 20:
                seaborn.catplot(x=col, data=category_df, alpha=0.7, kind='count')
                self.save_bytes_image(image_list)
        res = {
            'output' : category_df.head(10).round(3).to_json(orient='table'),
            'result' : image_list,
            'description' : description,
            'type' : method
        }
        intermediate_df.append(category_df.head(10).round(3))
        return res

    def compute_covariance_matrix(self, loaded_dataset, intermediate_df, description, method):
        import numpy as np
        import pandas as pd 
        import sys
        
        df_initial = loaded_dataset

        if df_initial is None:
            res = {
                'result': "Null dataframe needs numeric values",
                'output': "Null dataframe needs numeric values",
                'description': "Null dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        df = loaded_dataset.select_dtypes(include='number')

        if df.empty == True or df.isnull().values.all() == True:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : 'error'
            }
            return res

        """ convert numerical columns in df to matrix representation"""

        df_matrix = df.as_matrix() 
        """ This usage is set to be deprecated soon"""

        data = {"CovMeanDot":[]}
        covariance = np.cov(df_matrix)
        mean = np.mean(df_matrix, axis=0)
        inv = np.linalg.inv(covariance)

        """ checks if covariance matrix is singular or not"""
        if not (np.linalg.cond(covariance) < 1/sys.float_info.epsilon):
        	"""handle it"""
        	res = {
        		'output': "Matrix is singular",
        		'result': "Matrix is singular",
        		'description' : "Matrix is singular",
        		'type' : 'error'
        	}

        	return res
        else:
            inv = np.linalg.inv(covariance)

        dot = np.dot(np.dot(mean, inv), mean)
        data["CovMeanDot"].append(dot)
        res = {
            'output': pd.DataFrame(data).to_json(orient='table'),
            'result': pd.DataFrame(data).to_json(orient='table'),
            'description' : description,
            'type': method
        }

        intermediate_df.append(pd.DataFrame(data))
        return res

    def compute_percentiles_range(self, loaded_dataset, intermediate_df, description, method):
        from pandas.api.types import is_numeric_dtype
        import pandas as pd 
        import numpy as np

        lowerPercentile = 5
        upperPercentile = 95
        df = loaded_dataset
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if len(quantitativeColumns) == 0:
            res = {
                'output': "Dataframe needs numeric values",
                'result': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        data = {"Percentile"+str(lowerPercentile):[],"Percentile"+str(upperPercentile):[],"columnName":quantitativeColumns}
        for c in quantitativeColumns:
            data["Percentile"+str(lowerPercentile)].append(np.percentile(df[[c]],lowerPercentile))
            data["Percentile"+str(upperPercentile)].append(np.percentile(df[[c]],upperPercentile))
        res = {
            'output': pd.DataFrame(data).to_json(orient='table'),
            'result': pd.DataFrame(data).to_json(orient='table'),
            'description' : description,
            'type': method
        }
        intermediate_df.append(pd.DataFrame(data))
        return res

    def conditional_frequence_distribution(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset.select_dtypes(include='object')

        if df.empty == True:
            res = {
                'output': "Dataframe has no object values",
                'result' : "Dataframe has no object values",
                'description' : "Dataframe has no object values",
                'type' : "error"
            }
            return res

        res_df = calcConditionalFreqDist(df)

        res = {
            'output': df.head(10).to_json(orient='table'),
            'result': res_df.head(10).to_json(orient='table'),
            'description' : description,
            'type': method
        }
        intermediate_df.append(df.head(10))
        return res

    def corr_heatmap(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt
        import seaborn 

        df = loaded_dataset

        corr = df.select_dtypes(include='number').corr()
        image_list = []
        plt.clf()

        seaborn.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,annot=True, annot_kws={"size": 8}, square=True)
        self.save_bytes_image(image_list)
        res = {
            'output' : corr.round(3).to_json(orient='table'),
            'result' : image_list,
            'description' : description,
            'type' : method
        }
        intermediate_df.append(corr.round(3))
        return res

    def corr(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset

        if df is None:
            res = {
                'result': "Null dataframe needs numeric values",
                'output': "Null dataframe needs numeric values",
                'description': "Null dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        try:
            numerical_df = df.select_dtypes(include='number')
        except ValueError:
            res = {
                'result': "Dataframe needs numeric values",
                'output': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        if (numerical_df.empty == True or df.isnull().values.all() == True):
            res = {
                'result': "Dataframe needs numeric values",
                'output': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }

            return res

        correlations = numerical_df.corr()
        res = {
            'result' : correlations.round(3).to_json(orient='table'),
            'output' : correlations.round(3).to_json(orient='table'),
            'description' : description,
            'type' : method
        }

        intermediate_df.append(correlations.round(3))
        return res

    def decision_tree_classifier(self, loaded_dataset, intermediate_df, description, method):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset

        if df is None:
            res = {
                'output': "Null Dataframe needs numeric values",
                'result' : "Null Dataframe needs numeric values",
                'description' : "Null Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
        if len(quantitativeColumns) <= 1 or df.isnull().values.all() == True:
            res = {
                'output': "Dataframe needs numeric values",
                'result' : "Dataframe needs numeric values",
                'description' : "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        res = {}
        try:
            X = df[quantitativeColumns[:-1]]
            y = df[[quantitativeColumns[-1]]].values.ravel()
            classifier = DecisionTreeClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_test)
            data = {'accuracyScore': []}
            data['accuracyScore'].append(accuracy_score(y_test, prediction, normalize=False))
            res = {
                'output': pd.DataFrame(data).head(10).to_json(orient='table'),
                'result': pd.DataFrame(data).head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(data).head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result' : str(e),
                'description' : str(e),
                'type' : 'error'
            }
        finally:
            return res

    def decision_tree_regressor(self, loaded_dataset, intermediate_df, description, method):
        from sklearn.tree import DecisionTreeRegressor
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset

        tree_reg1 = DecisionTreeRegressor(random_state=42)
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if df.isnull().any().any() == True:
            res = {
                'output': "Dataframe needs numeric values",
                'result': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res
        if len(quantitativeColumns) == 0:
            res = {
                'output': "Dataframe needs numeric values",
                'result': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        X = df[quantitativeColumns[:-1]]
        y = df[[quantitativeColumns[-1]]]
        tree_reg1.fit(X, y)
        y_pred1 = tree_reg1.predict(X)
        out_df = X.copy()
        out_df["Expected-"+quantitativeColumns[-1]] = y
        out_df["Predicted-"+quantitativeColumns[-1]] = y_pred1
        res = {
            'output': out_df.head(10).to_json(orient='table'),
            'result': out_df.head(10).to_json(orient='table'),
            'description' : description,
            'type': method
        }
        intermediate_df.append(out_df.head(10))
        return res

    def demo_hstack(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        import numpy as np
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset

        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if len(quantitativeColumns) == 0:
            res = {
                'output': "Dataframe needs numeric values",
                'result': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type': 'error'
            }
            return res
        x = df[quantitativeColumns[0]].values.ravel()
        y = df[quantitativeColumns[1]].values.ravel()
        x1 = 1 / x
        y1 = 1 / y
        x2, y2 = np.dot(np.random.uniform(size=(2, 2)), np.random.normal(size=(2, len(x))))
        u = np.hstack([x1, x2])
        v = np.hstack([y1, y2])
        res = {
            'output': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
            'result': pd.DataFrame([u, v]).head(10).to_json(orient='table'),
            'description' : description,
            'type': method
        }
        intermediate_df.append(pd.DataFrame([u, v]).head(10))
        return res

    def demo_log_space(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        from sklearn.model_selection import train_test_split
        from pandas.api.types import is_numeric_dtype
        import numpy as np

        res = {}
        try:
            df = loaded_dataset
            quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

            if len(quantitativeColumns) == 0:
                res = {
                    'output': "Dataframe needs numeric values",
                    'result': "Dataframe needs numeric values",
                    'description': "Dataframe needs numeric values",
                    'type' : 'error'
                }
                return res

            X = df[quantitativeColumns[:-1]]
            y = df[[quantitativeColumns[-1]]].values.ravel()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            data = {'svc__C': np.logspace(-3, 2, 6), 'svc__gamma': np.logspace(-3, 2, 6) / X_train.shape[0]}
            res = {
                'output': pd.DataFrame(data).head(10).to_json(orient='table'),
                'result': pd.DataFrame(data).head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(data).head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': method
            }
        finally:
            return res

    def demo_mat_show(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        import pandas as pd 
        from sklearn.metrics import confusion_matrix
        import numpy as np
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset
        image_list = []

        data = {'confusion': []}
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if len(quantitativeColumns) == 0:
            res = {
                'output': "Dataframe needs numeric values",
                'result': "Dataframe needs numeric values",
                'description': "Dataframe needs numeric values",
                'type' : 'error'
            }
            return res

        try:
            yy_test = df[quantitativeColumns[0]].values.ravel()
            yy_pred = df[quantitativeColumns[1]].values.ravel()
            confusion = confusion_matrix(yy_test, yy_pred)
            data['confusion'].append(confusion)
            plt.matshow(confusion)
            plt.title('Confusion matrix')
            plt.gray()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

            self.save_bytes_image(image_list)
            plt.clf()

            invert_colors = np.ones(confusion.shape) * confusion.max()
            plt.matshow(invert_colors - confusion)
            plt.title('Confusion matrix')
            plt.gray()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

            self.save_bytes_image(image_list)

            res = {
                'output': image_list,
                'result': pd.DataFrame(data).head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(data).head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def dist_quant_category(self, loaded_dataset, intermediate_df, description, method):
        import seaborn
        import pandas as pd

        res = {}
        try:
            image_list = []
            clean_dataset = pd.DataFrame()
            df = loaded_dataset

            for col in loaded_dataset.columns:
                if df[col].dtype == 'object' or df[col].value_counts().count() <= 20:
                    clean_dataset[col] = df[col].astype('category')
                else:
                    clean_dataset[col] = df[col]

            numerical_cols = clean_dataset.select_dtypes(include='number').columns
            category_cols = clean_dataset.select_dtypes(include='category').columns

            if (len(numerical_cols) == 0 or len(category_cols) == 0):
                res = {
                    'output': "Dataframe needs numeric AND category values",
                    'result': "Dataframe needs numeric AND category values",
                    'description': "Dataframe needs numeric AND category values",
                    'type' : 'error'
                }
                return res

            count = 0
            for col1, col2 in zip(numerical_cols, category_cols):
                if clean_dataset[col2].value_counts().count() <= 10:
                    seaborn.catplot(col2, col1, data=clean_dataset)
                    self.save_bytes_image(image_list)
                    count+=1
                    if count >= 5:
                        break
            res = {
                'output' : clean_dataset.head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }
            intermediate_df.append(clean_dataset.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': method
            }
        finally:
            return res

    def dist_two_categories(self, loaded_dataset, intermediate_df, description, method):
        import itertools
        import pandas as pd 
        import seaborn 

        res = {}
        try:
            image_list = []
            clean_dataset = pd.DataFrame()
            df = loaded_dataset

            for col in df.columns:
                if df[col].dtype == 'object' or df[col].value_counts().count() <= 20:
                    clean_dataset[col] = df[col].astype('category')

            category_df = clean_dataset.select_dtypes(include='category')

            if category_df.empty == True:
                res = {
                    'output': "Dataframe has no categorical values",
                    'result' : "Dataframe has no categorical values",
                    'description' : "Dataframe has no categorical values",
                    'type' : "error"
                }
                return res

            count = 0
            for col1, col2 in itertools.combinations(category_df.columns, 2):
                if category_df[col1].value_counts().count() <= 10 and			 category_df[col2].value_counts().count() <= 5:
                    seaborn.catplot(col1, data=category_df, hue=col2, kind='count')
                    self.save_bytes_image(image_list)
                    count += 1
                    if count >= 5:
                        break
            res = {
                'output' : category_df.head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }
            intermediate_df.append(category_df.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': method
            }
        finally:
            return res

    def dist_num(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt
        
        res = {}
        try:
            image_list = []
            df = loaded_dataset

            """cols that have numerical values"""
            numerical_df = df.select_dtypes(include='number')

            """checks to see if input df has any numerical values"""
            if numerical_df.empty == True:
                res = {
                    'output': "Dataframe has no numerical values",
                    'result' : "Dataframe has no numerical values",
                    'description' : "Dataframe has no numerical values",
                    'type' : "error"
                }
                return res
            count = 0
            for col in numerical_df:
                fig, ax = plt.subplots()
                ax.hist(numerical_df[col])
                plt.xlabel(col)
                plt.ylabel("Dist")
                plt.title('Histogram of ' + col)
                self.save_bytes_image(image_list)
                count += 1
                if count >= 5:
                    break
            res = {
                'output' : numerical_df.head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }
            intermediate_df.append(numerical_df.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': method
            }
        finally:
            return res

    def drop_cols(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset
        dropped_columns = []
        df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]

        """df2 holds the columns that have more than 30% NaN entries - if empty - algo should be run"""
        for c in df.columns:
            if c not in df2.columns:
                dropped_columns.append(c)

        if len(dropped_columns) == 0:
            res = {
                'output': df.describe().round(3).to_json(orient='table'),
                'result' : df.describe().round(3).to_json(orient='table'),
                'description' : "Dataframe has less than 30% NaN entries",
                'type' : "error"
            }
            return res
        loaded_dataset = df2
        res = {
            'output' : df2.describe().round(3).to_json(orient='table'),
            'result' : df2.describe().round(3).to_json(orient='table'),
            'description' : description,
            'type' : method
        }
        intermediate_df.append(df2.describe().round(3))
        return res

    def drop_rows(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset
        if df.isnull().values.any() == False:
            res = {
                'output': df.head(10).to_json(orient='table'),
                'result' : df.head(10).to_json(orient='table'),
                'description' : "Dataframe has no rows with NaN entries",
                'type' : "error"
            }
            return res

        new_df = loaded_dataset.dropna()

        res = {
            'output' : new_df.head(10).round(3).to_json(orient='table'),
            'result' : new_df.describe().round(3).to_json(orient='table'),
            'description' : description,
            'type' : method
        }
        intermediate_df.append(new_df.head(10).round(3))
        return res

    def eval_model_predictions(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        import numpy as np 

        df = loaded_dataset.select_dtypes(include='number')
        if df.empty == True:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        predictions = df.iloc[:,-1].values
        labels = df.iloc[:,-2].values
        res = {
            'output': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
            'result': pd.DataFrame(np.equal(predictions,labels)).to_json(orient='table'),
            'description' : description,
            'type': method
        }
        intermediate_df.append(pd.DataFrame(np.equal(predictions,labels)))
        return res

    def extra_trees_classifier(self, loaded_dataset, intermediate_df, description, method):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_val_score
        import pandas as pd 
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset

        try:
            ETC = ExtraTreesClassifier()
            quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
            if (len(quantitativeColumns) == 0):
                res = {
                    'output': "Dataframe needs numeric values",
                    'result': "Dataframe needs numeric values",
                    'description': "Dataframe needs numeric values",
                    'type' : 'error'
                }
                return res
            X = df[quantitativeColumns[:-1]]
            y = df[[quantitativeColumns[-1]]].values.ravel()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            ETC = ExtraTreesClassifier()
            ETC.fit(X_train, y_train)
            prediction = ETC.predict(X_test)
            scores = cross_val_score(ETC,X_train,y_train,cv=2)

            res = {
                'output': pd.DataFrame(scores).to_json(orient='table'),
                'result': pd.DataFrame(scores).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            intermediate_df.append(pd.DataFrame(scores))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def firstTen(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset

        try:
            samples = df.head(10)
            res = {
                'result' : samples.round(3).to_json(orient='table'),
                'output' : samples.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }
            intermediate_df.append(samples.round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def fit_decision_tree(self, loaded_dataset, intermediate_df, description, method):
        df = loaded_dataset

        res = {}
        try:
            X_train, X_test, y_train, y_test = df
            reg = fit_model(X_train, y_train)
            res = {
                'output' : y_train.head(10).round(3).to_json(orient='table'),
                'result' : "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']),
                'description' : description,
                'type' : method
            }
            intermediate_df.append(y_train.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def des(self, loaded_dataset, intermediate_df, description, method):
        res = {}
        try:
            descriptive_statistics = loaded_dataset.describe(include='all')
            res = {
                'result' : descriptive_statistics.round(3).to_json(orient='table'),
                'output' : descriptive_statistics.round(3).to_json(orient='table'),
                'description' : description,
                'type' : 'group-statistics'
            }
            intermediate_df.append(descriptive_statistics.round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def initialize_kmeans_cluster(self, loaded_dataset, intermediate_df, description, method):
        df= loaded_dataset
        res = {}
        try:
            res_df = self.initializeClustersForKmeans(df)
            res = {
                'output': df.head(10).to_json(orient='table'),
                'result': res_df.head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(df.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def matrix_norm(self, loaded_dataset, intermediate_df, description, method):
        from pandas.api.types import is_numeric_dtype
        import numpy.linalg as LA
        import pandas as pd

        df = loaded_dataset

        nCols = [c for c in list(df) if is_numeric_dtype(df[c])]

        """if length is 0 that means no columns contained any numerical data"""
        if len(nCols) == 0:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            data = {"columnName":[],"NumpyLinalgNorm":[]}
            for nc in nCols:
                data["columnName"].append(nc)
                data["NumpyLinalgNorm"].append(LA.norm(df[[nc]].values))
            res = {
                'output': pd.DataFrame(data).to_json(orient='table'),
                'result': pd.DataFrame(data).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(data))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    """test this with non umeric values - then check if needed"""
    def mean(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 

        df = loaded_dataset.select_dtypes(include='number')

        if df.empty == True:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        try:
            new_df = pd.DataFrame(df.mean(), columns=['mean'])

            res = {
                'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
                'result' : new_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }
            intermediate_df.append(new_df.round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def num_boxplot(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        import seaborn 

        df = loaded_dataset

        numerical_cols = df.select_dtypes(include='number').columns

        if len(numerical_cols) == 0:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        try:
            image_list = []
            for num_var in numerical_cols:
                plt.clf()
                ax = seaborn.boxplot(y=num_var, data=df)
                plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
                plt.xticks(rotation=45)
                self.save_bytes_image(image_list)
                if len(image_list) >= 5:
                    break
            res = {
                'output' : df.select_dtypes(include='number').head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }

            intermediate_df.append(df.select_dtypes(include='number').head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def outer_join(self, loaded_dataset, intermediate_df, description, method):
        df1 = loaded_dataset

        try:

            res = {
                'output': df1.head(10).to_json(orient='table'),
                'result': df1.head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }

            intermediate_df.append(df1.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def plot_via_limit(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        import numpy as np 

        df = loaded_dataset.select_dtypes(include='number')

        if df.empty == True or len(df.columns) < 2:
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            image_list = []
            p_steps = 100
            v_steps = 100
            P = [item for sublist in df[df.columns[[0]]].values.tolist() for item in sublist]
            V = [item for sublist in df[df.columns[[1]]].values.tolist() for item in sublist]
            x = np.arange(-np.pi, np.pi, 2*np.pi/p_steps)
            y = []
            for i in range(len(x)): y.append([])
            for p, v in zip(P, V):
                i = int((p+np.pi)/(2*np.pi/p_steps))
                y[i].append(v)

            means = [ np.mean(np.array(vs)) for vs in y ]
            stds = [ np.std(np.array(vs)) for vs in y ]
            plt.plot(x, means)
            plt.plot([-2, -2], [-0.99, 0.99], 'k:', lw=1)
            plt.plot([2, 2], [-0.99, 0.99], 'k:', lw=1)
            plt.xlim([-np.pi, np.pi])
            plt.ylim([-1,1])
            plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
            plt.xlabel(r"Phase [rad]")
            plt.ylabel(r"V mean")
            self.save_bytes_image(image_list)

            res = {
                'output': df.head(10).to_json(orient='table'),
                'result': image_list,
                'description' : description,
                'type': method
            }

            intermediate_df.append( df.head(10))
        except Exception as e: 
           res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        return res

    def plot(self, loaded_dataset, intermediate_df, description, method):
        import numpy as np 
        import matplotlib.pyplot as plt 
        import matplotlib.gridspec as gridspec

        res = {}
        try:
            df = loaded_dataset
            image_list = []

            samples = dict()
            alt_df = df.select_dtypes(include='number')

            if (alt_df.empty == True):
                res = {
                    'output': "Dataframe has no numeric values",
                    'result': "Dataframe has no numeric values",
                    'description' : "Dataframe has no numeric values",
                    'type' : "error"
                }
                return res

            h, w = alt_df.shape
            for a in np.arange(h):
                samples[a] = ((alt_df.iloc[[a]].values).ravel())[:4]
                if len(samples) >= 5:
                    break
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(1, len(samples))
            gs.update(wspace=0.05, hspace=0.05)
            for i, sample in samples.items():
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(2, 2), cmap='Greys_r')
            self.save_bytes_image(image_list)
            res = {
                'output': df.head(10).to_json(orient='table'),
                'result': image_list,
                'description' : description,
                'type': method
            }
            intermediate_df.append(df.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }

        return res

    def predict_test(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 

        res = {}
        try:
            df = loaded_dataset
            """ print ("DF ROWS: \n", df.shape[0])"""
            if (df.shape[0] < 2):
                res = {
                    'output': "Dataframe has less than two rows",
                    'result': "Dataframe has less than two rows",
                    'description' : "Dataframe has less than two rows",
                    'type' : "error"
                }
                return res

            reg = df[-1]
            X_train, X_test, y_train, y_test = df[-2]
            y_predict = reg.predict(X_test)
            new_df = pd.DataFrame()
            new_df['predicted'] = y_predict
            new_df['actual'] = list(y_test)
            res = {
                'output' : new_df.head(10).round(3).to_json(orient='table'),
                'result' : "predicted vs. actual test done, see Output Data Frame Tab.",
                'description' : description,
                'type' : method
            }
            intermediate_df.append(new_df.head(10).round(3))
        except Exception as e:
            res = {
                'output' : str(e),
                'result' : str(e),
                'description' : str(e),
                'type' : 'error'
            }
        finally:
            return res

    def probability_density_plot(self, loaded_dataset, intermediate_df, description, method):
        from pandas.api.types import is_numeric_dtype
        import pandas as pd 
        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        df = loaded_dataset
        image_list = []

        data = {'rv':[], 'pdf':[]}
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if (len(quantitativeColumns) == 0):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            x = df[quantitativeColumns[0]].values.ravel()
            for k in [1, 2]:
                rv = chi2(k)
                pdf = rv.pdf(x)
                data['rv'].append(rv)
                data['pdf'].append(pdf)
                plt.plot(x, pdf, label="$k=%s$" % k)
            plt.legend()
            plt.title("PDF ($\chi^2_k$)")
            self.save_bytes_image(image_list)
            output_df = pd.DataFrame(data)
            res = {
                'output': output_df.head(10).to_json(orient='table'),
                'result': image_list,
                'description' : description,
                'type': method
            }
            intermediate_df.append(output_df.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def quantitative_bar_plot(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        from pandas.api.types import is_numeric_dtype

        df = loaded_dataset
        image_list = []
        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if (len(quantitativeColumns) == 0):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            x = df[quantitativeColumns[0]].values.ravel()
            y = df[[quantitativeColumns[1]]].values.ravel()
            plt.figure()
            plt.title("Plot Bar")
            plt.bar(range(len(x)), y, align="center")
            plt.xticks(range(len(x)), rotation=90)
            plt.xlim([-1, len(x)])
            self.save_bytes_image(image_list)
            res = {
                'output': df.head(10).to_json(orient='table'),
                'result': image_list,
                'description' : description,
                'type': method
            }
            intermediate_df.append(df.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def random_forest_classifier(self, loaded_dataset, intermediate_df, description, method):
        from pandas.api.types import is_numeric_dtype
        import pandas as pd 
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        res = {}
        try:
            df = loaded_dataset.select_dtypes(include='number')

            forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=17)
            quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

            if (len(quantitativeColumns) == 0):
                res = {
                    'output': "Dataframe has no numeric values",
                    'result': "Dataframe has no numeric values",
                    'description' : "Dataframe has no numeric values",
                    'type' : "error"
                }
                return res

            X_train = df[quantitativeColumns[:-1]]
            y_train = df[[quantitativeColumns[-1]]].values.ravel()
            res = {
                'output': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
                'result': pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(cross_val_score(forest,X_train,y_train,cv=5)))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def rank_sum(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 
        from scipy import stats
        import itertools

        current_df = loaded_dataset
        if not isinstance(current_df, pd.DataFrame):
            current_df = current_df.to_frame()

        numerical_df = current_df.select_dtypes(include='number')

        if (numerical_df.empty == True):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            res_df = pd.DataFrame(columns=(numerical_df.columns), index=(numerical_df.columns))
            for col1, col2 in itertools.combinations(numerical_df, 2):
                z_stat, p_val = stats.ranksums(numerical_df[col1], numerical_df[col2])
                res_df[col1][col2] = p_val
                res_df[col2][col1] = p_val
            res = {
                'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
                'result' : res_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }
            intermediate_df.append(res_df.round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def scatterplot_regression(self, loaded_dataset, intermediate_df, description, method):
        import itertools
        import seaborn 
        import matplotlib.pyplot as plt 

        df = loaded_dataset
        numerical_df = df.select_dtypes(include='number')

        if (numerical_df.empty == True):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            image_list = []
            count = 0
            for col1, col2 in itertools.combinations(numerical_df, 2):
                plt.clf()
                seaborn.regplot(df[col1], df[col2])
                self.save_bytes_image(image_list)
                plt.show()
                count+=1
                if count >= 5:
                    break
            res = {
                'output' : numerical_df.head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }
            intermediate_df.append(numerical_df.head(10).round(3))
        except Exception as e:
            res = {
                'output' : str(e),
                'result' : str(e),
                'description' : str(e),
                'type' : 'error'
            }
        finally: 
            return res

    def shuffle_split(self, loaded_dataset, intermediate_df, description, method):
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import train_test_split

        df = loaded_dataset
        numerical_df = df.select_dtypes(include='number')

        if (numerical_df.empty == True):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            features = numerical_df[numerical_df.columns[0:3]]
            predicted_variables = numerical_df[numerical_df.columns[-1]]

            X_train, X_test, y_train, y_test = train_test_split(features, predicted_variables, test_size=0.2,random_state=100)
            new_df = (X_train, X_test, y_train, y_test)
            res = {
                'output' : X_train.head(10).round(3).to_json(orient="table"),
                'result' : "split into training and testing set",
                'description' : description,
                'type' : method
            }
            intermediate_df.append(X_train.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def stack_ftgrid(self, loaded_dataset, intermediate_df, description, method):
        import matplotlib.pyplot as plt 
        import seaborn

        df = loaded_dataset

        numerical_cols = df.select_dtypes(include='number').columns
        category_cols = df.select_dtypes(include='object').columns

        if (len(numerical_cols) == 0 or len(category_cols) == 0):
            res = {
                'output': "Dataframe has no numeric or category values",
                'result': "Dataframe has no numeric or category values",
                'description' : "Dataframe has no numeric or category values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            image_list = []
            for cat_var in category_cols:
                if df[cat_var].value_counts().count() <= 5:
                    for num_var in numerical_cols:
                        plt.clf()
                        fig = seaborn.FacetGrid(df,hue=cat_var)
                        fig.map(seaborn.kdeplot,num_var,shade=True)
                        oldest = df[num_var].max()
                        fig.set(xlim=(0, oldest))
                        fig.add_legend()
                        self.save_bytes_image(image_list)
                        if len(image_list) >= 5:
                            break
            res = {
                'output' : df.head(10).round(3).to_json(orient='table'),
                'result' : image_list,
                'description' : description,
                'type' : method
            }
            intermediate_df.append(df.head(10).round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def test_linear_regression(self, loaded_dataset, intermediate_df, description, method):
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        from pandas.api.types import is_numeric_dtype
        from sklearn.model_selection import cross_val_score

        linear_regression = LinearRegression()
        df = loaded_dataset

        quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]

        if (len(quantitativeColumns) == 0):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            data = df[quantitativeColumns[:-1]]
            target = df[[quantitativeColumns[-1]]].values

            res = {
            'output': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
            'result': pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)).to_json(orient='table'),
            'description' : description,
            'type': method
            }

            intermediate_df.append(pd.DataFrame(cross_val_score(linear_regression, data, target, cv=10)))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res            

    def top5cat(self, loaded_dataset, intermediate_df, description, method):
        category_df = loaded_dataset.select_dtypes(include='object')

        if (category_df.empty == True):
            res = {
                'output': "Dataframe has no category values",
                'result': "Dataframe has no category values",
                'description' : "Dataframe has no category values",
                'type' : "error"
            }
            return res

        try:
            for col in category_df:
                samples = category_df[col].value_counts().head(5)
                intermediate_df.append(samples.round(3))
                res = {
                    'output' : samples.round(3).to_json(orient='table'),
                    'result' :samples.round(3).to_json(orient='table'),
                    'description' : description,
                    'type' : method
                }
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def unique_column_values(self, loaded_dataset, intermediate_df, description, method):
        import pandas as pd 

        test = {}

        df = loaded_dataset
        alt_df = df.select_dtypes(include='number')

        if (alt_df.empty == True):
            res = {
                'output': "Dataframe has no numeric values",
                'result': "Dataframe has no numeric values",
                'description' : "Dataframe has no numeric values",
                'type' : "error"
            }
            return res

        res = {}
        try:
            for column in alt_df:
                test[column] = alt_df[column].dropna().unique()
            
            res = {
                'output': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
                'result': pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10).to_json(orient='table'),
                'description' : description,
                'type': method
            }
            intermediate_df.append(pd.DataFrame(dict([ (k,pd.Series(test[k])) for k in test.keys() ])).head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type': 'error'
            }
        finally:
            return res

    def variance(self, loaded_dataset, intermediate_df, description, method):
        res = {}
        try:
            new_df = loaded_dataset.var()
            res = {
                'output' : loaded_dataset.head(10).round(3).to_json(orient='table'),
                'result' : new_df.round(3).to_json(orient='table'),
                'description' : description,
                'type' : method
            }
            intermediate_df.append(new_df.round(3))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type' : "error"
            }
        finally:
            return res

    def word_to_vec(self, loaded_dataset, intermediate_df, description, method):
        df= loaded_dataset

        res = {}
        try:
            res_df = self.calcWordVec(df)
            res = {
                'output': res_df.head(10).to_json(orient='table'),
                'result': res_df.head(10).to_json(orient='table'),
                'description' : description,
                'type': method 
            }
            intermediate_df.append(res_df.head(10))
        except Exception as e:
            res = {
                'output': str(e),
                'result': str(e),
                'description' : str(e),
                'type' : "error"
            }
        finally:
            return res

if __name__ == "__main__":
    import pandas as pd 
    import numpy as np 

    c = CodeBlock()
    loaded_ds = pd.DataFrame(np.random.uniform(low=1, high=10, size=(10,3)), columns=['a', 'b', 'c'])
    res = c.word_to_vec(loaded_ds, [], '', 'method')
    print(res)