# -*- coding: utf-8 -*-

"""
This module is used for creating a class that can run 5 classifiers using grid search with
many defaults hyper parameters. Then the results can be analyzed in 7 different levels in order for finding
the best model and best hyper parameters or combine all/some models to ensemble prediction.
Available levels:
1. Score for each model (according to the input function scoring) - method:ScoreSummery
2. Detailed results: a dictionary with the following: model name, the y_pred (prediction),
                     best parameters found in grid search, the full cv_results from the grid search of the model
3. Hyper parameters results: Analyze the change in hyper parameters per model. Shows a chart for every
                             hyper parameter that the x-axis is the changing value of the hyper parameter and
                             the y-axis is the average change in scoring.
4. A full classification report dataframe. Contains the precision, recall, f1-score and more of every model best
                                           parameters and for the combined models.
                                            method:GetClassificationReport
5. Sliced classification report: Suppose we want to look on the precision of a specific label (or few labels) only.
                                 This tool can extract only the relevant data from the classification dataframe and
                                 return the data as a dataframe  even show it on a chart
                                 method:GetSpecificLabelScore
6. Feature Importance: Dataframe and a chart to explain the feature importance for models that support it
7. Confusion Matrix: Show a graphical confusion matrix for each model
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from sklearn.metrics import accuracy_score, make_scorer, classification_report
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import datetime
import time
import charts


class MegaClassifier:
    """
    This MegaClassifier runs the data on 5 models with grid search on default hyperparameters (that can be changed).
    Available classifiers: DecisionTree, LogisticRegression, RandomForest, SVC, XGBOOST
    The grid search also contains cross-validation.
    The results can be seen in 6 different levels; see below for details.

    Available methods:

    Classics methods:
    fit: get X, y for fit. It fits all models if RelevantModels='all'.
         Can get a list of models that will be used instead.
        If SaveEachRes = True (default=False), then every model will save the grid results to a file.
    predict: Run predict on all models. Returns a dataframe with y_pred on all models + y_true + Average column
             + a column that sums the probability squared for each class and returns the class with the highest score.
             The predict methods also create the results dictionary (read more in the GetResults method).
    predict_proba: Run predict_proba on all models. Then it updates the result dictionary by the predict_proba data.
                    Unlike the predict methods, the return dataframe contains the y_true and the  column that sums
                    the probability squared for each class and returned the class with the highest score.

    Get/set methods:

    GetGridHyperParameters: Returns a dictionary contains hyperparameters used for grid search in a specific model.
    SetGridHyperParameters: Gets a dictionary. Update the hyperparameters dictionary. and then update the relevant
                            grid search

    Explore Outputs methods:

    ScoreSummery: Return a dataframe with the best score for every model add PredAllModelsByProba column that sums
                 the probability squared for each class and returned the class with the highest score.
    GetResults: Return a dictionary of all the models' results (the default is all models) or a specific model.
                Each dictionary contains the following:
                             {'Classifier':name of the classifier,
                                'Score':The scoring after fitting and predict,
                                'y_pred':An array with the predicted classes,
                                'Best_param':Best parameters for the model (decided during the fit phase),
                                'cv_results':Read the cv_results in the grid search documentation.
                                             gets the result of every run in the grid search and the cross validation}
    ParamInsight: Works on a specific model. It takes every hyperparameter and every value it gets and shows the score
            it gets when we group by the parameter and value. That allows us to understand the effect this
            hyperparameter has on the scoring.
            It returns a grid of charts for each parameter that shows the mean score (after cross-validation) and
            the standard deviation of the cross-validation
    GetClassificationReport: Return a multi-model classification report in the form of a dataframe
    GetSpecificLabelScore: Slice the classification dataframe (get it by GetClassificationReport methods) by
                           specific labels only and  specific score types only
    GetFeatureImportance: return a dataframe with the feature importance for every model that supports this attribute
                          As a default, it also shows a chart with the combined results (normalized)
    ShowConfusionMatrix - Show a confusion matrix for each model
"""

    def __init__(self, scoring=accuracy_score, ShortCrossValidParts=5, LongCrossValidParts=3, Class_weight='balanced',
                 MultiClass=False, BigDataSet=False, PathForOutFile='', verbose=0, RandomSeed=1234):
        """
        scoring - Scoring function to use for all models
        ShortCrossValidParts = int. Number of cross validation parts in FAST time running models
        LongCrossValidParts = int. Number of cross validation parts in SLOW time running models
        Class_weight = dictionary. class weights for every model except XGBOOST (that does not support weights)
        MultiClass = bool. Gets True if this is a multiclass problem. Multiclass can't use some of the hyper parameters
        BigDataSet = bool. if it gets true then it runs less hyper parameters to avoid long fitting time
        PathForOutFile = string. Gets a path string if a result summery is needed for every model fit.
        verbose = int.  1 : the computation time for each fold and parameter candidate is displayed
                        2 : the score is also displayed
                        3 : the fold and candidate parameter indexes are also displayed together with the
                            starting time of the computation.
        RandomSeed = int. Random seed
        """
        self.Random = RandomSeed  # Random seed
        self.multiclass = MultiClass  # Gets True if this is a multiclass. False for boolean
        self.bigDataSet = BigDataSet  # Gets True if this is a big database so the grid should be downsized
        self.OutPath = ''  # if the path is not empty it is the path to put the result files
        if len(PathForOutFile) > 1:
            self.OutPath = PathForOutFile + '\\'
        self.Weights = Class_weight  # Class weights does not work in XGBOOST
        self.verbose = verbose
        self.classifiers = {}  # A dictionary of all available classifiers
        self.GridClassifiers = {}  # A dictionary of all available Grids objects
        self.Parameters = {}  # A dictionary of all the parameters that each classifier gets
        self.Score = make_scorer(scoring)  # The score used in the grid search
        self.OriginalScoring = scoring  # The basic matrices as received from the user
        self.Scv = ShortCrossValidParts  # Number of cross validation parts for FAST time running models
        self.Lcv = LongCrossValidParts  # Number of cross validation parts for SLOW time running models
        self.cv = {}
        self.BestParam = {}  # Holds a dictionary that holds the best parameters for every classifier
        self.Label2Num = preprocessing.LabelEncoder()  # Transformer for label encoding

        # Holds a list of possible classifiers. It does not change when we change the relevant models
        self.PossibleModels = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'SVC', 'XGBOOST']
        # Holds a list of relevant classifiers as given by the user or all for all classifiers
        self.RelevantModel = self.PossibleModels

        # Result will be in the following format:
        # {'Classifier':{'Score':,'y_pred':,'Best_param':,'cv_results':},,,,}
        # Updated when using predict or predict_proba
        self.results = {}

        # OutputDF Fill up after predict or predict_proba.
        # Contains all the y_pred from all models + y_true + PredAllModelsByProba
        # PredAllModelsByProba - sums the probability squared for each class and return the class with the highest score
        # Updated when using predict or predict_proba
        self.OutputDF = pd.DataFrame()
        # Holds the  feature important summery in a dataframe. Updated when using predict
        self.featuresImport = pd.DataFrame()
        # Dataframe that contains: precision,recall,f1-score and more per each label+model
        # Updated when using predict
        self.ClassReportDF = pd.DataFrame()
        # True if there are less then 2 classes. Then don't allow: fir, predict and the rest
        self.NumOfClassesLessThen2 = False

        # Initiate the models by running the following methods
        self.__DefaultsGridParameters()
        self.__InitClassifier()
        self.__InitCV()
        self.__BuildGridSearch()

    # Update the relevant parameter per model, big data and multiclass
    def __DefaultsGridParameters(self):
        if self.bigDataSet:
            self.Parameters['DecisionTree'] = {'criterion': ['gini'], 'max_depth': [None, 50],
                                               'min_samples_leaf': [10, 50], 'max_features': ['auto'],
                                               'random_state': [self.Random], 'class_weight': [self.Weights],
                                               'ccp_alpha': [0.0, 0.01, 0.02]}
            self.Parameters['LogisticRegression'] = [
                {'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'], 'C': [10, 1, 0.1, 0.01],
                 'random_state': [self.Random], 'class_weight': [self.Weights]},
                {'solver': ['saga'], 'penalty': ['none'],
                 'random_state': [self.Random], 'class_weight': [self.Weights]}]
            self.Parameters['RandomForest'] = {'n_estimators': [50, 100], 'criterion': ['gini', 'entropy'],
                                               'max_depth': [10, 50],
                                               'min_samples_leaf': [5, 10, 50], 'max_features': ['auto'],
                                               'random_state': [self.Random], 'class_weight': [self.Weights],
                                               'ccp_alpha': [0.0, 0.01, 0.02], 'max_samples': [None, 0.2]}
            self.Parameters['SVC'] = {'C': [10, 1, 0.1, 0.01], 'kernel': ['poly', 'rbf', 'sigmoid'],
                                      'random_state': [self.Random], 'class_weight': [self.Weights]}
            self.Parameters['XGBOOST'] = {'max_depth': [10, 50], 'min_child_weight': [5, 10], 'seed': [self.Random]}

        else:  # In case this is not a big dataset
            self.Parameters['DecisionTree'] = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 50],
                                               'min_samples_leaf': [1, 3, 5, 10],
                                               'max_features': ['auto', 'log2', None],
                                               'random_state': [self.Random], 'class_weight': [self.Weights],
                                               'ccp_alpha': [0.0, 0.01, 0.02]}
            if self.multiclass:  # In case this is not a big dataset and with multiclass
                self.Parameters['LogisticRegression'] = [
                    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [10, 1, 0.1, 0.01],
                     'random_state': [self.Random], 'class_weight': [self.Weights]},
                    {'solver': ['lbfgs'], 'penalty': ['none'],
                     'random_state': [self.Random], 'class_weight': [self.Weights]}]

            else:
                self.Parameters['LogisticRegression'] = [
                    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [10, 1, 0.1, 0.01],
                     'random_state': [self.Random], 'class_weight': [self.Weights]},
                    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [10, 1, 0.1, 0.01],
                     'random_state': [self.Random], 'class_weight': [self.Weights]},
                    {'solver': ['lbfgs'], 'penalty': ['none'],
                     'random_state': [self.Random], 'class_weight': [self.Weights]}]

            self.Parameters['RandomForest'] = {'n_estimators': [50, 100, 200], 'criterion': ['gini', 'entropy'],
                                               'max_depth': [10, 50],
                                               'min_samples_leaf': [1, 3, 5, 10],
                                               'max_features': ['auto', 'log2', None], 'oob_score': [True, False],
                                               'random_state': [self.Random], 'class_weight': [self.Weights],
                                               'ccp_alpha': [0.0, 0.01, 0.02], 'max_samples': [None, 0.5]}
            self.Parameters['SVC'] = {'C': [10, 1, 0.1, 0.01], 'kernel': ['poly', 'rbf', 'sigmoid'],
                                      'degree': [3, 5, 10],
                                      'random_state': [self.Random], 'class_weight': [self.Weights]}
            self.Parameters['XGBOOST'] = {'n_estimators': [100, 300], 'learning_rate': [0.1],
                                          'gamma': [0.001, 0.01, 0.1, 1, 5],
                                          'max_depth': [6, 15, 30], 'lambda': [1e-5, 1e-2, 0.1, 1, 100],
                                          'min_child_weight': [5, 10],
                                          'alpha': [1e-5, 1e-2, 0.1, 1, 100], 'seed': [self.Random]}

    # Creates an instance of each classifier
    def __InitClassifier(self):
        self.classifiers['XGBOOST'] = XGBClassifier(n_estimators=500, random_state=self.Random, seed=self.Random,
                                                    use_label_encoder=False)
        self.classifiers['RandomForest'] = RandomForestClassifier(max_depth=2, random_state=self.Random)
        self.classifiers['DecisionTree'] = DecisionTreeClassifier(random_state=self.Random)
        self.classifiers['SVC'] = SVC(gamma='auto', random_state=self.Random, probability=True)
        self.classifiers['LogisticRegression'] = LogisticRegression(random_state=self.Random, max_iter=10000)

    # Define the cross validation parts for every model
    def __InitCV(self):
        self.cv['XGBOOST'] = self.Lcv
        self.cv['RandomForest'] = self.Lcv
        self.cv['DecisionTree'] = self.Scv
        self.cv['SVC'] = self.Scv
        self.cv['LogisticRegression'] = self.Scv

    # Creates a grid search for every classification path
    def __BuildGridSearch(self):
        for clf in self.classifiers.keys():
            self.GridClassifiers[clf] = GridSearchCV(self.classifiers[clf], self.Parameters[clf], self.Score,
                                                     cv=self.cv[clf], verbose=self.verbose)

    def fit(self, X, y=None, RelevantModels='all', SaveEachRes=False):
        """
        Fits all relevant models.
        RelevantModels - A list of models to fit. The default is 'all', then all models will get fitted.
                         Available models: ['DecisionTree','LogisticRegression','RandomForest','SVC','XGBOOST']
        SaveEachRes = If True then every time a model gets fit.
                      It will save the grid search parameter: cv_results_ to a file.
        """
        X_new = X.copy()
        y_new = self.Label2Num.fit_transform(y)  # Update the y labels to an array of numbers. Avoid strings in labels
        # If there are less then 2 classes then don't do fit and stop attempts to do predict
        if len(np.unique(y_new)) < 2:
            self.NumOfClassesLessThen2 = True
            return
        else:
            self.NumOfClassesLessThen2 = False

        # Update the models that are relevant. 'all' means all models. Else use a list of models
        self.__RelevantModels(RelevantModels)

        for cls in self.RelevantModel:
            print('Start fitting ' + cls + ' model.')
            print('Current time: ' + self.__GetLocalTime())

            current_time = datetime.datetime.now()  # Count the time for fitting

            self.GridClassifiers[cls].fit(X_new, y_new)
            self.BestParam[cls] = self.GridClassifiers[cls].best_params_

            TimeFitting = self.__ShowTimeInMin((datetime.datetime.now() - current_time))
            if SaveEachRes:
                self.__Save2File(cls)
            print(cls + ' model done fitting.' + '\n Time for fitting: ' + TimeFitting)
        return self

    def predict(self, X, y):
        """
        Run predict on every model that was fitted. Fills the results dictionary
        Return a dataframe with: y_true, PredAllModelsByProba
                                 PredAllModelsByProba = sums the probability squared for each class and return
                                 the class with the highest score.
        """
        AccumSumProba = None
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return
        X_new = X.copy()
        y_new_label = y.copy()  # Array of y with labels
        y_newNum = self.Label2Num.transform(y_new_label)  # Array of y after label encoder
        self.OutputDF = pd.DataFrame()  # Restart OutputDF
        self.OutputDF['y_true'] = y_new_label
        for mdl in self.RelevantModel:
            y_pred = self.GridClassifiers[mdl].predict(X_new)
            y_predDecode = self.Label2Num.inverse_transform(y_pred)
            self.OutputDF[mdl] = y_predDecode
            Score_result = self.GridClassifiers[mdl].score(X_new, y_newNum)
            self.results[mdl] = {'Classifier': mdl,
                                 'Score': Score_result,
                                 'y_pred': y_predDecode,
                                 'Best_param': self.GridClassifiers[mdl].best_params_,
                                 'cv_results': self.GridClassifiers[mdl].cv_results_}

        # Add prediction by predict proba
        NumOfClasses = len(np.unique(y_newNum))
        # AccumSumProba = np.zeros((len(y_newNum), NumOfClasses))
        Flag = True
        for mdl in self.RelevantModel:
            y_pred = self.GridClassifiers[mdl].predict_proba(X_new)
            if Flag:
                AccumSumProba = (y_pred ** 2)  # We use **2 to give more weight to the highest probabilities
                Flag = False
            else:
                AccumSumProba += (y_pred ** 2)  # We use **2 to give more weight to the highest probabilities

        y_predictAccum = np.argmax(AccumSumProba, axis=1)  # Y in numbers
        y_predictLabels = self.Label2Num.inverse_transform(y_predictAccum)  # Y in labels
        self.OutputDF['PredAllModelsByProba'] = y_predictLabels

        self.results['All'] = {'Classifier': 'All',
                               'Score': self.OriginalScoring(y_newNum, y_predictAccum),
                               'y_pred': y_predictLabels}
        # Update the classification report dataframe
        self.__UpdateClassificationReport()
        self.__UpdateFeatureImportance(X_new)

        return self.OutputDF

    def predict_proba(self, X, y):
        """
        Run predict_proba on every model that was fitted. Fills the results dictionary that includes the y_pred columns
        Return a dataframe with: y_true,y_pred for every model, y_average (avg. of all y_pred), PredAllModelsByProba
                                 PredAllModelsByProba = column that sums the probability squared for each class and
                                 return the class with the highest score.
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return

        X_new = X.copy()
        y_new_label = y.copy()  # Array of y with labels
        y_newNum = self.Label2Num.transform(y_new_label)  # Array of y after label encoder
        self.OutputDF = pd.DataFrame()  # Restart OutputDF
        NumOfClasses = len(np.unique(y_newNum))
        AccumSum = np.zeros((len(y_newNum), NumOfClasses))
        self.OutputDF['y_true'] = y_new_label

        for mdl in self.RelevantModel:
            y_pred = self.GridClassifiers[mdl].predict_proba(X_new)
            AccumSum += y_pred ** 2  # We use **2 to give more weight to the highest probabilities
            y_predDecode = self.Label2Num.inverse_transform(np.argmax(y_pred, axis=1))
            Score_result = self.GridClassifiers[mdl].score(X_new, y_newNum)
            self.results[mdl] = {'Classifier': mdl,
                                 'Score': Score_result,
                                 'y_pred': y_predDecode,
                                 'Best_param': self.GridClassifiers[mdl].best_params_,
                                 'cv_results': self.GridClassifiers[mdl].cv_results_}

        y_predictAccum = np.argmax(AccumSum, axis=1)  # Y in numbers
        y_predictLabels = self.Label2Num.inverse_transform(y_predictAccum)  # Y in labels
        self.OutputDF['PredAllModelsByProba'] = y_predictLabels

        # Update the ALL element in results dictionary
        self.results['All'] = {'Classifier': 'All',
                               'Score': self.OriginalScoring(y_newNum, y_predictAccum),
                               'y_pred': y_predictLabels}

        return self.OutputDF

    def ScoreSummery(self):
        """
        Return a data frame that contains the score for each model
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return

        OutDF = pd.DataFrame()
        for mdl in self.RelevantModel:
            OutDF[mdl] = pd.Series(self.results[mdl]['Score'])

        OutDF['PredAllModelsByProba'] = pd.Series(self.results['All']['Score'])
        OutDF.index.name = self.OriginalScoring.__name__

        return OutDF

    def ParamInsight(self, ModelName):
        """
        Works on a specific model. Takes every hyper parameter and every value it gets and shows the score
        it gets when we group by the parameter and value. It shows the effect each hyper parameter has on the scoring.
        It returns a grid of charts for each parameters that shows the mean score (after cross validation) and
        the standard deviation of the cross validation
        ModelName - string of the model needed
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return

        # Get the parameters dictionary out of the results for a specific model
        CurrParam = self.results[ModelName]['cv_results']['params'][0]
        if 'class_weight' in CurrParam:
            del CurrParam['class_weight']
        if 'random_state' in CurrParam:
            del CurrParam['random_state']
        if 'seed' in CurrParam:
            del CurrParam['seed']
        # create a list of parameters with a prefix (so we can use it to refer to each column)
        ParmLst = ['param_' + parm for parm in list(CurrParam.keys())]
        # Create a list of relevant columns
        colLst = ['mean_test_score', 'std_test_score']
        colLst.extend(ParmLst)
        # Create a dataframe with only the relevant columns
        df = pd.DataFrame(self.results[ModelName]['cv_results'])
        df = df[colLst]
        NumberOfParam = len(CurrParam)

        # Creates a subplots with the mean score and the mean of the standard deviation each parameter gets
        # when we group by by that parameter.

        NumOfCol = 4  # number of columns in the row
        NumOfRows = 1 + (2 * NumberOfParam) // NumOfCol
        if NumOfRows < 2:
            NumOfRows = 2
        fig, ax = plt.subplots(NumOfRows, NumOfCol, figsize=(25, NumOfRows * 7))
        # Use the following arguments to improve the look of the line in the charts
        kwargs = {'linewidth': 3, 'marker': '*', 'markersize': 15, 'markerfacecolor': 'red',
                  'markeredgecolor': '#411a20'}

        iRow = 0
        jCol = -1
        # Go over all the parameters
        for param in ParmLst:
            # Create a chart for the mean score on the cross validation runs by a specific parameter value
            CurrDf = df.groupby(param).mean()['mean_test_score']
            iRow, jCol = self.__nextij(iRow, jCol, NumOfCol)
            ax[iRow, jCol].plot(CurrDf, **kwargs)
            ax[iRow, jCol].set_title(param[6:] + ' score', fontdict={'fontsize': 20, 'color': '#411a20'})
            # Do the same of standard deviation
            CurrDf = df.groupby(param).mean()['std_test_score']
            iRow, jCol = self.__nextij(iRow, jCol, NumOfCol)
            ax[iRow, jCol].plot(CurrDf, **kwargs)
            ax[iRow, jCol].set_title(param[6:] + ' _SD', fontdict={'fontsize': 20, 'color': '#411a20'})

        plt.show()

    def GetGridHyperParameters(self, modelName):
        return self.Parameters[modelName]

    def SetGridHyperParameters(self, mdlName, HyperParamDic):
        """
        Adjust the hyper parameters dictionary and initiate the grid search instance
        mdlName - string.The model name that need adjustment
        HyperParamDic - Dictionary of grid search hyper parameters for the specific model
        """
        self.Parameters[mdlName] = HyperParamDic
        self.GridClassifiers[mdlName] = GridSearchCV(self.classifiers[mdlName], self.Parameters[mdlName], self.Score,
                                                     cv=self.cv)

    def GetResults(self, modelName='All'):
        """
        Returns the results dictionary.
        If a specific model is given then it returns only that dictionary.
        If 'All' is given (default) then returns a dictionary of results dictionaries.
        Each results dictionary contains the following:
            {'Classifier':name of the classifier,
            'Score':The scoring after fitting and predict,
            'y_pred':An array with the predicted classes,
            'Best_param':Best parameters for the model (decided during the fit phase),
            'cv_results':Read the cv_results in the grid search documentation.
                            ItGets the result of every run in the grid search and the cross validation}
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return
        if modelName == 'All':
            return self.results
        else:
            return self.results[modelName]

    def GetClassificationReport(self):
        return self.ClassReportDF

    # Update the self.ClassReportDF that contains the classification report dataframe
    def __UpdateClassificationReport(self):
        FirstFlag = True
        for clf in self.results.keys():
            # Creates a dataframe from the classification report per model and append all to one dataframe
            y_pred = self.results[clf]['y_pred']
            CRDic = classification_report(y_pred, self.OutputDF['y_true'], output_dict=True)
            tempDF = pd.DataFrame.from_dict(CRDic)
            tempDF.reset_index(inplace=True)
            tempDF.rename(columns={'index': 'Score_type'}, inplace=True)
            tempDF.insert(0, 'Classifier', pd.Series())
            tempDF['Classifier'] = clf
            # First clf create the output dataframe and the rest just append
            if FirstFlag:
                self.ClassReportDF = tempDF.copy()
                FirstFlag = False
            else:
                self.ClassReportDF = self.ClassReportDF.append(tempDF)

    def GetSpecificLabelScore(self, ListOfScoreTypes, ListOfLabels, ShowChart=False):
        """
        Slice the classification dataframe (get it by GetClassificationReport methods) by  specific labels only
        and  specific score types only
        :param ListOfScoreTypes: List. List of scores needed from classification dataframe
        :param ListOfLabels: List. List of labels needed from classification dataframe
        :param ShowChart: Bool. If True then show a barchart of the result
        :return: dataframe with the desired slice
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return
        colList = ['Classifier', 'Score_type']
        for lbl in ListOfLabels:
            if lbl in self.ClassReportDF.columns:
                colList.append(lbl)

        SlicedDf = self.ClassReportDF[self.ClassReportDF['Score_type'].isin(ListOfScoreTypes)][colList]
        if ShowChart:
            NumOfLabels = len(ListOfLabels)
            NewTitle = ["\n\nLabel: " + sub for sub in ListOfLabels]
            SlicedDf.sort_values(by=ListOfLabels, ascending=False).plot(x='Classifier', kind='bar', subplots=True,
                                                                        sharex=False, figsize=(15, 8 * NumOfLabels),
                                                                        xlabel='\nModels\n', rot=0, sort_columns=True,
                                                                        title=NewTitle)
        return SlicedDf

    def ShowConfusionMatrix(self, FigSize=(7, 5)):
        """
        Show the confusion matrix of all models.

        :param FigSize: tuple of 2 integers. Changing the figsize.
        :return: Nothing
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return
        for clf in self.RelevantModel:
            classes = self.Label2Num.inverse_transform(self.GridClassifiers[clf].classes_)
            charts.ClassicGraphicCM(self.OutputDF[clf], self.OutputDF['y_true'], classes, title='\nModel: ' + clf,
                                    fig_size=FigSize, ClassReport=False, ReturnAx=True)

    def __UpdateFeatureImportance(self, X):
        """
        Calculate a dataframe of feature importance per model.
        +Add a summery column + Add a normalize summery column (also sort by this column, Descending)
        The result dataframe is saved at: self.featuresImport

        :param X: X train array
        :return: Nothing
        """
        flag = True
        for clf in self.RelevantModel:
            CurrClf = self.GridClassifiers[clf].best_estimator_
            try:
                FI = CurrClf.feature_importances_
                featuresDic = {}  # a dict to hold feature_name: feature_importance
                for feature, importance in zip(X.columns, FI):
                    featuresDic[feature] = importance  # add the name/value pair
                if flag:  # In case this is the first loop then initialize the dataframe
                    self.featuresImport = pd.DataFrame(FI, index=X.columns, columns=[clf])
                    flag = False
                else:
                    self.featuresImport[clf] = pd.Series(FI, index=X.columns)
            except AttributeError:
                pass
        # Sum all columns
        self.featuresImport['SumOfCol'] = self.featuresImport.sum(axis=1)
        # Add a normalized column (value/sum of all values)
        self.featuresImport['SumOfColNormalize'] = self.featuresImport['SumOfCol'] / self.featuresImport[
            'SumOfCol'].sum()
        self.featuresImport.sort_values(by='SumOfColNormalize', ascending=False, inplace=True)

    def GetFeatureImportance(self, TopFeatures=10, ShowChart=True):
        """
        Return a dataframe with the feature importance for all models that have that feature.
        Also includes a summery column for all the models importance.
        + a Normalized summery column (the value divided by the sum of the column values)
        If ShowChart=True then it also shows a chart of the normalized summery column, sorted

        :param TopFeatures: int. How many features should be in the chart
        :param ShowChart: bool. True means show the chart
        :return: Dataframe
        """
        if self.NumOfClassesLessThen2:
            print('Less then 2 classes in fitting. Method stop.')
            return
        if ShowChart:
            pltDf = self.featuresImport.head(TopFeatures)['SumOfColNormalize']
            charts.BarCharts([pltDf], ['Feature importance - sum of models'], WithPerc=3, LabelPrecision=2)
        return self.featuresImport

    # Save the results dictionary of a specific model to a file
    def __Save2File(self, cls):
        df = pd.DataFrame(self.GridClassifiers[cls].cv_results_)
        df.insert(0, 'Classifier', cls)
        df.to_csv(self.OutPath + 'result_' + cls + '.csv')

    # Returns the next 2D index of an array
    @staticmethod
    def __nextij(CurrRow, CurrCol, NumOfColumns):
        if CurrCol < (NumOfColumns - 1):
            return CurrRow, CurrCol + 1
        else:
            return CurrRow + 1, 0

    # Update the RelevantModel list
    def __RelevantModels(self, RelevantModel):
        if RelevantModel == 'all':
            self.RelevantModel = ['DecisionTree',
                                  'LogisticRegression',
                                  'RandomForest',
                                  'SVC',
                                  'XGBOOST']
        else:
            self.RelevantModel = RelevantModel

    # prints the time in h:m:s format
    @staticmethod
    def __ShowTimeInMin(DateTimeVal):
        seconds = DateTimeVal.total_seconds()
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return ' h: ' + str(int(hours)) + '  m: ' + str(int(minutes)) + '  s: ' + str(seconds)

    # Return the local time in the form of H:M:S
    @staticmethod
    def __GetLocalTime():
        return str(time.localtime().tm_hour) + ':' + str(time.localtime().tm_min) + ':' + str(time.localtime().tm_sec)


class MultiMegaClassifiers:
    """
    This class is used when we want to use the Mega classifier on different segments of a dataset.

    For example:
    If we have a dataset that contains a feature of the area. What if each area contains data that seems very different.
    We want to run a model for each area separately. We also want to use the MegaClassifier to find the best model for
    each area. This class helps to manage the data and the processes.
    """

    def __init__(self, SavePath=''):
        self.MultiMC = {}
        self.path = SavePath
        self.FirstModel = True
        self.BestSliceModel = {}
        self.NumOfModels = 0

        self.AllResult = {}
        self.ScoreDf4All = pd.DataFrame()
        self.ClassReportAll = pd.DataFrame()
        self.Feature_all = pd.DataFrame()
        self.SliceByColumn = ''

    def insertModel(self, MC_model, strName):
        self.MultiMC[strName] = MC_model
        self.SaveModel()
        print('\nModel: ' + str(strName) + ' Inserted and saved.\n')

    def ReadMultiMCFromFile(self, path=''):
        if path == '':
            with open(self.path + '/MultiMC.MC', 'rb') as MultiMCFile:
                self.MultiMC = pickle.load(MultiMCFile)
                print('Read completed MultiMC.MC')
        else:
            with open(path, 'rb') as MultiMCFile:
                self.MultiMC = pickle.load(MultiMCFile)
                print('Read completed')

    def ReadMultiMCFromManyFiles(self, PathList):
        """
        This method takes a path list of saved multi MegaClassifier object and join them in one model.
        We use this method if we split the tests. Every split saved the result in different path.

        :param PathList: A list of strings that are the paths to the saved model
        :return:
        """
        for path in PathList:
            # Read from each path
            with open(path, 'rb') as MultiMCFile:
                currMultiMC = pickle.load(MultiMCFile)
            # Every path may have more than one model
            for MultiModel in currMultiMC.keys():
                self.insertModel(currMultiMC[MultiModel], MultiModel)

        print('Read completed')

    def CreateCombinedData(self):
        mdl = ''
        for mdl in self.MultiMC.keys():
            if self.FirstModel:
                self.__InsertFirstModel(self.MultiMC[mdl], mdl)
            else:
                self.__InsertNoneFirstModel(self.MultiMC[mdl], mdl)

        self.NumOfModels = len(self.MultiMC[mdl].RelevantModel)
        self.ScoreDf4All['Max score'] = self.ScoreDf4All.iloc[:, 0:self.NumOfModels].max(axis=1)
        self.ScoreDf4All['BestModel'] = self.ScoreDf4All.iloc[:, 0:self.NumOfModels].idxmax(axis=1, skipna=True)
        self.FirstModel = True
        print("Done")

    # def GetBestModelAndParameters(self):

    def SaveModel(self):
        # Save the model dictionary
        with open(self.path + '/MultiMC.MC', 'wb') as MultiMCFile:
            pickle.dump(self.MultiMC, MultiMCFile)

    def BuildBestModelPerSlice(self):
        BestModelDic = pd.Series(self.ScoreDf4All.BestModel.values, index=self.ScoreDf4All.Slice).to_dict()
        for Slice in BestModelDic.keys():
            BestClassifier = BestModelDic[Slice]
            BestEstimator = clone(self.MultiMC[Slice].GridClassifiers[BestClassifier].estimator)
            self.BestSliceModel[Slice] = {'Best name': BestClassifier, 'Best parameters': BestEstimator.get_params,
                                          'Best estimator': BestEstimator}
        return self.BestSliceModel

    def UseBestModel2Predict(self, X):

        Flag = True
        for Slice in self.BestSliceModel:
            x = X[X[self.SliceByColumn] == Slice]
            y_pred = self.BestSliceModel[Slice]['Best estimator'].predict(x)
            CurrModel = pd.DataFrame(y_pred, index=x.index.tolist(), columns=['y_predict'])
            if Flag:
                yPred_DF = CurrModel
                Flag = False
            else:
                yPred_DF = pd.concat(yPred_DF, CurrModel)

        yPred_DF = yPred_DF.reindex(X.index.tolist())

        return yPred_DF

    def __InsertFirstModel(self, MC_model, strName):
        self.AllResult[strName] = MC_model.GetResults()
        self.ScoreDf4All = MC_model.ScoreSummery()
        if not isinstance(self.ScoreDf4All, pd.DataFrame):
            self.ScoreDf4All = pd.DataFrame()
            return
        self.ScoreDf4All['Slice'] = strName
        # Classification report
        self.ClassReportAll = MC_model.GetClassificationReport()
        self.ClassReportAll['Slice'] = strName

        # Feature importance
        self.Feature_all = MC_model.GetFeatureImportance(ShowChart=False)
        self.Feature_all['Slice'] = strName

        self.FirstModel = False

    def __InsertNoneFirstModel(self, MC_model, strName):
        self.AllResult['Slice'] = MC_model.GetResults()
        # Score
        CurrScore = MC_model.ScoreSummery()
        if not isinstance(CurrScore, pd.DataFrame):
            return
        CurrScore['Slice'] = strName
        self.ScoreDf4All = self.ScoreDf4All.append(CurrScore)
        # Classification report
        CurrClassReport = MC_model.GetClassificationReport()
        CurrClassReport['Slice'] = strName
        self.ClassReportAll = self.ClassReportAll.append(CurrClassReport)

        # Feature importance
        Curr_Feature = MC_model.GetFeatureImportance(ShowChart=False)
        Curr_Feature['Slice'] = strName
        self.Feature_all = self.Feature_all.append(Curr_Feature)
