# -*- coding: utf-8 -*-

"""
Those modules contain mainly Pandas transformers. Get dataframe as input and returns dataframe
Pandas transformers classes:
    PandasTransformer - General purpose transformer. Get as an input a regular sklearn transformer and returns
                        a dataframe.
    P_StandardScaler -  StandardScaler that returns a dataframe.
    P_MaxAbsScaler -    MaxAbsScaler that returns a dataframe.
    P_MinMaxScaler -    MinMaxScaler that returns a dataframe.
    P_SimpleImputer -   SimpleImputer that returns a dataframe.
    P_SelectKBest -     Returns a dataframe + can deal with negative number if chosen
    P_LabelEncoder -    Label encoder to all relevant columns (that are not numeric and not in the not transformed list)

Other transformers:
    BinaryDownSizeTransformer - Use when a data downsize is needed.
"""

import pandas as pd

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
# noinspection PyProtectedMember
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, f_regression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


class PandasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, TransFormerObject, columns=None):
        """
        Transformer that gets a DataFrame as an input and use sklearn transformer on a specific columns

        :param TransFormerObject:  A sklearn transformer sent by the user with the desired parameters
        :param columns: list. List of columns names to apply the transformation on.
                              If empty it will work on all numeric columns
        :return: DataFrame, with the transformed values in the wanted columns.

        ### Original code written by Dror geva and general purposes by Gal merom ###

        """

        if columns is None:
            columns = []
        self.columns = columns
        self.Transformer_model = TransFormerObject

    def fit(self, X, y=None):
        # In case columns were not defined then find all the numeric columns in the dataframe
        if len(self.columns) == 0:
            for col in X.columns:
                if is_numeric_dtype(X[col].dtype):
                    self.columns.append(col)

        self.Transformer_model.fit(X[self.columns])
        return self

    def transform(self, X):
        X_new = X.copy()
        scaled_cols = self.Transformer_model.transform(X_new[self.columns])
        X_new.loc[:, self.columns] = scaled_cols
        return X_new

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None)
        return self.transform(X)


class P_StandardScaler(PandasTransformer):
    def __init__(self, columns=None):
        """
        Like a StandardScaler. returns  a dataframe (not numpy)

        columns - list, list of columns names to apply the transformation on.
                  If empty it will work on all numeric columns
        :return: DataFrame, with the transformed values in the wanted columns.
        """

        if columns is None:
            columns = []

        self.columns = columns
        self.Transformer_model = StandardScaler()


class P_MaxAbsScaler(PandasTransformer):
    def __init__(self, columns=None):
        """
        Like a MaxAbsScaler. returns  a dataframe (not numpy)

        columns - list, list of columns names to apply the transformation on.
                  If empty it will work on all numeric columns
        :return: DataFrame, with the transformed values in the wanted columns.
        """
        if columns is None:
            columns = []
        self.columns = columns
        self.Transformer_model = MaxAbsScaler()


class P_MinMaxScaler(PandasTransformer):
    def __init__(self, columns=None):
        """
        Like a MinMaxScaler, but it returns  a dataframe
        columns - list.list of columns names to apply the transformation on.If empty it will work on all numeric columns
        :return: DataFrame, with the transformed values in the wanted columns.
        """
        if columns is None:
            columns = []

        self.columns = columns
        self.Transformer_model = MinMaxScaler()


class P_SimpleImputer(PandasTransformer):
    def __init__(self, columns=None, **kwargs):
        """
    Like a SimpleImputer, but it returns  a dataframe
    columns - list, list of columns names to apply the transformation on.If empty it will work on all numeric columns
    :return: DataFrame, with the transformed values in the wanted columns.
    """
        if columns is None:
            columns = []

        self.columns = columns
        self.Transformer_model = SimpleImputer(**kwargs)


class P_SelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func=f_classif, k=10, DealWithNegValues=0):
        """
        Like a SelectKBest, but it returns  a dataframe
        score_func - callable, default = f_classif
                    Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single
                     array with scores.
        k - int or “all”, default=10
            Number of top features to select. The “all” option bypasses selection, for use in a parameter search.
        DealWithNegValues - What to do with negative values:
                                                            0 - Don't do anything
                                                            1 - Use MinMaxScaler
                                                            2 - Add minimum value for each column
        :return: DataFrame, with the selected columns.
        """
        self.score_func = score_func
        self.k = k
        self.Transformer_model = SelectKBest(score_func=self.score_func, k=self.k)
        self.NegValueProcess = DealWithNegValues

    def fit(self, X, y=None, **kwargs):
        X_fit_new = X.copy()

        if self.NegValueProcess == 1:
            MinMax = P_MinMaxScaler()
            X_fit_new = MinMax.fit_transform(X_fit_new, y)
        elif self.NegValueProcess == 2:
            for col in X_fit_new.columns:
                if X_fit_new[col].min() < 0:
                    X_fit_new[col] = X_fit_new[col] + abs(X_fit_new[col].min())

        # run the following in all cases
        self.Transformer_model.fit(X_fit_new, y, **kwargs)
        return self

    def transform(self, X):
        X_new = X.copy()
        X_np = self.Transformer_model.transform(X_new)

        # The results are np array we change them back to dataframe
        mask = self.Transformer_model.get_support()  # list of booleans
        new_features = []  # The list of  K best features
        # noinspection PyTypeChecker
        for Flag, feature in zip(mask, X_new.columns):
            if Flag:
                new_features.append(feature)

        X_newDF = pd.DataFrame(X_np, columns=new_features, index=X_new.index)
        return X_newDF

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y=y, **kwargs)
        return self.transform(X)


class P_LabelEncoder:
    """
    A dataframe LabelEncoder can get columns that shouldn't be transformed
    """

    def __init__(self, ListOfColNotToTransformed=None):
        """

        :param ListOfColNotToTouch: List of columns headers that should not be transformed
        """
        self.ColLabelsDict = defaultdict(LabelEncoder)
        self.TransformedCol = {}
        if ListOfColNotToTransformed is None:
            self.ColNotToTouch = []
        else:
            self.ColNotToTouch = ListOfColNotToTransformed

    def fit_transform(self, Dataframe):
        df = Dataframe.copy()
        # Encoding the variable
        Output = df.apply(lambda x: self.ColConvertor(x, ActionType='fit_transform'))
        return Output

    def transform(self, Dataframe):
        df = Dataframe.copy()
        Output = df.apply(lambda x: self.ColConvertor(x, ActionType='transform'))
        return Output

    def inverse_transform(self, Dataframe):
        df = Dataframe.copy()
        # Inverse the encoded
        Output = df.apply(lambda x: self.ColConvertor(x, ActionType='inverse_transform'))
        return Output

    def ColConvertor(self, Col, ActionType):
        if Col.name in self.ColNotToTouch:
            return Col
        colName = Col.name
        NumCol = is_numeric_dtype(Col.dtype)
        if NumCol and not ActionType == 'inverse_transform':
            self.TransformedCol[colName] = False
            return Col
        else:
            self.TransformedCol[colName] = True

        if ActionType == 'fit_transform':
            return self.ColLabelsDict[colName].fit_transform(Col)
        elif ActionType == 'transform':
            return self.ColLabelsDict[colName].transform(Col)

        elif ActionType == 'inverse_transform':
            if self.TransformedCol[colName]:
                return self.ColLabelsDict[colName].inverse_transform(Col)
            else:
                return Col


class BinaryDownSizeTransformer(BaseEstimator, TransformerMixin):
    """
  This transformer reduce a dataframe, so we will get a predefined proportional
  between the "positive value" that we seek and the rest.
  For example: if we have only 10% people that have "yes" in the BUY column, and we want
  the dataframe to contain 25% people who bought then the positiveLabel will be "yes"
  and the TargetCol will be BUY the proportion will be 0.25
  
  PropDesiredOfPos = The proportional of the positive value (number between 0 and 1)
  positiveLabel = What value is considered "positive"
  TargetCol = What is the column to search for the positiveLabel
  Direction = if value is 1 then the records of the non-positive will be removed from the end of
              the dataframe records toward the beginning (meaning the last records).
              If the value 2 is given then use random.
              if value is 0 : Remove the first records (good for time series)
  """

    def __init__(self, PropDesiredOfPos, positiveLabel, TargetCol, Direction=2):
        self.PosLabel = positiveLabel
        self.TargetCol = TargetCol
        self.Prop = PropDesiredOfPos
        self.Direction = Direction

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        posDf = X_new[X_new[self.TargetCol] == self.PosLabel]
        OtherDf = X_new[X_new[self.TargetCol] != self.PosLabel]
        numOfPos = len(posDf)
        numOfOther = len(OtherDf)
        FutureNumOfOther = int((numOfPos / self.Prop) - numOfPos)
        if FutureNumOfOther > numOfOther:
            print('There are ' + str(numOfPos) +
                  ' with the label asked. And there are only ' +
                  str(numOfOther) + ' with other labels.\n Use must enter different proportion.')
            return
        else:
            if self.Direction == 1:
                ReducedOtherDf = OtherDf.iloc[range(FutureNumOfOther, 0, -1), :]
            elif self.Direction == 2:
                tmp = OtherDf.sample(frac=1)
                ReducedOtherDf = tmp.iloc[range(0, FutureNumOfOther), :]
            else:
                ReducedOtherDf = OtherDf.iloc[range(0, FutureNumOfOther), :]
        AllData = pd.concat([posDf, ReducedOtherDf])
        if self.Direction == 2:
            AllData = AllData.sample(frac=1)
        return AllData
