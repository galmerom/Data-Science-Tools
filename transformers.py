# -*- coding: utf-8 -*-

"""
This modules contains mainly Pandas transformers. Get dataframe as input and returns dataframe
Pandas transformers classes:
    PandasTransformer - General purpose transformer. Get as an input a regular sklearn transformer and returns
                        a dataframe.
    P_StandardScaler -  StandardScaler that returns a dataframe.
    P_MaxAbsScaler -    MaxAbsScaler that returns a dataframe.
    P_MinMaxScaler -    MinMaxScaler that returns a dataframe.
    P_SimpleImputer -   SimpleImputer that returns a dataframe.

Other transformers:
    DownSizeTransformer - Use when a data downsize is needed.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, TransformerMixin


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

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(self, X)


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
    Like a MinMaxScaler but it returns  a dataframe
    columns - list, list of columns names to apply the transformation on.If empty it will work on all numeric columns
    :return: DataFrame, with the transformed values in the wanted columns.
    """
        if columns is None:
            columns = []

        self.columns = columns
        self.Transformer_model = MinMaxScaler()


class P_SimpleImputer(PandasTransformer):
    def __init__(self, columns=None, **kwargs):
        """
    Like a SimpleImputer but it returns  a dataframe
    columns - list, list of columns names to apply the transformation on.If empty it will work on all numeric columns
    :return: DataFrame, with the transformed values in the wanted columns.
    """
        if columns is None:
            columns = []
        from sklearn.impute import SimpleImputer
        self.columns = columns
        self.Transformer_model = SimpleImputer(**kwargs)


class DownSizeTransformer(BaseEstimator, TransformerMixin):
    """
  This transformer reduce a dataframe so we will get a predefined proportional
  between the "positive value" that we seeks and the rest.
  For example: if we have only 10% people that have "yes" in the BUY column and we want 
  the dataframe to contain 25% people who bought then the positiveLabel will be "yes"
  and the TargetCol will be BUY the proportion will be 0.25
  
  PropDesiredOfPos = The proportional of the positive value (number between 0 and 1)
  positiveLabel = What value is considered "positive"
  TargetCol = What is the column to search for the positiveLabel
  Direction = if value is 1 then the records of the non positive will be taken from the end of
              the dataframe records toward the beginning (meaning the last records).
              If the value 2 is given then use random.
              Other value that is not 1 will take the first records
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
