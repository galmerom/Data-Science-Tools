##################################################################################################################
# This module includes all kind of codes snippets that we use often. 
# The following functions are available:
# 
# File handlings:
#   ReadCsvDirectory2Pandas - used for reading many csv files into one dataframe
#   PickleSave - Save an object as pickle
#   PickleLoad - Load object from pickle
#   OpenZipFilesInDirectory - Open all zip files in a directory to the same directory or to a new one
#   MoveFilesWithSpecificExtention - Move files between directories that contains a specific extention.
#
# Scoring:
#   Scoring - Gets 2 series and return r^2 and RMSE and if asked it also show a chart
#   ErrScoreSlicer - show the perentage score with LOT (limit of detection) by category
# Column manipulation:
#   pdChangeColLoc - Change the location of a column in dataframe
#   CategValueSeries -Gets a series and a list of bins and returns a series with categories (bins), similar to
#                       histogram bins.
#   AddBasicFeatures - Add difference and ratio columns between columns (from a given list)
# Model related:
#   FindFeatureImportanceRMSE - Takes a model and the input and show the importance of the CHANGE in each feature
#                               in terms of RMSE
# Compare dataframes:
#   FindDiffBetweenDfByKey - Gets 2 dataframes and find the extra records that are in table2 compare to table1 by
#                              key list
# Other:
#   PandasIf - do a simple if with pandas series if(condition,TrueValue,FalseValue)
########################################################################################################################

# Imports
import os
import sys
import zipfile
import shutil
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import pickle


def ReadCsvDirectory2Pandas(DirectoryPath, **kwargs):
    """
    Gets a directory path that contains csv files and returns a dataframe that
    contains all the concated data from all the files. At the end of the dataframe it
    adds 2 new columns:
    1. Called "FileName" that includes the name of the original file.
    2. Called "OrigIndex" that include the original index from the original file.
    
    parameters:
    DirectoryPath str. The directory path
    **kwarg dictionary. Contains all  arguments that is needed for pd.read_csv
    
    Example for calling the function:
    argDic={'squeeze':True}
    df = ReadCsvDirectory2Pandas('/gdrive/...",**argDic)

    Returns dataframe
    """
    if DirectoryPath[-1] != "/":
        DirectoryPath = DirectoryPath + "/"
    First_Flag = True
    if len(os.listdir(DirectoryPath)) == 0:
        print('Empty directory')
        return
    for f in os.listdir(DirectoryPath):
        if os.path.isfile(DirectoryPath + f):
            if First_Flag:
                data = pd.read_csv(DirectoryPath + f, **kwargs)
                data['FileName'] = f
                First_Flag = False
            else:
                tmpdata = pd.read_csv(DirectoryPath + f, **kwargs)
                tmpdata['FileName'] = f
                data = pd.concat([data, tmpdata], axis=0, copy=True)
    data = data.reset_index()
    data = data.rename({'index': 'OrigIndex'}, axis=1)
    return data


def NoNegative(Inpseries):
    """
    Gets a pandas/numpy series, change all negative values to zero
    """
    Outseries = np.where(Inpseries < 0, 0, Inpseries)
    return Outseries


def ConcatDataFrameDict(DFdic, AddOriginCol=True):
    """
    Gets a dictionary of dataframes and perform a concat (on the index axis)

    :param DFdic dictionary. Dictionary with dataframes as the values
    :param AddOriginCol bool. If true then add 'OriginalKey' column that
                             contains the key of each dataframe from the dict.
    Return dataframe. Dataframe that concat all the dictionary dataframes
    """
    FirstFlag = True
    for key in DFdic.keys():
        if FirstFlag:
            OutDf = DFdic[key]
            if AddOriginCol:
                OutDf['OriginalKey'] = key
            FirstFlag = False
        else:
            tempdf = DFdic[key]
            if AddOriginCol:
                tempdf['OriginalKey'] = key
            OutDf = pd.concat([OutDf, tempdf])
    return OutDf


def Scoring(y_true, y_pred, colorSer=None, WithChart=False, Figsize=(15, 7), ylabel='Predicted values',
            xlabel='Actual values',
            Title='Actual ver. predicted', LOD=0.00001, OutLierType='Manual', DBSCAN_Parm={'eps': 5, 'min_samples': 5},
            ShowOutliertxtFrom=9999, OutlierXMinMax=None, MaxOutlier=100, AnnotFontSize=12, PercThreshold=0):
    """
    This fucnction gets 2 series and compare them wirh the following scores: R^2 and RMSE.
    It can also draw a chart if needed.
    input parameters:
    y_true              series. The actual values
    y_pred              series. The predicted values
    WithChart           bool. Show a chart or not. If true, the following parameters that can be changed:
        Figsize         tuple. chart size
        ylabel          string. y-axis description
        xlabel          string. x-axis description
        Title           string. Title of chart
    LOD                 float. LOD = Limit of detection. Under this number we assume that the value that we got is zero
    OutLierType         string. Can get "DBSCAN" and then it will use DBSCAN algorithm to find outliers.
    DBSCAN_Parm         dict. Get the input parameters for the DBSCAN model such as eps and min_samples
    ShowOutliertxtFrom  float. Show annotation text to outlier point if the value of (y_true/y_pred) is greater
                               than 1+ ShowOutliertxtFrom or smaller than 1 - ShowOutliertxtFrom.
                                To avoid showing very small outliars we filter out any true values that
                               are between min and max given in the OutlierXMinMax parameter.
                               The annotation text includes (index,true value, pred value)
    OutlierXMinMax      tuple.(Min_Value,Max_value)Used with ShowOutliertxtFrom to filter out only outliers that
                                their x value is between the min and max value.
    AnnotFontSize       int. The font size of the annotation
    MaxOutlier          int. Max number of outliers to show
    PercThreshold       float. When using this parameter the Percent scoring will have 2 values one for an average of
                        RMSE for y_true under this value+ another value for the average value of Percent scoring
                         for y_true over this value.
    Returns:            tuple. (string that show the results, float.R^2 result,float RMSE result,
                        if ShowOutliertxtFromis not default then it returns the ourliers dataframe )
                        The result string includes r^2, rmse, Percent scoring (if shows the average for all records
                        of abs(y_true-y_pred)/y_pred when ever the values of each series is under the LOD barrier it
                        takes either zero or LOD value).(check __ErorCalc function for exact algorithm)
                        If PercThreshold is not zero then it also shows the average RMSE of scores upto the Threshold
                        and an average percentage of all values over that Threshold.
                        The Threshold is defined by the True series.
    """
    r2 = '{:.3f}'.format(r2_score(y_true, y_pred))
    rmse = '{:.3f}'.format(np.sqrt(mean_squared_error(y_true, y_pred)))
    # In case the input is numpy instead of a series
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true, name='y_true')
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, name='y_pred')
    joinedDF = pd.concat([y_true, y_pred], axis=1)
    joinedDF = joinedDF.rename(columns={joinedDF.columns[0]: 'y_true', joinedDF.columns[1]: 'y_pred'})
    col1 = joinedDF.columns[0]
    col2 = joinedDF.columns[1]
    joinedDFUnderThreshold = joinedDF[joinedDF['y_true'] < PercThreshold]
    joinedDFOverThreshold = joinedDF[joinedDF['y_true'] >= PercThreshold]
    PercScore = joinedDFOverThreshold.apply(lambda x: __ErorCalc(x[col1], x[col2], LOD), axis=1).mean()

    if len(joinedDFUnderThreshold) > 0:
        RMSE_under_Threshold = '{:.3f}'.format(
            np.sqrt(mean_squared_error(joinedDFUnderThreshold['y_true'], joinedDFUnderThreshold['y_pred'])))

    Diff = y_true - y_pred
    if len(joinedDFUnderThreshold) > 0:
        ReturnStr = 'R-squared: ' + str(r2) + '   RMSE:' + str(rmse) + '   Percent scoring: ' + \
                    str('{:.1%}'.format(PercScore) + '   RMSE under ' + str(
                        PercThreshold) + ': ' + RMSE_under_Threshold)
    else:
        ReturnStr = 'R-squared: ' + str(r2) + '   RMSE:' + str(rmse) + '   Percent scoring: ' + str(
            '{:.1%}'.format(PercScore))

    colorDic = {}  # This dict. is only used if we use chart with colors
    if WithChart:
        MaxValue = max(max(y_true), max(y_pred))
        MinValue = min(min(y_true), min(y_pred))
        MaxValue = MaxValue + 0.05 * (
                    MaxValue - MinValue)  # add a bit to the right so the max point will not be on the end of the chart

        ###### Find Outlier #######

        if ShowOutliertxtFrom != 9999:
            if OutlierXMinMax is None:
                Mask = [True] * len(y_true.index)
            else:
                Mask = (y_true >= OutlierXMinMax[0]) & (y_true <= OutlierXMinMax[1])
            MaxLimit = pd.Series((1 + ShowOutliertxtFrom), index=y_true[Mask].index)
            MinLimit = pd.Series((1 - ShowOutliertxtFrom), index=y_true[Mask].index)
            TempDF = pd.DataFrame()
            TempDF['y_pred'] = y_pred[Mask]
            TempDF['y_true'] = y_true[Mask]
            TempDF['TrueOverPred'] = np.where(TempDF['y_pred'] != 0, TempDF['y_true'] / TempDF['y_pred'], 0)
            TempDF['Outlier'] = (~(TempDF['TrueOverPred'].between(MinLimit, MaxLimit)))
            TempDF['Max'] = MaxLimit
            TempDF['Min'] = MinLimit
            print('First 5 outliers')
            print(TempDF[TempDF['Outlier']].head(5).to_markdown())
        ####### find and deal with colors ########
        if isinstance(colorSer, pd.Series):
            if len(colorSer.unique()) >= 10:
                colorlist = list(colors.ColorConverter.colors.keys())
            else:
                colorlist = list(mcolors.TABLEAU_COLORS)  # This list contains only 10 colors with big contrast
            colorDic = dict(zip(colorSer.unique(), colorlist[0:len(
                colorSer.unique())]))  # create a dictionary with unique values and colors
            ColorInput = colorSer.map(colorDic)
        else:
            ColorInput = None

        ###### start plotting ######
        plt.figure(figsize=Figsize)
        scatter = plt.scatter(x=y_true, y=y_pred, c=ColorInput, label=ColorInput)
        markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colorDic.values()]
        plt.legend(markers, colorDic.keys(), loc='best', numpoints=1)
        plt.plot([MinValue, MaxValue], [MinValue, MaxValue], 'k-', color='r')

        if ShowOutliertxtFrom != 9999:
            for indx in range(0, len(TempDF[TempDF['Outlier']].iloc[0:MaxOutlier])):
                txt = "(" + str(indx) + "," + str(TempDF.iloc[indx].y_true.round(1)) + "," + str(
                    TempDF.iloc[indx].y_pred.round(1)) + ")"
                plt.annotate(txt,
                             (round(TempDF.iloc[indx].y_true * 1.015, 3), round(TempDF.iloc[indx].y_pred * 1.015, 3)),
                             fontsize=AnnotFontSize)

        if OutLierType == 'DBSCAN':
            dbs = DBSCAN(**DBSCAN_Parm)
            df = pd.DataFrame()
            df['x'] = y_true
            df['y'] = y_pred
            cluster = pd.Series(dbs.fit_predict(df[['x', 'y']]))
            Outliar = cluster[cluster == -1]
            df2 = df.iloc[Outliar.index.tolist()]
            SmallChangeInY = (df['y'].max() - df['y'].min()) * 0.03
            SmallChangeInX = (df['x'].max() - df['x'].min()) * 0.03

            for indx in df2.index:
                txt = "(" + str(indx) + "," + str(df2.loc[indx]['x'].round(1)) + "," + str(
                    df2.loc[indx]['y'].round(1)) + ")"
                plt.annotate(txt, (df2.loc[indx]['x'] - SmallChangeInX, df2.loc[indx]['y'] - SmallChangeInY),
                             fontsize=AnnotFontSize)
            print(df2.to_markdown())
        # Set x and y axes labels
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plt.xlim(MinValue, MaxValue)
        plt.ylim(MinValue, MaxValue)

        plt.title(Title + '\n' + ReturnStr)

        plt.show()
    if ShowOutliertxtFrom != 9999:
        return (ReturnStr, float(r2), float(rmse), float(PercScore), TempDF[TempDF['Outlier']])
    else:
        return (ReturnStr, float(r2), float(rmse), float(PercScore))


def __ErorCalc(y_true, y_pred, LOD):
    """
    Gets one value y_true and one y_pred and LOD value.
    The result usually is abs(y_pred-y_true)/y_true.
    sometimes one of the values is less than LOD, and then we
    change the algorithm a bit.
    """
    if y_true >= LOD > y_pred:
        return abs(LOD - y_true) / y_true
    elif y_true < LOD and y_pred < LOD:
        return 0
    elif y_true < LOD <= y_pred:
        return abs(y_pred - LOD) / LOD
    else:
        return abs(y_pred - y_true) / y_true


def CategValueSeries(InputSer, BucketList, NewSeriesName='CatgSer', AddLeadingZeros=False):
    """
    This function gets a series and a list of values and returns a series with categories.
    (if the value is equal to an element in the bucket list, then it will get the bucket of the smaller one)
    Example:
    Input series = [-10,0,12,13,25,10,3,60]
    BucketList = [0,10,20,30]
    The output will be:
    Output series = [' Less than -10','0-10','10-20','10-20','20-30','10-20','0-10','30+']
    :param InputSer     pd.series  The input series
    :param BucketList   list. A list that defines the buckets limits
    :param NewSeriesName string. The name of the new series
    :param AddLeadingZeros bool. If True then the output string will contain leading zeros for easy sorting
    return pd.series
    
    Example of use:
    train_df['Param_categ'] = CategValueSeries(train_df['Param'],[0,10,20,30,40,60])

    """
    Bucklst = sorted(BucketList)
    # Find the most digits in the list
    MaxNumOfDigits = len(str(max(abs(x) for x in Bucklst)))
    # Create a new list to store the output.
    output_list = Bucklst
    if AddLeadingZeros:
        output_list = []
        # Iterate over the input list and add leading zeros to each number.
        for num in Bucklst:
            output_list.append("{0:0{1}}".format(num, MaxNumOfDigits))
    # Build a dataframe to use for calculations
    Outdf = pd.DataFrame()
    Outdf['Original'] = InputSer
    Outdf[NewSeriesName] = ""
    # Deal with the values BEFORE the first element and AFTER the last element in the bucket list
    Outdf.loc[Outdf['Original'] < Bucklst[0], NewSeriesName] = ' Less than ' + str(output_list[0])
    Outdf.loc[Outdf['Original'] > Bucklst[-1], NewSeriesName] = str(output_list[-1]) + '+'
    # Deal with the values BETWEEN the first element and the last
    for i in range(0, len(Bucklst) - 1):
        FirstElem = str(Bucklst[i])
        if len(FirstElem) == 1 and FirstElem != '0':
            FirstElem = '0' + FirstElem
        Outdf.loc[Outdf['Original'].between(Bucklst[i], Bucklst[i + 1], inclusive='left'), NewSeriesName] = str(
            output_list[i]) + '-' + str(output_list[i + 1])
    return Outdf[NewSeriesName]


def ErrScoreSlicer(df, TrueSer, PredSer, SliceDic, LOD=0):
    """
    Takes True and pred series and give them a score then find the mean values per category.
    The category can be given as another series if: 
    CutByCategory is a string (The category column name in the dataframe) or
    CutByCategory is a list of values and the list will be used to make the category
    Parameters:
    df             dataframe. The input dataframe
    TrueSer        string. The true value column name
    PredSer        string. The Predicted value column name
    SliceDic       dict. {series,list of values} get a dictionary with column name and a list 
                   that indicates how to slice 
                   for example: [0,10,30] The categories will be: less than 0,0-10,10-30,30+
    LOD             double. LOD=Limit of detection that will be the smallest number that we suppose
                    to detect. Smaller numbers will get the values of LOD for scoring
    Returns (dataframe with the results, dataframe with the ungroup values)
    example how to use:
    Slicer = {'Param1':[0,10,20,40],'Param2':[0,5,10,20,40,60]}
    Sumdf,df2 = ErrScoreSlicer(test_df,'TrueCol','Pred_col',Slicer,LOD)
    """
    GroupByList = []
    df2 = df.copy()
    for col in SliceDic.keys():
        catgName = col + '_Category'
        df2[catgName] = CategValueSeries(df2[col], SliceDic[col])
        GroupByList.append(catgName)
    df2['Score'] = df2.apply(lambda x: __ErorCalc(x[TrueSer], x[PredSer], LOD), axis=1)
    df2['LOD'] = LOD
    df2['Diff'] = abs(df2[TrueSer] - df2[PredSer])

    RelvFields = [col for col in GroupByList]
    RelvFields.extend([TrueSer, PredSer, 'Score', 'LOD', 'Diff'])
    df2 = df2[RelvFields]

    # summerize
    OutDF = df2.groupby(GroupByList).agg(['count', 'mean'])
    OutDF.columns = [''.join(col).strip() for col in OutDF.columns.values]
    OutDF = OutDF.rename({'Scoremean': 'Score', 'Diffmean': 'Rmse', 'Diffcount': 'NumOfElements'}, axis=1)
    OutDF = OutDF[['Score', 'Rmse', 'NumOfElements']]
    OutDF['Score'] = OutDF['Score'].astype(float).map("{:.2%}".format)
    OutDF = OutDF.sort_index()
    df2['Score_perc'] = df2['Score'].astype(float).map("{:.2%}".format)
    return OutDF, df2


def PandasIf(whereCond, IfTrue, IfFalse):
    """
    Simple if statement using pandas
    Example: PandasIf(df[A]==0,0,df2[B]/df[A] )
    """
    return IfFalse.where(whereCond, IfTrue)


def AddBasicFeatures(df, ListOfCol):
    """
    Gets a list of columns and adds columns with the difference and ration between each column
    df          dataframe. The dataframe to work on (returns a copy)
    ListOfCol   List. List of columns to use
    Return dataframe with the added columns
    """
    df2 = df.copy()
    ColLeft = ListOfCol.copy()  # ColLeft contains the columns that where not used yet
    for col in ListOfCol:
        ColLeft.pop(0)
        for SecondField in ColLeft:
            df2[col + '_minus_' + SecondField] = df2[col] - df[SecondField]
            df2[col + '_over_' + SecondField] = np.where(df[SecondField] == 0, 0, df2[col] / df[SecondField])
    return df2


def FindFeatureImportanceRMSE(X, y, model, diffValinPercList, showchart=True, MaxFeatures=20):
    """
    This function shows the change of overall RMSE for every change of X percentage per feature.
    It takes every feature and multiplies it by (1+perentage of change), runs predict, and shows the CHANGE in RMSE
    compared to the base prediction (no change in values). Then it sorts the features by the change in RMSE
    in absolute value in descending order.
    The multiplication of features is done one by one. Between each prediction, we reset the features
    to the original values.
    For example, if there are 5 features and 2 values in the diffValinPercList, there will be 11 predictions.
    The first for the base prediction(no change) and the rest=(num of features * num of diffValinPercList)

    Parameters:
    X,y                 Dataframe or series. X is used for the prediction, and y is used for calculating RMSE between
                        y and the prediction
    model               model. The model that we will run the command prediction on
    diffValinPercList   list. A list of values to change the features by. For example: diffValinPercList=[0.01,0.1]
                        every feature will multiply by 0.01 and then by 0.1
    showchart           bool. If true, then a bar chart with results will show.
    MaxFeatures         int. The number of features to show in the bar chart 
    """
    OutDF = pd.DataFrame(columns=['Feature', 'diff_Value_Perc', 'Base_RMSE', 'DiffVal_RMSE', 'DiffMaxBase',
                                  'Abs_DiffMaxBase', 'ActualInpMaxOverMin'])
    startTime = dt.datetime.now()
    BaseRMSE = np.sqrt(mean_squared_error(y, model.predict(X).flatten()))
    TimePerIterInMin = (dt.datetime.now() - startTime).total_seconds() / 60.0
    NumOfIter = len(X.columns) * len(diffValinPercList)
    print('Number of iterations: ' + str(NumOfIter) + ' expected time in min. :' + str(TimePerIterInMin * NumOfIter))
    counter = 0
    for diffValue in diffValinPercList:
        for col in X.columns:
            tempdf = X.copy()
            tempdf[col] = (1 + diffValue) * tempdf[col]
            CurrRMSE = np.sqrt(mean_squared_error(y, model.predict(tempdf).flatten()))
            counter += 1
            ExpectedTimeLeftInMin = round(
                (NumOfIter - counter) * (((dt.datetime.now() - startTime).total_seconds() / 60.0) / (counter + 1)), 1)
            print('Iteration: ' + str(counter) + ' out of : ' + str(
                NumOfIter) + ' Curr Feature:' + col + ' Diff Value: ' +
                  str(diffValue) + ' Exp. Min. left: ' + str(ExpectedTimeLeftInMin))
            tempDic = {'Feature': col, 'diff_Value_Perc': diffValue, 'Base_RMSE': BaseRMSE,
                       'DiffVal_RMSE': CurrRMSE, 'DiffMaxBase': (CurrRMSE - BaseRMSE),
                       'Abs_DiffMaxBase': abs(CurrRMSE - BaseRMSE),
                       'ActualInpMaxOverMin': abs(tempdf[col].max() / tempdf[tempdf[col] != 0][col].min())}
            tmpDF = pd.DataFrame.from_dict(tempDic)
            tmpDF.index = [OutDF.index.max() + 1]
            OutDF = pd.concat([OutDF, tmpDF], axis=0)
            OutDF = OutDF.sort_values(by='ActualInpMaxOverMin', ascending=False)
    ### create charts to show the results ######
    if showchart:
        for diff in np.sort(OutDF.diff_Value_Perc.unique()):
            mask = OutDF['diff_Value_Perc'] == diff
            fig = plt.figure()
            ax = OutDF[mask].set_index(['Feature']).iloc[0:MaxFeatures].plot.bar(y='ActualInpMaxOverMin',
                                                                                 figsize=(20, 5), fontsize=16,
                                                                                 legend=False)
            title = 'Change of RMSE per a change of ' + "{0:.1%}".format(diff)
            ax.set_title(title, pad=20, fontdict={'fontsize': 24})
            ax.set_ylabel('Change of RMSE', fontdict={'fontsize': 20})
    return OutDF


def PickleSave(PathPlusName, Object, ShowSucc=True):
    """
    Save an object as pickle
    Parameters:
    PathPlusName    string. Include the path + file name
    Object          object. The Object to pickle
    ShowSucc        bool. If True then show success message

    """
    try:
        with open(PathPlusName, 'wb') as f:
            pickle.dump(Object, f)
            if ShowSucc:
                print('Pickle saved')
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


def PickleLoad(PathPlusName):
    """
    Load an object from pickled file
    Parameters:
    PathPlusName    string. Include the path + file name
    
    Returns the object
    """
    try:
        with open(PathPlusName, 'rb') as f:
            return pickle.load(f)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


def OpenZipFilesInDirectory(DirectoryPath, DestinationPath=None):
    """
    Get a directory with zip files and extract them. Ignore non zip files and directories.
    The files are extracted to the destination directory if given (if not exists then it create the directory).
    If DestinationPath not given then it extacts to the same directory.

    :param DirectoryPath:       string. The path for the directory contaning the zip files
    :param DestinationPath:     string. The path were the extracted files will be saved. If None then use DirectoryPath
    :return: Nothing
    """
    counter = 0  # count the number of zip files

    cwd = os.getcwd()  # keep the name of the current working directory
    if not os.path.isdir(DirectoryPath):
        print('DirectoryPath: ' + str(DirectoryPath) + ' not exists. \nFiles were not extracted')
        return

    for file in os.listdir(DirectoryPath):  # get the list of files
        CurrPath = os.path.join(DirectoryPath, file)
        if zipfile.is_zipfile(CurrPath):  # if it is a zipfile, extract it
            with zipfile.ZipFile(CurrPath) as item:  # treat the file as a zip
                counter += 1
                item.extractall(DestinationPath)  # extract it in the working directory
    print(str(counter) + ' zip files extracted.')
    os.chdir(cwd)


def MoveFilesWithSpecificExtention(InDirectory, OutDirectory, Extension, includePoint=True):
    """
    Move files between directories that contains a specific extention
    :param InDirectory:     string. The path to the input directory
    :param OutDirectory:    string. The path to the Output directory
    :param Extension:       string. The extention to look for.
    :param includePoint:    bool. If true then the extention must include a point at the start of the extention
                            if a point is not included then it is added. If False then an extention without point
                            can be fetched.
    :return: none
    example how to use:
    Util.MoveFilesWithSpecificExtention(OutDirc,OutDirc+'/csvfiles','csv')
    """
    if includePoint and not Extension.startswith("."):
        actualExtention = '.' + Extension
    else:
        actualExtention = Extension
    if not os.path.isdir(OutDirectory):
        os.mkdir(OutDirectory)
    counter = 0
    for file in os.listdir(InDirectory):
        if file.endswith(actualExtention):
            counter += 1
            shutil.move(os.path.join(InDirectory, file), os.path.join(OutDirectory, file))
    print(str(counter) + ' files copied to OutDirectory')


def pdChangeColLoc(df, Col2Move, Col2MoveBefore=None):
    """
    Change the location of a column in a dataframe
    :param df               dataframe. The relevant dataframe
    :param Col2Move         string or a list. The name of the column to move or a list of columns to move
    :param Col2MoveBefore   string. The Col2Move column will be before this column. 
                            If it ramains None then the Col2Move will be moved to the end
    return dataframe
    """
    colList = list(df.columns)
    if Col2MoveBefore not in colList and Col2MoveBefore is not None:
        print('Col2MoveBefore: ' + Col2MoveBefore + ' is not a column in the dataframe.')
        return
    if isinstance(Col2Move, list):
        for elem in Col2Move:
            df = pdChangeColLoc(df, elem, Col2MoveBefore)
        return df
    else:
        if Col2Move not in colList:
            print('Col2Move: ' + Col2Move + ' is not a column in the dataframe.')
            return

        oldindex = colList.index(Col2Move)
        val = colList.pop(oldindex)
        if Col2MoveBefore is None:
            colList.append(val)
        else:
            newindex = colList.index(Col2MoveBefore)
            colList.insert(newindex, val)
        return df[colList]


def FindDiffBetweenDfByKey(df1, df2, keyList):
    """
    Gets 2 dataframes and a list of keys.
    The function returns a dataframe with the records that are in df2 but not in df1
    :param df1: dataframe. The first dataframe
    :param df2: dataframe. The second dataframe
    :param keyList: list. The list of keys to compare the dataframes by
    :return: dataframe. The records that are in df2 but not in df1
    """
    df1 = df1.drop_duplicates(subset=keyList, keep='first')
    df2 = df2.drop_duplicates(subset=keyList, keep='first')
    df1 = df1.set_index(keyList)
    df2 = df2.set_index(keyList)
    df2 = df2.drop(df1.index, errors='ignore')
    return df2.reset_index()
