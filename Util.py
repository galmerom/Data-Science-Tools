#####################################################################################################
# This module includes all kind of codes snippets that we use often. 
# The following functions are available:
# 
# File handlings:
#   ReadCsvDirectory2Pandas - used for reading many csv files into one dataframe
#
# Scoring:
#   Scoring - Gets 2 series and return r^2 and RMSE and if asked it also show a chart
#####################################################################################################

#Imports
import os
import pandas as pd
from sklearn.metrics import mean_squared_error , r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors

def ReadCsvDirectory2Pandas(DirectoryPath,**kwargs):
    '''
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
    '''
    if DirectoryPath[-1]!="/":
        DirectoryPath = DirectoryPath+"/"
    First_Flag=True
    if len(os.listdir(DirectoryPath))==0:
        print('Empty directory')
        return
    for f in os.listdir(DirectoryPath):
        if os.path.isfile(DirectoryPath+f):
            if First_Flag:
                data = pd.read_csv(DirectoryPath + f,**kwargs)
                data['FileName'] = f
                First_Flag=False
            else:
                tmpdata = pd.read_csv(DirectoryPath + f,**kwargs)
                tmpdata['FileName'] = f
                data = pd.concat([data,tmpdata],axis=0,copy=True)
    data = data.reset_index()
    data = data.rename({'index':'OrigIndex'},axis=1)
    return data


def NoNegative(Inpseries):
    """
    Gets a pandas/numpy series, change all negative values to zero
    """
    Outseries = np.where(Inpseries< 0, 0, Inpseries)
    return Outseries




from matplotlib import colors
import matplotlib.colors as mcolors
def Scoring(y_true,y_pred,colorSer=None,WithChart=False,Figsize=(15,7),ylabel='Predicted values',xlabel='Actual values',Title='Actual ver. predicted',
            LOD=0.00001,ShowOutliertxtFrom=9999,YtrueOutlierMin=0,AnnotFontSize = 12 ):
    '''
    This fucnction gets 2 series and compare them wirh the following scores: R^2 and RMSE.
    It can also draw a chart if needed.
    input parameters:
    y_true series. The actual values
    y_pred series. The predicted values
    WithChart bool. Show a chart or not
    In case there is a chart, then there are default parameters that can be changed:
    Figsize tuple. chart size
    ylabel string. y axis description
    xlabel string. x axis description
    Title string. Title of chart
    LOD float. LOD = Limit of detection. Under this number we assume that the value that we got is zero
    clrTpl tuple. (series,color dictionary) The first elemnent is the series to map. 
                                            The second is a dictionary that maps values (unique values in the series) to colors.
    ShowOutliertxtFrom float. : Show annotation text to outlier point if the value of (y_true/y_pred) is greater than 1+ ShowOutliertxtFrom
                               or smaller than 1 - ShowOutliertxtFrom. To avoid showing very small outliars we filter out any true values that
                               are less than YtrueOutlierMin.
                               The annotation text includes (index,true value, pred value)
    YtrueOutlierMin float. : Used with ShowOutliertxtFrom to filter out small outliers
    AnnotFontSize int.     : The font size of the annotation
    Returns: tuple. (string that show the results, float.R^2 result,float RMSE result, if colors were used then it returns a dictionary
                    between unique series values and the colors that were picked auto.)
                    The result string includes r^2, rmse, Percent scoring (if shows the average for all records of abs(y_true-y_pred)/y_pred
                    when ever the values of each series is under the LOD barrier it takes either zero or LOD value. check __ErorCalc function
                    for exact algorithm
    '''
    r2='{:.3f}'.format(r2_score(y_true, y_pred))
    rmse = '{:.3f}'.format(np.sqrt(mean_squared_error(y_true, y_pred)))
    joinedDF = pd.concat([y_true, y_pred], axis=1)
    col1=joinedDF.columns[0]
    col2 = joinedDF.columns[1]
    PercScore = joinedDF.apply(lambda x: __ErorCalc(x[col1], x[col2],LOD),axis=1).mean()

    Diff = y_true-y_pred

    ReturnStr = 'R-squared: '+str(r2)+'   RMSE:'+str(rmse) + '   Percent scoring: ' + str('{:.1%}'.format(PercScore))   
                                                                                  
    colorDic = {} # This dict. is only used if we use chart with colors
    if WithChart:
        MaxValue=max(max(y_true),max(y_pred))
        MinValue=min(min(y_true),min(y_pred))
        MaxValue = MaxValue+0.05*(MaxValue-MinValue)# add a little to the right so the max point will not be on the end of the chart
        
        ###### Find Outlier #######
        if ShowOutliertxtFrom != 9999:
            MaxLimit = pd.Series((1+ShowOutliertxtFrom), index=y_true[y_true>YtrueOutlierMin].index) 
            MinLimit = pd.Series((1-ShowOutliertxtFrom), index=y_true[y_true>YtrueOutlierMin].index)
            TempDF = pd.DataFrame()
            TempDF['y_pred'] = y_pred[y_true>YtrueOutlierMin]
            TempDF['y_true'] = y_true[y_true>YtrueOutlierMin]
            TempDF['TrueOverPred'] = np.where (TempDF['y_pred']!=0,TempDF['y_true']/TempDF['y_pred'],0)
            TempDF['Outlier'] = (~(TempDF['TrueOverPred'].between(MinLimit, MaxLimit)))
            TempDF['Max'] = MaxLimit
            TempDF['Min'] = MinLimit
            print(TempDF[TempDF['Outlier']])
        ####### find and deal with colors ########
        if isinstance(colorSer, pd.Series):
            if len(colorSer.unique())>=10:
                colorlist =list(colors.ColorConverter.colors.keys())
            else:
                colorlist = list(mcolors.TABLEAU_COLORS) # This list contains only 10 colors with big contrast 
            colorDic = dict(zip(colorSer.unique(),colorlist[0:len(colorSer.unique())])) # create a dictionary with unique values and colors
            ColorInput = colorSer.map(colorDic)
        else:
            ColorInput = None

        ###### start plotting ######
        plt.figure(figsize=Figsize)
        scatter=plt.scatter(x=y_true,y=y_pred,c=ColorInput ,label = ColorInput)
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colorDic.values()]
        plt.legend(markers,colorDic.keys(),loc='best', numpoints=1)
        plt.plot([MinValue, MaxValue], [MinValue, MaxValue], 'k-', color = 'r')

        if ShowOutliertxtFrom != 9999:
            OulierSeiries = TempDF[TempDF['Outlier']]
            if len(TempDF[TempDF['Outlier']])>10: # show no more than 10 values 
                print("show only the first 10 outliers")
                OulierSeiries = TempDF[TempDF['Outlier']].iloc[:20]
                
            for indx in TempDF[TempDF['Outlier']].index:
                txt= "(" + str(indx) + "," + str(TempDF.loc[indx].y_true.round(1)) + "," + str(TempDF.loc[indx].y_pred.round(1))+")"
                plt.annotate(txt, (TempDF.loc[indx].y_true*1.015, TempDF.loc[indx].y_pred*1.015),fontsize=AnnotFontSize)
        
        # Set x and y axes labels
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        plt.xlim(MinValue,MaxValue)
        plt.ylim(MinValue,MaxValue)

        plt.title(Title+'\n'+ReturnStr)
        
        plt.show()
    return ( ReturnStr,float(r2),float(rmse),str(colorDic))

def __ErorCalc(y_true,y_pred,LOD):
    """
    Gets one value y_true and one y_pred and LOD value.
    The result usually is abs(y_pred-y_true)/y_pred.
    sometimes one of the values is less than LOD and the we 
    change the algorithm a bit.
    """
    if y_true >= LOD and y_pred < LOD:
        return (abs(LOD-y_true)/y_true)
    elif y_true < LOD and y_pred < LOD:
        return 0
    elif y_true < LOD and y_pred >=LOD:
        return (abs(y_pred-LOD)/LOD)
    else:
        return (abs(y_pred-y_true)/y_pred)
