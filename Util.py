#####################################################################################################
# This module includes all kind of codes snippets that we use often. 
# The following functions are available:
# 
# File handlings:
#   ReadCsvDirectory2Pandas - used for reading many csv files into one dataframe
#
#####################################################################################################

#Imports
import os
import pandas as pd

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
