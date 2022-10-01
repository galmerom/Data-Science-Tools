########################################################
# The following module dealls with SQL and all kinds of fits that are required when working with dataframes and SQL.
# It contains the following functions:
# InsertMissingFields - Compares a database table and a dataframe and does the following:
#                       1. Add a field to the database table that appears in the dataframe and not in the database table.
#                       2. If the capitalization of the dataframe is different then in the database then it changes the dataframe column
#                          to fit the database table capitalization.
#                       3. Optional. It converts the dataframe columns type to fit the database table columns type.
# BuildSQLAndSend - Allow us to send 1 record to a database table with the following features:
#                   1. If the record key already exists then it only UPDATEs the record. If not then it INSERT a new record.
#                   2. If the record is existed and a table with the same name and the word archieve exists in the database, then it saves
#                      the existing record to the archieve table and only then it updates the record 
####################################

# Imports
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_bool_dtype
import datetime as dt
import pytz

def InsertMissingFields(df,DB_tableName,connection,typeConverDic=None,adjustFieldType=True,errorType='coerce'):
    """
    This function compares a dataframe to a database table and does the following:
    1. If a column appears in the dataframe but not in the database table, it adds it to the
       database table (but not vice versa).
    2. If the same column exists in different capitalization in the dataframe ver. the database table, then 
       it changes the capitalization to fit the database table.
    3. It can convert the dataframe column type to the database type. The conversion is only done by 
       finding if the column type is a string, date, or numeric. Change it to a
       double/float type if a numeric is required. There is no conversion between double/float and an int or vice versa. If the
       type is different from numeric/date/string, it does nothing.
    
    The function returns a NEW dataframe with the converted columns

    parameters:
    :param df               dataframe. The input dataframe.
    :param DB_tableName     string. The name of the database table
    :param connection       database connection
    :param typeConverDic    dictionary. A dictionary that convert the database type to a category of: 'object' for strings,
                            'Numeric' for numeric types, 'bool' for boolean, 'Datetime' for dates. If none are given then
                             it uses the default dictionary based on Mysql types.

    :param adjustFieldType  bool. If True (default) it adjust the datafram column to fit the database table column types.
                            It support the following conversions (DB: database type, DF: dataframe type):
                            (DB: string, DF: not string), (DB: Datetime, DF: string), (DB: Numeric, DF: string)
                            , (DB: bool, DF: string), (DB: Numeric, DF: bool)
    :param errorType        bool. In case the conversion is not working what should be done. 
                            Can get one of the following: {‘ignore’, ‘raise’, ‘coerce’}
                            If ‘raise’, then invalid parsing will raise an exception.
                            If ‘coerce’, then invalid parsing will be set as NaN.   
                            If ‘ignore’, then invalid parsing will return the input.

    Returns a copy of the dataframe after changes
    """
    df2 = df.copy()
    ConvTypeDic = typeConverDic
    if ConvTypeDic is None:
        ConvTypeDic = {'varchar':'object','char':'object','varbinary':'object','tinytext':'object','text':'object',
                       'mediumtext':'object','longtext':'object','tinyint':'Numeric','smallint':'Numeric','mediumint':'Numeric',
                       'int':'Numeric','integer':'Numeric','bigint':'Numeric','float':'Numeric','double':'Numeric',
                       'double precision':'Numeric','decimal':'Numeric','dec':'Numeric','year':'Numeric','date':'Datetime',
                       'datetime':'Datetime','time':'Datetime'}
    DfCol = list(df2.columns)
    DfCol_lower = [x.lower() for x in DfCol] 
    SQL = 'SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = "'+DB_tableName+'"'
    DBtable = pd.read_sql(SQL,connection)
    TableColumns = DBtable.COLUMN_NAME.unique()
    TableColumnsLower = [x.lower() for x in TableColumns]
    for col in DfCol_lower:
        # Find the original name before lower case
        Colindx = [i for i,x in enumerate(DfCol_lower) if x == col][0]
        OriginalColName = DfCol[Colindx]

        # Find the generic type of the column
        GenerType,DBType = __findGenericType(df2[OriginalColName])
        # Check if the column exists in the table
        if col not in TableColumnsLower:
            SQL = 'ALTER TABLE ' + str(DB_tableName) + ' ADD ' + str(OriginalColName) + ' ' + str(DBType)
            connection.execute(SQL)
            print('Column: "' + str(OriginalColName)+ '" added to table: ' + str(DB_tableName))
        else:
            # In case the same column appears in different capitalization in dataframe ver. the DB table
            # Then convert the dataframe name to fit the table name
            if OriginalColName not in TableColumns:
                DBColindx = [i for i,x in enumerate(TableColumnsLower) if x == col][0]
                DBOriginalColName = TableColumns[DBColindx]
                df2 = df2.rename({OriginalColName:DBOriginalColName},axis=1)
                OriginalColName=DBOriginalColName
                print ('Dataframe column: "' +str(OriginalColName)+'" changed to "' + str(DBOriginalColName) + 
                       '" to fit database table column name')
            # column exist. If asked adjust the type
            if adjustFieldType:
                colType = DBtable[DBtable['COLUMN_NAME'].str.lower()==col]['DATA_TYPE'].iloc[0]
                DbGenerType = __FindTablColGenType(col,colType,ConvTypeDic,DB_tableName)
                if DbGenerType=='Other':
                    continue
                if DbGenerType=='object' and GenerType!='object':
                    try:
                        df2[OriginalColName] = df2[OriginalColName].astype(str)
                        print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a string type as in database table.')
                    except Exception as e:
                        print('The conversion of column: "' +str(OriginalColName)+ 
                              '"  to string type, in the dataframe, did not succeed. The following error generated:' + str(e))
                elif DbGenerType=='Datetime' and GenerType=='object':
                    try:
                        df2[OriginalColName] = pd.to_datetime(df2[OriginalColName], errors = errorType)
                        print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a date type as in database table.')                        
                    except Exception as e:
                        print('The conversion of column: "' +str(OriginalColName)+ 
                              '"  to date type, in the dataframe, did not succeed. The following error generated:' + str(e))
                elif DbGenerType=='Numeric' and GenerType=='object':
                    try:
                        df2[OriginalColName] = pd.to_numeric(df2[OriginalColName], errors = errorType)
                        print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a numeric type as in database table.')
                    except Exception as e:
                        print('The conversion of column: "' +str(OriginalColName)+ 
                              '"  to numeric type, in the dataframe, did not succeed. The following error generated:' + str(e))
                elif DbGenerType=='bool' and GenerType=='object':
                    try:
                        df2[OriginalColName] = df2[OriginalColName].astype(bool)
                        print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a bool type as in database table.')
                    except Exception as e:
                        print('The conversion of column: "' +str(OriginalColName)+ 
                              '"  to bool type, in the dataframe, did not succeed. The following error generated:' + str(e))
                elif DbGenerType=='bool' and GenerType=='Numeric':
                    # Gets True if the numeric value is not zero and False if Zero
                    df2[OriginalColName] = df2[OriginalColName] != 0
                    print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a bool type as in database table.')
                elif DbGenerType=='Numeric' and GenerType=='bool':
                    df2[OriginalColName] = np.where(df2[OriginalColName],1,0)
                    print('Column: "' +str(OriginalColName)+ '" in dataframe converted to a numeric type as in database table.')
                else:
                    continue
                    
    return df2

def __findGenericType(PandasSer):
    GenerType = 'Other'
    DBType = 'text'
    if is_string_dtype(PandasSer):
        GenerType='object'
        DBType = 'text'
    elif is_bool_dtype(PandasSer):
        GenerType='bool'
        DBType = 'bool'
    elif is_numeric_dtype(PandasSer):
        GenerType='Numeric'
        DBType = 'double'
    elif is_datetime64_any_dtype(PandasSer):
        GenerType='Datetime'
        DBType = 'Datetime'

    return GenerType,DBType

def __FindTablColGenType(col,colType,ConvTypeDic,DB_tableName):
    if colType not in ConvTypeDic.keys():
        print('Column: ' + str(col)+ ' in the database table: ' + str(DB_tableName) +' has a type ' + 
              str(colType) +' that is not supported.')
        return 'Other'
    else:
        return  ConvTypeDic[colType]   



##################################################################################################################
# The following functions are used for sending SQL messages to update dada tables in a database.
# It checks if the record exist by checking if a record with the same keys exists.
# If there is such a record it uses the SQL Update statement to update the record.
# If not it uses the INSERT INTO statement.
# It also copy the record to an archive table (with  name: original table name + "_archive"). The archive table
# should contains  the same columns + "ArchiveDate" column as Datetime type.
# Use this function to send SQL: BuildSQLAndSend
# 
# Example of using this module:
# Table2Upd = 'clients_devices'
# Rec = {'idDevice':DeviceID,'idClients':1,'idSite':1,'comment':'Welcome new Device','Start_date':Now}
# KeyRec={'idDevice':DeviceID}
# sql=BuildSQLAndSend(Rec,KeyRec,Table2Upd,connection)

# The Rec dictionary uses keys as the table columns and the values are the values we want to enter the record.
# The KeyRec dictionary is the dictionary that contains keys of the primary keys in the table. (the algorithm will
# use KeyRec dictionary to look if the record is already exists in the database.
##################################################################################################################




def CrtUpdateSQL(RecDic, KeyDic, TblName):
    """
    Creates an UPDATE Sql statement that gets the record in RecDic, the table
    name and the keyDic for the WHERE clause and creates the statement.
    Inputs:
    TblName str. table name
    RecDic dict. Dictionary where the keys are the headers in the table and
    the values are the values of the record.

    Return: str. An SQL statement
    """
    sql = "UPDATE " + TblName + " SET "
    # Update the SET paragraph
    valuesStr = ''
    for i in RecDic.keys():
        key = str(i)
        PreVal = RecDic[i]
        if isinstance(PreVal, str) or isinstance(PreVal, dt.datetime):
            valuesStr = valuesStr + key + ' = "' + str(PreVal) + '", '
        else:
            valuesStr = valuesStr + key + ' = ' + str(PreVal) + ', '

    valuesStr = valuesStr[0:-2]
    # Update the where paragraph
    WhereStr = ''
    for i in KeyDic.keys():
        key = str(i)
        PreVal = KeyDic[i]
        if isinstance(PreVal, str) or isinstance(PreVal, dt.datetime):
            WhereStr = WhereStr + key + ' = "' + str(PreVal) + '" and '
        else:
            WhereStr = WhereStr + key + ' = ' + str(PreVal) + ' and '

    WhereStr = WhereStr[0:-4]
    sql = sql + valuesStr + " WHERE " + WhereStr
    return sql


def CrtInsertSQL(RecDic, TblName):
    """
    Creates an INSERT INTO Sql statement that gets the record in RecDic and
    the table name and creates the statement.
    Inputs:
    TblName str. table name
    RecDic dict. Dictionary where the keys are the headers in the table and
    the values are the values of the record.
    Return str. An SQL statement
    """
    sql = "INSERT INTO " + TblName + " ("
    Keystr = ''
    for i in RecDic.keys():
        Keystr = Keystr + str(i) + ', '
    Keystr = Keystr[0:-2]
    sql = sql + Keystr + ") VALUES ("
    # Update the SET paragraph
    valuesStr = ''
    for i in RecDic.keys():
        PreVal = RecDic[i]
        if isinstance(PreVal, str) or isinstance(PreVal, dt.datetime):
            valuesStr = valuesStr + '"' + str(PreVal) + '", '
        else:
            valuesStr = valuesStr + str(PreVal) + ', '

    valuesStr = valuesStr[0:-2]
    sql = sql + valuesStr + ") "
    return sql


# Checks if a record exist based on key parameters
def CheckIfExists(keyDic, TableName, connection, Debug=False):
    """
    This function gets a keyDic and a table name and find out if there is
    a record in the database that contains this key.
    Input:
    keyDic Dict. Contains the headers and the values of a table
                 as keys and values in a dictionary.
                 The keys are checked agaianst the DB.
    TableName str. The name of the table to check
    connection database conncetion.
    Debug bool. If True then print all SQL statements
    Return: bool. True if the record exists in the table, False if not
    """
    cursor = connection.cursor()
    valuesStr = ''
    for i in keyDic.keys():
        key = str(i)
        PreVal = keyDic[i]
        if isinstance(PreVal, str) or isinstance(PreVal, dt.datetime):
            valuesStr = valuesStr + key + ' = "' + str(PreVal) + '" and '
        else:
            valuesStr = valuesStr + key + ' = ' + str(PreVal) + ' and '

    valuesStr = valuesStr[0:-4]

    sql = 'SELECT * FROM ' + TableName + ' WHERE ' + valuesStr
    if Debug:
        print('CheckIfExists:\n' + sql)
    cursor.execute(sql)
    records = cursor.fetchall()
    # check if the cursor is from pymysql using sqlalchemy
    if str(type(cursor))=="<class 'pymysql.cursors.Cursor'>":
        Desc=cursor.description
        header=[]
        for i in range(0,len(Desc)):
            header.append(Desc[i][0])
    else:
        header = cursor.column_names

    if len(records) > 0:
        rec = dict(zip(header, records[0]))
        for key in rec.keys():
            if rec[key] is None:
                rec[key] = 'null'
        if Debug:
            print('Rec after zip:\n' + str(rec))
        return True, rec
    else:
        return False, 0


def sendSQL(SQL, connection):
    """
    Gets an SQL atatement and a connection object and send it to the DB.
    Inputs:
    SQL str. SQL atatement
    connection mysql.connector object for connecting to DB
    """
    cursor = connection.cursor()
    cursor.execute(SQL)
    connection.commit()


def ArchiveRecord(Record, Archive_table, connection, Debug=False, DefaultTimeZone='Israel'):
    """
    Take the new record and insert it to an archive table with ArchiveDate.
     The archive table should contain all fields + "ArchiveDate" that contains the date of the update
    :param Record dict. A dictionary that contains the header and the value for each record
    :param Archive_table: string. The name of the archive table
    :param connection: the connection to the database
    :param Debug: bool. If True then it prints the SQL statements. Helps when trying to debug
    :param DefaultTimeZone: string. To add the updated time we need the Timezone
    :return:
    """
    Rec = Record.copy()
    Rec['ArchiveDate'] = str(dt.datetime.now(pytz.timezone(DefaultTimeZone)).strftime("%Y-%m-%d %H:%M:%S"))
    if Debug:
        print('Archive record:\n' + str(Rec))
    SQL = CrtInsertSQL(Rec, Archive_table)
    if Debug:
        print('Archive SQL:\n' + SQL)
    sendSQL(SQL, connection)


def BuildSQLAndSend(Rec, KeyRec, Table2Upd, connection, Archive_table=True, Debug=False, DefaultTimeZone='Israel'):
    """
    Gets the record in dictionary format, the key record, the table name and the cursor
    Update the record in the database, if the record exists it overwrite the given fields
    param: Rec dict. A dictionary that contains the header and the value for each record
    param: KeyRec dict. Dictionary that contains only the keys of the table header:value
    param: Table2Upd string. The name of the table to update
    param: connection database connection
    param: Archive_table bool. If true then it send the old record to the
             archive table which has the name: Table2Upd +  "_archive"
    param: Debug bool. If true then print all the SQL statements. False print nothing
    param: DefaultTimeZone string. The timezone used to calculate now() only for the archive date
    return the SQL as a string
    """
    OldRecExist, OldRec = CheckIfExists(KeyRec, Table2Upd, connection, Debug)
    if OldRecExist:
        if Archive_table:
            # send the same record to the archive table
            ArchiveRecord(Rec, Table2Upd + '_archive', connection, Debug, DefaultTimeZone)
        SQL = CrtUpdateSQL(Rec, KeyRec, Table2Upd)
    else:
        SQL = CrtInsertSQL(Rec, Table2Upd)
    if Debug:
        print('Main SQL:\n' + SQL)
    sendSQL(SQL, connection)
    print('SQL sent')
    return SQL
