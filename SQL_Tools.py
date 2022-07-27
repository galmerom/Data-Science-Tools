##################################################################################################################
# This module is udes for sending SQL messages to update dada tables in a database.
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

import datetime as dt
import pytz


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
