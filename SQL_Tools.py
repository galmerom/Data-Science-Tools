import datetime as dt


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
def CheckIfExists(keyDic, TableName, mycursor):
    """
    This function gets a keyDic and a table name and find out if there is
    a record in the database that contains this key.
    Input:
    keyDic Dict. Contains the headers and the values of a table
                 as keys and values in a dictionary.
                 The keys are checked agaianst the DB.
    TableName str. The name of the table to check
    Return: bool. True if the record exists in the table, False if not
    """
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
    mycursor.execute(sql)
    records = mycursor.fetchall()
    if len(records) > 0:
        return True
    else:
        return False


def sendSQL(SQL, mycursor):
    """
    Gets an SQL atatement and a connection object and send it to the DB.
    Inputs:
    SQL str. SQL atatement
    connection mysql.connector object for connecting to DB
    """
    cursor = mycursor
    cursor.execute(SQL)



def BuildSQLAndSend(Rec, KeyRec, Table2Upd, cursor):
    """
    Gets the record in dictionary format, the key record, the table name and the cursor
    Update the record in the database, if the record exists it overwrite the given fields
    Rec dict. A dictionary that contains the header and the value for each record
    KeyRec dict. Dictionary that contains only the keys of the table header:value
    Table2Upd string. The name of the table to update
    cursor database cursor
    return the SQL as a string
    """
    if CheckIfExists(KeyRec, Table2Upd, cursor):
        SQL = CrtUpdateSQL(Rec, KeyRec, Table2Upd)
        print("Record to be updated to  " + Table2Upd)
    else:
        SQL = CrtInsertSQL(Rec, Table2Upd)
        print("Record to be inserted to  " + Table2Upd)
    # print(SQL)
    sendSQL(SQL, cursor)
    print('SQL sent')
    return SQL
