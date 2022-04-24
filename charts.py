# -*- coding: utf-8 -*-
"""
This model helps to build complicated charts. It contains the following functions:

BarCharts - Used for building bar charts. The bar charts can be build as one or more charts in sub plots.
StackBarCharts - Used to build a stack bar chart.
HistCharts - Used for building a histogram charts
pairplotVerCol - Used for comparing each 2 features against the target feature. Return a grid of scatter charts
                 with X and y as the features and the value as the target features. Charts are made by matplotlib
pairplotVerColSNS - The same as pairplotVerCol but charts made with seaborn library
AnomalyChart - Use this chart for showing inertia when using k - means
plotCM - Plotting graphical confusion matrix, can also shows classification report
ClassicGraphicCM - like plotCM except it does not get a model and perform a predict (gets y_pred and classes instead)
PlotFeatureImportance - Plot feature importance and return a dataframe
Show_AucAndROC - Show AUC value and if a classifier model is given it also show the ROC chart
BuildMuliLineChart - Built a chart with 2 or more lines. First line is on the left axis and the rest on the right axis

"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def BarCharts(InpList, TitleList, NumRows=1, NumCol=1, ChartType='bar', ChartSize=(15, 5), Fsize=15, TitleSize=30,
              WithPerc=0, XtickFontSize=15, Colorcmap='plasma', Xlabelstr=['', 15], Ylabelstr=['', 15], PadValue=0.3,
              LabelPrecision=0, txt2show=[("", 10)], RotAngle=45, SaveCharts=False):
    """
    Builds a one or more bar charts (use the NumRows and NumCol to determine the grid)
    The charts can be customized using the following parameters:


    :param InpList:  list of dataframes to show
    :param TitleList:  List of titles to appear on the top of the charts
    :param NumRows:  Number of rows of charts
    :param NumCol:  Number of columns of charts
    :param ChartType:  chart type to show default = bar
    :param ChartSize:  The size of each chart
    :param Fsize:  Font size of the data labels
    :param TitleSize: Font size of the title
    :param WithPerc:
                0 or default = data labels + Normalized Percentage
                1 = Only Normalized Percentage
                2 = Only values
                3 = Only Percentage
                Normalized Percentage = The value of column/ sum of all columns
    :param XtickFontSize: The size of the fonts of the x ticks labels
    :param Colorcmap: The color scheme used. Schemas can be found here:
                      https://matplotlib.org/examples/color/named_colors.html
    :param Xlabelstr: Gets a list. First element is the X axis label and the second element is the font size
    :param Ylabelstr: Gets a list. First element is the Y axis label and the second element is the font size
    :param PadValue: Float. the amount of empty space to put around the value label
    :param LabelPrecision: integer. The number of digits after the period in the label value
    :param txt2show: Gets a list of tuples. Each tuple is for each chart. Every tuple must have 4 values:
                     (string to show, font size,position correction of x,position correction of y) for example:
                     txt2show=[('50% of people are men',10,0.1,-0.1)]
                     The position correction values are in percentage of the chart.
                     So if we want to move the textbox 20% (of the chart length) to the right lets put in
                     the third place the value 0.2
    :param RotAngle: The angle for the x axis labels
    :param SaveCharts: If True then every time this function is called the chart is also saved as jpeg
  """

    i = 0
    j = 0
    RemarkAvail = True
    if len(txt2show) == 1 and txt2show[0][0] == "":
        RemarkAvail = False

    if NumRows > 1 or NumCol > 1:
        fig, axes = plt.subplots(nrows=NumRows, ncols=NumCol, figsize=ChartSize)

    if NumRows == 1 and NumCol == 1:
        ax = InpList[0].plot(kind=ChartType, title=TitleList[0], cmap=Colorcmap, figsize=ChartSize)
        ax.title.set_size(TitleSize)
        ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=RotAngle)
        ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
        ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
        if ChartType == 'barh':
            MaxVal = __add_Horizontal_value_labels(ax, Fsize, WithPerc, PadValue=PadValue)
        else:
            MaxVal = __add_value_labels(ax, Fsize, WithPerc, PadValue=PadValue, precision=LabelPrecision)

        if RemarkAvail:
            __AddTextOnTheCorner(ax, txt2show[0])
    elif NumRows == 1:
        for i in range(len(InpList)):
            ax = InpList[i].plot(kind=ChartType, ax=axes[i], title=TitleList[i], cmap=Colorcmap, figsize=ChartSize)
            ax.title.set_size(TitleSize)
            ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=RotAngle)
            ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
            ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
            # ax.set_xticklabels(labels)
            # ax.set_xticklabels(labels)
            if ChartType == 'barh':
                MaxVal = __add_Horizontal_value_labels(ax, Fsize, WithPerc, PadValue=PadValue)
            else:
                MaxVal = __add_value_labels(ax, Fsize, WithPerc, PadValue=PadValue, precision=LabelPrecision)
            if RemarkAvail:
                __AddTextOnTheCorner(ax, txt2show[i])
    else:
        for counter in range(len(InpList)):
            ax = InpList[counter].plot(kind=ChartType, ax=axes[i][j], title=TitleList[counter], cmap=Colorcmap,
                                       figsize=ChartSize)
            ax.title.set_size(TitleSize)
            ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=RotAngle)
            ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
            ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
            # ax.set_xticklabels(labels)
            if ChartType == 'barh':
                MaxVal = __add_Horizontal_value_labels(ax, Fsize, WithPerc, PadValue=PadValue)
            else:
                MaxVal = __add_value_labels(ax, Fsize, WithPerc, PadValue=PadValue, precision=LabelPrecision)

            if RemarkAvail:
                __AddTextOnTheCorner(ax, txt2show[counter])
            counter += 1
            if j < (NumCol - 1):
                j += 1
            else:
                j = 0
                i += 1

    if SaveCharts:
        __SaveCharts(plt, TitleList[0])
    plt.show()


"""Add data labels. Values and percentages"""


def __add_value_labels(ax, Fsize=15, WithPerc=0, spacing=5, PadValue=0.3, precision=0):
    """
    Add labels to the end of each bar in a bar chart.
    :param ax: (matplotlib.axes.Axes): The matplotlib object containing the axes of the plot to annotate.
    :param Fsize: int. The font size
    :param WithPerc: int.
                0 or default = data labels + Normalized Percentage
                1 = Only Normalized Percentage
                2 = Only values
                3 = Only Percentage
                Normalized Percentage = The value of column/ sum of all columns
    :param spacing: int. The distance between the labels and the bars.
    :param PadValue: float The amount of space around the text
    :param precision: int. Number of digits after the dot to show in the label
    :return: The maximum value
    """

    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    Max = max(totals)
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top

        # Use Y value as label and format number with one decimal place
        strValFormat = "{:,." + str(precision) + "f}"
        strPercFormat = "{:." + str(precision) + "%}"
        CompleteLabel = strValFormat + '\n' + strPercFormat
        label = CompleteLabel.format(y_value, y_value / total)
        if WithPerc == 2:
            label = strValFormat.format(y_value)
        elif WithPerc == 1:
            label = strPercFormat.format(y_value / total)
        elif WithPerc == 3:
            label = strPercFormat.format(y_value)

        x_value = rect.get_x() + rect.get_width() / 4
        y_value = rect.get_height() / 2
        ax.text(x_value, y_value, label, style='italic',
                bbox={'facecolor': 'bisque', 'alpha': 1, 'pad': PadValue, 'boxstyle': 'round'}, fontsize=Fsize)
    return Max


def __add_Horizontal_value_labels(ax, Fsize=15, WithPerc=0, spacing=5, PadValue=0.3):
    """
    Add labels to the end of each bar in a bar chart. For horizontal bars

    :param ax (matplotlib.axes.Axes):   The matplotlib object containing the axes
                                        of the plot to annotate.
    :param spacing (int): The distance between the labels and the bars.
    :param PadValue (float): The amount of space around the text
    """
    totals = []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    Max = max(totals)
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values

        # If value of bar is negative: Place label below bar
        if x_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top

        # Use x value as label and format number with one decimal place

        label = "{:,.0f}\n{:.0%}".format(x_value, x_value / total)
        if WithPerc == 2:
            label = "{:,.0f}".format(x_value)
        elif WithPerc == 1:
            label = "{:.0%}".format(x_value / total)

        x_value = rect.get_width() / 2
        y_value = rect.get_y() + rect.get_height() / 4
        ax.text(x_value, y_value, label, style='italic',
                bbox={'facecolor': 'bisque', 'alpha': 1, 'pad': PadValue, 'boxstyle': 'round'}, fontsize=Fsize)
    return Max


"""Add the small remark in the corner"""


def __AddTextOnTheCorner(ax, str2Show):
    """
    Add a remark on the chart (usually at the corner).
    :param ax:  A plt object
    :param str2Show: tuple: (string to show, font size,position correction of x,position correction of y)
    :return: nothing
    """
    if len(str2Show) == 4:
        text, FontSize, correctionX, correctionY = str2Show
        ax.text(0 + correctionX, 1 + correctionY, text, transform=ax.transAxes,
                bbox={'facecolor': 'gold', 'alpha': 1, 'pad': 0.3, 'boxstyle': 'round'}, fontsize=FontSize)


def StackBarCharts(InpList, TitleList, NumRows=1, NumCol=1, ChartType='bar', ChartSize=(15, 5), Fsize=10, TitleSize=30,
                   WithPerc=0, XtickFontSize=15, ColorInt=0, Xlabelstr=['', 15], Ylabelstr=['', 15], PadValue=0.3,
                   StackBarPer=False, txt2show=[("", 10)], TopValFactor=1.1, SaveCharts=False):
    """
  Parameters:
      :param InpList =  List of tuples.Dataframes to show. Each element in the list is a tuple.
                        The tuple looks like this (df,xCol,LegendCol,ValueCol):
                        df=dataframes
                        xCol = The name of the column we want to use for the X axis
                        LegendCol = The name of the column we want to use as a LegendCol
                        ValueCol = The name of the column we want to use as the values

      :param TitleList = List of titles to appear on the top of the charts
      :param NumRows = Number of rows of charts
      :param NumCol = Number of columns of charts
      :param ChartType = chart type to show default = bar
      :param ChartSize = The size of each chart
      :param Fsize =  Font size of the data labels
      :param TitleSize = Font size of the title
      :param WithPerc =
                        0 or default = data labels + Percentage
                        1 = Only percentage
                        2= Only values
      :param XtickFontSize = The size of the fonts of the x ticks labels
      :param ColorInt =Currently there are 5 color pallets use (0,1,2,3,4) to run them
      :param Xlabelstr = Gets a list. First element is the X axis label and the second
                         element is the font size
      :param Ylabelstr = Gets a list. First element is the Y axis label and the second element is the font size
      :param PadValue = The padding of the data labels bbox
      :param StackBarPer =  If true then the stack bar is showing 100% stack bar.
                            If false then it is a regular values stack bar
      :param txt2show = List of tuples. Each tuple contains (string,integer,integer,integer).
                        The text will show on the chart in a box. The second parameter (integer)
                         is the font size. The third parameter is the correction in the location
                          of the box in the X-axis. The last integer is the correction on the y-axis.
      :param TopValFactor: float. The max value of the y-axis is determined by the max value in the chart * TopValFactor
      :param SaveCharts = Bool. If True then it will save the chart as a jpeg file (use for presentations)

    """

    if ColorInt > 4:
        ColorInt = 0

    i = 0
    j = 0

    if NumRows > 1 or NumCol > 1:
        fig, axes = plt.subplots(nrows=NumRows, ncols=NumCol, figsize=ChartSize)

    if NumRows == 1 and NumCol == 1:
        ax, maxVal = __CreateStackBarDetails(InpList[0], TitleList[0], PadVal=PadValue, StackBarPer=StackBarPer,
                                             ChartSizeVal=ChartSize, FsizeVal=Fsize, WithPerc=WithPerc,
                                             ColorInt=ColorInt)
        ax.title.set_size(TitleSize)
        ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=45)
        ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
        ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
        ax.set_ylim(top=maxVal * TopValFactor)
        __AddTextOnTheCorner(ax, txt2show[0])
    # elif NumRows == 1:
    #     for i in range(len(InpList)):
    #         ax = InpList[i].plot(kind=ChartType, ax=axes[i], title=TitleList[i], stacked=True)
    #         ax.title.set_size(TitleSize)
    #         ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=45)
    #         ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
    #         ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
    #         # ax.set_xticklabels(labels)
    #         __add_value_labels(ax, Fsize, WithPerc)
    # else:
    #     for counter in range(len(InpList)):
    #         ax = InpList[counter].plot(kind=ChartType, ax=axes[i][j], title=TitleList[counter], cmap=Colorcmap,
    #                                    figsize=ChartSize, stacked=True)
    #         ax.title.set_size(TitleSize)
    #         ax.xaxis.set_tick_params(labelsize=XtickFontSize, rotation=45)
    #         ax.set_xlabel(Xlabelstr[0], fontsize=Xlabelstr[1])
    #         ax.set_ylabel(Ylabelstr[0], fontsize=Ylabelstr[1])
    #         # ax.set_xticklabels(labels)
    #         __add_value_labels(ax, Fsize, WithPerc)
    #         counter += 1
    #         if j < (NumCol - 1):
    #             j += 1
    #         else:
    #             j = 0
    #             i += 1

    if SaveCharts:
        __SaveCharts(plt, TitleList[0])
    plt.show()


"""Create the stack bar from scratch"""


def __CreateStackBarDetails(tupleParam, titleVal, TitleSize=20, PadVal=0.3, StackBarPer=False, ChartSizeVal=(10, 7),
                            FsizeVal=10, WithPerc=0, ColorInt=0):
    """
    xCol = The column we want to be in x axis
    LegendCol = The column that we want to be the legend
    ValueCol  = The column that we want to be the values
  """
    DataLabelLocation = []
    dfOriginal, xCol, LegendCol, ValueCol = tupleParam

    # Copy the original dataframe so if we add records it will not have an effect on the original
    df = dfOriginal.copy()

    # Add records with zero values to a combination of xCol and LegendCol that is not exist in the original dataframe
    for xColVal in df[xCol].unique():
        for LegendValue in df[LegendCol].unique():
            if len(df[(df[xCol] == xColVal) & (df[LegendCol] == LegendValue)]) == 0:
                tempDic = {xCol: [xColVal], LegendCol: [LegendValue], ValueCol: [0]}
                tmpDF = pd.DataFrame.from_dict(tempDic)
                tmpDF.index = [df.index.max() + 1]
                df = df.append(tmpDF)
    df = df.sort_values(by=xCol)

    fig, ax = plt.subplots(figsize=ChartSizeVal)

    LegendVal = df[LegendCol].drop_duplicates()

    margin_bottom = np.zeros(len(df[xCol].drop_duplicates()))
    # colors = ["#CC0000", "#FF8000","#FFFF33","#66FFB2","#66FFB2"]
    colors = ["#A20101", "#F44E54", "#FF9904", "#FDDB5E", "#BAF1A1", "76AD3B"]
    colors2 = ["#3C3C86", "#863C5A", "#865A3C", "#DB6262", "#DCE4FC"]
    colors3 = ["#DCE4FC", "#FCE7DC", "#524F48", "#DCfCE4", "#7C6D6E"]
    colors4 = ["#4E0035", "#803468", "#B2027A", "#354D01", "#F703AA"]
    colors5 = ["#28180D", "#D24647", "#D27D46", "#D2B546", "#E6B798"]
    ColorList = [colors, colors2, colors3, colors4, colors5]

    for num, Leg in enumerate(LegendVal):
        values = list(df[df[LegendCol] == Leg].loc[:, ValueCol])
        x = df[df[LegendCol] == Leg].plot.bar(x=xCol, y=ValueCol, ax=ax, stacked=True,
                                              bottom=margin_bottom,
                                              color=ColorList[ColorInt][num], label=Leg,
                                              title=titleVal)
        margin_bottom += values

    if StackBarPer:
        __ReArrangeStackBar2percent(ax)

    if StackBarPer:
        __add_value_labels2StackBar(x, PadValue=PadVal, WithPerc=3, Fsize=FsizeVal)
        ax.set_ylim(0, 1)
    else:
        __add_value_labels2StackBar(x, PadValue=PadVal, Fsize=FsizeVal, WithPerc=WithPerc)

    return ax, max(margin_bottom)


"""This function change the data to be 100% stack bar"""


def __ReArrangeStackBar2percent(ax):
    RecTotal = {}
    # Update the total for every column
    for i in ax.patches:
        if i.get_x() in RecTotal:
            RecTotal[i.get_x()] += (i.get_height())
        else:
            RecTotal[i.get_x()] = i.get_height()

    # Starts the dic. with zeros per column
    RecValues = {}
    for i in ax.patches:
        RecValues[i.get_x()] = 0

    # update the height according to percentage
    counter = 0
    for i in ax.patches:
        PercVal = float(i.get_height() / RecTotal[i.get_x()])
        i.set_y(RecValues[i.get_x()])
        i.set_height(PercVal)
        RecValues[i.get_x()] = RecValues[i.get_x()] + PercVal  # keeps in dictionary the last value for the column

    return ax


"""This function deals with the labels"""


def __add_value_labels2StackBar(ax, Fsize=12, WithPerc=0, spacing=2, PadValue=0.3, StackBarPer=False):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    totals = {}
    for i in ax.patches:
        if i.get_x() in totals:
            totals[i.get_x()] += (i.get_height())
        else:
            totals[i.get_x()] = i.get_height()

    y_value = {}
    for i in ax.patches:
        y_value[i.get_x()] = 0

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value[rect.get_x()] += rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value[rect.get_x()] < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if WithPerc == 0:
            label = "{:,.0f} -> {:.0%}".format(rect.get_height(), rect.get_height() / totals[rect.get_x()])
        elif WithPerc == 2:
            label = "{:,.0f}".format(rect.get_height())
        elif WithPerc == 1:
            label = "{:.0%}".format(rect.get_height() / totals[rect.get_x()])
        elif WithPerc == 3:
            label = "{:.0%}".format(rect.get_height())

        x_value = rect.get_x() + rect.get_width() / 4

        ax.text(x_value, y_value[rect.get_x()] - 0.5 * rect.get_height(), label, style='italic',
                bbox={'facecolor': 'bisque', 'alpha': 1, 'pad': PadValue, 'boxstyle': 'round'}, fontsize=Fsize)


def __SaveCharts(pltObject, FileName):
    # noinspection PyBroadException
    try:
        pltObject.savefig(FileName + '.jpg', dpi=300)
    except:
        return


def HistCharts(InpList, TitleList, NumRows, NumCol, ChartSize=(25, 15), Fsize=15, TitleSize=30, WithPerc=0, binSize=50,
               SaveCharts=False):
    """
    Parameters:
      InpList = list of dataframes to show
      TitleList = List of titles to appear on the top of the charts
      NumRows = Number of rows of charts
      NumCol = Number of columns of charts
      ChartSize = The size of each chart
      Fsize =  Font size of the data labels
      TitleSize = Font size of the title
      WithPerc =
                0 or default = data labels + Percentage
                1 = Only percentage
                2= Only values
      binSize = int. How many bins to use for the histogram
      SaveCharts = Bool. If True then it will save the chart as a jpeg file (use for presentations)
  """

    i = 0
    j = 0

    empty = pd.DataFrame.from_dict({'a': (5, 4)})

    if NumRows > 1 and NumCol > 1:
        fig, axes = plt.subplots(nrows=NumRows, ncols=NumCol, figsize=ChartSize)

    if NumRows == 1 and NumCol == 1:
        if type(InpList[0]) != type(empty):
            InpList[0] = pd.DataFrame(InpList[0])
        ax = InpList[0].plot(kind='hist', title=TitleList[0], cmap='plasma', bins=binSize, figsize=ChartSize)
        ax.title.set_size(TitleSize)
        ax.xaxis.set_tick_params(labelsize=10, rotation=45)
        # __add_value_labels(ax,Fsize,WithPerc)
    elif NumRows == 1:
        for i in range(len(InpList)):
            ax = InpList[i].plot(kind='hist', ax=axes[i], title=TitleList[i], cmap='plasma', bins=binSize,
                                 figsize=ChartSize)
            ax.title.set_size(TitleSize)
            ax.xaxis.set_tick_params(labelsize=10, rotation=45)
            # __add_value_labels(ax,Fsize,WithPerc)
    else:
        for counter in range(len(InpList)):
            ax = InpList[counter].plot(kind='hist', ax=axes[i][j], title=TitleList[counter], cmap='plasma',
                                       bins=binSize)
            ax.title.set_size(TitleSize)
            ax.xaxis.set_tick_params(labelsize=10, rotation=45)
            # __add_value_labels(ax,Fsize,WithPerc)
            counter += 1
            if j < (NumCol - 1):
                j += 1
            else:
                j = 0
                i += 1

    if SaveCharts:
        __SaveCharts(plt, TitleList[0])
    plt.show()


def pairplotVerCol(DF, TargetCol, Figsize=(15, 5), Xlabelstr=15, Ylabelstr=15, RotAngle=45, C='DarkBlue', S=30):
    """
    Show a chart for each feature against the target column. Using matplotlib.

    :param DF: Dataframe as an input.
    :param TargetCol: string. The target column.
    :param Figsize: tuple, The figure size.
    :param Xlabelstr: string. The label of the x-axis.
    :param Ylabelstr: string. The label of the y-axis.
    :param RotAngle: integer. The rotation of the labels in the x-axis.
    :param  C:  In case of a scatter plot. Color of data points. Can get a name of color, an RGB or even a column name.
                See scatter matplotlib documentation
    :param S: In case of a scatter plot how big should be the points. See scatter matplotlib documentation
    :return: nothing
    """
    warnings.filterwarnings("ignore", message="More than 20 figures have been opened")

    for col in DF.drop([TargetCol], axis=1).columns:
        # noinspection PyBroadException
        try:
            tempDF = DF[[col, TargetCol]]
            if is_bool_dtype(DF[col].dtype):
                ax = tempDF.boxplot(by=col, column=TargetCol, figsize=Figsize)
            elif is_numeric_dtype(DF[col].dtype):
                ax = tempDF.plot(col, TargetCol, kind='scatter', figsize=Figsize, c=C, s=S)
            elif is_string_dtype(DF[col].dtype):
                ax = tempDF.boxplot(by=col, column=TargetCol, figsize=Figsize)

            # From here it is common for all data types
            ax.set_title(col + ' ver. ' + TargetCol)
            ax.xaxis.set_tick_params(labelsize=15, rotation=RotAngle)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.set_ylabel(TargetCol, fontsize=18)
            ax.set_xlabel(col, fontsize=18)
            ax.title.set_size(18)
            ax.plot()
        except:
            print('Not able to show a chart for column: ' + str(col) + '\t Data type:' + str(DF[col].dtype))


def pairplotVerColSNS(DF, TargetCol, Figsize=(15, 5), Xlabelstr=15, Ylabelstr=15, RotAngle=45, S=50,
                      UseTargetAsHue=False, ChangeAxis=False,Savepng=False):
    """
    Show a chart for each feature against the target column. Using matplotlib.

    :param DF: Dataframe as an input
    :param TargetCol: string. The target column.
    :param Figsize: tuple, The figure size.
    :param Xlabelstr: string. The label of the x-axis.
    :param Ylabelstr: string. The label of the y-axis.
    :param RotAngle: integer. The rotation of the labels in the x-axis.
    :param S: In case of a scatter plot: how big should be the points.
    :param UseTargetAsHue: bool. If true then use the target column value also as the hue value of the chart.
                           (determine the colors based on the values)
    :param ChangeAxis: bool. If false then f(x) is on the y asix (default) if true then change the axis so f(x)is on the x-axis
    :param Savepng: bool. If True then every chart will be saved in png format
    :return: nothing
    """
    warnings.filterwarnings("ignore", message="More than 20 figures have been opened")

    for col in DF.drop([TargetCol], axis=1).columns:
        plt.figure(figsize=Figsize)
        plt.title(col + ' ver. ' + TargetCol)
        # Find out which column should be on which axis
        X=col
        Y=TargetCol
        if ChangeAxis:
          Y=col
          X=TargetCol
          
        # noinspection PyBroadException
        try:
            tempDF = DF[[col, TargetCol]]
            if is_bool_dtype(DF[col].dtype):
                ax = sns.boxplot(x=X, y=Y, data=tempDF)
            elif is_numeric_dtype(DF[col].dtype):
                if UseTargetAsHue:
                    ax = sns.scatterplot(x=X, y=Y, data=tempDF, s=S, hue=TargetCol)
                else:
                    ax = sns.scatterplot(x=X, y=Y, data=tempDF, s=S)
            elif is_string_dtype(DF[col].dtype):
                ax = sns.boxplot(x=X, y=Y, data=tempDF)
            if Savepng:
              plt.savefig(col+' ver ' + TargetCol + '.png')
        except:
            print('Not able to show a chart for column: ' + str(col) + '\t Data type:' + str(DF[col].dtype))


"""# Anomaly chart"""


def AnomalyChart(X, model):
    """
  The function gets a np array X and an outlier model, AFTER FITTING, such as:
   Isolation forest
   Local Outlier Factor (LOF)
   One-Class Svm

   It draws a contour chart with the outliers as a black dots
  """
    n = int(model.get_params()['contamination'] * len(X))
    xx1, xx2 = np.meshgrid(np.linspace(-7, 7, 100),
                           np.linspace(-7, 7, 100))
    Z = model.decision_function(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    ax = plt.subplot(1, 1, 1)
    # Background colors
    ax.contourf(xx1, xx2, Z,
                levels=np.linspace(Z.min(), 0, 6),
                cmap=plt.cm.Blues_r)
    # Threshold contour
    a = ax.contour(xx1, xx2, Z,
                   levels=[0],
                   linewidths=2, colors='red')
    # Inliers coloring
    ax.contourf(xx1, xx2, Z,
                levels=[0, Z.max()],
                colors='orange')
    # Inliers scatter
    b = ax.scatter(X[:-n, 0], X[:-n, 1],
                   c='white', s=30, edgecolor='k')
    # Outliers scatter
    c = ax.scatter(X[-n:, 0], X[-n:, 1],
                   c='black', s=30, edgecolor='k')
    ax.set_xlim((-7, 7))
    ax.set_ylim((-7, 7))


plt.show()

"""# Inertia elbow chart"""


def __calc_inertia(k, model, data):
    model = model(n_clusters=k).fit(data)
    return model.inertia_


def InertiaElbow(data, model, StartFrom=1, EndAt=10, AddLabels=False):
    inertias = [(k, __calc_inertia(k, model, data)) for k in range(StartFrom, EndAt)]
    plt.figure(figsize=(10, 5))
    plt.plot(*zip(*inertias), linewidth=3, marker='*', markersize=15, markerfacecolor='red', markeredgecolor='#411a20')
    plt.title('Inertia vs. k', fontdict={'fontsize': 20, 'color': '#411a20'})
    plt.xlabel('k', fontdict={'fontsize': 20, 'color': '#411a20'})
    plt.ylabel('Inertia', fontdict={'fontsize': 20, 'color': '#411a20'})
    if AddLabels:
        for i, j in inertias:
            plt.annotate(str(int(j)), xy=(i, j), xytext=(i + 0.2, j + 0.2))


# Confusion matrix
def plotCM(X, y_true, modelName,
           normalize=False,
           title=None,
           cmap=plt.cm.Blues,
           precisionVal=2,
           titleSize=15,
           fig_size=(7, 5),
           InFontSize=15,
           LabelSize=15,
           ClassReport=True,
           RemoveColorBar=False,
           ShowAUCVal=False,
           pos_label=1):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input:
      X:            The input dataframe
      y_true:       Target column
      modelName:    The model used to predict AFTER FIT
      normalize:    If True then normalize the by row
      title:        string. Chart title
      cmap:         color map
      precisionVal: Precision values (0.00 = 2)
      titleSize:    Title font size
      fig_size:     Figure size
      InFontSize:   The font of the values inside the table
      LabelSize:    Label font size (the classes names on the axes)
      ClassReport:  If true add a classification report at the bottom
      RemoveColorBar: bool. If True then don't show the color bar
      ShowAUCVal: bool. If true then show the auc value and ROC chart
      pos_label: str. The positive value for calculating the AUC
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    y_pred = modelName.predict(X)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = modelName.classes_

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if not RemoveColorBar:
        ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title
           )
    ax.xaxis.set_tick_params(labelsize=LabelSize)
    ax.yaxis.set_tick_params(labelsize=LabelSize)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.' + str(precisionVal) + 'f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontdict={'fontsize': InFontSize})
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true)) - 0.5)
    plt.ylim(len(np.unique(y_true)) - 0.5, -0.5)
    plt.xlabel(xlabel='Predicted label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.ylabel(ylabel='True label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.title(title + '\n', fontdict={'fontsize': titleSize, 'color': '#411a20'})
    plt.show()
    if ClassReport:
        print('\n\nClassification_report\n*********************\n')
        print(classification_report(y_true=y_true,
                                    y_pred=y_pred))
    if ShowAUCVal:
        Show_AucAndROC(y_true, y_pred, pos_label, modelName,X)


def ClassicGraphicCM(y_pred, y_true, ModelClasses, normalize=False, title=None, cmap=plt.cm.Blues, precisionVal=2,
                     titleSize=15, fig_size=(7, 5), InFontSize=15, LabelSize=15, ClassReport=True, ReturnAx=False,
                     RemoveColorBar=False, ShowAUCVal=False, pos_label=1):
    """
    This function prints and plots the confusion matrix. WITHOUT using the model (no prediction needed)
    Normalization can be applied by setting `normalize=True`.
    Input:
        y_Pred:         Prediction array
        y_true:         Target array
        ModelClasses:   A list of classes as they appear in model.classes_
        normalize:      If True then normalize the by row
        title:          Chart title
        cmap:           color map
        precisionVal:   Precision values (0.00 = 2)
        titleSize:      Title font size
        fig_size:       Figure size
        InFontSize:     The font of the values inside the table
        LabelSize:      Label font size (the classes names on the axes)
        ClassReport:    If true add a classification report at the bottom
        ReturnAx: Bool. If true then don't show the confusion matrix and return the figure
        RemoveColorBar: bool. If True then don't show the color bar
        ShowAUCVal: bool. If true then show the auc value
        pos_label: str. The positive value for calculating the AUC

    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ModelClasses

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if not RemoveColorBar:
        ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title
           )
    ax.xaxis.set_tick_params(labelsize=LabelSize)
    ax.yaxis.set_tick_params(labelsize=LabelSize)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.' + str(precisionVal) + 'f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontdict={'fontsize': InFontSize})
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true)) - 0.5)
    plt.ylim(len(np.unique(y_true)) - 0.5, -0.5)
    plt.xlabel(xlabel='Predicted label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.ylabel(ylabel='True label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.title(title + '\n', fontdict={'fontsize': titleSize, 'color': '#411a20'})
    if ReturnAx:
        return plt
    else:
        plt.show()
    if ClassReport:
        print('\n\nClassification_report\n*********************\n')
        print(classification_report(y_true=y_true,
                                    y_pred=y_pred))
    if ShowAUCVal:
        Show_AucAndROC(y_true, y_pred, pos_label)


def Show_AucAndROC(y_true, y_pred, pos_label=1, cls=None, X=None):
    """
    Shows the AUC value, and if a classification model is given, it also offers a plot of the ROC curve
    Source code: https://medium.com/@kunanba/what-is-roc-auc-and-how-to-visualize-it-in-python-f35708206663


    :param cls: classifier model. If no classifier is given then it will only show the AUC value
    :param y_true: The actual values (ground true)
    :param y_pred: The predicted values
    :param pos_label: The label that is considered a positive value
    :param X: array. Used for predict proba
    :return: nothing

    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
    result = auc(fpr, tpr)
    print('\n\n AUC value: ' + str(result))
    if cls is not None:
        probas = cls.predict_proba(X)[:, 1]
        roc_values = []
        for thresh in np.linspace(0, 1, 100):
            preds = __get_preds(thresh, probas)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            roc_values.append([tpr, fpr])
        tpr_values, fpr_values = zip(*roc_values)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(fpr_values, tpr_values)
        ax.plot(np.linspace(0, 1, 100),
                np.linspace(0, 1, 100),
                label='baseline',
                linestyle='--')
        plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12)


def __get_preds(threshold, probabilities):
    return [1 if prob > threshold else 0 for prob in probabilities]


def PlotFeatureImportance(X, model, TopFeatures=10, ShowChart=True, Label_Precision=2):
    """
    Show feature importance as a chart and returns a dataframe
    :param X: dataframe. The dataframe used for the fitting
    :param model: classifier. The model used for the fitting
    :param TopFeatures: int. The number of features to show in the chart
    :param ShowChart: bool. If true, then show the chart
    :param Label_Precision: int. The number of digits after the period in the value label
    :return: Dataframe and show chart
    """
    FI = model.feature_importances_
    featuresDic = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X.columns, FI):
        featuresDic[feature] = importance  # add the name/value pair

    # Create a data frame to return
    FI_DF = pd.DataFrame(FI, index=X.columns, columns=['feature_importance'])
    FI_DF = FI_DF.sort_values(by='feature_importance', ascending=False)
    pltDf = FI_DF.head(TopFeatures)
    BarCharts([pltDf], ['Feature importance'], WithPerc=3, LabelPrecision=Label_Precision)
    return FI_DF

def BuildMuliLineChart(df, YFields, FieldDescription=None, rollinWindow=1, FirstAxisLimit=None, SecondAxisLimit=None,
                       XField='dataframe_Index', figsize=(20, 7), linewidth=0, colors=['none'], LabelSizes=(14, 14),
                       yLabels=('FirstField', 'SecondLabel'), LegendBboxCorr=(0.96, 0.965),AnnotLst={},MarkerWidth=2,title=("",16),
                       marker="o"):
    """
    Build a chart with 1 or more lines where the first line gets the left axis and the rest gets the right axis
    Input:
        df = dataframe. The dataframe
        YFields =           List of strings. List of all the fields in the dataframe that we want to see as lines
        FieldDescription =  List of strings. List of descriptions for each field used in the legend. If None then use
                            the name of the fields instead.
        rollinWindow =      integer. The rolling window is used for calculating moving averages. The integer is the
                            number of periods to do the average on.The default is 1 which means no moving average
        FirstAxisLimit =    List. A list of 2 units for the first Y axis. The first is the minimum and the second
                            is the maximum. The default none means no limits
        SecondAxisLimit =   List. A list of 2 units for the first Y axis. The first is the minimum and the second
                            is the maximum. The default none means no limits
        XField =            string. The name of the field to use as the X-axis. If none then use the index.
        figsize =           tuple with a pair of int. Used as the figure size in inches
        linewidth =         integer. The line width. Default =  no line only dots
        colors =            List of strings. List of names allowed by mathplotlib to be line colors.
                            The default allows 6 lines with 6 colors
        LabelSizes =        tuple with a couple of integer. First element is the font size of the x-label.
                            The second is for the y-axis
        yLabels =           tuple with a couple of integer. First elemnt = left y axis label,
                            second  = right y axis label
        LegendBboxCorr =    tuple with a couple of floats. Used to correct the legend label to place it in the
                            right position
        AnnotLst =          Dictionary with a tuple as values and integer as keys. Show strings next to the points of a specific line.
                            The general form: {Line number: ([List of strings], Font size)}
                            First element of a tupple is a list of strings and the second is an integer.
                            It looks like this {0:['Point1,'Point2],20}. The keys of the dictionary refer to the line index
                            in the YFields list. So, the first line gets a value of zero.
                            The values of the dictionary: First elemnt is the list of strings to show
                            second element is the font size.
                            If AnnotLst is empty then nothing will happen
        MarkerWidth =       int or list of int.The size of the marker (usually the size of the point). If int then the size will be the same for all lines.
                            If list then the first element will get the first value and so on.
        title =             tuple. First element is a string that will be the figure title. The second element is the font size.
        marker =            string or a list of string that define the way the marker of the point will look like. if there is only one string then all lines will
                            get the same marker. If not then first line will get the first element in the list and so on.
                            check marker types in the following link: https://matplotlib.org/stable/api/markers_api.html
    return: fig
    """
    NumOfLines = len(YFields)
    lines = []
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=figsize)
    if FieldDescription is None:
        FieldDescription = YFields

    if XField == 'dataframe_Index':
        x = df.index
        xLabel = 'index'
        if not df.index.name is None:
            if len(df.index.name) > 0:
                xLabel = df.index.name
    else:
        x = df[XField]
        xLabel = XField
    if yLabels == ('FirstField', 'SecondLabel'):
        y_labels = (YFields[0], YFields[1])
    else:
        y_labels = yLabels

    if colors == ['none']:
        colors = ['red', 'blue','black' , 'green', 'yellow', 'pink']
    
    # make a plot of the first line
    # Find the type and size of the marker for the first line
    if isinstance(marker, list):
      currMarker=marker[0]
    else:
      currMarker=marker
    if isinstance(MarkerWidth, list):
      currMarWdth=MarkerWidth[0]
    else:
      currMarWdth = MarkerWidth

    # draw the actual first line
    lines.append(
        ax.plot(x, df[YFields[0]].rolling(rollinWindow).mean(), color=colors[0], marker=currMarker, linewidth=linewidth,
                label=FieldDescription[0],markeredgewidth=currMarWdth))
    # set x-axis label
    ax.set_xlabel(xLabel, fontsize=LabelSizes[0])

    # set y-axis label
    ax.set_ylabel(y_labels[0], fontsize=LabelSizes[1], color="red")

    # set y-axis limits
    ax.set_ylim(FirstAxisLimit)
    if len(AnnotLst)>0:
        if 0 in AnnotLst.keys():
            for i, txt in enumerate(AnnotLst[0][0]):
                sizeOfFonts=AnnotLst[0][1]
                plt.annotate(txt, (x[i], df[YFields[0]].iloc[i]),fontsize=sizeOfFonts)
    # Add lines from the second line
    for DrawLine in range(NumOfLines - 1):
        Inx = DrawLine + 1
        
        # Find the type and size of the marker for the first line
        if isinstance(marker, list):
          currMarker=marker[Inx]
        else:
          currMarker=marker
        if isinstance(MarkerWidth, list):
          currMarWdth=MarkerWidth[Inx]
        else:
          currMarWdth = MarkerWidth

        ax2 = ax.twinx()
        lines.append(ax2.plot(x, df[YFields[Inx]].rolling(rollinWindow).mean(), color=colors[Inx], marker=currMarker,
                              linewidth=linewidth, label=FieldDescription[Inx],markeredgewidth=currMarWdth))
        ax2.set_ylim(SecondAxisLimit)
        if len(AnnotLst)>0:
            if Inx in AnnotLst.keys():
                for i, txt in enumerate(AnnotLst[Inx][0]):
                    sizeOfFonts=AnnotLst[Inx][1]
                    plt.annotate(txt, (x[i], df[YFields[Inx]].iloc[i]),fontsize=sizeOfFonts)

    ax2.set_ylabel(y_labels[1], color="blue", fontsize=LabelSizes[1])
    
 
    
    fig.legend(lines, labels=FieldDescription, loc="upper right", borderaxespad=0.1, title="Legend",
               bbox_to_anchor=LegendBboxCorr, shadow=True)
    fig.suptitle(title[0], fontsize=title[1])
    plt.show()
    return fig,ax
