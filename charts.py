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
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype
from sklearn.metrics import confusion_matrix, classification_report


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
                MaxVal = __add_Horizontal_value_labels(ax, Fsize, WithPerc, PadValue)
            else:
                MaxVal = __add_value_labels(ax, Fsize, WithPerc, PadValue, precision=LabelPrecision)

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
        ax, maxVal = __CreateStackBarDetails(InpList[0], TitleList[0], PadValue, StackBarPer=StackBarPer,
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


def pairplotVerCol(DF, TargetCol, Figsize=(15, 5), Xlabelstr=15, Ylabelstr=15, RotAngle=45, C='DarkBlue', S=20):
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


def pairplotVerColSNS(DF, TargetCol, Figsize=(15, 5), Xlabelstr=15, Ylabelstr=15, RotAngle=45, PointSize=20,
                      UseTargetAsHue=False):
    """
    Show a chart for each feature against the target column. Using matplotlib.

    :param DF: Dataframe as an input
    :param TargetCol: string. The target column.
    :param Figsize: tuple, The figure size.
    :param Xlabelstr: string. The label of the x-axis.
    :param Ylabelstr: string. The label of the y-axis.
    :param RotAngle: integer. The rotation of the labels in the x-axis.
    :param PointSize: In case of a scatter plot: how big should be the points.
    :param UseTargetAsHue: bool. If true then use the target column value also as the hue value of the chart.
                           (determine the colors based on the values)
    :return: nothing
    """
    warnings.filterwarnings("ignore", message="More than 20 figures have been opened")

    for col in DF.drop([TargetCol], axis=1).columns:
        plt.figure(figsize=Figsize)
        plt.title(col + ' ver. ' + TargetCol)

        # noinspection PyBroadException
        try:
            tempDF = DF[[col, TargetCol]]
            if is_bool_dtype(DF[col].dtype):
                ax = sns.boxplot(x=col, y=TargetCol, data=tempDF)
            elif is_numeric_dtype(DF[col].dtype):
                if UseTargetAsHue:
                    ax = sns.scatterplot(x=col, y=TargetCol, data=tempDF, size=PointSize, hue=TargetCol)
                else:
                    ax = sns.scatterplot(x=col, y=TargetCol, data=tempDF, size=PointSize)
            elif is_string_dtype(DF[col].dtype):
                ax = sns.boxplot(x=col, y=TargetCol, data=tempDF)

        except:
            print('Not able to show a chart for column: ' + str(col) + '\t Data type:' + str(DF[col].dtype))


"""## Examples calling pairplot"""

# warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
# pairplotVerColSNS(CityData[CityData['adr']<900],'adr')

# warnings.filterwarnings("ignore", message="More than 20 figures have been opened")
# pairplotVerCol(CityData,'adr')

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
           ClassReport=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input:
      X:            The input dataframe
      y_true:       Target column
      modelName:    The model used to predict AFTER FIT
      normalize:    If True then normalize the by row
      title:        Chart title
      cmap:         color map
      precisionVal: Precision values (0.00 = 2)
      titleSize:    Title font size
      fig_size:     Figure size
      InFontSize:   The font of the values inside the table
      LabelSize:    Label font size
      ClassReport:  If true add a classification report at the bottom

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


def ClassGraphicCM(y_pred, y_true,
                   ModelClasses,
                   normalize=False,
                   title=None,
                   cmap=plt.cm.Blues,
                   precisionVal=2,
                   titleSize=15,
                   fig_size=(7, 5),
                   InFontSize=15,
                   LabelSize=15,
                   ClassReport=True,
                   ReturnAx=False):
    """
    This function prints and plots the confusion matrix.
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
        LabelSize:      Label font size
        ClassReport:    If true add a classification report at the bottom
        ReturnAx: Bool. If true then don't show the confusion matrix and return the figure

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
