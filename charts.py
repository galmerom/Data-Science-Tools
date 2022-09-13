# -*- coding: utf-8 -*-
"""
This model helps to build complicated charts. It contains the following functions:

BarCharts - Used for building bar charts. The bar charts can be built as one or more charts in subplots.
StackBarCharts - Used to build a stacked bar chart.
HistCharts - Used for creating histogram charts
pairplotVerCol - Used for comparing every 2 features against the target feature. Return a grid of scatter charts
                 with X and y as the features and the value as the target features. Charts are made by matplotlib
pairplotVerColSNS - The same as pairplotVerCol but charts made with seaborn library
AnomalyChart - Use this chart to show inertia when using k - means
plotCM - Plotting graphical confusion matrix can also show classification report
ClassicGraphicCM - like plotCM, except it does not get a model and perform a predict (gets y_pred and classes instead)
PlotFeatureImportance - Plot feature importance and return a dataframe
Show_AucAndROC - Show AUC value, and if a classifier model is given, it also shows the ROC chart
BuildMuliLineChart - Built a chart with two or more lines. The first line is on the left axis, the rest are on the
                        right axis
PolyFitResults - Build ski-learn curve fit for polynomials until the 5th degree with intercept and without.
                (no intercept means that when x=0 also y=0)
Scatter - Used for creating a scatter that uses x and y as the location of the point. It also uses DBSCAN to show
            outliers values.
"""


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
from matplotlib.colors import colorConverter
from sklearn.cluster import DBSCAN

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def BarCharts(InpList, TitleList, NumRows=1, NumCol=1, ChartType='bar', ChartSize=(15, 5), Fsize=15, TitleSize=30,
              WithPerc=0, XtickFontSize=15, Colorcmap='plasma', Xlabelstr=['', 15], Ylabelstr=['', 15], PadValue=0.3,
              LabelPrecision=0, txt2show=[("", 10)], RotAngle=45, SaveCharts=False):
    """
    Builds one or more bar charts (use the NumRows and NumCol to determine the grid)
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
    :param Xlabelstr: Gets a list. The first element is the X-axis label and the second element is the font size
    :param Ylabelstr: Gets a list. The first element is the Y-axis label and the second element is the font size
    :param PadValue: Float. the amount of space to put around the value label
    :param LabelPrecision: integer. The number of digits after the period in the label value
    :param txt2show: Gets a list of tuples. Each tuple is for each chart. Every tuple must have 4 values:
                     (string to show, font size,position correction of x,position correction of y) for example:
                     txt2show=[('50% of people are men',10,0.1,-0.1)]
                     The position correction values are in the percentage of the chart.
                     So if we want to move the textbox 20% (of the chart length) to the right let's put in
                     the third place the value 0.2
    :param RotAngle: The angle for the x-axis labels
    :param SaveCharts: If True, then every time this function is called, the chart is also saved as a jpeg

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

    :param ax (matplotlib.axes.Axes):   The matplotlib object containing the plot's axes to annotate.
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
        :param Xlabelstr = Gets a list. The first element is the X-axis label and the second
                           element is the font size
        :param Ylabelstr = Gets a list. The first element is the Y-axis label and the second element is the font size
        :param PadValue = The padding of the data labels bbox
        :param StackBarPer =  If true, the stack bar is showing 100%.
                              If false, then it is a regular values stack bar
        :param txt2show = List of tuples. Each tuple contains (string,integer,integer,integer).
                          The text will show on the chart in a box. The second parameter (integer)
                           is the font size. The third parameter is the correction in the box's location on the X-axis.
                           The last integer is the correction on the y-axis.
        :param TopValFactor: float. The max value of the y-axis is determined by the max value
                                    in the chart * TopValFactor
        :param SaveCharts = Bool. If True, then it will save the chart as a jpeg file (use for presentations)


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
    xCol = The column we want to be in the x-axis
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
    """
    Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the plot's axes to annotate.
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
      InpList = List of dataframes to show
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
      SaveCharts = Bool. If True, then it will save the chart as a jpeg file (use for presentations)

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
    :param RotAngle: integer. The rotation of the labels on the x-axis.
    :param  C:  In case of a scatter plot. Color of data points. Can get a name of color, an RGB, or even a column name.
                See scatter matplotlib documentation
    :param S: In the case of a scatter plot, how big should the points be. See scatter matplotlib documentation
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
                      UseTargetAsHue=False, ChangeAxis=False, Savepng=False):
    """
    Show a chart for each feature against the target column. Using matplotlib.

    :param DF: Dataframe as an input
    :param TargetCol: string. The target column.
    :param Figsize: tuple, The figure size.
    :param Xlabelstr: string. The label of the x-axis.
    :param Ylabelstr: string. The label of the y-axis.
    :param RotAngle: integer. The rotation of the labels on the x-axis.
    :param S: In the case of a scatter plot: how big should the points be.
    :param UseTargetAsHue: bool. If true, then use the target column value as the chart's hue value.
                           (determine the colors based on the values)
    :param ChangeAxis: bool. If false, then f(x) is on the y axis (default) if true, then change the axis so f(x)
                             is on the x-axis
    :param Savepng: bool. If True, then every chart will be saved in png format
    :return: nothing

    """
    warnings.filterwarnings("ignore", message="More than 20 figures have been opened")

    for col in DF.drop([TargetCol], axis=1).columns:
        plt.figure(figsize=Figsize)
        plt.title(col + ' ver. ' + TargetCol)
        # Find out which column should be on which axis
        X = col
        Y = TargetCol
        if ChangeAxis:
            Y = col
            X = TargetCol

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
                plt.savefig(col + ' ver ' + TargetCol + '.png')
        except:
            print('Not able to show a chart for column: ' + str(col) + '\t Data type:' + str(DF[col].dtype))


"""# Anomaly chart"""


def AnomalyChart(X, model,ylim=(-7, 7),xlim=(-7, 7),FigSize=(10,10)):
    """
  The function gets a np array X and an outlier model, AFTER FITTING, such as:
   Isolation forest
   Local Outlier Factor (LOF)
   One-Class Svm

   It draws a contour chart with the outliers as  black dots.
   Parameters:
   X            dataframe. The input dataframe
   model        model. one of the 3 models above, alreadt fitted.
   ylim,xlim    tuple. The axis limitation for y and x
   FigSize      tuple. The figure size in inches
  """
    n = int(model.get_params()['contamination'] * len(X))
    xx1, xx2 = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                           np.linspace(ylim[0], ylim[1], 100))
    Z = model.decision_function(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
    fig, ax = plt.subplots(1,1, figsize=FigSize)

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
    b = ax.scatter(X.iloc[:-n, 0], X.iloc[:-n, 1],
                   c='white', s=30, edgecolor='k')
    # Outliers scatter
    c = ax.scatter(X.iloc[-n:, 0], X.iloc[-n:, 1],
                   c='black', s=30, edgecolor='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    

"""# Inertia elbow chart"""


def __calc_inertia(k, model, data):
    """
    Used for fit the model to k clusters
    k       int. The number of clusters to use by the model
    model   model. The inertia model
    data    dataframe. The input dataframe
    Returns the odel.inertia_ for each k input
    """
    model = model(n_clusters=k).fit(data)
    return model.inertia_


def InertiaElbow(data, model, StartFrom=1, EndAt=10, AddLabels=False):
    """
    Gets a dataframe and the inertia modeland create a chart to help find where the "elbow" is.
    data        dataframe. The input dataframe
    model       model. The inertia model
    StartFrom   int. This is the minimum k value
    EndAt       int. This is the maximum k value
    AddLabels   bool. If true, then add labels for each point
    Returns nothing
    """
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
def plotCM(X, y_true, modelName, normalize=False, title=None, cmap=plt.cm.Blues, precisionVal=2, titleSize=15,
           fig_size=(7, 5), InFontSize=15, LabelSize=15, ClassReport=True, RemoveColorBar=False,  ShowAUCVal=False,
           pos_label=1):
    """
     This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input:
      X:            The input dataframe
      y_true:       Target column
      modelName:    The model used to predict AFTER FIT
      normalize:    If True, then normalize the by row
      title:        string. Chart title
      cmap:         color map
      precisionVal: Precision values (0.00 = 2)
      titleSize:    Title font size
      fig_size:     Figure size
      InFontSize:   The font of the values inside the table
      LabelSize:    Label font size (the classes names on the axes)
      ClassReport:  If true, add a classification report at the bottom
      RemoveColorBar: bool. If True, then don't show the color bar
      ShowAUCVal: bool. If true, then show the AUC value and ROC chart
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
        Show_AucAndROC(y_true, y_pred, pos_label, modelName, X)


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
        normalize:      If True, then normalize the by row
        title:          Chart title
        cmap:           color map
        precisionVal:   Precision values (0.00 = 2)
        titleSize:      Title font size
        fig_size:       Figure size
        InFontSize:     The font of the values inside the table
        LabelSize:      Label font size (the classes names on the axes)
        ClassReport:    If true, add a classification report at the bottom
        ReturnAx: Bool. If true, then don't show the confusion matrix and return the figure
        RemoveColorBar: bool. If True, then don't show the color bar
        ShowAUCVal: bool. If true, then show the AUC value
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
    It shows the AUC value, and if a classification model is given, it also offers a plot of the ROC curve.
    Source code: https://medium.com/@kunanba/what-is-roc-auc-and-how-to-visualize-it-in-python-f35708206663


    :param cls: classifier model. If no classifier is given, then it will only show the AUC value
    :param y_true: The actual values (ground true)
    :param y_pred: The predicted values
    :param pos_label: The label that is considered a positive value
    :param X: array. Used to predict proba
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
    :param TopFeatures: int. The number of features shown in the chart
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
                       yLabels=('FirstField', 'SecondLabel'), LegendBboxCorr=(0.96, 0.965), AnnotLst={}, MarkerWidth=2,
                       title=("", 16),
                       marker="o", ReturnArtistOnly=False, SavePath=None, showTable=False):
    """
    Build a chart with 1 or more lines where the first line gets the left axis, and the rest gets the right axis
    Input:
        df = dataframe. The dataframe
        YFields =           List of strings. List of all the fields in the dataframe that we want to see as lines
        FieldDescription =  List of strings. List of descriptions for each field used in the legend. If None, then use
                            the name of the fields instead.
        rollinWindow =      integer. The rolling window is used for calculating moving averages. The integer is the
                            number of periods to do the average on. The default is 1, which means no moving average
        FirstAxisLimit =    List. A list of 2 units for the first Y axis. The first is the minimum, and the second
                            is the maximum. The default none means no limits.
        SecondAxisLimit =   List. A list of 2 units for the first Y axis. The first is the minimum, and the second
                            is the maximum. The default none means no limits.
        XField =            string. The name of the field to use as the X-axis. If none, then use the index.
        figsize =           tuple with a pair of int. Used as the figure size in inches
        linewidth =         integer. The line width. Default =  no line only dots
        colors =            List of strings. List of names allowed by mathplotlib to be line colors.
                            The default allows 6 lines with 6 colors.
        LabelSizes =        tuple with a couple of integers. The first element is the font size of the x-label.
                            The second is for the y-axis
        yLabels =           tuple with a couple of integers. First element = left y-axis label,
                            second  = right y-axis label
        LegendBboxCorr =    tuple with a couple of floats. Used to correct the legend label to place it in the
                            right position
        AnnotLst =          Dictionary with a tuple as values and integer as keys. Show strings next to the points of a
                            specific line.
                            The general form: {Line number: ([List of strings], Font size)}
                            The first element of a tuple is a list of strings, and the second is an integer.
                            It looks like this {0:['Point1,'Point2],20}. The dictionary's keys refer to the line
                            index in the YFields list. So, the first line gets a value of zero.
                            The values of the dictionary: The first element is the list of strings to show
                            the second element is the font size.
                            If AnnotLst is empty, then nothing will happen.
        MarkerWidth =       int or list of int.The size of the marker (usually the size of the point). If int, then the
                            size will be the same for all lines. If list, then the first element will get the first
                             value, and so on.
        title =             tuple. The first element is a string that will be the figure title. The second element is
                            the font size.
        marker =            string or a list of strings that define how the marker of the point will look.
                            If there is only one string, all lines will
                            get the same marker. If not, then the first line gets the 1st element in the list and so on.
                            check marker types in the following link: https://matplotlib.org/stable/api/markers_api.html
        ReturnArtistOnly=   bool. If False  (default), show the chart before the end of the function. If True, then
                            don't show the chart and only return the
                            artist that can be used to add more lines to the chart.
        SavePath =          str. If the path is different from None, save the chart as an image (png format).
                            The SavePath contains the path and the file name of the image
        showTable =         bool. If true, then show a table with the values from the dataframe. Show only YFields.
    return: fig, ax1,ax2
    code example:
    tempdf=df[(df['Param1']=='xxx')&(df['Param2']>0)]
    yfields = ['Param1','Param2','Param3','Param4']
    charts.BuildMuliLineChart(tempdf, yfields,XField='Param5',linewidth=1,LegendBboxCorr=(0.8,0.75))

    """

    warnings.filterwarnings('ignore')
    NumOfLines = len(YFields)
    lines = []
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots(figsize=figsize)
    if FieldDescription is None:
        FieldDescription = YFields

    if XField == 'dataframe_Index':
        x = df.index
        xLabel = 'index'
        if df.index.name is not None:
            if len(df.index.name) > 0:
                xLabel = df.index.name
    else:
        x = df[XField]
        xLabel = XField
    if yLabels == ('FirstField', 'SecondLabel'):
        if NumOfLines > 1:
            y_labels = (YFields[0], YFields[1])
        else:
            y_labels = (YFields[0])
    else:
        y_labels = yLabels

    if colors == ['none']:
        colors = ['red', 'blue', 'black', 'green', 'yellow', 'pink']

    # make a plot of the first line
    # Find the type and size of the marker for the first line
    if isinstance(marker, list):
        currMarker = marker[0]
    else:
        currMarker = marker
    if isinstance(MarkerWidth, list):
        currMarWdth = MarkerWidth[0]
    else:
        currMarWdth = MarkerWidth

    # draw the actual first line
    lines.append(
        ax.plot(x, df[YFields[0]].rolling(rollinWindow).mean(), color=colors[0], marker=currMarker, linewidth=linewidth,
                label=FieldDescription[0], markeredgewidth=currMarWdth))
    # set x-axis label
    ax.set_xlabel(xLabel, fontsize=LabelSizes[0])

    # set y-axis label
    ax.set_ylabel(y_labels[0], fontsize=LabelSizes[1], color="red")

    # set y-axis limits
    ax.set_ylim(FirstAxisLimit)
    if len(AnnotLst) > 0:
        if 0 in AnnotLst.keys():
            for i, txt in enumerate(AnnotLst[0][0]):
                sizeOfFonts = AnnotLst[0][1]
                plt.annotate(txt, (x[i], df[YFields[0]].iloc[i]), fontsize=sizeOfFonts)
    # Add lines from the second line
    for DrawLine in range(NumOfLines - 1):
        Inx = DrawLine + 1

        # Find the type and size of the marker for the first line
        if isinstance(marker, list):
            currMarker = marker[Inx]
        else:
            currMarker = marker
        if isinstance(MarkerWidth, list):
            currMarWdth = MarkerWidth[Inx]
        else:
            currMarWdth = MarkerWidth

        ax2 = ax.twinx()
        lines.append(ax2.plot(x, df[YFields[Inx]].rolling(rollinWindow).mean(), color=colors[Inx], marker=currMarker,
                              linewidth=linewidth, label=FieldDescription[Inx], markeredgewidth=currMarWdth))
        ax2.set_ylim(SecondAxisLimit)
        if len(AnnotLst) > 0:
            if Inx in AnnotLst.keys():
                for i, txt in enumerate(AnnotLst[Inx][0]):
                    sizeOfFonts = AnnotLst[Inx][1]
                    plt.annotate(txt, (x[i], df[YFields[Inx]].iloc[i]), fontsize=sizeOfFonts)
    if NumOfLines > 1:
        ax2.set_ylabel(y_labels[1], color="blue", fontsize=LabelSizes[1])

    fig.legend(lines, labels=FieldDescription, loc="upper right", borderaxespad=0.1, title="Legend",
               bbox_to_anchor=LegendBboxCorr, shadow=True)
    fig.suptitle(title[0], fontsize=title[1])
    if not ReturnArtistOnly:
        plt.show()
    if SavePath is not None:
        fig.savefig(SavePath, dpi=300)

    if showTable:
        if XField == 'dataframe_Index':
            Field2show = YFields.copy()
        else:
            Field2show = YFields.copy()
            Field2show.insert(0, XField)

        print(df[Field2show].to_markdown())
    if NumOfLines == 1:
        return fig, ax
    else:
        return fig, ax, ax2


##### start PolyFitResults ##########

#### possibe fit functions #####
# line


def __poly_1(x, a, b):
    return a + b * x


# parabola
def __poly_2(x, a, b, c):
    return a + b * x + c * (x ** 2)


def __poly_3(x, a, b, c, d):
    return a + b * x + c * (x ** 2) + d * (x ** 3)


def __poly_4(x, a, b, c, d, e):
    return a + b * x + c * (x ** 2) + d * (x ** 3) + e * (x ** 4)


def __poly_5(x, a, b, c, d, e, f):
    return a + b * x + c * (x ** 2) + d * (x ** 3) + e * (x ** 4) + f * (x ** 5)


def __poly_1_no_inter(x, b):
    return b * x


# parabola
def __poly_2_no_inter(x, b, c):
    return b * x + c * (x ** 2)


def __poly_3_no_inter(x, b, c, d):
    return b * x + c * (x ** 2) + d * (x ** 3)


def __poly_4_no_inter(x, b, c, d, e):
    return b * x + c * (x ** 2) + d * (x ** 3) + e * (x ** 4)


def __poly_5_no_inter(x, b, c, d, e, f):
    return b * x + c * (x ** 2) + d * (x ** 3) + e * (x ** 4) + f * (x ** 5)


##### start main function followed by scoring function #####
def PolyFitResults(XInput, yInput, showCharts=True, figureSize=(25, 5), ColorSeries=None):
    """
    Takes the X,Y series and tries to find the coefficients that can adopt X to y using polynomial regression.
    The output is 10 charts that try to fit the polynomial regression.
    Inputs: XInput,yInput both are pandas series that we are looking for the coefficients that by given XInput we will
            get yInput.
    showCharts bool or string. It supports the following:
                               True(default) bool. = Show all charts
                               False bool. = Don't show charts,
                               'Include_inter' string. = Show only charts with intercept,
                               'No_inter' string. = Show only charts without intercept (if x=0, then y=0)
    Figsize tuple. It gets a tuple like this: (x,y), where x is the width of the figure (in inches), and y is the length
                    (in inches)
    Returns:
    The function returns a tuple of 3 objects:
    Curves dataframe = for each record in x and y, we get the values of the 10 polynomials that tried to fit and
                        "connect the dots." This is a dataframe that summarizes the results
    curvesDic dictionary = A dictionary of all the coefficients of all 10 polynomials. The user can pick the best one.
    BestOpt string = The string of the best result (that gives the least RMSE). This string can be used to get
                     the values from the curvesDic dictionary

    """
    X = XInput
    y = yInput
    boundsVal = (0, np.inf)
    popt1, _ = curve_fit(__poly_1, X, y)
    popt2, _ = curve_fit(__poly_2, X, y)
    popt3, _ = curve_fit(__poly_3, X, y)
    popt4, _ = curve_fit(__poly_4, X, y)
    popt5, _ = curve_fit(__poly_5, X, y)
    popt1_no_inter, _ = curve_fit(__poly_1_no_inter, X, y)
    popt2_no_inter, _ = curve_fit(__poly_2_no_inter, X, y)
    popt3_no_inter, _ = curve_fit(__poly_3_no_inter, X, y)
    popt4_no_inter, _ = curve_fit(__poly_4_no_inter, X, y)
    popt5_no_inter, _ = curve_fit(__poly_5_no_inter, X, y)
    # popspecial = curve_fit(poly_special, X, y)
    curves = pd.DataFrame(X.apply(__poly_1, a=popt1[0], b=popt1[1]))
    curves.rename({curves.columns[0]: 'CF1'}, axis=1, inplace=True)
    curves['CF2'] = X.apply(__poly_2, a=popt2[0], b=popt2[1], c=popt2[2])
    curves['CF3'] = X.apply(__poly_3, a=popt3[0], b=popt3[1], c=popt3[2], d=popt3[3])
    curves['CF4'] = X.apply(__poly_4, a=popt4[0], b=popt4[1], c=popt4[2], d=popt4[3], e=popt4[4])
    curves['CF1_no_inter'] = X.apply(__poly_1_no_inter, b=popt1_no_inter[0])
    curves['CF2_no_inter'] = X.apply(__poly_2_no_inter, b=popt2_no_inter[0], c=popt2_no_inter[1])
    curves['CF3_no_inter'] = X.apply(__poly_3_no_inter, b=popt3_no_inter[0], c=popt3_no_inter[1], d=popt3_no_inter[2])
    curves['CF4_no_inter'] = X.apply(__poly_4_no_inter, b=popt4_no_inter[0], c=popt4_no_inter[1], d=popt4_no_inter[2],
                                     e=popt4_no_inter[3])
    curves['X_Input'] = XInput
    curves['y_Input'] = yInput
    curves = curves.sort_values('X_Input')

    # find colors
    if isinstance(ColorSeries, pd.Series):
        colorlist = list(mcolors.ColorConverter.colors.keys())
        colorDic = dict(zip(ColorSeries.unique(), colorlist[0:len(
            ColorSeries.unique())]))  # create a dictionary with unique values and colors
        ColorInput = ColorSeries.map(colorDic)
        print(str(colorDic))
    else:
        ColorInput = 'xkcd:black'

    # find the figure size
    if showCharts is True and figureSize == (25, 5):
        figureSize = (25, 10)

    # Create charts
    if showCharts is not False:
        if showCharts is True or showCharts == 'Include_inter':
            if showCharts == 'Include_inter':
                fig, axs = plt.subplots(1, 4, figsize=figureSize)
                axs[0].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0].plot(curves.X_Input, curves['CF1'], linewidth=3, color='green')
                axs[0].set_title('\n' + 'CF1' + '\n' + _Scoring(curves, 'y_Input', 'CF1'))
                axs[0].legend(['y_true', 'CF1: ${:.2f}+{:.2f}x$'.format(*popt1)], loc='best')
                axs[1].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1].plot(curves.X_Input, curves['CF2'], linewidth=3, color='green')
                axs[1].set_title('\n' + 'CF2' + '\n' + _Scoring(curves, 'y_Input', 'CF2'))
                axs[1].legend(['y_true', 'CF2: ${:.2f}+{:.2f}x+{:.2f}x^2$'.format(*popt2)], loc='best')
                axs[2].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[2].plot(curves.X_Input, curves['CF3'], linewidth=3, color='green')
                axs[2].set_title('\n' + 'CF3' + '\n' + _Scoring(curves, 'y_Input', 'CF3'))
                axs[2].legend(['y_true', 'CF3: ${:.2f}+{:.2f}x+{:.2f}x^2+{:.2f}x^3$'.format(*popt3)], loc='best')
                axs[3].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[3].plot(curves.X_Input, curves['CF4'], linewidth=3, color='green')
                axs[3].set_title('\n' + 'CF4' + '\n' + _Scoring(curves, 'y_Input', 'CF4'))
                axs[3].legend(['y_true', 'CF4: ${:.2f}+{:.2f}x+{:.2f}x^2+{:.2f}x^3+{:.2f}x^4$'.format(*popt4)],
                              loc='best')
            else:
                fig, axs = plt.subplots(2, 4, figsize=figureSize)
                axs[0, 0].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0, 0].plot(curves.X_Input, curves['CF1'], linewidth=3, color='green')
                axs[0, 0].set_title('\n' + 'CF1' + '\n' + _Scoring(curves, 'y_Input', 'CF1'))
                axs[0, 0].legend(['y_true', 'CF1: ${:.2f}+{:.2f}x$'.format(*popt1)], loc='best')
                axs[0, 1].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0, 1].plot(curves.X_Input, curves['CF2'], linewidth=3, color='green')
                axs[0, 1].set_title('\n' + 'CF2' + '\n' + _Scoring(curves, 'y_Input', 'CF2'))
                axs[0, 1].legend(['y_true', 'CF2: ${:.2f}+{:.2f}x+{:.2f}x^2$'.format(*popt2)], loc='best')
                axs[0, 2].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0, 2].plot(curves.X_Input, curves['CF3'], linewidth=3, color='green')
                axs[0, 2].set_title('\n' + 'CF3' + '\n' + _Scoring(curves, 'y_Input', 'CF3'))
                axs[0, 2].legend(['y_true', 'CF3: ${:.2f}+{:.2f}x+{:.2f}x^2+{:.2f}x^3$'.format(*popt3)], loc='best')
                axs[0, 3].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0, 3].plot(curves.X_Input, curves['CF4'], linewidth=3, color='green')
                axs[0, 3].set_title('\n' + 'CF4' + '\n' + _Scoring(curves, 'y_Input', 'CF4'))
                axs[0, 3].legend(['y_true', 'CF4: ${:.2f}+{:.2f}x+{:.2f}x^2+{:.2f}x^3+{:.2f}x^4$'.format(*popt4)],
                                 loc='best')

        if showCharts is True or showCharts == 'No_inter':
            if showCharts == 'No_inter':
                fig, axs = plt.subplots(1, 4, figsize=figureSize)
                axs[0].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[0].plot(curves.X_Input, curves['CF1_no_inter'], linewidth=3, color='green')
                axs[0].set_title('\n' + 'CF1_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF1_no_inter'))
                axs[0].legend(['y_true', 'CF1_no_inter: {:.2f}x$'.format(*popt1_no_inter)], loc='best')
                axs[1].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1].plot(curves.X_Input, curves['CF2_no_inter'], linewidth=3, color='green')
                axs[1].set_title('\n' + 'CF2_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF2_no_inter'))
                axs[1].legend(['y_true', 'CF2_no_inter: {:.2f}x+{:.2f}x^2$'.format(*popt2_no_inter)], loc='best')
                axs[2].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[2].plot(curves.X_Input, curves['CF3_no_inter'], linewidth=3, color='green')
                axs[2].set_title('\n' + 'CF3_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF3_no_inter'))
                axs[2].legend(['y_true', 'CF3_no_inter: {:.2f}x+{:.2f}x^2+{:.2f}x^3$'.format(*popt3_no_inter)],
                              loc='best')
                axs[3].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[3].plot(curves.X_Input, curves['CF4_no_inter'], linewidth=3, color='green')
                axs[3].set_title('\n' + 'CF4_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF4_no_inter'))
                axs[3].legend(
                    ['y_true', 'CF4_no_inter: {:.2f}x+{:.2f}x^2+{:.2f}x^3+{:.2f}x^4$'.format(*popt4_no_inter)],
                    loc='best')
            else:
                axs[1, 0].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1, 0].plot(curves.X_Input, curves['CF1_no_inter'], linewidth=3, color='green')
                axs[1, 0].set_title('\n' + 'CF1_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF1_no_inter'))
                axs[1, 0].legend(['y_true', 'CF1_no_inter: {:.2f}x$'.format(*popt1_no_inter)], loc='best')
                axs[1, 1].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1, 1].plot(curves.X_Input, curves['CF2_no_inter'], linewidth=3, color='green')
                axs[1, 1].set_title('\n' + 'CF2_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF2_no_inter'))
                axs[1, 1].legend(['y_true', 'CF2_no_inter: {:.2f}x+{:.2f}x^2$'.format(*popt2_no_inter)], loc='best')
                axs[1, 2].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1, 2].plot(curves.X_Input, curves['CF3_no_inter'], linewidth=3, color='green')
                axs[1, 2].set_title('\n' + 'CF3_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF3_no_inter'))
                axs[1, 2].legend(['y_true', 'CF3_no_inter: {:.2f}x+{:.2f}x^2+{:.2f}x^3$'.format(*popt3_no_inter)],
                                 loc='best')
                axs[1, 3].scatter(curves.X_Input, curves.y_Input, c=ColorInput)
                axs[1, 3].plot(curves.X_Input, curves['CF4_no_inter'], linewidth=3, color='green')
                axs[1, 3].set_title('\n' + 'CF4_no_inter' + '\n' + _Scoring(curves, 'y_Input', 'CF4_no_inter'))
                axs[1, 3].legend(
                    ['y_true', 'CF4_no_inter: {:.2f}x+{:.2f}x^2+{:.2f}x^3+{:.2f}x^4$'.format(*popt4_no_inter)],
                    loc='best')

    curvesDic = {'CF1': popt1, 'CF2': popt2, 'CF3': popt3, 'CF4': popt4, 'CF1_no_inter': popt1_no_inter,
                 'CF2_no_inter': popt2_no_inter, 'CF3_no_inter': popt3_no_inter, 'CF4_no_inter': popt4_no_inter}
    BestR2 = 0
    BestCol = 'CF1'  # as default take the first column as best column
    for col in curvesDic.keys():
        tmp = r2_score(curves['y_Input'], curves[col])
        if tmp > BestR2:
            BestR2 = tmp
            BestCol = col

    return (curves, curvesDic, BestCol)


def _Scoring(df, y_true, y_pred):
    r2 = '{:.3f}'.format(r2_score(df[y_true], df[y_pred]))
    rmse = '{:.3f}'.format(np.sqrt(mean_squared_error(df[y_true], df[y_pred])))
    return 'R-squared: ' + str(r2) + '   RMSE:' + str(rmse)




def Scatter(dframe, x, y, ClrSeries=None, SizeSeries=None, Title='Default', equalAxis=False, ShowEqualLine=False, markersize=40,
            ShowOutliar=False, OutFont=8,figsize=(20, 7), DBSCAN_Parm={'eps': 5, 'min_samples': 3}, TitleFontSize=20, XAxisLimit=None,
            YAxisLimit=None, LegendFontSize=14, SizeBins=5, BasicSize=40,
            FindBoundries=False, BoundriesBins=20, Bound_SD_max=1, Bound_SD_min=1, BoundryPolyLevel=3,BinsType='EqualPoints'):
    """
    Show a scatter chart from the dataframe that can also show outliers using the DBSCAN model and upper
    and lower boundaries.
    How does the boundaries algorithm work?
    It divides the x-axis into bins (BoundriesBins) and takes the mean x for every bin. The ymin and max are calculated
    as the mean y of each bin plus or minus the SD (standard deviation)*factors
    factors=(Bound_SD_max for max line and Bound_SD_min for min line)
    Then using the points, we use curve fit to find the equation of each line. The polynomial level is determined by
    BoundryPolyLevel parameter (Maximum value is 5)
    Parameters:
    dframe              dataframe. The input dataframe
    x               string. The name of the series that should be on the x-axis
    y               string. The name of the series that should be on the y-axis
    ClrSeries       string. The name of the series that will be used for different colors for each unique value
    SizeSeries      string. The name of the series that will be used for changing the marker size. The algorithm will
                    put each value in a bin. Each bin will get different size
    SizeBins        int. The amount of bins to use when calculating the marker size. Will be used only if SizeBins
                    is not None
    BasicSize       int. Use for caculation of marker size:  Marker size = Number of bin * BasicSize
    Title           string. The chart title
    equalAxis       bool. If True, both axes will have the same minimum and maximum values. For example, it can be used
                        when comparing y_true and y_pred
    markersize      int. The scale of the marker
    ShowOutliar     bool. If True, use the DBSCAN model to find outliers and show the outlier values and index.
    OutFont         int. If ShowOutliar=True, this parameter is the font size that shows the outlier values
    figsize         tuple. A tuple that describes the chart size in inches. (x in inches, y in inches)
    DBSCAN_Parm     dictionary. Used as the parameters for the DBSCAN model
    TitleFontSize   int. The title's fonts size
    XAxisLimit      tuple. (Minumum X-axis value, Maximum X-axis value)
    YAxisLimit      tuple. (Minumum Y axis value, Maximum Y axis value)
    LegendFontSize  int. The font size of the legend. It also changes the marker next to the text in the legend
                        with the same proportions.
    FindBoundries    bool. If true, then it will show upper and lower boundaries.
    BoundriesBins    int. The number of bins that the x-axis should divide to.
    Bound_SD_max     float. The factor to multiply the SD for the MAX boundary line
    Bound_SD_min     float. The factor to multiply the SD for the MIN boundary line
    BoundryPolyLevel int. The level of the polynomial that is used to curve fit the boundaries
    return nothing if FindBoundries=False
    return BoundDF,minEquation,maxEquation if FindBoundries=True
    BoundDF is a dataframe that shows for each bin the x mean and for the y: max, min, mean, and the calculated value
                of y_mean+factor*SD and  y_mean-factor*SD
    minEquation is a six position list that includes the factors of the equation of the minimum line
    maxEquation is a six position list that includes the factors of the equation of the maximum line
    the list looks like this[x**0,x**1,x**2,x**3,x**4,x**5].
    For example if the equation is y=ax+b then the list will look like [b,a,0,0,0,0]
    Example of how to use:
    Scatter(df,'weight','height','gender',ShowOutliar=True,DBSCAN_Parm = {'eps':2,'min_samples':5},markersize=40)
    """
    df=dframe.copy()
    warnings.filterwarnings('ignore')

    ####### Build the COLORS series ###############
    if not isinstance(ClrSeries, type(None)):
        ClrSer = df[ClrSeries]
        UnqVal = ClrSer.unique()
        if len(UnqVal) >= 10:
            colorlist = list(colorConverter.colors.keys())
        else:
            colorlist = list(mcolors.TABLEAU_COLORS)  # This list contains only 10 colors with big contrast
        colorDic = dict(zip(ClrSer.unique(),
                            colorlist[0:len(ClrSer.unique())]))  # create a dictionary with unique values and colors
        ColorInput = ClrSer.map(colorDic)
    else:
        ColorInput = None
    ########## Build the SIZE series ###############
    if not isinstance(SizeSeries, type(None)):
        SizeSer = pd.Series(dframe[SizeSeries],index=dframe.index)
        BinSize = (SizeSer.max()-SizeSer.min())/SizeBins
        SizeSer = (((SizeSer-SizeSer.min())/BinSize).astype('int')+1)*BasicSize
    else:
        SizeSer = markersize
    ##### In case the user wants equal axis (x and y the same max and min)#########    
    if equalAxis or ShowEqualLine::
        MaxValue = max(max(df[x]), max(df[y]))
        MinValue = min(min(df[x]), min(df[y]))

        diffMaxMin = MaxValue - MinValue
        MaxValue = MaxValue + 0.05 * (
            diffMaxMin)  # add a little to the right so the max point will not be on the end of the chart

    ###### start plotting ######

    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.scatter(x=df[x], y=df[y], c=ColorInput, label=ColorInput, s=SizeSer)
    ######### Build legend if color series is given ##########
    if not isinstance(ClrSeries, type(None)):
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', markersize=10 * (LegendFontSize / 14)) for
            color in colorDic.values()]
        plt.legend(markers, colorDic.keys(), loc='best', numpoints=1, fontsize=LegendFontSize)
    ##### Plot diagonal line across the chart #####
    if ShowEqualLine:
        plt.plot([MinValue, MaxValue], [MinValue, MaxValue], 'k-', color='r')
    
    ###### show values of outliers #######
    if ShowOutliar:
        dbs = DBSCAN(**DBSCAN_Parm)

        cluster = pd.Series(dbs.fit_predict(df[[x, y]]))
        Outliar = cluster[cluster == -1]
        df2 = df.iloc[Outliar.index.tolist()]
        SmallChangeInY = (df[y].max() - df[y].min()) * 0.03
        SmallChangeInX = (df[x].max() - df[x].min()) * 0.03

        for indx in df2.index:
            txt = "(" + str(indx) + "," + str(df2.loc[indx][x].round(1)) + "," + str(df2.loc[indx][y].round(1)) + ")"
            plt.annotate(txt, (df2.loc[indx][x] - SmallChangeInX, df2.loc[indx][y] - SmallChangeInY), fontsize=OutFont)

    # Set x and y axes labels
    plt.ylabel(y)
    plt.xlabel(x)
    ##### Add a diagonal line across the chart ###########
    if ShowEqualLine:
        plt.xlim(MinValue, MaxValue)
        plt.ylim(MinValue, MaxValue)
    ##### Add x-axis limits if given ##############
    if not isinstance(XAxisLimit, type(None)):
        plt.xlim(XAxisLimit[0], XAxisLimit[1])
    ##### Add y-axis limits if given ##############
    if not isinstance(YAxisLimit, type(None)):
        plt.ylim(YAxisLimit[0], YAxisLimit[1])
    #### Build title #######
    titlestr = Title
    if Title == 'Default':
        titlestr = 'Scatter of ' + str(x) + ' (x) against ' + str(y) + ' (y)'
    ###### Plot boundries ###################
    if FindBoundries:
        BoundDF, EquationsDic = __findBoundries(df, x, y,
                                                            BoundriesBins, DBSCAN_Parm,
                                                            Bound_SD_max, Bound_SD_min, BoundryPolyLevel,BinsType)
        minEquation=EquationsDic['Min']
        maxEquation=EquationsDic['Max']
        xBound = BoundDF['X_mean'].append(pd.Series(df[x].max())) # add the last point
        plt.plot(xBound, __CalibList(xBound, minEquation), color='red')
        plt.plot(xBound, __CalibList(xBound, maxEquation), color='red')
        df['MinBound'] = df[x].apply(lambda w:  __CalibList(w, minEquation))
        df['MaxBound'] = df[x].apply(lambda w:  __CalibList(w, maxEquation))
        df['InBound'] = df[y].between(df['MinBound'],df['MaxBound'])
        OutOfBound = df[~df['InBound']]
        if len(OutOfBound)>0:
            # print('Points out of boundries:\n\n')
            # print(OutOfBound[[x,y,'MinBound','MaxBound','InBound']].reset_index().to_markdown(index=False,tablefmt="grid"))
            plt.scatter(OutOfBound[x], OutOfBound[y],marker='o', s=markersize*4, facecolors='none', edgecolors='r')
    plt.title(titlestr, fontsize=TitleFontSize)

    plt.show()
    if not isinstance(SizeSeries, type(None)):
        print('Size series '+ SizeSeries+':'+str([BinSize*i for i in range(0,SizeBins)] ) ) 
    if FindBoundries:
        return BoundDF, EquationsDic,OutOfBound


def __findBoundries(df, x, y, BoundriesBins, DBSCAN_Parm, Bound_SD_max, Bound_SD_min, BoundPolyLvl,BinsType):
    """
    Gets dataframe and the x and y + numbers of bins and finds the maximum and minimum boundaries
    param: df:            dataframe. The dataframe that is the input
    param: x:             string. The x column
    param y:              string. The y column
    param BoundriesBins: int. The number of bins to use for the line
    param DBSCAN_Parm:   tuple. The parameters used for the DBSCAN model that removes outliers
    param Bound_SD_max:  float. The upper boundary  = y mean per bin + Bound_SD_max * SD of each bin
    param Bound_SD_min:  float. The lower boundary  = y mean per bin - Bound_SD_min * SD of each bin
    param BoundPolyLvl:  int. The level of the polynomial used to draw the boundaries
                            The higher the level, the curve becomes more fit.
                            MAX level=5
    return Dataframe. Dataframe with x mean for every bin and y min and y max + a column with the bin tuple

    """
    # Find relevant points only
    # dbs = DBSCAN(**DBSCAN_Parm)
    # cluster = pd.Series(dbs.fit_predict(df[[x]],df[[y]]))
    # RelevPoints = df[cluster != -1]
    # notRelevantPoints=df[cluster == -1]
    # print(notRelevantPoints[[x,y]].to_markdown())
    RelevPoints=df
    # Find bins from max and min of the x-axis
    TupleBins = FindBins(df,x,y,BoundriesBins,BinsType)
    # Find min and max point for every bin
    OutDF = pd.DataFrame(columns=['Bin', 'Y_Min', 'Y_max', 'X_mean', 'Y_Mean_Less_x_SD', 'Y_Mean_plus_x_SD'])
    
    #initiate CurrDF so that prevDF can get a value
    if BinsType == 'Linear':
        CurrDF=df[(df[x] >= TupleBins[0][0]) & (df[x] <= TupleBins[0][1])]
    else:
        CurrDF=df.sort_values(by=x).reset_index().iloc[TupleBins[0][0]:TupleBins[0][1]]
    
    # Start moving accross all bins
    for bin in TupleBins:
        prevDF=CurrDF
        if BinsType == 'Linear':
            CurrDF = df[(df[x] >= bin[0]) & (df[x] <= bin[1])]
        else:
            CurrDF=df.sort_values(by=x).reset_index().iloc[bin[0]:bin[1]]
        if len(CurrDF) > 0:
            currYMax = CurrDF[y].max()
            currYMin = CurrDF[y].min()
            currXavg = CurrDF[x].mean()
            currYMean = CurrDF[y].mean()
            currYSd = CurrDF[y].std()
            if  np.isnan(currYSd):
                currYSd=prevDF[y].std()
            currYminWithSD = currYMean - Bound_SD_min * currYSd
            currYmaxWithSD = currYMean + Bound_SD_max * currYSd
            OutDF = OutDF.append({'Bin': bin, 'Y_Min': currYMin, 'Y_max': currYMax, 'X_mean': currXavg,
                                  'Y_Mean': currYMean, 'Y_Sd': currYSd, 'Y_Mean_Less_x_SD': currYminWithSD,
                                  'Y_Mean_plus_x_SD': currYmaxWithSD}, ignore_index=True)
    curves, curvesDic, BestOpt = PolyFitResults(OutDF['X_mean'], OutDF['Y_Mean_Less_x_SD'], showCharts=False)
    minEquation = __CompleteSet("", curvesDic['CF' + str(BoundPolyLvl)])
    curves, curvesDic, BestOpt = PolyFitResults(OutDF['X_mean'], OutDF['Y_Mean_plus_x_SD'], showCharts=False)
    maxEquation = __CompleteSet("", curvesDic['CF' + str(BoundPolyLvl)])
    curves, curvesDic, BestOpt = PolyFitResults(OutDF['X_mean'], OutDF['Y_Mean'], showCharts=False)
    MeanEquation = __CompleteSet("", curvesDic['CF' + str(BoundPolyLvl)])
    EquationsDic={'Max':maxEquation,'Min':minEquation,'Mean':MeanEquation}
    
    return OutDF, EquationsDic

def FindBins(df,x,y,BoundriesBins,BinsType):
    df2=df.copy()
    Bins = []
    if BinsType=='Linear':
        CurrVal = df2[x].min()
        while CurrVal <= df2[x].max():
            Bins.append(CurrVal)
            CurrVal += (df2[x].max() - df2[x].min()) / BoundriesBins
        Bins.append(df2[x].max())
        TupleBins = [(Bins[i], Bins[i + 1]) for i in range(0, len(Bins) - 1)]
    elif BinsType=='EqualPoints':
        df2 = df.sort_values(by=x).reset_index().drop(['index'],axis=1)
        NumberOfPoints = len(df2)
        binSize= NumberOfPoints / BoundriesBins
        df2['Group'] = (df2.index / binSize).astype(int)
        df2.reset_index(inplace=True)
        df2 = df2.rename(columns = {'index':'counter'})
        DF = df2.groupby(['Group']).counter.agg(['min','max']).reset_index()
        Bins = list([(DF[DF['Group']==i]['min'].iloc[0],DF[DF['Group']==i]['max'].iloc[0]) for i in range(0,DF.Group.max()+1)])

        TupleBins=Bins
    return  TupleBins  

    
def __CalibList(x, CalbList, MustBePositive=False):
    """
    Gets alist of polynomial parameters (CalibList) that is used for calibration. Also gets the X value and
    returns the f(x) value of the calibration.
    :param x: double/int. The X value in the calibration polynom
    :param CalbList:list. The list that contains the polynomial parameters
    :param MustBePositive: in case the f(x) should only be positive then it transforms the negative values to zero
    :return: f(x)
    """
    reslt = CalbList[0] * (x ** 0) + CalbList[1] * (x ** 1) + CalbList[2] * (x ** 2) \
            + CalbList[3] * (x ** 3) + CalbList[4] * (x ** 4) + CalbList[5] * (x ** 5)
    if MustBePositive:
        reslt = np.where(reslt < 0, 0, reslt)
    return reslt


def __CompleteSet(Title, ParamList):
    """
     This function takes a list of values and return a 6 values list putting
     zeroes in "empty places. Also, if the title say "no_inter" then
     it adds a zero at the beggining of the list.
     Input: Title string. - The title of the parameters list
            ParamList list. - The parameters list
     returns list with 6 parameters that represent X at the power of zero
                                                 untill the power of 5
    """
    OutList = []
    if 'no_inter' in Title:
        OutList = [0]

    OutList.extend(ParamList)
    for x in range(len(OutList), 6):
        OutList.append(0)
    return OutList
