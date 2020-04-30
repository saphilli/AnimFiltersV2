#
# Copyright 2018 Michal Mach
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

__version__ = '2.0'
__author__ = 'Michal Mach, Sarah Phillips'

import sys
import os
import math
from functools import partial
from distutils.util import strtobool

from collections import OrderedDict

import math
import numpy as np
import copy as copy
from PySide2 import QtCore
from PySide2 import QtWidgets
from PySide2 import QtUiTools
import shiboken2

import maya.cmds as cmds
import maya.OpenMayaUI as MayaUI
import maya.api.OpenMayaAnim as oma
import maya.api.OpenMaya as om
from scipy.signal import butter, filtfilt, medfilt, gaussian, convolve
from scipy.interpolate import InterpolatedUnivariateSpline

# Where is this script?
SCRIPT_LOC = os.path.split(__file__)[0]
maya_useNewAPI = True


def add_keys(anim_curve, key_dict):
    # type: (unicode, dict) -> None
    """
    Add keyframes to animation curve

    :param anim_curve: animation curve name
    :param key_dict: dictionary of keyframes in {frame_number (float): value (float)} format
    :return: None
    """

    unit = om.MTime.uiUnit()
    nodeNattr = cmds.listConnections(anim_curve, d=True, s=False, p=True)[0]
    selList = om.MSelectionList()
    selList.add(nodeNattr)
    mplug = selList.getPlug(0)
    dArrTimes = om.MTimeArray()
    dArrVals = om.MDoubleArray()

    if 'rotate' in nodeNattr:
        for i in key_dict.keys():
            dArrTimes.append(om.MTime(float(i), unit))
            dArrVals.append(om.MAngle.uiToInternal(key_dict[i]))
    else:
        for i in key_dict.keys():
            dArrTimes.append(om.MTime(float(i), unit))
            dArrVals.append(key_dict[i])

    crvFnc = oma.MFnAnimCurve(mplug)
    crvFnc.addKeys(dArrTimes, dArrVals, crvFnc.kTangentAuto, crvFnc.kTangentAuto)


def copy_original_curves():
    # type: () -> tuple
    """
    Copy selected animation curves to clipboard and return their names with start and end frames
    so we can paste them back later

    :return: list of anim curve names, start frame, end frame
    :rtype: list, float, float
    """
    anim_curves = cmds.keyframe(q=True, sl=True, name=True)
    if anim_curves is None:
        cmds.headsUpMessage("No animation keys/curve selected! Select keys to filter, please!")
        cmds.warning("No animation keys/curve selected! Select keys to filter, please!")
        return None, None, None
    anim_keys = cmds.keyframe(q=True, sl=True, timeChange=True)
    start, end = int(anim_keys[0]), int(anim_keys[len(anim_keys) - 1])
    cmds.copyKey(anim_curves, t=(start, end)) 
    return anim_curves, start, end


def paste_clipboard_curves(anim_curves, start, end):
    # type: (list, float, float) -> None
    """
    Paste original anim curves we stored when the preview button was pressed

    :param anim_curves: list of animation curves
    :param start: start frame
    :param end: end frame
    :return: None
    """
    cmds.pasteKey(anim_curves, t=(start, end), o="replace")


def median_filter(raw_anim_curves, window_size=15):
    if raw_anim_curves is None:
        cmds.headsUpMessage("No animation keys/curve selected! Select keys to filter, please!")
        cmds.warning("No animation keys/curve selected! Select keys to filter, please!")
        return
    if window_size % 2 == 0:
        window_size += 1
    processed_curves = {}
    for key in raw_anim_curves.keys():
        start, end = min(raw_anim_curves[key]), max(raw_anim_curves[key])
        x = []
        for i in range(start, end):
            x.append(raw_anim_curves[key][i])
        y = medfilt(x, window_size)
        processed_keys = {}
        for i in range(start, end):
            processed_keys[str(i)] = y[i - start]
        processed_curves[key] = processed_keys
    return processed_curves


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    # ensure cutoff frequency doesn't overflow sampling frequency
    if cutoff > nyq:
        cutoff = nyq
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    padlen = 3 * max(len(a), len(b))

    # Switch padding method based on time range length
    if len(data) >= padlen:
        y = filtfilt(b, a, data)
    else:
        y = filtfilt(b, a, data, method="gust")
        cmds.warning("Working on a short segment, selecting longer time range could improve results.")
    return y


def butterworth_filter(raw_anim_curves, fs=30.0, cutoff=5.0, order=5):
    if raw_anim_curves is None:
        cmds.headsUpMessage("No animation keys/curve selected! Select keys to filter, please!")
        cmds.warning("No animation keys/curve selected! Select keys to filter, please!")
        return

    processed_curves = {}
    for key in raw_anim_curves.keys():
        start, end = min(raw_anim_curves[key]), max(raw_anim_curves[key])
        curve_keys = raw_anim_curves[key].keys()
        curve_keys.sort()
        x = []
        T = end-start
        nsamples = ((fs * T) / 30.0) + 1
        t_space = np.linspace(start, end, nsamples, endpoint=True)
        reached_end = False
        for t_sample in t_space:
            if int(t_sample) >= end and not reached_end:
                value = raw_anim_curves[key][int(t_sample)]
                reached_end = True
            else:
                value = raw_anim_curves[key][int(t_sample)] + (
                        raw_anim_curves[key][int(t_sample) + 1] - raw_anim_curves[key][int(t_sample)]) * (
                                t_sample - int(t_sample))
            x.append(value)
        y = butter_lowpass_filter(x, cutoff, fs, order)
        processed_keys = {}
        key_index = 0
        for t_sample in t_space:
            value = y[key_index]
            processed_keys[str(t_sample)] = value
            key_index += 1
        processed_curves[key] = processed_keys
    return processed_curves


def resample_keys(kv, thresh):
    start = float(min(kv.keys()))
    end = float(max(kv.keys()))
    startv = float(kv[start])
    endv = float(kv[end])
    total_error = 0
    offender = -1
    outlier = -1
    for k, v in kv.items():
        offset = (k - start) / (end - start)
        sample = (offset * endv) + ((1 - offset) * startv)
        delta = abs(v - sample)
        total_error += delta
        if delta > outlier:
            outlier = delta
            offender = k
    if total_error < thresh or len(kv.keys()) == 2:
        return [{start: startv, end: endv}]
    else:
        s1 = {kk: vv for kk, vv in kv.items() if kk <= offender}
        s2 = {kk: vv for kk, vv in kv.items() if kk >= offender}
        return resample_keys(s1, thresh) + resample_keys(s2, thresh)


def rejoin_keys(kvs):
    result = {}
    for item in kvs:
        result.update(item)
    return result


def decimate(keys, tolerance):
    return rejoin_keys(resample_keys(keys, tolerance))


def adaptive_filter(raw_anim_curves, tolerance_value):
    processed_curves = {}
    for key in raw_anim_curves.keys():
        processed_keys = decimate(raw_anim_curves[key], tolerance_value)
        processed_curves[key] = processed_keys
    return processed_curves
    

def laplacian_filter(raw_anim_curves, factor, sigma):
     """
    Author: Sarah Phillips
    Based on "The Cartoon Animation Filter" by Wang et al. 2006
    Processes animation curves and introduces Exaggeration, Anticipation and Follow-Through 
    args:
        raw_anim_curves (dict): dict object containing keyframes of each animation curve 
        factor (int): value that determines the strength of the filter
        sigma (str): width of the filter
    returns: Processed animation curve data 
    rtype: dict
    """
    if raw_anim_curves is None:
        cmds.headsUpMessage("No animation keys/curve selected! Select keys to filter, please!")
        cmds.warning("No animation keys/curve selected! Select keys to filter, please!")
        return
    filtered_curves = {}
    for key in raw_anim_curves.keys():
        #get dict representing one of the selected anim curves 
        curve = copy.deepcopy(raw_anim_curves[key])
        ys = [] 
        xs = curve.keys()
        xs.sort()
        for x in xs:
            ys.append(curve[x])
        spline = InterpolatedUnivariateSpline(xs,ys)
        deriv_1st = spline.derivative()
        deriv_2nd = deriv_1st.derivative()
        deriv_2nd_vals = []
        for x in xs:
            deriv_2nd_vals.append(deriv_2nd(x))
        #convolve the second derivative curve with a gaussian to smooth
        kernel = gaussian(51,sigma)
        smoothed_deriv = np.convolve(deriv_2nd_vals, kernel, mode='same')
        filtered_keyframes = {}
        for x,s in zip(xs, smoothed_deriv):
            filtered_keyframes[x]= spline(x) - (factor*s)     
        filtered_curves[key] = filtered_keyframes
    return filtered_curves


def slow_in_slow_out(raw_anim_curves,skip, factor,controller,controllerState):    
    """
    Author: Sarah Phillips
    Processes animation curves and introduces Slow-In Slow-Out effects
    args:
        raw_anim_curves (dict): dict object containing keyframes of each animation curve 
        skip (int): value that determines how many sections of the curve will not be processed
        controller (str): name of the animation curve that will be used as the controlling curve
        controllerState (bool): the state of the Controller Curve checkbox
    returns: Processed animation curve data 
    rtype: dict
    """
    if raw_anim_curves is None:
        cmds.headsUpMessage("No animation keys/curve selected! Select keys to filter, please!")
        cmds.warning("No animation keys/curve selected! Select keys to filter, please!")
        return
    filtered_curves = {}

    for curve_key in raw_anim_curves.keys():
        curve = raw_anim_curves[curve_key] 
        curve_length = len(curve.keys())
        ys = [] 
        xs = curve.keys()
        xs.sort()
        for x in xs:
            ys.append(curve[x])
        first_x = xs[0]
        last_x =xs[-1]
        stationary_points = []

        #check if the Controller Curve checkbox is checked
        if controllerState is False:
            curve_name = curve_key
        else:
            curve_name = controller

        #find the slopes of the tangents for each frame on the curve
        tangents = cmds.keyTangent(curve_name,q=1,oa=True,t=(first_x,last_x))
        for t,x in zip(tangents,xs):
            #if the slope of the tangent is 0: 
            #add the frame value to list of SPs
            if round(t) == 0:
                stationary_points.append(x)

        #skip the curve if there is only one or no points or the entire curve is constant 
        number_sps = len(stationary_points)
        if number_sps < 2 or number_sps == curve_length:
            filtered_curves[curve_key] = curve
            continue

        #discard any clusters of stationary points 
        reduced_sps = []
        neighbours = []
        for i in range(number_sps):
            current = stationary_points[i]
            #track neighbouring points and add only the middle value
            if current+1 in stationary_points or current-1 in stationary_points:
                neighbours.append(current)
                if current+1 not in stationary_points:
                    middle = int(len(neighbours)/2)
                    reduced_sps.append(neighbours[middle])
                    neighbours = []
                continue 
            reduced_sps.append(current)

        i=0
        while i+1 < len(reduced_sps):
            current_tp = reduced_sps[i]
            next_tp = reduced_sps[i+1]
            cmds.selectKey(cl=True)
            #get tangents of all points between current_tp and next_tp 
            cmds.selectKey(curve_key,k=True,add=True,t=(current_tp+1,next_tp-1))
            slopes = cmds.keyTangent(curve_name,q=1,ia=True,t=(current_tp+1,next_tp-1))
            if slopes is None:
                continue
            points = range(current_tp+1,next_tp) 

            steepest=0
            ##loop to find the steepest points between the SPs
            for t,x in zip(slopes,points):
                #if the slope is decreasing
                avg = sum(slopes)/len(slopes)
                if avg<0:
                    steepest = min(slopes)
                #else if slope is increasing
                elif avg>0:
                    steepest = max(slopes)
                sum_of_steepest_frames = 0
                count = 0
                steepest_rounded = steepest
                for k in range(0,len(slopes)):
                    if slopes[k] == steepest_rounded:
                        sum_of_steepest_frames +=points[k]
                        count+=1
                if count == 0:
                    i+=1+int(skip)
                    continue
                else:
                    steepest = int(sum_of_steepest_frames/count)

            #perform the scale operation
            cmds.scaleKey(curve_key,ssk=1,autoSnap=0,t=(current_tp+1,next_tp),shape=1,iub=False,ts=factor,tp=steepest,fs=factor,fp=steepest,vs=1,vp=0)
            i+=1+int(skip)
            


def blendKeys(window, anim_curves):
    """
    Author: Sarah Phillips
    Extends the selected animation curves before filtering 
    args:
        window(int): number of frames to extend the animation curves by before the first frame and after the last frame
        anim_curves (dict): dict object containing keyframes of each animation curve 
    returns: Extended animation curve data 
    rtype: dict
    """
        for key in anim_curves:
            curve = anim_curves[key]
            xs = curve.keys()
            last_key = max(xs)+1
            first_key = min(xs)-1
            i=0
            while i < window:
                if i>= len(xs):
                    break
                curve[last_key+i] = curve[xs[i]]
                curve[first_key-i] = curve[last_key-1-i]
                i+=1
        return anim_curves

def apply_curves(original_curves, processed_curves=None):
    for curve_name in original_curves.keys():
        start, end = min(original_curves[curve_name]), max(original_curves[curve_name])
        cmds.cutKey(curve_name, time=(start + 0.001, end - 0.001), option="keys", cl=True)
        if processed_curves is None:
            add_keys(curve_name, original_curves[curve_name])
        else:
            add_keys(curve_name, processed_curves[curve_name])
            
            
def select_curves(anim_curves, first_key_only=False):
    cmds.selectKey(cl=True)
    for curve_name in anim_curves.keys():
        start, end = min(anim_curves[curve_name]), max(anim_curves[curve_name])
        if first_key_only:
            cmds.selectKey(curve_name, t=(start, start), add=True)
        else:
            cmds.selectKey(curve_name, t=(start, end), add=True)


def loadAnimFiltersUI(uifilename, parent=None):
    """Properly Loads and returns UI files - by BarryPye on stackOverflow"""
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(uifilename)
    uifile.open(QtCore.QFile.ReadOnly)
    ui = loader.load(uifile, parent)
    uifile.close()
    return ui



class AnimFiltersUI(QtWidgets.QMainWindow):
    def __init__(self):
        mainUI = SCRIPT_LOC + "/animFilters.ui"
        MayaMain = shiboken2.wrapInstance(long(MayaUI.MQtUtil.mainWindow()), QtWidgets.QWidget)
       
        super(AnimFiltersUI, self).__init__(MayaMain)

        # main window load / settings
        self.MainWindowUI = loadAnimFiltersUI(mainUI, MayaMain)
        self.MainWindowUI.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.MainWindowUI.destroyed.connect(self.onExitCode)
        self.MainWindowUI.show()

        # init settings
        script_name = os.path.basename(__file__)
        script_base, ext = os.path.splitext(script_name)  # extract basename and ext from filename
        self.settings = QtCore.QSettings("MayaAnimFilters", script_base)

       
        # connect sliders and spinners
        self.MainWindowUI.thresholdSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.thresholdSpinBox, 1000.0))
        self.MainWindowUI.thresholdSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.thresholdSlider, 1000.0))
        self.MainWindowUI.multiSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.multiSpinBox, 1.0))
        self.MainWindowUI.multiSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.multiSlider, 1.0))

        self.MainWindowUI.butterSampleFreqSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.butterSampleFreqSpinBox, 1.0))
        self.MainWindowUI.butterSampleFreqSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.butterSampleFreqSlider, 1.0))
        self.MainWindowUI.butterCutoffFreqSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.butterCutoffFreqSpinBox, 100.0))
        self.MainWindowUI.butterCutoffFreqSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.butterCutoffFreqSlider, 100.0))
        self.MainWindowUI.butterOrderSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.butterOrderSpinBox, 1.0))
        self.MainWindowUI.butterOrderSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.butterOrderSlider, 1.0))
        
        self.MainWindowUI.factorSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged,self.MainWindowUI.factorSlider, 4.0))
        self.MainWindowUI.factorSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.factorSpinBox, 4.0))

        self.MainWindowUI.sigmaSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged,self.MainWindowUI.sigmaSlider, 10.0))
        self.MainWindowUI.sigmaSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.sigmaSpinBox, 10.0))

        self.MainWindowUI.blendWindowSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged,self.MainWindowUI.blendWindowSlider, 1.0))
        self.MainWindowUI.blendWindowSlider.valueChanged.connect(
            partial(self.sliderChanged,self.MainWindowUI.blendWindowSpinBox, 1.0))
        
        self.MainWindowUI.slowInSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.slowInSpinBox, 1.0))
        self.MainWindowUI.slowInSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.slowInSlider, 1.0))

        self.MainWindowUI.slowInScaleSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.slowInScaleSpinBox, 100.0))
        self.MainWindowUI.slowInScaleSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.slowInScaleSlider, 100.0))

        self.MainWindowUI.medianSlider.valueChanged.connect(
            partial(self.sliderChanged, self.MainWindowUI.medianSpinBox, 1.0))
        self.MainWindowUI.medianSpinBox.valueChanged.connect(
            partial(self.spinBoxChanged, self.MainWindowUI.medianSlider, 1.0))


 

        # connect buttons
        self.MainWindowUI.previewButton.clicked.connect(self.previewFilter)
        self.MainWindowUI.cancelButton.clicked.connect(self.cancelFilter)
        self.MainWindowUI.applyButton.clicked.connect(self.applyFilter)
        self.MainWindowUI.deleteInbetweenButton.clicked.connect(self.deleteInbetweens)
        self.MainWindowUI.evenOutButton.clicked.connect(self.evenOutKeys)
        self.MainWindowUI.selectAllButton.clicked.connect(self.select_all_curves)
        self.MainWindowUI.resetButton.clicked.connect(self.resetValues)
        self.MainWindowUI.bufferCurvesCheckBox.stateChanged.connect(self.bufferCurvesChanged)
        self.MainWindowUI.filterAllGroupBox.toggled.connect(self.filterAllChanged)
        self.MainWindowUI.excludeGroupBox.toggled.connect(self.excludeChanged)
        self.MainWindowUI.loopedAnimCheckBox.toggled.connect(self.loopedAnimChanged)
        self.MainWindowUI.addButton.clicked.connect(self.addExcludedCurve)
        self.MainWindowUI.removeButton.clicked.connect(self.removeExcludedCurve)
        self.MainWindowUI.removeAllButton.clicked.connect(self.removeAllExcludedCurves)
        self.MainWindowUI.controllingCurveGroupBox.toggled.connect(self.controllerChanged)
        self.MainWindowUI.addControl.clicked.connect(self.addControllerCurve)
        self.MainWindowUI.removeControl.clicked.connect(self.removeControllerCurve)


        # initialize variables
        self.animCurvesBuffer = None
        self.animCurvesProcessed = None
        self.previewActive = False
        self.start = None
        self.end = None
        self.originalCurves = None
        self.bufferCurvesState = None
        self.loopedAnimState = False
        self.filterAll = None
        self.excludeState = False
        self.excludedCurves = []
        self.controllerState = False
        self.controllingCurve = None
        self.restoreSettings()

    def restoreSettings(self):
        self.bufferCurvesState = self.settings.value("bufferCurves")
        self.filterAll = self.settings.value("filterAll")
        # Is the tool ran for the first time? Store the initial checkbox value.
        if self.bufferCurvesState is None:
            self.bufferCurvesState = self.MainWindowUI.bufferCurvesCheckBox.isChecked()
            self.settings.setValue("bufferCurves", self.bufferCurvesState)
        else:
            self.MainWindowUI.bufferCurvesCheckBox.setChecked(strtobool(self.bufferCurvesState))

        if self.filterAll is None:
            self.filterAll = self.MainWindowUI.filterAllGroupBox.isChecked()
            self.settings.setValue("filterAll", self.filterAll)
        else:
            self.MainWindowUI.filterAllGroupBox.setChecked(strtobool(self.filterAll))
        
        if self.loopedAnimState is None:
            self.loopedAnimState = self.MainWindowUI.loopedAnimCheckBox.isChecked()
            self.settings.setValue("loopedAnim", self.loopedAnimState)
        else:
            self.MainWindowUI.loopedAnimCheckBox.setChecked(self.loopedAnimState)

        self.MainWindowUI.excludeGroupBox.setChecked(self.excludeState)
        self.MainWindowUI.controllingCurveGroupBox.setChecked(self.controllerState)


    def bufferCurvesChanged(self):
        self.bufferCurvesState = self.MainWindowUI.bufferCurvesCheckBox.isChecked()
        self.settings.setValue("bufferCurves", self.bufferCurvesState)

    def filterAllChanged(self):
        self.filterAll = self.MainWindowUI.filterAllGroupBox.isChecked()
        self.settings.setValue("filterAll", self.filterAll)
    
    def excludeChanged(self):
        self.excludeState = self.MainWindowUI.excludeGroupBox.isChecked()
        self.settings.setValue("exclude",self.excludeState)
    
    def controllerChanged(self):
        self.controllerState = self.MainWindowUI.controllingCurveGroupBox.isChecked()
        self.settings.setValue("control",self.controllerState)
    
    def loopedAnimChanged(self):
        self.loopedAnimState = self.MainWindowUI.loopedAnimCheckBox.isChecked()
        self.settings.setValue("filterAll", self.loopedAnimState)

    def dynamicChanged(self):
        self.dynamicState = self.MainWindowUI.dynamicCheckBox.isChecked()
        self.MainWindowUI.staticCheckBox.setChecked(not(self.dynamicState))
        self.settings.setValue("dynamic", self.dynamicState)
        self.settings.setValue("static", not(self.dynamicState))

    def staticChanged(self):
        self.dynamicState = not(self.MainWindowUI.staticCheckBox.isChecked())
        self.MainWindowUI.dynamicCheckBox.setChecked(self.dynamicState)
        self.settings.setValue("dynamic", self.dynamicState)
        self.settings.setValue("static", not(self.dynamicState))


    def switchTabs(self, state=False):
        for i in range(0, self.MainWindowUI.tabWidget.count()):
            if not i == self.MainWindowUI.tabWidget.currentIndex():
                self.MainWindowUI.tabWidget.setTabEnabled(i, state)

    def addExcludedCurve(self):
        curves = cmds.keyframe(q=True,name=True,sl=True)
        for c in curves:
            self.excludedCurves.append(c)
            if self.MainWindowUI.listWidget.findItems(c,QtCore.Qt.MatchExactly) == []:
                self.MainWindowUI.listWidget.addItem(c)

          
    def removeExcludedCurve(self):
        curves = self.MainWindowUI.listWidget.selectedItems()
        if curves == None:
            return
        for c in curves:
            name = c.text()
            row = self.MainWindowUI.listWidget.row(c)
            self.excludedCurves.remove(name)
            self.MainWindowUI.listWidget.takeItem(row)
            self.MainWindowUI.listWidget.removeItemWidget(c)


    def removeAllExcludedCurves(self):
        self.excludedCurves = []
        self.MainWindowUI.listWidget.clear()

    def removeControllerCurve(self):
        curve = self.MainWindowUI.controllingCurveList.selectedItems()
        if curve == None:
            return
        self.MainWindowUI.controllingCurveList.clear()
        self.controllingCurve = None


    def addControllerCurve(self):
        if self.controllingCurve is not None:
            return
        curve = cmds.keyframe(q=True,name=True,sl=True)
        if len(curve)>1:
            cmds.headsUpMessage("Please select only one animation curve to control the scale of any other animation curves.")
            return
        elif curve is None:
            cmds.headsUpMessage("Please select the keys of the animation curve you want to add as a controller curve in the graph editor.")
            return
        self.controllingCurve = curve[0]
        self.MainWindowUI.controllingCurveList.addItem(curve[0])


    # switch state of buttons based on the Preview button state
    def switchButtons(self, state=True):
        self.MainWindowUI.previewButton.setEnabled(not state)
        self.MainWindowUI.cancelButton.setEnabled(state)
        self.MainWindowUI.applyButton.setEnabled(state)
        self.MainWindowUI.evenOutButton.setEnabled(not state)
        self.MainWindowUI.deleteInbetweenButton.setEnabled(not state)


    # update spin box value when slider changes
    def sliderChanged(self, target, multiplier, sliderValue):
        target.setValue(sliderValue / multiplier)
        if self.previewActive:
            self.refreshFilter()

    # update slider value when spin box value changes
    def spinBoxChanged(self, target, multiplier, spinBoxValue):
        target.setValue(spinBoxValue * multiplier)
        if self.previewActive:
            self.refreshFilter()



    def select_all_curves(self):
        cmds.select(clear=True)
        curves = cmds.ls(typ='animCurve')
        if curves is None:
            cmds.error("No animation curves are present to select keys from.")
        for curve in curves:
            if self.excludeState is True:
                if curve in self.excludedCurves:
                    continue
            keys = cmds.keyframe(curve,q=True)
            if keys is None:
                continue
            elif len(keys) > 3:
                cmds.select(curve,add=True)
                cmds.selectKey(curve,add=True)

    def select_curves_to_filter(self,frameBased=True):
        curves = []
        if self.filterAll is True:
            self.select_all_curves()
        else:
            curves = cmds.keyframe(q=True, sl=True, name=True)
            if curves is None:
                cmds.warning("Error: No keys selected, please select at least 4 keys on a curve and try again.")
                return 0
            keys_to_filter = []
            for curve in curves:
                if self.excludeState is True:
                    if curve in self.excludedCurves:
                        curves.remove(curve)
                        continue
                keys = cmds.keyframe(curve,q=True, sl=True)
                min_key = min(keys)
                max_key = max(keys)
                if keys is None:
                    continue
                window_length = 0
                if frameBased:
                    window_length = max_key - min_key
                else:
                    window_length - len(keys)
                if window_length >= 3:
                    keys_to_filter.append((curve,min_key,max_key))
                elif window_length < 3:
                    s = str(" Skipping the curve " + curve +" as not enough keys (<4) selected on curve.")
                    cmds.warning(s)
            cmds.select(cl=1)
            for curve_keys in (keys_to_filter):
                cmds.select(curve_keys[0], add=True)
                cmds.selectKey(curve_keys[0], add=True, t=(curve_keys[1],curve_keys[2]))
        

    def removeNoisyPeaks(self):
        """
        Author: Sarah Phillips
        Removes noisy points from animation curve
        ***function removed from animFilterTool
        returns: Smoothed animation curve data 
        rtype: dict
        """
        self.animCurvesBuffer = None
        if self.select_curves_to_filter() == 0:
            return
        self.animCurvesBuffer = self.getRawCurves()
        if self.animCurvesBuffer is None:
            return      
        self.originalCurves, self.start, self.end = copy_original_curves()
        if self.originalCurves is None:
            return
        cmds.undoInfo(swf=False)
        self.MainWindowUI.statusBar().showMessage("UNDO is SUSPENDED in Preview Mode")
        self.switchTabs(False)
        self.previewActive = True
        self.switchButtons(True)
        self.animCurvesProcessed = self.animCurvesBuffer.copy()
        for key in self.animCurvesBuffer:
            curve = self.animCurvesBuffer[key]
            for x in range(min(curve.keys()),max(curve.keys())-1):
                next_x = x+1
                second_next = x+2
                current_y = curve[x]
                if current_y < curve[next_x] and curve[second_next] <= current_y:
                    curve[next_x] = current_y
                elif current_y > curve[next_x] and curve[second_next] >= current_y:
                    curve[next_x] = current_y
                elif curve[second_next] < curve[next_x] and current_y <= curve[second_next]:
                    curve[next_x] = current_y
                elif curve[second_next] > curve[next_x] and current_y >= curve[second_next]:
                    curve[next_x] = current_y
        apply_curves(self.animCurvesBuffer, processed_curves=None)

    def deleteInbetweens(self):
        """
        Author: Sarah Phillips
        Rounds all keys on the animation curves to the nearest whole frame 
        returns: Animation curve data
        rtype: dict
        """
        self.animCurvesBuffer = None
        if self.select_curves_to_filter() == 0:
            return
        self.animCurvesBuffer = self.getRawCurves(frameBased=False)
        if self.animCurvesBuffer is None:
            return      
        self.originalCurves, self.start, self.end = copy_original_curves()
        if self.originalCurves is None:
            return
        cmds.undoInfo(swf=False)
        self.MainWindowUI.statusBar().showMessage("UNDO is SUSPENDED in Preview Mode")
        self.switchTabs(False)
        self.previewActive = True
        self.switchButtons(True)
        self.animCurvesProcessed = self.animCurvesBuffer.copy()
        for key in self.animCurvesBuffer:
            curve = self.animCurvesBuffer[key]
            xs = curve.keys()
            for x in xs:
                f = float(x)
                y = copy.deepcopy(curve[x])
                if f.is_integer() is False:
                    curve[int(x)] = y
                    curve.pop(x)
            self.animCurvesProcessed[key] = curve
        apply_curves(self.animCurvesBuffer, processed_curves=None)

    def evenOutKeys(self):
        """
        Author: Sarah Phillips
        Places keys on every frame of the animation curves
        Removes keys which lie in between whole frames
        returns: animation curve data 
        rtype: dict
        """
        self.animCurvesBuffer = None
        if self.select_curves_to_filter() == 0:
            return
        self.animCurvesBuffer = self.getRawCurves()
        if self.animCurvesBuffer is None:
            return      
        self.originalCurves, self.start, self.end = copy_original_curves()
        if self.originalCurves is None:
            return
        cmds.undoInfo(swf=False)
        self.MainWindowUI.statusBar().showMessage("UNDO is SUSPENDED in Preview Mode")
        self.switchTabs(False)
        self.previewActive = True
        self.switchButtons(True)
        self.animCurvesProcessed = self.animCurvesBuffer.copy()
        for key in self.animCurvesBuffer:
            curve = copy.deepcopy(self.animCurvesBuffer[key])
            start = round(min(curve.keys()))
            end = round(max(curve.keys()))
            curve_length = end-start
            sample_space = np.linspace(start,end,num=curve_length+1,endpoint=True)
            xs = curve.keys()
            xs.sort()
            ys = []
            for x in xs:
                ys.append(curve[x])
            spline = InterpolatedUnivariateSpline(xs,ys)
            resampled_curve = {}
            for i in sample_space:
                resampled_curve[i] = spline(i)
            self.animCurvesProcessed[key] = resampled_curve
        apply_curves(self.animCurvesBuffer, processed_curves=None)

    # grab selected anim curves when the preview button is pressed
    def previewFilter(self):
        self.animCurvesBuffer = None
        if self.bufferCurvesState is True:
            cmds.bufferCurve(animation='keys', overwrite=True)
        if self.select_curves_to_filter() == 0:
            return
        self.animCurvesBuffer = self.getRawCurves()
        if self.animCurvesBuffer is None:
            return    
        self.originalCurves, self.start, self.end = copy_original_curves()
        if self.originalCurves is None:
            return
        cmds.undoInfo(swf=False)
        self.MainWindowUI.statusBar().showMessage("UNDO is SUSPENDED in Preview Mode")
        self.switchTabs(False)
        self.switchButtons(True)
        self.previewActive = True
        self.refreshFilter()

    def getRawCurves(self, frameBased=True):
        result_curves = {}
        anim_curves = cmds.keyframe(q=True, sl=True, name=True)
        if anim_curves is None:
            cmds.headsUpMessage("No animation keys/curve selected! Please select the keys you want to filter")
            cmds.warning("No animation keys/curve selected! Please select the keys you want to filter")
            return None
        for anim_curve in anim_curves:
            anim_keys = cmds.keyframe(anim_curve,q=True,sl=True, timeChange=True)
            if anim_keys is None:
                anim_curves.remove(anim_curve)
                continue
            window_length = 0
            if frameBased:
                start, end = int(min(anim_keys)),int(max(anim_keys))
                window_length = end-start
            else:
                window_length = len(anim_keys)
            if ((window_length) < 3):
                cmds.warning(" Skipping the curve " + anim_curve +" as the range of frames selected on the curve is too low (<4)")
                continue
            anim_dict = {}
            if frameBased:
                for i in range(start, end + 1): 
                    anim_dict[i] = cmds.keyframe(anim_curve, q=True, time=(i, i), ev=True)[0] 
            else: 
                for i in anim_keys: 
                        anim_dict[i] = cmds.keyframe(anim_curve, q=True, time=(i, i), ev=True)[0] 
            result_curves[anim_curve] = anim_dict
       
        return result_curves
        

    def refreshFilter(self):
        unfiltered_curves = copy.deepcopy(self.animCurvesBuffer)
        if self.loopedAnimState:
                unfiltered_curves = blendKeys(self.MainWindowUI.blendWindowSpinBox.value(),unfiltered_curves)
        if self.MainWindowUI.tabWidget.currentIndex() == 0:
            self.animCurvesProcessed = adaptive_filter(unfiltered_curves,
                                                       self.MainWindowUI.thresholdSpinBox.value() *
                                                       self.MainWindowUI.multiSpinBox.value()) 
        elif self.MainWindowUI.tabWidget.currentIndex() == 1:
            self.animCurvesProcessed = butterworth_filter(unfiltered_curves,
                                                          self.MainWindowUI.butterSampleFreqSpinBox.value(),
                                                          self.MainWindowUI.butterCutoffFreqSpinBox.value(),
                                                          self.MainWindowUI.butterOrderSpinBox.value())     
        elif self.MainWindowUI.tabWidget.currentIndex() == 2:
            apply_curves(self.animCurvesBuffer, processed_curves=None)  
            self.animCurvesProcessed = None
            slow_in_slow_out(unfiltered_curves, self.MainWindowUI.slowInSpinBox.value(),self.MainWindowUI.slowInScaleSpinBox.value(),self.controllingCurve,self.controllerState)


        elif self.MainWindowUI.tabWidget.currentIndex() == 3:
            self.animCurvesProcessed = laplacian_filter(unfiltered_curves, 
                                                        self.MainWindowUI.factorSpinBox.value(),
                                                        self.MainWindowUI.sigmaSpinBox.value())
        elif self.MainWindowUI.tabWidget.currentIndex() == 4:
            self.animCurvesProcessed = median_filter(unfiltered_curves, 
                                                            self.MainWindowUI.medianSpinBox.value())
        if self.loopedAnimState:
            for key in self.animCurvesProcessed:
                first_orig = min(self.animCurvesBuffer[key].keys())
                first_proc = min(self.animCurvesProcessed[key].keys())
                for i in range(int(first_proc),int(first_orig)):
                    self.animCurvesProcessed[key].pop(i)
                last_orig = max(self.animCurvesBuffer[key].keys())
                last_proc = max(self.animCurvesProcessed[key].keys())
                for x in range(last_orig,last_proc+1):
                    self.animCurvesProcessed[key].pop(x)
        if self.animCurvesProcessed is None:
            select_curves(self.animCurvesBuffer, True)  
        else:
            apply_curves(self.animCurvesBuffer, processed_curves=self.animCurvesProcessed)    
            select_curves(self.animCurvesProcessed, True)                                               

    
    
    def resetValues(self):
        if self.MainWindowUI.tabWidget.currentIndex() == 0:
            self.MainWindowUI.multiSpinBox.setValue(0.5)
            self.MainWindowUI.thresholdSpinBox.setValue(0.5)
        elif self.MainWindowUI.tabWidget.currentIndex() == 1:
            self.MainWindowUI.butterSampleFreqSpinBox.setValue(30.0)
            self.MainWindowUI.butterCutoffFreqSpinBox.setValue(7.0)
            self.MainWindowUI.butterOrderSpinBox.setValue(5)
        elif self.MainWindowUI.tabWidget.currentIndex() == 2:
            self.MainWindowUI.slowInSpinBox.setValue(2)
            self.MainWindowUI.slowInScaleSpinBox.setValue(0.7)
        elif self.MainWindowUI.tabWidget.currentIndex() == 3:
            self.MainWindowUI.factorSpinBox.setValue(2.00)
            self.MainWindowUI.sigmaSpinBox.setValue(3.00)
            self.MainWindowUI.windowSizeSpinBox.setValue(0.20)
        elif self.MainWindowUI.tabWidget.currentIndex() == 4:
            self.MainWindowUI.medianSpinBox.setValue(35)


        
    def cancelFilter(self):
        paste_clipboard_curves(self.originalCurves, self.start, self.end)
        select_curves(self.animCurvesBuffer)
        self.switchButtons(False)
        self.switchTabs(True)
        self.animCurvesBuffer = None
        self.animCurvesProcessed = None
        self.previewActive = False
        cmds.undoInfo(swf=True)
        self.MainWindowUI.statusBar().showMessage("")

    def applyFilter(self):
        #apply original curve for undo step
        cmds.undoInfo(swf=True)
        cmds.undoInfo(openChunk=True, chunkName = "Apply curve changes")
        try:
            apply_curves(self.animCurvesBuffer)
        finally:
            cmds.undoInfo(closeChunk=True)
        #apply processed curve for undo step
        cmds.undoInfo(swf=True)
        cmds.undoInfo(openChunk=True)
        try:
            if self.animCurvesProcessed is not None:
                apply_curves(self.animCurvesBuffer, self.animCurvesProcessed)
            select_curves(self.animCurvesBuffer)
        finally:
            cmds.undoInfo(closeChunk=True)
        self.switchButtons(False)
        self.switchTabs(True)
        self.animCurvesBuffer = None
        self.animCurvesProcessed = None
        self.previewActive = False
        self.controllingCurve = None
        self.MainWindowUI.statusBar().showMessage("")

    def onExitCode(self):
        if self.previewActive:
            apply_curves(self.animCurvesBuffer)
            select_curves(self.animCurvesBuffer)
        self.animCurvesBuffer = None
        self.animCurvesProcessed = None
        cmds.undoInfo(swf=True)


def main():
    """Command within Maya to run this script"""
    if not (cmds.window("animFiltersWindow", exists=True)):
        AnimFiltersUI()
    else:
        sys.stdout.write("Tool is already open!")
