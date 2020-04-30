Animation Curve Filters V2 for Maya 2017+
#
#Copyright 2020 Sarah Phillips

This program is free software; you can redistribute it and/or modify
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


This program is an extended version of the original Animation Curve Filters tool developed by Michal Mach 2018.
The original version of this program is available at http://www.michalmach.com/tools/animation-curve-filters-for-maya/

Installation
============
Unzip the script files to your Maya user scripts directory (typically Documents/maya/scripts).

Open a Python tab in the Maya's Script Editor, paste the following code and drag it to a shelf to create a button:

from animFilters import animFilters
reload(animFilters)
animFilters.main()

How to use
==========
This tool is designed to be used with Maya's Graph Editor. 
1) Open the Graph Editor (Windows>Animation Editors>Graph Editor) and select animation curves/keys you want to filter.
2) Choose one of the filter tabs and hit the Preview button to see a preview of the filtered curves.
3) The parameters to the filters can be changed in preview mode by using the sliders or entering values. 
The curves will update with any changes to the parameters. 
4) Once you're satisfied with the result, click Apply.
5) If you don't like the result, you can click Cancel and original curves will be restored.
6) Closing the tool in Preview mode restores the original curves, so don't forget to hit Apply before closing!

Please be aware that "Undo" is disabled in Preview mode.

The 'Auto Buffer Curves' checkbox stores the Buffer Curves when you click the Preview button, so you can see
your original curves drawn in grey color with the filtered curves.
You have to enable View / Display Buffer Curves option in the Graph Editor to see them.

The "Exclude Animation Curves" checkbox allows you to select animation curves that will be excluded from any filters 
applied, even when selected. Check the checkbox and select the animation curves you want to be left out of the filter. 
Clicking the "+" button will add the names of the animation curves to the list in the interface. Make sure you have selected
keys on the animation curves you want to add, otherwise the curve will not be added to the list.
To remove a curve from the animation list, simply click on the name of the curve in the interface and click the "-" button. 
Clicking "Remove All" will remove all curves from the exclusion list. Alternatively, the checkbox can be unchecked and the 
curves will no longer be excluded, but the list will be saved.

The "Filter All" checkbox will cause all animation curves in the scene to be filtered without you having to select them.
For scenes where there is more than one animated character/object, it is recommended to leave this off. 

The "Blend First and Last Keys" option, when checked, will attempt to keep the first and last key frames selected in place when 
a filter is applied. This is recommended for looped animations when applying the Exaggeration filter.

The "Select All Curves" button selects all animation curves in the scene.

The "Round Keys to Nearest Frame" option will round all selected key frames to the nearest whole frame. 
The "Set Keys on Every Frame" option will set a key on every frame, and remove any keys that are located between frames.



Requirements
============
The Exaggeration, Butterworth and Median filters require SciPy and NumPy Python modules to function.
Publicly available versions of these modules are not compatible with Maya's Python interpreter (mayapy).
Eric Vignola is one of them and here's his Google Drive folder with many interesting modules:
https://drive.google.com/drive/folders/0BwsYd1k8t0lEMjBCa2N1Z25KZXc

You only need to download these two:
scipy-0.19.1-cp27-none-win_amd64.whl
numpy-1.13.1+mkl-cp27-none-win_amd64.whl

After downloading the files, you can simply unzip them to the site-packages folder inside your Maya installation.
Typically "c:\Program Files\Autodesk\Maya20xx\Python\Lib\site-packages" on Windows.
On a Mac, it is recommending to install these packages using "pip".

Official guide for installing packages: https://packaging.python.org/tutorials/installing-packages/

Remember to use Maya's version of python interpreter (mayapy) and "pip".

Change Log
==========
2020-04-30 (ver 2.0) - modified version with Exaggeration and Slow-In Slow-Out filters 
2018-10-23 (ver 1.0) - original release with Adaptive, Butterworth and Median filters

