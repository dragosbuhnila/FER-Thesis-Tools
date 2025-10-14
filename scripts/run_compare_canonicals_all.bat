@echo off

set PYTHON_EXE="C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe"
set SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py"
set MERGING_SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/scripts/make_images_grid.py"
@REM  3 - diff_comparison 1v1; 4 - organize_folders
set METHOD=3

REM GradCam LAYER30 ConvNext vs FEDMAR
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_grad fedmar

REM GradCam LAYER30 ConvNext vs MARFRO
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_grad marfro

REM Bubbles ConvNext vs FEDMAR
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_bub fedmar

REM Bubbles ConvNext vs MARFRO
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_bub marfro

REM External Perturbation ConvNext vs FEDMAR
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_ext fedmar

REM External Perturbation ConvNext vs MARFRO
%PYTHON_EXE% %SCRIPT% %METHOD% convnext_ext marfro

REM FEDMAR vs MATVIN
%PYTHON_EXE% %SCRIPT% %METHOD% fedmar matvin

REM MARFRO vs MATVIN
%PYTHON_EXE% %SCRIPT% %METHOD% marfro matvin

REM FEDMAR vs MARFRO
%PYTHON_EXE% %SCRIPT% %METHOD% fedmar marfro

REM maschi vs femmine (men vs women)
%PYTHON_EXE% %SCRIPT% %METHOD% men women

REM upper tail vs lower tail (best vs worst)
%PYTHON_EXE% %SCRIPT% %METHOD% best worst

@REM Subsequently you should run merge_images_in_grid.py, setting the NUMBER_OF_IMAGES to 3 or 4 based on whether you use both meandif and difmean or just one of them

exit