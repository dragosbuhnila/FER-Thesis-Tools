@echo off

set PYTHON_EXE="C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe"
set SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py"
set MERGING_SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/scripts/make_images_grid.py"
@REM  3 - diff_comparison 1v1; 4 - organize_folders
set METHOD=4

REM FEDMAR vs MARFRO
%PYTHON_EXE% %SCRIPT% %METHOD% fedmar marfro

@REM REM FEDMAR vs MATVIN
@REM %PYTHON_EXE% %SCRIPT% %METHOD% fedmar matvin

@REM REM MARFRO vs MATVIN
@REM %PYTHON_EXE% %SCRIPT% %METHOD% marfro matvin

@REM REM maschi vs femmine (men vs women)
@REM %PYTHON_EXE% %SCRIPT% %METHOD% men women

@REM REM upper tail vs lower tail (best vs worst)
@REM %PYTHON_EXE% %SCRIPT% %METHOD% best worst

@REM REM Bubbles ConvNext vs FEDMAR
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_bub fedmar

@REM REM Bubbles ConvNext vs MARFRO
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_bub marfro

@REM REM External Perturbation ConvNext vs FEDMAR
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_ext fedmar

@REM REM External Perturbation ConvNext vs MARFRO
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_ext marfro

@REM REM GradCam LAYER30 ConvNext vs FEDMAR
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_grad fedmar

@REM REM GradCam LAYER30 ConvNext vs MARFRO
@REM %PYTHON_EXE% %SCRIPT% %METHOD% convnext_grad marfro

exit