@echo off

set PYTHON_EXE="C:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/.venv/Scripts/python.exe"
set SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/compare_canonicals.py"
set MERGING_SCRIPT="c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/d3 Masks/scripts/make_images_grid.py"
@REM  3 - diff_comparison 1v1; 4 - organize_folders_aggregatedly; 5 - organize_folders_granularly
set METHOD=5

REM maschi vs femmine (men vs women)
%PYTHON_EXE% %SCRIPT% %METHOD% men women

REM upper tail vs lower tail (best vs worst)
%PYTHON_EXE% %SCRIPT% %METHOD% best worst

exit