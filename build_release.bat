@echo off
setlocal

set "ROOT=%~dp0"
set "CXX=g++"
set "WINDRES=windres"
set "OUT=%ROOT%PixelLane.exe"
set "RES=%ROOT%pixellane_res.o"

where "%CXX%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: g++ was not found on PATH.
    exit /b 1
)

where "%WINDRES%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: windres was not found on PATH.
    exit /b 1
)

echo Building version resource...
del /q "%ROOT%simd_image_lab_res.o" "%RES%" >nul 2>&1
"%WINDRES%" "%ROOT%pixellane.rc" -O coff -o "%RES%"
if errorlevel 1 exit /b 1

echo Building PixelLane release...
"%CXX%" -std=c++17 -O2 -static -static-libgcc -static-libstdc++ ^
    -mwindows "%ROOT%gui_demo.cpp" "%RES%" ^
    -o "%OUT%" ^
    -ldwmapi -lcomdlg32 -lgdi32 -luser32 -lshell32
if errorlevel 1 exit /b 1

del /q "%RES%" >nul 2>&1
del /q "%ROOT%SIMD_Image_Lab.exe" >nul 2>&1
echo Release built: "%OUT%"
exit /b 0
