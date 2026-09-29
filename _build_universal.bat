@echo off
cls

net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :request_admin

goto :main_code

:request_admin
echo [INFO] Requesting administrator privileges...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"

if %ERRORLEVEL% NEQ 0 (
    echo ==========================================================================
    echo [ERROR] PowerShell failed to restart as admin.
    echo Please right-click this file and choose "Run as administrator".
    echo ==========================================================================
    pause
    exit /b 1
)

exit /b 0

:main_code
cls
setlocal enabledelayedexpansion

cd /d "%~dp0"

set OUTPUT_EXE=Cortex.exe

echo ==========================================================================
echo Building an Autonomous AI Controller (C++17 / CUDA 13.0 / WinAPI / WinRT)
echo Architecture support: Turing, Ampere, Ada Lovelace, Hopper, Blackwell
echo ==========================================================================

:choose_config
echo.
echo Select build configuration:
echo 1. Release (Max speed, NO debug info, for users)
echo 2. Debug   (Your original setup: Max speed + debug info/symbols)
echo.
set /p CONFIG_CHOICE="Enter your choice (1 or 2): "

if "%CONFIG_CHOICE%"=="1" (
    set "CONFIG_MODE=Release"
    set "NVCC_OPT_FLAGS="
    set "MSVC_OPT_FLAGS="
    set "LINKER_FLAGS="
    set "OMP_DLL_NAME=libomp140.x86_64.dll"
    goto msvc_init
)
if "%CONFIG_CHOICE%"=="2" (
    set "CONFIG_MODE=Debug"
    set "NVCC_OPT_FLAGS=-g"
    set "MSVC_OPT_FLAGS=/Zi"
    set "LINKER_FLAGS=/DEBUG /DEBUGTYPE:pdata"
    set "OMP_DLL_NAME=libomp140d.x86_64.dll"
    goto msvc_init
)

echo [ERROR] Invalid choice. Please try again.
goto choose_config

:msvc_init
echo.
echo [INFO] Selected Configuration: !CONFIG_MODE!
echo ==========================================================================

where cl.exe >nul 2>nul
if %ERRORLEVEL%==0 (
	echo [INFO] MSVC environment is already initialized.
	goto check_cuda
)

echo [INFO] Searching for Visual Studio 2022 / MSVC Build Tools...

set "VCVARS_PATH="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
	for /f "tokens=*" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do (
		if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
			set "VCVARS_PATH=%%i\VC\Auxiliary\Build\vcvarsall.bat"
			set "VS_INSTALL_DIR=%%i"
			goto msvc_found
        )
	)
)

set "VS_PATHS[0]=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
set "VS_PATHS[1]=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
set "VS_PATHS[2]=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
set "VS_PATHS[3]=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

for /L %%i in (0,1,3) do (
	call set "CURRENT_PATH=%%VS_PATHS[%%i]%%"
	if defined CURRENT_PATH (
		if exist "!CURRENT_PATH!" (
			set "VCVARS_PATH=!CURRENT_PATH!"
			for %%A in ("!CURRENT_PATH!\..\..\..\..") do set "VS_INSTALL_DIR=%%~fA"
			goto msvc_found
		)
	)
)

echo [INFO] Checking specific Visual Studio path...

if exist "D:\Program_files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
	set "VCVARS_PATH=D:\Program_files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
	set "VS_INSTALL_DIR=D:\Program_files\Microsoft Visual Studio\2022\Community"
	goto msvc_found
)

echo [ERROR] Visual Studio 2022 C++ compiler (vcvarsall.bat) not found 
echo [ERROR] Please install Visual Studio 2022.
goto error

:msvc_found
echo [INFO] Found Build Tools at: "%VCVARS_PATH%"
echo [INFO] Initializing x64 developer environment...
call "%VCVARS_PATH%" amd64 >nul
if %ERRORLEVEL% NEQ 0 (
	echo [ERROR] Failed to initialize MSVC environment.
	goto error
)

:check_cuda
if "%CUDA_PATH%"=="" (
	echo [ERROR] Environment variable CUDA_PATH not found!
	echo Make sure CUDA Toolkit is installed on your computer.
	goto error
)

set NVCC="%CUDA_PATH%\bin\nvcc.exe"

echo [INFO] Path to CUDA: %CUDA_PATH%
echo [INFO] Output file: %OUTPUT_EXE%
echo [INFO] Starting compilation...

del *.obj >nul 2>nul

echo [INFO] Compiling CUDA files via NVCC...
for %%F in (main cs ca) do %NVCC% -c %%F.cu -o %%F_cuda.obj ^
	-rdc=true ^
	--ptxas-options=-v ^
	-std=c++17 ^
	-O3 ^
	--use_fast_math ^
	!NVCC_OPT_FLAGS! ^
	--extended-lambda ^
	--cudart=static ^
	-allow-unsupported-compiler ^
	-gencode arch=compute_75,code=sm_75 ^
	-gencode arch=compute_80,code=sm_80 ^
	-gencode arch=compute_86,code=sm_86 ^
	-gencode arch=compute_89,code=sm_89 ^
	-gencode arch=compute_90,code=sm_90 ^
	-gencode arch=compute_100,code=sm_100 ^
	-gencode arch=compute_100,code=compute_100 ^
	-Xcompiler /W3 ^
	-Xcompiler /openmp:llvm ^
	-Xcompiler /EHsc ^
	-Xcompiler /Zc:__cplusplus ^
	-Xcompiler /Zc:preprocessor ^
	-Xcompiler /DNOMINMAX ^
	-Xcompiler /O2 ^
	-Xcompiler /Oi ^
	-Xcompiler /Ot ^
	-Xcompiler /permissive- ^
	-Xcompiler "!MSVC_OPT_FLAGS!" || goto error
if %ERRORLEVEL% NEQ 0 goto error

echo [INFO] Compiling C++ files via MSVC...
cl.exe /c cs.cpp ca.cpp ^
	/std:c++17 ^
	/W3 ^
	/openmp:llvm ^
	/EHsc ^
	/Zc:__cplusplus ^
	/Zc:preprocessor ^
	/DNOMINMAX ^
	/O2 ^
	/Oi ^
	/Ot ^
	/permissive- ^
	!MSVC_OPT_FLAGS!
if %ERRORLEVEL% NEQ 0 goto error

echo [INFO] Performing CUDA Device Linking...
%NVCC% -dlink ^
	main_cuda.obj cs_cuda.obj ca_cuda.obj ^
	-o cuda_dlink.obj ^
	-gencode arch=compute_75,code=sm_75 ^
	-gencode arch=compute_80,code=sm_80 ^
	-gencode arch=compute_86,code=sm_86 ^
	-gencode arch=compute_89,code=sm_89 ^
	-gencode arch=compute_90,code=sm_90 ^
	-gencode arch=compute_100,code=sm_100 ^
	--cudart=static || goto error
if %ERRORLEVEL% NEQ 0 goto error

echo [INFO] Linking everything together into %OUTPUT_EXE% via MSVC...
cl.exe *.obj ^
	/Fe:%OUTPUT_EXE% ^
	/link ^
	/OPT:REF ^
	/OPT:ICF ^
	!LINKER_FLAGS! ^
	/LIBPATH:"%CUDA_PATH%\lib\x64" ^
	cudart_static.lib ^
	cuda.lib ^
	User32.lib Gdi32.lib Ole32.lib OleAut32.lib windowsapp.lib d3d11.lib
if %ERRORLEVEL% NEQ 0 goto error

del *.obj *.exp *.lib >nul 2>nul

echo [INFO] Searching and copying LLVM OpenMP DLL (!OMP_DLL_NAME!)...
set "DLL_FOUND=0"
if defined VS_INSTALL_DIR (
    set "REDIST_ROOT=!VS_INSTALL_DIR!\VC\Redist\MSVC"
    if exist "!REDIST_ROOT!" (
        for /f "delims=" %%D in ('dir /b /ad "!REDIST_ROOT!"') do (
            set "TARGET_DLL_PATH=!REDIST_ROOT!\%%D\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\!OMP_DLL_NAME!"
            if exist "!TARGET_DLL_PATH!" (
                copy /Y "!TARGET_DLL_PATH!" ".\" >nul
                echo [SUCCESS] Copied !OMP_DLL_NAME! to build directory.
                set "DLL_FOUND=1"
                goto dll_copy_done
            )
        )
    )
)

:dll_copy_done
if "!DLL_FOUND!"=="0" (
    echo [WARNING] !OMP_DLL_NAME! was not found automatically in VC\Redist.
    echo Please make sure to copy it manually to run the executable safely.
)

echo ==========================================================================
echo %OUTPUT_EXE% successfully compiled in !CONFIG_MODE! mode.
echo Nice to meet you, carbon-based unit.
echo ==========================================================================
pause
exit /b 0

:error
echo ==========================================================================
echo Assembly of %OUTPUT_EXE% failed.
echo Check the compiler/linker error log above.
echo ==========================================================================
pause
exit /b 1
