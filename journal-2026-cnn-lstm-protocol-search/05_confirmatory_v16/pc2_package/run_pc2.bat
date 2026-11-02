@echo off
REM ==========================================================================
REM  run_pc2.bat  -  second PC (RTX 3090): runs Parts F, G, I, J and K of the
REM  v16 review experiments on its own, independently of the lab PC.
REM  Double-click it. Results go to results_v16_pc2 in this folder.
REM  Restart it any time: finished runs are kept and skipped.
REM ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion

REM ---------------- settings ------------------------------------------------
set "ENV_NAME=NCmapss_GPU"
set "NB=Review_Experiments_v16_PC2.ipynb"
set "KERNEL=v16pc2"
set "RES=results_v16_pc2"
set "MAX_FIX=6"
REM  Optional: folder that holds the data, if it is not in or above this folder
REM set "V16_DATA_ROOT=D:\data"
REM --------------------------------------------------------------------------

cd /d "%~dp0"
set "V16_SMOKE=" & set "V16_DRY=" & set "V16_PARTS=" & set "V16_RESULTS_DIR="
if not exist "%NB%" ( echo %NB% not found in %CD% & pause & exit /b 1 )
for %%d in (logs runs backups) do if not exist %%d mkdir %%d
if exist STOP_AUTOFIX.txt del STOP_AUTOFIX.txt
for %%f in (ds05_seq_selected.json fd003_l2_selected.json) do (
    if not exist "%RES%\%%f" ( echo %RES%\%%f is missing - copy the whole PC2_package folder. & pause & exit /b 1 )
)

REM ---------------- find the conda environment by its folder ----------------
set "ENVDIR="
for %%r in ("%USERPROFILE%\.conda\envs" "%USERPROFILE%\anaconda3\envs" "%USERPROFILE%\miniconda3\envs" "%ProgramData%\anaconda3\envs" "%ProgramData%\miniconda3\envs" "%LOCALAPPDATA%\anaconda3\envs" "%LOCALAPPDATA%\miniconda3\envs") do (
    if not defined ENVDIR if exist "%%~r\%ENV_NAME%\python.exe" set "ENVDIR=%%~r\%ENV_NAME%"
)
if not defined ENVDIR (
    echo Environment %ENV_NAME% not found. Open an Anaconda Prompt, run: conda env list
    echo and put the exact name in ENV_NAME at the top of this file.
    pause & exit /b 1
)
echo Environment: %ENVDIR%
set "PATH=%ENVDIR%;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\bin;%ENVDIR%\Scripts;%ENVDIR%\bin;%PATH%"
set "PY=%ENVDIR%\python.exe"
"%PY%" -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('TensorFlow', tf.__version__, '| GPU:', g); assert g" || (
    echo TensorFlow does not see a GPU in %ENV_NAME%. Check the environment before running.
    pause & exit /b 1
)
"%PY%" -c "import papermill, h5py, scipy, sklearn" 2>nul || (
    echo Installing papermill and helpers into %ENV_NAME% ...
    "%PY%" -m pip install papermill ipykernel h5py scipy scikit-learn || ( echo pip install failed. & pause & exit /b 1 )
)
"%PY%" -m ipykernel install --user --name %KERNEL% --display-name "Python (%ENV_NAME% v16 PC2)" >nul 2>&1

set "HAVE_CLAUDE=1"
where claude >nul 2>&1 && goto claude_ok
for %%d in ("%USERPROFILE%\.local\bin" "%APPDATA%\npm" "%LOCALAPPDATA%\Programs\claude" "%LOCALAPPDATA%\AnthropicClaude") do (
    if exist "%%~d\claude.exe" set "PATH=%%~d;!PATH!"
    if exist "%%~d\claude.cmd" set "PATH=%%~d;!PATH!"
)
where claude >nul 2>&1 || set "HAVE_CLAUDE=0"
:claude_ok
if "%HAVE_CLAUDE%"=="0" echo WARNING: Claude Code not found - crashes will not be auto-fixed.

start "v16 PC2 monitor" powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0monitor_v16.ps1" -Res "%RES%"

REM ---------------- run / fix loop ------------------------------------------
set /a ATTEMPT=0
set /a SAME=0
:run
set /a ATTEMPT+=1
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"
set "LOG=logs\run_pc2_!TS!_a!ATTEMPT!.log"
set "OUTNB=runs\executed_pc2_a!ATTEMPT!.ipynb"
echo.
echo [%date% %time%] Attempt !ATTEMPT!  log=!LOG!
"%PY%" -m papermill "%NB%" "!OUTNB!" -k %KERNEL% --log-output --no-progress-bar > "!LOG!" 2>&1
if not errorlevel 1 goto success

echo [%date% %time%] Run failed. See !LOG!
if !ATTEMPT! GTR %MAX_FIX% ( echo Reached MAX_FIX=%MAX_FIX% attempts. & goto giveup )
if "%HAVE_CLAUDE%"=="0" goto giveup
powershell -NoProfile -Command "$l=(Select-String -Path '!LOG!' -Pattern '(Error|Exception):' | Select-Object -Last 1).Line; if(-not $l){$l='no error line'}; $l.Trim()" > "logs\sig_!ATTEMPT!.txt"
set /a PREV=!ATTEMPT!-1
if exist "logs\sig_!PREV!.txt" (
    fc /b "logs\sig_!ATTEMPT!.txt" "logs\sig_!PREV!.txt" >nul 2>&1 && set /a SAME+=1 || set /a SAME=0
)
if !SAME! GEQ 2 ( echo The same error repeated after two fixes. & goto giveup )
copy /y "%NB%" "backups\nb_before_fix_a!ATTEMPT!_!TS!.ipynb" >nul
copy /y Review_Experiments_v16_PC2.py "backups\py_before_fix_a!ATTEMPT!_!TS!.py" >nul
set "PROMPT=logs\claude_prompt_a!ATTEMPT!.txt"
> "!PROMPT!" (
    echo Read V16_FIX_INSTRUCTIONS.md in this folder and follow it exactly.
    echo Attempt number: !ATTEMPT!
    echo Results folder: %RES%
    echo Log of the failed run: !LOG!
    echo Executed notebook of the failed run, with the error cell: !OUTNB!
    echo Find the cause of the error, fix it in %NB% and Review_Experiments_v16_PC2.py, record it in V16_FIX_NOTES.md, then stop.
)
echo [%date% %time%] Asking Claude Code to fix it ...
call claude -p --permission-mode acceptEdits --max-turns 40 --allowedTools "Read" "Edit" "Write" "Grep" "Glob" "NotebookEdit" "Bash(python:*)" "Bash(pip:*)" "Bash(nvidia-smi:*)" < "!PROMPT!" > "logs\claude_fix_a!ATTEMPT!.log" 2>&1
type "logs\claude_fix_a!ATTEMPT!.log"
if exist STOP_AUTOFIX.txt ( type STOP_AUTOFIX.txt & goto giveup )
goto run

:success
echo.
echo [%date% %time%] Second-PC run finished without errors.
copy /y "!OUTNB!" "Review_Experiments_v16_PC2_EXECUTED.ipynb" >nul
copy /y "!OUTNB!" "%RES%\Review_Experiments_v16_PC2_EXECUTED.ipynb" >nul
copy /y "!LOG!" "%RES%\run_log.txt" >nul
if exist V16_FIX_NOTES.md copy /y V16_FIX_NOTES.md "%RES%\" >nul
powershell -NoProfile -Command "Compress-Archive -Path '%RES%' -DestinationPath '%RES%_!TS!.zip' -Force"
echo Results zip: %RES%_!TS!.zip
echo Bring this zip to the lab PC and merge it with merge_v16.py (see README_PC2.txt).
pause
exit /b 0

:giveup
echo.
echo [%date% %time%] Stopped. Finished runs are safe in %RES%\runs.jsonl; run this file again to resume.
echo Last log: !LOG!
pause
exit /b 1
