@echo off
REM ==========================================================================
REM  run_v16.bat  -  runs Review_Experiments_v16.ipynb unattended on the lab PC
REM
REM  Usage (double-click, or from a terminal in this folder):
REM     run_v16.bat smoke     short real-data test of every part (about 1 h)
REM     run_v16.bat full      the full review run (about 40 GPU-hours)
REM     run_v16.bat pc2 ENV   second PC: runs only Parts F, G, I, J, K into
REM                           results_v16_pc2 (ENV = its conda environment name);
REM                           merge into the lab results later with merge_v16.py
REM
REM  What it does:
REM   1. activates the conda environment and installs papermill if missing
REM   2. opens a monitor window (progress, GPU, last log lines; keeps the PC awake)
REM   3. executes the notebook with papermill, saving outputs as it goes
REM   4. if the run crashes, backs up the notebook, asks Claude Code to fix the
REM      cause, and runs again. Finished runs are kept in results_v16\runs.jsonl,
REM      so each retry resumes where the last one stopped.
REM   5. on success: checks summary.json, saves the executed notebook and zips
REM      the results.
REM ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion

REM ---------------- settings (edit if the lab PC differs) -------------------
set "CONDA_ROOT=%USERPROFILE%\anaconda3"
set "ENV_NAME=NCmapss"
if not "%~2"=="" set "ENV_NAME=%~2"
set "NB=Review_Experiments_v16.ipynb"
set "KERNEL=v16env"
set "MAX_FIX=6"
REM  Optional: folder that contains N-Cmapps, env_eweda, Nasa_battery_dataset
REM set "V16_DATA_ROOT=C:\Users\AUC\Documents\thesiswork"
REM --------------------------------------------------------------------------

cd /d "%~dp0"
set "MODE=%~1"
if "%MODE%"=="" set /p "MODE=Mode - type smoke or full: "
set "V16_PARTS="
set "V16_RESULTS_DIR="
if /i "%MODE%"=="smoke" (
    set "V16_SMOKE=1"
    set "RES=results_v16_smoke"
) else if /i "%MODE%"=="full" (
    set "V16_SMOKE="
    set "RES=results_v16"
) else if /i "%MODE%"=="pc2" (
    set "V16_SMOKE="
    set "RES=results_v16_pc2"
    set "V16_PARTS=F,G,I,J,K"
    set "V16_RESULTS_DIR=results_v16_pc2"
) else (
    echo Unknown mode "%MODE%". Use smoke, full or pc2.
    pause & exit /b 1
)
if /i "%MODE%"=="pc2" (
    if not exist "%RES%" mkdir "%RES%"
    for %%f in (ds05_seq_selected.json fd003_l2_selected.json) do (
        if not exist "%RES%\%%f" if exist "pc2_seed\%%f" copy /y "pc2_seed\%%f" "%RES%\" >nul
        if not exist "%RES%\%%f" ( echo %RES%\%%f is missing - copy the pc2_seed folder next to this file. & pause & exit /b 1 )
    )
)
set "V16_DRY="

if not exist "%NB%" ( echo %NB% not found in %CD% & pause & exit /b 1 )
if not exist logs mkdir logs
if not exist runs mkdir runs
if not exist backups mkdir backups
if exist STOP_AUTOFIX.txt del STOP_AUTOFIX.txt

REM ---------------- environment ---------------------------------------------
if not exist "%CONDA_ROOT%\Scripts\activate.bat" for %%r in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\Anaconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
    if exist "%%~r\Scripts\activate.bat" set "CONDA_ROOT=%%~r"
)
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo Anaconda not found at %CONDA_ROOT%. Edit CONDA_ROOT at the top of this file.
    pause & exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 ( echo Could not activate conda env %ENV_NAME%. & pause & exit /b 1 )

python -c "import papermill" 2>nul || (
    echo Installing papermill into %ENV_NAME% ...
    python -m pip install papermill ipykernel || ( echo pip install failed. & pause & exit /b 1 )
)
python -m ipykernel install --user --name %KERNEL% --display-name "Python (%ENV_NAME% v16)" >nul 2>&1

set "HAVE_CLAUDE=1"
where claude >nul 2>&1 && goto claude_ok
REM not on PATH: try the usual install locations (native installer, npm)
for %%d in ("%USERPROFILE%\.local\bin" "%APPDATA%\npm" "%LOCALAPPDATA%\Programs\claude" "%LOCALAPPDATA%\AnthropicClaude") do (
    if exist "%%~d\claude.exe" set "PATH=%%~d;!PATH!"
    if exist "%%~d\claude.cmd" set "PATH=%%~d;!PATH!"
)
where claude >nul 2>&1 || set "HAVE_CLAUDE=0"
:claude_ok
if "%HAVE_CLAUDE%"=="1" for /f "delims=" %%c in ('where claude') do echo Claude Code: %%c
if "%HAVE_CLAUDE%"=="0" echo WARNING: Claude Code not found on PATH - crashes will not be auto-fixed.

REM ---------------- monitor window ------------------------------------------
start "v16 monitor" powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0monitor_v16.ps1" -Res "%RES%"

REM ---------------- run / fix loop ------------------------------------------
set /a ATTEMPT=0
set /a SAME=0
:run
set /a ATTEMPT+=1
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"
set "V16_DRY="
set "LOG=logs\run_%MODE%_!TS!_a!ATTEMPT!.log"
set "OUTNB=runs\executed_%MODE%_a!ATTEMPT!.ipynb"
echo.
echo [%date% %time%] Attempt !ATTEMPT!  mode=%MODE%  log=!LOG!
papermill "%NB%" "!OUTNB!" -k %KERNEL% --log-output --no-progress-bar > "!LOG!" 2>&1
if not errorlevel 1 goto success

echo [%date% %time%] Run failed. See !LOG!
if !ATTEMPT! GTR %MAX_FIX% ( echo Reached MAX_FIX=%MAX_FIX% attempts. & goto giveup )
if "%HAVE_CLAUDE%"=="0" goto giveup

REM stop if the same error comes back three times in a row
powershell -NoProfile -Command "$l=(Select-String -Path '!LOG!' -Pattern '(Error|Exception):' | Select-Object -Last 1).Line; if(-not $l){$l='no error line'}; $l.Trim()" > "logs\sig_!ATTEMPT!.txt"
set /a PREV=!ATTEMPT!-1
if exist "logs\sig_!PREV!.txt" (
    fc /b "logs\sig_!ATTEMPT!.txt" "logs\sig_!PREV!.txt" >nul 2>&1 && set /a SAME+=1 || set /a SAME=0
)
if !SAME! GEQ 2 ( echo The same error repeated after two fixes. & goto giveup )

REM back up the notebook, then ask Claude Code to fix the cause
copy /y "%NB%" "backups\nb_before_fix_a!ATTEMPT!_!TS!.ipynb" >nul
if exist Review_Experiments_v16.py copy /y Review_Experiments_v16.py "backups\py_before_fix_a!ATTEMPT!_!TS!.py" >nul
set "PROMPT=logs\claude_prompt_a!ATTEMPT!.txt"
> "!PROMPT!" (
    echo Read V16_FIX_INSTRUCTIONS.md in this folder and follow it exactly.
    echo Attempt number: !ATTEMPT!
    echo Run mode: %MODE%  ^(results folder: %RES%^)
    echo Log of the failed run: !LOG!
    echo Executed notebook of the failed run, with the error cell: !OUTNB!
    echo Find the cause of the error, fix it in %NB% and Review_Experiments_v16.py, record it in V16_FIX_NOTES.md, then stop.
)
echo [%date% %time%] Asking Claude Code to fix it ^(log: logs\claude_fix_a!ATTEMPT!.log^) ...
call claude -p --permission-mode acceptEdits --max-turns 40 --allowedTools "Read" "Edit" "Write" "Grep" "Glob" "NotebookEdit" "Bash(python:*)" "Bash(pip:*)" "Bash(nvidia-smi:*)" < "!PROMPT!" > "logs\claude_fix_a!ATTEMPT!.log" 2>&1
type "logs\claude_fix_a!ATTEMPT!.log"
if exist STOP_AUTOFIX.txt (
    echo Claude Code could not fix this automatically:
    type STOP_AUTOFIX.txt
    goto giveup
)
goto run

REM ---------------- success -------------------------------------------------
:success
echo.
echo [%date% %time%] Notebook finished without errors.
copy /y "!OUTNB!" "Review_Experiments_v16_EXECUTED_%MODE%.ipynb" >nul
copy /y "!OUTNB!" "%RES%\Review_Experiments_v16_EXECUTED.ipynb" >nul
copy /y "!LOG!" "%RES%\run_log.txt" >nul
if exist V16_FIX_NOTES.md copy /y V16_FIX_NOTES.md "%RES%\" >nul
if not exist "%RES%\summary.json" (
    echo WARNING: %RES%\summary.json is missing - check that Part L ran.
) else (
    echo Results summary: %RES%\summary.json
)
powershell -NoProfile -Command "Compress-Archive -Path '%RES%' -DestinationPath '%RES%_!TS!.zip' -Force"
echo Executed notebook : Review_Experiments_v16_EXECUTED_%MODE%.ipynb
echo Results zip       : %RES%_!TS!.zip
echo Send the zip and the executed notebook back for the paper.
pause
exit /b 0

REM ---------------- give up -------------------------------------------------
:giveup
echo.
echo [%date% %time%] Stopped. Finished runs are safe in %RES%\runs.jsonl;
echo running this file again resumes from there.
echo Last log: !LOG!
if exist V16_FIX_NOTES.md echo Fix notes: V16_FIX_NOTES.md
pause
exit /b 1
