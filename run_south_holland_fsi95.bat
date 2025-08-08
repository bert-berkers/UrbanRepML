@echo off
REM Batch script to run South Holland FSI 95% experiment

echo ========================================
echo Running South Holland FSI 95%% Experiment
echo ========================================

REM Run the experiment
python scripts\experiments\run_experiment.py ^
  --experiment_name south_holland_fsi95 ^
  --city south_holland ^
  --fsi_percentile 95 ^
  --resolutions 8,9,10 ^
  --run_training ^
  --epochs 100

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Experiment failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Experiment completed successfully!
echo ========================================
pause