@echo off

For /f "tokens=1-3 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%b-%%a)
For /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)

SET SCRIPT_TO_RUN=run.py
SET MAP_NAME=CollectMineralShards
SET MODEL_NAME=test_model_mineral
SET SCRIPT_ARGS=--map_name %MAP_NAME% --model_name %MODEL_NAME% --training=False --if_output_exists continue

SET LOG_FILE=eval_logs/%mydate%_%mytime%_%MAP_NAME%_%MODEL_NAME%.log

type nul > %LOG_FILE%

echo "Starting script..." >> %LOG_FILE% 2>&1
echo "Using %MODEL_NAME%, on map %MAP_NAME%." >> %LOG_FILE% 2>&1
python %SCRIPT_TO_RUN% %SCRIPT_ARGS% >> %LOG_FILE% 2>&1
echo "Finished script." >> %LOG_FILE% 2>&1
