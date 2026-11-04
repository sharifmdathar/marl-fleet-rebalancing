@echo off
REM MARL Fleet Rebalancing - Training Script for WSL
REM This script uses mise to run Python training commands

echo ========================================
echo MARL Fleet Rebalancing - Training Suite
echo ========================================
echo.

:MENU
echo Select an option:
echo 1. Train IPPO (all checkpoints: 40/80/200 iters x 5 seeds)
echo 2. Train MAPPO (all checkpoints: 40/80/200 iters x 5 seeds)
echo 3. Exit
echo.

set /P CHOICE="Enter choice (1-3): "

if "%CHOICE%"=="1" goto TRAIN_IPPO
if "%CHOICE%"=="2" goto TRAIN_MAPPO
if "%CHOICE%"=="3" goto EXIT

echo Invalid choice. Please try again.
pause
cls
goto MENU

:TRAIN_IPPO
echo.
echo ========================================
echo Training IPPO checkpoints
echo ========================================
echo.

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo ippo --iterations 40 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 40 --name ippo-bandra-40-s%%s"
)

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo ippo --iterations 80 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 80 --name ippo-bandra-80-s%%s"
)

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo ippo --iterations 200 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 40 --name ippo-bandra-200-s%%s"
)

echo.
echo IPPO training complete!
pause
cls
goto MENU

:TRAIN_MAPPO
echo.
echo ========================================
echo Training MAPPO checkpoints
echo ========================================
echo.

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo mappo --iterations 40 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 40 --name mappo-bandra-40-s%%s"
)

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo mappo --iterations 80 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 80 --name mappo-bandra-80-s%%s"
)

for /L %%s in (0,1,4) do (
    echo ----------------------------------------
    echo seed %%s
    echo ----------------------------------------
    wsl bash -c "cd /home/luffy/marl-fleet-rebalancing &&mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo mappo --iterations 200 --eval-interval 10 --eval-episodes 12 --checkpoint-interval 40 --name mappo-bandra-200-s%%s"
)

echo.
echo MAPPO training complete!
pause
cls
goto MENU

:EXIT
echo.
echo Exiting...
timeout /t 2 >nul
