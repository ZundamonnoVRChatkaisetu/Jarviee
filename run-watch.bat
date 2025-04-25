@echo off
REM スクリプト所在ディレクトリに移動
cd /d "%~dp0"

REM PowerShell スクリプトを実行（実行ポリシーは一時的に Bypass）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch-and-push.ps1"
