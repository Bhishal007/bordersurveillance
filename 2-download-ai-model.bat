@echo off
title Download AI Model
echo Downloading Llama 3.2 to Pendrive...
:: Ensure it saves to the pendrive, not the C: drive
set OLLAMA_MODELS=%~dp0models
"%~dp0ollama.exe" pull llama3.2
echo Download Complete!
pause