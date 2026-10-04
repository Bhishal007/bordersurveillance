@echo off
title Portable Ollama Server
echo Starting Omni-Agent Local Server...
:: Set the models directory to the 'models' folder on the pendrive
set OLLAMA_MODELS=%~dp0models
:: Run the executable located in the same folder
"%~dp0ollama.exe" serve
pause