@echo off
REM Скрипт для подготовки файлов для развертывания на другом ПК (Windows)

set DEPLOY_DIR=deployment_package
if not exist "%DEPLOY_DIR%" mkdir "%DEPLOY_DIR%"

echo 📦 Подготовка файлов для развертывания...

REM Копируем необходимые файлы
copy app.py "%DEPLOY_DIR%\" >nul
copy meat_classifier.pth "%DEPLOY_DIR%\" >nul
copy requirements.txt "%DEPLOY_DIR%\" >nul
copy check_setup.py "%DEPLOY_DIR%\" >nul
copy README.md "%DEPLOY_DIR%\" >nul
copy DEPLOY.md "%DEPLOY_DIR%\" >nul

echo ✅ Файлы скопированы в директорию: %DEPLOY_DIR%
echo.
echo 📋 Список файлов:
dir /b "%DEPLOY_DIR%"
echo.
echo 💡 Теперь можно скопировать папку '%DEPLOY_DIR%' на другой ПК

