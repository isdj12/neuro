#!/bin/bash
# Скрипт для подготовки файлов для развертывания на другом ПК

DEPLOY_DIR="deployment_package"
mkdir -p "$DEPLOY_DIR"

echo "📦 Подготовка файлов для развертывания..."

# Копируем необходимые файлы
cp app.py "$DEPLOY_DIR/"
cp meat_classifier.pth "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp check_setup.py "$DEPLOY_DIR/"
cp README.md "$DEPLOY_DIR/"
cp DEPLOY.md "$DEPLOY_DIR/"

echo "✅ Файлы скопированы в директорию: $DEPLOY_DIR"
echo ""
echo "📋 Список файлов:"
ls -lh "$DEPLOY_DIR" | grep -v "^total"
echo ""
echo "💡 Теперь можно скопировать папку '$DEPLOY_DIR' на другой ПК"
echo "   Размер: $(du -sh $DEPLOY_DIR | cut -f1)"

