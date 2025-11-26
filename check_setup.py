#!/usr/bin/env python3
"""
Скрипт для проверки готовности окружения к работе приложения.
"""

import sys
import os

def check_python_version():
    """Проверяет версию Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Требуется Python 3.8+, установлена версия {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Проверяет наличие необходимых библиотек."""
    required = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package} установлен")
        except ImportError:
            print(f"❌ {package} не установлен")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_model():
    """Проверяет наличие файла модели."""
    model_path = "meat_classifier.pth"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # Размер в МБ
        print(f"✅ Модель найдена: {model_path} ({size:.2f} МБ)")
        return True
    else:
        print(f"❌ Модель не найдена: {model_path}")
        return False

def main():
    print("🔍 Проверка окружения для классификатора мяса\n")
    print("=" * 50)
    
    all_ok = True
    
    # Проверка Python
    print("\n1. Проверка версии Python:")
    if not check_python_version():
        all_ok = False
    
    # Проверка зависимостей
    print("\n2. Проверка зависимостей:")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_ok = False
        print(f"\n💡 Установите недостающие пакеты: pip install {' '.join(missing)}")
    
    # Проверка модели
    print("\n3. Проверка модели:")
    if not check_model():
        all_ok = False
        print("\n💡 Убедитесь, что файл meat_classifier.pth находится в текущей директории")
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("\n✅ Все проверки пройдены! Приложение готово к работе.")
        print("💡 Запустите приложение: streamlit run app.py")
    else:
        print("\n❌ Обнаружены проблемы. Исправьте их перед запуском приложения.")
        sys.exit(1)

if __name__ == "__main__":
    main()

