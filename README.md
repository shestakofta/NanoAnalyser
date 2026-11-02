# NanoAnalyser_test


## Download NanoAnalyzer

| Windows | macOS | Linux |
|:---:|:---:|:---:|
| [![Windows](https://img.shields.io/badge/Download-Windows-0078D4?logo=windows&logoColor=white)](https://github.com/shestakofta/NanoAnalyser_test/releases/latest/download/NanoAnalyzer-Windows.zip) | [![macOS](https://img.shields.io/badge/Download-macOS-000000?logo=apple&logoColor=white)](https://github.com/shestakofta/NanoAnalyser_test/releases/latest/download/NanoAnalyzer-macOS-arm64.zip) | [![Linux](https://img.shields.io/badge/Download-Linux-FCC624?logo=linux&logoColor=black)](https://github.com/shestakofta/NanoAnalyser_test/releases/latest/download/NanoAnalyzer-Linux.tar.gz) |

macOS Intel: [Скачать](https://github.com/shestakofta/NanoAnalyser_test/releases/latest/download/NanoAnalyzer-macOS-Intel.zip)

## Запуск

Распакуйте архив целиком. Windows: запустите NanoAnalyzer.exe из папки NanoAnalyzer. macOS: откройте NPA ML.app (основная кнопка — Apple Silicon; для Intel используйте отдельную ссылку). Linux: запустите ./NanoAnalyzer/NanoAnalyzer. Python устанавливать не нужно. Linux требует glibc 2.35+ (Ubuntu 22.04+). macOS-приложение не нотариально заверено Apple, поэтому первый запуск может потребовать разрешения в настройках безопасности.

## Сборка и выпуск версий

Push в main и pull request автоматически собирают Windows x64, macOS ARM64, macOS Intel и Linux x64. Каждая сборка проверяется обработкой тестового изображения и созданием Excel-отчёта. Архивы доступны в Artifacts соответствующего запуска Actions.

Для публикации релиза обновите APP_VERSION в app.py при необходимости, создайте тег (например, `git tag v4.1.0`) и отправьте его командой `git push origin v4.1.0`. После успешной проверки всех четырёх сборок автоматически публикуется GitHub Release с архивами и SHA256SUMS.txt. Постоянные имена архивов позволяют кнопкам скачивать последний стабильный релиз. До первого успешного релиза кнопки недоступны.

Ручной запуск: Actions → Build and release NanoAnalyzer → Run workflow. Запуск на ветке создаёт артефакты; запуск на теге v* также публикует релиз.
