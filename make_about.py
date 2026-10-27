# -*- coding: utf-8 -*-
"""
make_about.py — встраивает текст окна «О программе» из файла !infoabout.txt в модуль about_text.py.

Запускается автоматически перед сборкой (build_exe.bat в Windows, сценарий GitHub Actions для macOS):
текст попадает внутрь программы и после сборки не меняется. Чтобы изменить текст — отредактируйте
!infoabout.txt (кодировка UTF-8) и соберите программу заново.
Файл ищется в папке resources/ (структура для сборки через GitHub), затем рядом с этим скриптом.
"""
import os
import sys

# Вывод в консоль — в UTF-8 при любой кодировке системы (в Windows вывод в файл или конвейер,
# например в GitHub Actions, по умолчанию идёт в cp1252 и падает на русских буквах).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

here = os.path.dirname(os.path.abspath(__file__))
for src in (os.path.join(here, "resources", "!infoabout.txt"), os.path.join(here, "!infoabout.txt")):
    if os.path.exists(src):
        break
else:
    raise SystemExit("Не найден !infoabout.txt (ни в resources/, ни рядом с make_about.py)")
with open(src, encoding="utf-8-sig") as fh:
    text = fh.read()
with open(os.path.join(here, "about_text.py"), "w", encoding="utf-8") as fh:
    fh.write("# -*- coding: utf-8 -*-\n# Создано make_about.py из !infoabout.txt. Не редактируйте вручную.\n")
    fh.write("TEXT = {!r}\n".format(text))
print("about_text.py: {} символов из {}".format(len(text), os.path.relpath(src, here)))
