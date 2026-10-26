# -*- coding: utf-8 -*-
"""
make_about.py — встраивает текст окна «О программе» из файла !infoabout.txt в модуль about_text.py.

Запускается автоматически из build_exe.bat перед сборкой: текст попадает внутрь NanoAnalyzer.exe
(в скомпилированном виде) и после сборки не меняется. Чтобы изменить текст — отредактируйте
!infoabout.txt (кодировка UTF-8) и соберите программу заново.
"""
import os

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "resources", "!infoabout.txt")
with open(src, encoding="utf-8-sig") as fh:
    text = fh.read()
with open(os.path.join(here, "about_text.py"), "w", encoding="utf-8") as fh:
    fh.write("# -*- coding: utf-8 -*-\n# Создано make_about.py из !infoabout.txt. Не редактируйте вручную.\n")
    fh.write("TEXT = {!r}\n".format(text))
print("about_text.py: {} символов из !infoabout.txt".format(len(text)))
