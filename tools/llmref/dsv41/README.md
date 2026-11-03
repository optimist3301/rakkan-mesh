# Эталон DeepSeek-V4.1-Flash

Инструменты, с которыми движок `internal/llm` сверяет свою реализацию
DeepSeek-V4.1-Flash. Что именно движок должен воспроизвести, описано в
[docs/dev/deepseek-v41.md](../../../docs/dev/deepseek-v41.md).

| Файл | Что делает |
|---|---|
| `upstream/` | Эталонный код разработчиков без изменений (MIT), ревизия в `upstream/REVISION` |
| `files.json` | Закреплённая ревизия, размеры и хеши файлов репозитория модели |
| `fetch.py` | Скачивает файлы этой ревизии с проверкой хеша |
| `kernel_cpu.py` | Ядра tilelang (только CUDA), переписанные на PyTorch для CPU |
| `test_kernel_cpu.py` | Сверяет `kernel_cpu` с прямыми определениями |
| `run.py` | Загружает неизменённый `model.py` с `kernel_cpu` и правилом top-K движка |
| `tiny.py` | Строит крошечную модель и снимает с неё эталонные выходы |
| `check_kernels.py` | На машине с GPU NVIDIA сверяет `kernel_cpu` с настоящими ядрами |

## Установка (CPU)

```sh
cd tools/llmref
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
.venv/bin/pip install -r dsv41/requirements-cpu.txt
```

`.venv` не попадает в git. `fetch.py` складывает загруженное в
`~/.cache/rakkan-llmref/dsv41/<ревизия>`. Скрипты запускаются из `dsv41/` с
активированным окружением: вызов `../.venv/bin/python` по относительному пути
Python 3.14 сопровождает предупреждениями.

## Проверка и пересборка фикстуры

```sh
cd tools/llmref/dsv41
. ../.venv/bin/activate
python test_kernel_cpu.py
(cd upstream/encoding && python -m pytest -q -p no:cacheprovider)   # формат чата, 55 тестов
python tiny.py ../../../internal/llm/testdata/dsv41-tiny
git status --short ../../../internal/llm/testdata/dsv41-tiny   # должно быть пусто
```

`tiny.py` детерминирован: повторный запуск с теми же версиями пакетов даёт
побитово те же файлы. Если фикстура изменилась, значит изменилось поведение
эталона или PyTorch — с этим надо разобраться, прежде чем коммитить.

## Проверка на GPU

`kernel_cpu` повторяет квантование эталона операция за операцией. Совпадает
ли он с настоящими ядрами, можно проверить только на GPU NVIDIA (Hopper или
Blackwell: tilelang и FP8/FP4 эталона рассчитаны на них).

```sh
cd tools/llmref/dsv41
python3 -m venv .venv
.venv/bin/pip install -r upstream/inference/requirements.txt
.venv/bin/python tiny.py --medium /tmp/medium
.venv/bin/python check_kernels.py --fixture /tmp/medium --out report.json
.venv/bin/python check_kernels.py --out report-tiny.json   # крошечная фикстура; tilelang может отвергнуть её размеры
```

**Ожидаемый результат:**
- квантование (`act_quant*`, `fp4_quant*`) совпадает на 1.0000;
- у умножений, внимания и Sinkhorn доля совпадений может быть меньше 1, но
  ошибки должны быть порядка округления bf16;
- для модели `argmax` совпадает на всех шагах.

Файлы `report*.json` нужно вернуть в проект: по ним решается, нужно ли
править `kernel_cpu`.

Без GPU `check_kernels.py --dry-run` проверяет сам скрипт: `kernel_cpu`
сравнивается сам с собой, и всё должно совпасть.
