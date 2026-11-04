# Эталон DeepSeek-V4.1-Flash

Инструменты, с которыми движок `internal/llm` сверяет свою реализацию
DeepSeek-V4.1-Flash. Это только инструменты разработки: узел запускает
модель без Python, а здесь снимаются эталонные значения для тестов Go.
Что именно движок должен воспроизвести, описано в
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
| `vectors.py` | Эталонные значения для Go-реализаций того, что эталон берёт из библиотек Python (токенайзер, хеш Engram) |
| `chat_vectors.py` | Эталонные промпты и разборы ответов формата чата (`encoding.py`) для порта на Go |
| `num_vectors.py` | Эталонные значения для арифметики FP8/FP4/bf16 движка (`internal/num`) |
| `layer_check.py` | Выходы одного настоящего слоя на эмбеддингах настоящего текста, для `internal/llm/dsv41/real_test.go` |
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
python vectors.py ../../../internal/llm/testdata/dsv41-vectors.json.gz
python num_vectors.py ../../../internal/num/testdata/dsv41-mx.json.gz
python chat_vectors.py ../../../internal/llm/testdata/dsv41-chat.json.gz
git status --short ../../../internal   # должно быть пусто
```

`tiny.py` детерминирован: повторный запуск с теми же версиями пакетов даёт
побитово те же файлы. Если фикстура изменилась, значит изменилось поведение
эталона или PyTorch — с этим надо разобраться, прежде чем коммитить.

## Проверка на настоящих слоях

```sh
cd tools/llmref/dsv41 && . ../.venv/bin/activate
DIR=$(python -c 'import fetch; print(fetch.default_cache(fetch.pins()["revision"]))')
python layer_check.py 2 $DIR/layer2.safetensors     # скачает шард слоя (6,9 ГиБ) и эмбеддингов
python layer_check.py 20 $DIR/layer20.safetensors
cd ../../.. && RAKKAN_DSV41_REAL=$DIR go test -run RealLayers -v ./internal/llm/dsv41
```

Нужно около 16 ГБ памяти. Эталон считает слой около 3 минут, тест на Go — около 2.

## Проверка на GPU

`kernel_cpu` повторяет квантование эталона операция за операцией. Совпадает
ли он с настоящими ядрами, можно проверить только на GPU NVIDIA. Ядра
tilelang считают в FP8 на тензорных ядрах (FP4-веса тоже переводятся в FP8),
и написаны они под Hopper. Хватит одной H100 на пару часов. На Ada (L40S,
RTX 4090) FP8 есть, но ядра могут не собраться.

На сервере нужны:
- CUDA 12.x и Python 3.10+;
- от 64 ГБ RAM и 100 ГБ диска для проверки настоящих слоёв;
- для тестов Vulkan — драйвер NVIDIA с Vulkan: пакеты `libnvidia-gl-*` и
  `vulkan-tools`, проверка — `vulkaninfo --summary`.

Ветка в удалённом репозитории может отставать: скопируйте рабочую копию,
затем на сервере запустите скрипт.

```sh
rsync -az --exclude 'tools/llmref/.venv*' --exclude 'gpu-results-*' ./ SERVER:rakkan-mesh/
ssh SERVER 'cd rakkan-mesh && tools/llmref/dsv41/gpu_server.sh --real'
scp 'SERVER:rakkan-mesh/gpu-results-*.tar.gz' .
```

`gpu_server.sh` делает всё сам:
- под root на Ubuntu ставит то, чего нет в голом образе: `python3-venv`,
  загрузчик Vulkan и библиотеки, которые грузит драйвер Vulkan NVIDIA
  (`libvulkan1 vulkan-tools libxext6 libx11-6 libegl1 libgl1`);
- ставит в `tools/llmref/.venv-gpu` PyTorch 2.15 с CUDA (тот же, на котором
  сделана фикстура), tilelang и `nvcc` из pip той же версии CUDA, что у
  PyTorch: более новый `nvcc` делает PTX, который старый `ptxas` не берёт;
- прогоняет `check_kernels.py` на крошечной и средней фикстуре;
- ставит Go нужной версии и прогоняет тесты движка, тесты Vulkan — на GPU
  NVIDIA, если его драйвер даёт Vulkan;
- с `--real` скачивает с Hub 16 ГиБ и сверяет слои 2 и 20 с эталоном на CPU и
  GPU.

У арендованных машин диск бывает маленьким (20 ГБ), а `/dev/shm` большим:
скрипт кладёт кэши (tilelang, Go, шарды Hub) в `/dev/shm`, если там есть
40 ГБ свободного места.

Каждый шаг пишет свой лог, ошибка одного шага не останавливает остальные.
Итог — в `summary.txt`, всё вместе — в архиве `gpu-results-*.tar.gz`. Его нужно
вернуть в проект: по `report-*.json` решается, нужно ли править `kernel_cpu`.
Без `--real` скрипт укладывается примерно в полчаса.

**Ожидаемый результат** (так было на H100 PCIe, 2026-11-04):
- квантование (`act_quant*`, `fp4_quant*`) и умножения FP8/FP4 совпадают на
  1.0000;
- `sparse_attn` совпадает на 0.99, расхождение — один шаг округления bf16;
  у Sinkhorn — один шаг float32. Ядра считают `exp2` и суммы по-своему, и
  повторять это побитово `kernel_cpu` не нужно;
- в проверке модели `argmax` совпадает не на всех шагах (13 и 15 из 18). По-видимому, в
  случайной модели индексатор выбирает top-k из почти равных оценок, и
  разница в последнем бите меняет выбор строк (`topk_idxs`), а за ним логиты.
  Расхождение больше, чем в самих ядрах, означало бы ошибку;
- тесты Go проходят, в том числе тесты Vulkan на GPU NVIDIA (побитово как CPU).

Без GPU `check_kernels.py --dry-run` проверяет сам скрипт: `kernel_cpu`
сравнивается сам с собой, и всё должно совпасть.
