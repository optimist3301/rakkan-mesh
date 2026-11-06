# HTTP API узла

Каждый узел (`rakkan-node daemon`) открывает локальный HTTP API на
unix-сокете (`-api rakkan.sock`). JS SDK — тонкая обёртка над ним, так что
клиента можно написать на любом языке, который умеет HTTP поверх
unix-сокета: Python (`requests-unixsocket`, `httpx`), Go (`net/http` с
`DialContext` на `unix`), `curl --unix-socket`.

- [Транспорт и соглашения](#транспорт-и-соглашения)
- [Модули: POST /v1/modules](#post-v1modules)
- [Задачи: POST /v1/run](#post-v1run)
- [Много задач: POST /v1/map](#post-v1map)
- [Блобы: POST /v1/blobs, GET /v1/blobs/{id}](#блобы)
- [Состояние: GET /v1/status, GET /v1/trust](#состояние)
- [Языковые модели: POST /v1/llm/models, GET /v1/llm/models/{id}, POST /v1/llm/generate](#языковые-модели)
- [Веб-страница узла](#веб-страница-узла)

Примеры с `curl` и их ответы получены на настоящем узле; примеры раздела о
языковых моделях показывают формат, значения в них условные.

## Транспорт и соглашения

- **Доступ.** Сокет создаётся с правами `0600`: обращаться к API может только
  пользователь, от которого запущен узел. Других механизмов аутентификации
  нет: доступ к сокету и есть доступ к узлу.
- **Длина пути.** Путь к сокету в Linux не длиннее 107 байт, иначе узел не
  сможет его открыть (`bind: invalid argument`).
- **Хост.** Любое имя хоста в URL игнорируется. В примерах используется
  `http://rakkan/…`.
- **Формат.** Тела запросов и ответов — JSON. Исключения — загрузка модулей и
  блобов (сырые байты) и скачивание блобов.
- **Двоичные поля** (`module`, `input`, `inputs`, `output`) в JSON
  кодируются в base64.
- **Потоковые ответы** (`/v1/map`, `/v1/llm/generate`) приходят в формате
  NDJSON (`application/x-ndjson`): один JSON-объект на строку, по мере
  готовности. Закрытие соединения клиентом отменяет работу.
- **Ошибки:**

```json
{"error": {"kind": "out_of_gas", "message": "out of gas: budget 100000"}}
```

| HTTP | Когда |
|---|---|
| 200 | успех (206 для запроса диапазона блоба) |
| 400 | неверный запрос, `kind: "bad_request"` |
| 404 | блоб не найден, `kind: "not_found"` |
| 409 | реплики не дали большинства, `kind: "no_consensus"` |
| 422 | задача не завершилась, `kind` — вид ошибки задачи (`out_of_gas`, `trap`, …; см. [tasks.md](tasks.md#код-выхода-и-ошибки)) |
| 503 | нет подходящих исполнителей: `no_capacity`, `not_enough_nodes` |

В потоковых ответах ошибки приходят строкой NDJSON того же вида. HTTP-код
к этому моменту уже 200.

```sh
S="curl -s --unix-socket rakkan.sock"
```

## POST /v1/modules

Сохраняет модуль на узле. Тело — байты модуля (до 8 МиБ).

```sh
$S --data-binary @echo.wasm http://rakkan/v1/modules
```
```json
{"hash":"50521fd60207422012eefd90c2241664d610b25f9da2479be24b21928eab2dab"}
```

`hash` — SHA-256 модуля в hex. Дальше модуль можно указывать по хешу. Узел
держит модули в кеше (256 МиБ), и после перезапуска узла модуль нужно
загрузить заново: на неизвестный хеш придёт 400 с `unknown module`.

## POST /v1/run

Запускает одну задачу.

| Поле | Тип | Смысл |
|---|---|---|
| `module` | base64 | модуль inline (сохраняется в кеш), или |
| `module_hash` | hex | хеш загруженного модуля |
| `entry` | string | точка входа, по умолчанию `run` |
| `input` | base64 | вход, или |
| `input_blob` | hex | ID блоба на этом узле как вход |
| `blobs` | hex[] | до 16 прикреплённых блобов (должны быть на этом узле) |
| `limits` | object | `gas`, `timeout_ms`, `memory_mb`, `max_input_bytes`, `max_output_bytes`; незаданные берутся по умолчанию |
| `replicas` | int | запустить на стольких разных узлах, принять ответ большинства |

```sh
$S -d "{\"module_hash\":\"$HASH\",\"input\":\"$(printf 'привет' | base64)\"}" http://rakkan/v1/run
```
```json
{"exit_code":0,"output":"0L/RgNC40LLQtdGC","output_size":12,"gas_used":51,"elapsed_ms":1.154,
 "node":"12D3KooWQecPNLRsukiPJVos8kj5YcpapCcA4Y5vw1TRT1XySCsP","attempts":1}
```

**Ответ:**

| Поле | Смысл |
|---|---|
| `exit_code` | код выхода задачи; 0 — успех |
| `output` | выход, base64 (пустая строка, если выход пришёл блобом) |
| `output_blob` | ID блоба с выходом, если выход больше 1 МиБ; скачивается через `GET /v1/blobs/{id}` |
| `output_size` | размер выхода в байтах |
| `logs`, `log_truncated` | лог задачи; признак обрезки |
| `gas_used`, `elapsed_ms` | расход gas, время на исполнителе |
| `node` | ID исполнителя |
| `attempts` | число исполнителей, понадобившихся из-за сбоев |
| `replicas` | `[{node, agreed, error?}]`, если задача реплицировалась или результат гостя проверялся |
| `checked` | результат гостя подтверждён независимым запуском |

Ошибка задачи:

```sh
$S -d "{\"module\":\"$(base64 -w0 spin.wasm)\",\"limits\":{\"gas\":100000}}" http://rakkan/v1/run
```
```
HTTP 422
{"error":{"kind":"out_of_gas","message":"out of gas: budget 100000"}}
```

Лимиты выше политики всех узлов:

```sh
$S -d "{\"module_hash\":\"$HASH\",\"limits\":{\"memory_mb\":512}}" http://rakkan/v1/run
```
```
HTTP 503
{"error":{"kind":"no_capacity","message":"no member can run this task: the task's limits exceed every executor's
 policy (requested limits exceed node policy: memory pages 8192 (ceiling 1024))"}}
```

Запрос держится открытым до результата. Если клиент закроет соединение,
задача отменяется.

## POST /v1/map

Запускает один модуль на многих входах.

| Поле | Тип | Смысл |
|---|---|---|
| `module_hash` | hex | модуль (только по хешу; загрузите его через `/v1/modules`) |
| `inputs` | base64[] | входы, или |
| `input_blobs` | hex[] | ID блобов как входы |
| `blobs` | hex[] | блобы, прикреплённые к каждой задаче |
| `entry`, `limits` | | как у `/v1/run` |
| `concurrency` | int | задач одновременно; по умолчанию все свободные слоты сети |
| `stop_on_exit_code` | int | закончить на первой задаче с этим кодом выхода |

Ответ — NDJSON. Строка на каждую задачу (результат как у `/v1/run` с полем
`index` или ошибка с полем `index`), по мере готовности и в любом порядке,
затем итоговая строка:

```sh
$S -d "{\"module_hash\":\"$HASH\",\"inputs\":[\"$(printf a|base64)\",\"$(printf bb|base64)\"]}" http://rakkan/v1/map
```
```
{"index":0,"exit_code":0,"output":"YQ==","output_size":1,"gas_used":51,"elapsed_ms":0.714,"node":"12D3KooW…","attempts":1}
{"index":1,"exit_code":0,"output":"YmI=","output_size":2,"gas_used":51,"elapsed_ms":0.992,"node":"12D3KooW…","attempts":1}
{"done":true,"reported":2,"total":2}
```

Строка ошибки: `{"index":3,"error":{"kind":"trap","message":"…"}}`.

`reported < total` в итоговой строке означает, что map остановлен раньше:
через `stop_on_exit_code` или из-за отмены. Ненужные задачи отменены и не
сообщаются. Тело запроса ограничено 256 МиБ (входы в base64). Большие входы
передавайте блобами.

## Блобы

### POST /v1/blobs

Сохраняет данные. Тело — сырые байты любого размера, можно потоком.

```sh
printf 'hello blob world' | $S --data-binary @- http://rakkan/v1/blobs
```
```json
{"id":"868768594da1a9ce88d4850a6120c91d325dde247fd8261c58fc601589e8d926","size":16}
```

ID зависит только от содержимого: одни и те же байты дают один ID на любом
узле.

### GET /v1/blobs/{id}

Отдаёт блоб потоком (`application/octet-stream`, `ETag` = ID). Поддерживает
`Range`:

```sh
$S -H 'Range: bytes=6-9' http://rakkan/v1/blobs/868768594da1a9ce88d4850a6120c91d325dde247fd8261c58fc601589e8d926
```
```
blob          (HTTP 206)
```

## Состояние

### GET /v1/status

```sh
$S http://rakkan/v1/status
```
```json
{"id":"12D3KooWQecPNLRsukiPJVos8kj5YcpapCcA4Y5vw1TRT1XySCsP","network":"12D3KooWSLA948LoLhp1K9bLviMpwnoVMw1tLG2njA7KveHkB1fj",
 "guest":false,"open":false,"load":{"slots":2,"busy":0,"queued":0,"capabilities":["image/1","tensor/1"]},"members":[]}
```

| Поле | Смысл |
|---|---|
| `id`, `network`, `version` | ID узла и сети, версия программы |
| `addrs` | адреса, которые узел сообщает другим (без `/p2p/…`) |
| `yggdrasil` | если включён: `addr` (`/ygg/…`) и `peers` — поднятых соединений с узлами Yggdrasil |
| `guest` | узел вошёл как гость |
| `open`, `pow_bits` | сеть открыта для гостей, сложность proof-of-work |
| `load` | слоты этого узла: `slots`, `busy`, `queued`, `capabilities` |
| `members[]` | известные узлы: `id`, `roles`, `guest`, `score` (у гостей), `connected`, `relayed`, `rtt_ms`, `addr` (удалённый адрес основного соединения: по нему видно, напрямую ли оно, через ретранслятор или через `/ygg/`), `last_seen`, `load` |
| `llm[]` | узлы, считающие стадии моделей, быстрые первыми: `id`, `device` (GPU), `speed` (ГБ/с), `free_mem`, `free_cache`, `shards` |

### GET /v1/trust

Реестр пиров: кредит и репутация. Поля: `id`, `served`, `consumed`,
`verified`, `failed`, `cheated`, `last_cheated` (если был пойман), `score`,
`member`; смысл — в [SDK](sdk.md#trust). У узла, ещё ни с кем не работавшего:

```sh
$S http://rakkan/v1/trust
```
```json
{"peers":[]}
```

## Языковые модели

Работают, если узел запущен с сервисом моделей (это делает любой демон).
Стадии считают узлы с флагом `-llm`.

### POST /v1/llm/models

Импортирует модель из блобов, загруженных заранее через `/v1/blobs`.

Формат llama2.c:

```json
{"checkpoint": "<id блоба>", "tokenizer": "<id блоба>", "name": "stories15M", "quant": "q8_0"}
```

Hugging Face:

```json
{"format": "hf", "config": "<id config.json>", "tokenizer": "<id tokenizer.json>",
 "tokenizer_config": "<id>", "generation_config": "<id>",
 "weights": ["<id model-00001-of-00002.safetensors>", "<id …>"], "quant": "q4_m"}
```

`quant`: `bf16`, `q8_0`, `q4_m`, `q4_0` или пусто (как в исходных файлах).

Ответ:

```json
{"model": "<ID модели>", "config": {"dim": …, "hidden_dim": …, "n_layers": …, "n_heads": …, "n_kv_heads": …,
 "vocab_size": …, "seq_len": …, "dtype": "q4_m", …}, "chat": true}
```

`chat` есть только у моделей Hugging Face: умеет ли модель формат чата.

DeepSeek-V4.1 можно импортировать прямо с Hugging Face Hub, ничего не загружая
в узел заранее:

```json
{"format": "hub", "repo": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "main", "name": "DeepSeek-V4.1-Flash"}
```

Узел скачивает только `config.json`, индекс и `tokenizer.json`. Ветка или тег
закрепляются как коммит. Манифест (около 10 КБ) хранит размер и хеш каждого
файла: LFS SHA-256 у весов, git SHA-1 у мелких файлов. Ещё в нём записано, в
каких файлах какой слой. Веса скачивают узлы, которые их обслуживают, и только
свои слои: сначала у участников сети, у которых файл уже есть, иначе с Hub.
Хеш проверяется при загрузке из любого источника. Оборванная загрузка
продолжается с того же места. Hub задаётся флагом демона `-hub` (по умолчанию
`https://huggingface.co`, пусто — только от участников), токен для закрытых
репозиториев — `HF_TOKEN`. В ответе дополнительно `repo` и `revision`.

### GET /v1/llm/models/{id}

Описание модели по её манифесту (узел берёт его у узлов, считающих стадии,
если у самого нет):

```json
{"model":"fe081d37…","name":"tiny-stories","arch":"llama","layers":4,"chat":false}
```

`arch` — `llama` для моделей llama-подобной архитектуры или имя другой
(`dsv41`); `chat` — у модели есть формат чата; `images` — понимает картинки;
`draft` — есть черновая модель. Неизвестная модель — 404 с
`{"error":{"kind":"not_found",…}}`.

### POST /v1/llm/prefetch

```json
{"model": "<ID>", "from": 0, "to": 20}
```

Заранее скачивает этому узлу файлы слоёв `[from, to)` модели с Hub (по
умолчанию всех). Отвечает `{"model": …, "bytes": …}`, когда файлы на месте. Если
запрос оборвался, загрузка продолжается. Нужно для больших файлов: таблица
Engram DeepSeek-V4.1 весит 101 ГБ, а стадия ждёт загрузки слоёв не дольше 10
минут.

DeepSeek-V4.1 импортируется и тем же запросом с `"format": "hf"`: узел узнаёт
модель по `config.json` (`text_config.model_type` = `deepseek_v41_text`).
`quant` для неё не задаётся: веса остаются в FP8/FP4 чекпойнта. Каждый блок
хранится отдельным блобом, таблицы Engram — тоже отдельными: стадия читает их
с диска по нескольку строк на токен. В ответе `config` — исходный
`config.json`. С `"chat": true` при генерации промпт оформляется форматом
чата DeepSeek (режим без рассуждений).

### POST /v1/llm/generate

```json
{"model": "<ID>", "prompt": "What is the capital of France?", "chat": true,
 "max_tokens": 50, "temperature": 0, "seed": 0, "stages": 0}
```

`"history"` — прежние реплики разговора, старые первыми, только вместе с
`"chat": true`: `[{"user": "…", "assistant": "…"}, …]`. Промпт тогда — весь
разговор в формате чата модели, а `prompt` — новое сообщение. Поддерживаются
форматы ChatML (`<|im_start|>`: Qwen и др.), Llama 3 и DeepSeek-V4.1; для
других форматов история — ошибка. Узел ничего не помнит между запросами:
историю каждый раз присылает клиент.

`"draft": true` для DeepSeek-V4.1 включает черновую модель (DSpark): она
предлагает следующие токены, а один проход конвейера проверяет их сразу
несколько. Текст тот же, что без неё, при любой температуре. В статистике
`passes` (проходов конвейера), `drafted` и `accepted` (черновых токенов
предложено и принято).

`"images"` — картинки для модели с кодировщиком изображений (DeepSeek-V4.1):
PNG или JPEG в base64 или как `data:`-URL с base64, до 16 штук. Ссылки на файлы и
адреса в сети узел не открывает. С `"chat": true` картинки идут в сообщении
перед текстом. Без `chat` каждое место картинки в `prompt` отмечается
строкой `<｜deepseek_image｜>`. Мест должно быть столько же, сколько картинок.
Кодировщик запускает узел-координатор: веса (970 МиБ) он загружает при первом
запросе с картинкой. В `prompt_tokens` входят позиции картинок. JPEG с
арифметическим кодированием и 12-битный узел не принимает.

```json
{"model": "<ID>", "prompt": "What is in the picture?", "chat": true,
 "images": ["data:image/png;base64,iVBORw0KGgo…"]}
```

Ответ — NDJSON: строка на токен, затем статистика.

```
{"token":<номер токена>,"text":"Par"}
{"token":<номер токена>,"text":"is"}
{"done":true,"stats":{"prompt_tokens":<…>,"tokens":2,"elapsed_ms":<…>,"tokens_per_s":<…>,
 "stages":[{"from":0,"to":14,"node":"12D3KooW…"},{"from":14,"to":24,"node":"12D3KooW…","witness":"12D3KooW…"}],
 "failovers":0,"caught":0}}
```

Ошибка во время генерации приходит строкой
`{"error":{"kind":"generation_failed"|"no_capacity","message":"…"}}`.
`text` токена может быть пустым: символ, занимающий несколько токенов,
выводится целиком с последним из них.

## Веб-страница узла

Отдельная программа `rakkan-web` (в том же архиве, что и узел) — такой же
клиент этого API, как SDK: подключается к сокету узла и раздаёт страницу с
участниками, узлами нейросетей и чатом с моделью сети. Сам узел о ней
ничего не знает.

```sh
rakkan-web                              # найдёт сокет установленного узла
rakkan-web -api /path/to/rakkan.sock    # или -config /path/to/node.json
rakkan-web -listen 0.0.0.0:7780 -token <пароль>
```

Без `-api` и `-config` программа ищет сокет там, куда его кладут
установщики (сначала установка для пользователя, потом для системы), и в
текущем каталоге. Сокет доступен только пользователю узла: при установке
для системы запускайте `sudo -u rakkan rakkan-web`.

Странице доступна часть API: `GET /v1/status`, `GET /v1/trust`,
`GET /v1/llm/models/{id}`, `POST /v1/llm/generate`. Задач и блобов через неё
нет.

Вкладка «Модели» загружает модель с Hugging Face Hub (`-hub`, по умолчанию
`https://huggingface.co`). Это делает сама `rakkan-web`: закрепляет ветку как
коммит, скачивает `config.json`, `tokenizer.json`, по возможности
`generation_config.json` и `tokenizer_config.json`, и веса — шарды из
`model.safetensors.index.json` или `.safetensors` в корне репозитория. Файлы
идут потоком с Hub прямо в `POST /v1/blobs` узла, на диск `rakkan-web` ничего
не пишет. SHA-256 каждого файла весов сверяется с тем, что публикует Hub.
Оборванная загрузка файла повторяется с начала, до трёх раз. Затем
`rakkan-web` вызывает `POST /v1/llm/models` с `"format": "hf"` и выбранным
`quant`. DeepSeek-V4.1 она узнаёт по `config.json` и передаёт узлу
`"format": "hub"`: веса скачают узлы, считающие стадии, каждый свои слои.
Токен Hugging Face для закрытых моделей страница передаёт только на время
загрузки и нигде не хранит (DeepSeek из закрытого репозитория требует
`HF_TOKEN` у самих узлов). Маршруты программы — `/app/models/inspect`,
`/app/models/import`, `/app/models/jobs`, `/app/models/jobs/{id}/cancel` —
защищены так же, как `/v1/`; загрузки помнятся, пока программа работает.

Защита. Любая страница в браузере может слать запросы на порт localhost,
поэтому на loopback-адресе `rakkan-web` отвечает только запросам с
заголовком `Host` вида `localhost`/`127.0.0.1`/`[::1]` (против DNS
rebinding), а запись принимает только JSON (чужая страница не пошлёт его без
CORS-preflight, которому никто не отвечает) и только со своего `Origin`. На
любом другом адресе нужен токен (`-token` или `RAKKAN_WEB_TOKEN`): запросы к
`/v1/` несут `Authorization: Bearer <токен>`, страница спрашивает его сама;
узлу токен не передаётся. Без токена такой адрес программа не откроет. По
HTTP токен идёт открытым текстом: вне доверенной сети ставьте перед ней
HTTPS-прокси.

С узлом 0.006-alpha страница работает, но без памяти разговора (старый узел
не знает поле `history`) и без имён моделей (нет `GET /v1/llm/models/{id}`);
она предупреждает об этом.
