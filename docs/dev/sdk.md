# JS SDK: `@rakkan/sdk`

Клиентская библиотека для Node.js 18+, без зависимостей, с типами TypeScript
(`index.d.ts`). Она говорит с локальным узлом через его unix-сокет, а узел
распределяет работу по сети.

```sh
npm install /путь/к/rakkan-mesh/sdk/js
```

```js
import { connect, TaskError } from '@rakkan/sdk';
```

- [connect](#connect)
- [Задачи: upload, run, map](#задачи)
- [Блобы: putBlob, getBlob, readBlob](#блобы)
- [Сеть: status, trust](#сеть)
- [Языковые модели: importModel, generate](#языковые-модели)
- [Типы](#типы)
- [Ошибки](#ошибки)
- [Отмена и таймауты](#отмена-и-таймауты)

## connect

```ts
connect(options?: { socket?: string }): Mesh
```

Возвращает клиента. Сокет ищется так: `options.socket`, затем переменная
окружения `RAKKAN_SOCKET`, затем `./rakkan.sock`. Подключение ленивое: если
узла нет, первый вызов отклоняется с `TaskError` вида `unavailable`.

Путь к unix-сокету в Linux не длиннее 107 байт. Если узел запущен с длинным
путём в `-api`, он не сможет открыть сокет. Используйте короткий или
относительный путь.

Один объект `Mesh` можно использовать из любого числа одновременных вызовов.

## Задачи

### upload

```ts
upload(wasm: Bytes): Promise<string>
```

Загружает модуль на узел и возвращает его хеш (SHA-256, hex). Повторная
загрузка того же модуля через тот же `Mesh` не отправляет байты. Вызывать
`upload` явно не обязательно: `run` и `map` принимают сам модуль и загружают
его сами. Явная загрузка полезна, чтобы дальше передавать хеш.

`Bytes` — это `Uint8Array` (в том числе `Buffer`), `ArrayBuffer` или строка
(UTF-8).

### run

```ts
run(options: RunOptions): Promise<Result>
```

Запускает одну задачу на лучшем доступном узле. Возможно, на этом же,
если у него есть свободный воркер.

| Параметр | Тип | Смысл |
|---|---|---|
| `module` | `Bytes` или хеш | модуль или хеш из `upload` (обязателен) |
| `input` | `Bytes` | вход задачи |
| `json` | любое | вход как `JSON.stringify(json)` вместо `input` |
| `inputBlob` | `string` | ID блоба как вход (данные уже на узле) |
| `blobs` | `string[]` | до 16 прикреплённых блобов, которые задача читает кусками |
| `entry` | `string` | другая точка входа вместо `run` |
| `limits` | `Limits` | лимиты задачи, см. [Типы](#типы) |
| `replicas` | `number` | запустить на стольких разных узлах и принять ответ большинства |
| `outputAsBlob` | `boolean` | большой выход не скачивать, вернуть только `outputBlob` |
| `signal` | `AbortSignal` | отмена |

Поведение:

- **Вход больше 256 КиБ** SDK сам загружает как блоб, а узел передаёт его
  исполнителю проверенными чанками.
- **Выход больше 1 МиБ** приходит как блоб. По умолчанию SDK скачивает его в
  `output`, а с `outputAsBlob` отдаёт только `outputBlob`, например чтобы
  передать результат следующей задаче.
- **Сбои исполнителя.** Если исполнитель упал, потерял связь или занят, узел
  повторяет задачу на другом (до 3 исполнителей) или ждёт освобождения.
  Сколько попыток понадобилось, видно в `attempts`.
- **Перезапуск узла.** Если узел перезапустился и забыл модуль, SDK загрузит
  его ещё раз и повторит запрос.
- **Ненулевой код выхода** — это обычный результат с `exitCode`, а не
  исключение. Исключение `TaskError` бросается только тогда, когда задача
  не смогла завершиться (см. [Ошибки](#ошибки)).

```js
const res = await mesh.run({
  module: readFileSync('task.wasm'),
  json: { n: 42 },
  limits: { gas: 2e9, timeoutMs: 5000, memoryMB: 32 },
});
if (res.exitCode !== 0) console.log('задача сообщила о проблеме:', res.logs);
```

**Проверка репликами:**

```js
const res = await mesh.run({ module, input, replicas: 3 });
// Задача шла на трёх разных узлах; принят ответ, совпавший у большинства
// (код выхода, выход и gas побайтово).
for (const r of res.replicas) console.log(r.node, r.agreed ? 'согласен' : 'расходится');
```

Без большинства вызов отклоняется с `no_consensus`, а если разных узлов
меньше, чем реплик, — с `not_enough_nodes`.

### map

```ts
map(options: MapOptions): AsyncGenerator<MapItem>
```

Запускает один модуль на многих входах по всей сети и отдаёт результаты по
мере готовности, **не по порядку**: у каждого элемента есть `index` входа.

| Параметр | Тип | Смысл |
|---|---|---|
| `module` | `Bytes` или хеш | модуль |
| `inputs` | `Bytes[]` | входы задач |
| `inputBlobs` | `string[]` | или ID блобов как входы |
| `blobs` | `string[]` | блобы, прикреплённые к каждой задаче (общие данные, исполнитель скачивает один раз) |
| `entry`, `limits`, `outputAsBlob`, `signal` | | как у `run` |
| `concurrency` | `number` | сколько задач одновременно; по умолчанию столько, сколько свободных слотов у всех узлов |
| `stopOnExitCode` | `number` | закончить map на первой задаче с этим кодом выхода; остальные отменяются |

Элемент — либо `{index, ...Result}`, либо `{index, error: TaskError}`.
Ошибка одной задачи не останавливает остальные.

```js
let total = 0;
for await (const item of mesh.map({ module, inputs })) {
  if (item.error) {
    console.error(`вход ${item.index}: ${item.error.kind}`);
    continue;
  }
  total += item.output.readUInt32LE(0);
}
```

Выход из цикла (`break`, `return`, исключение) или отмена `signal` закрывает
соединение, и узел отменяет задачи, которые ещё не закончились.

Если среди `inputs` есть вход больше 256 КиБ, SDK загружает все входы как
блобы. Повторы при сбоях исполнителей работают как у `run`; реплик у `map`
нет: для проверки используйте `run` с `replicas` (см.
[distributed.md](distributed.md#проверка-результатов)).

## Блобы

Блоб — неизменяемые данные любого размера на узле, адресуемые по хешу
содержимого: одни и те же байты дают один и тот же ID на любом узле. Данные
режутся на чанки по 1 МиБ, у каждого свой хеш. Исполнители скачивают блобы
с вашего узла и с других, у кого они есть, проверяя каждый чанк, а
оборванную передачу докачивают.

### putBlob

```ts
putBlob(data: BlobSource, options?: { signal?: AbortSignal }): Promise<{ id: string; size: number }>
```

Сохраняет данные на узле потоково. `BlobSource` — это `Bytes`, поток Node.js
(`Readable`), веб-поток (`ReadableStream`) или `{ path: 'файл' }`, чтобы
отправить файл с диска, не читая его в память.

```js
const { id } = await mesh.putBlob({ path: 'corpus.txt' });
```

### getBlob, readBlob

```ts
getBlob(id: string, options?: { start?: number; end?: number; signal?: AbortSignal }): Promise<NodeJS.ReadableStream>
readBlob(id: string, options?: { start?: number; end?: number; signal?: AbortSignal }): Promise<Buffer>
```

`getBlob` отдаёт поток, `readBlob` — весь блоб (или диапазон) в `Buffer`.
`start` и `end` задают диапазон байт, `end` включительно. Блоб читается с
локального узла. Неизвестный ID даёт `TaskError` вида `not_found`.

Блобы хранятся на узле в пределах бюджета `-store-gb`. Давно не
использованные блобы вытесняются, если место нужно новым, а блобы
работающих задач защищены от вытеснения.

## Сеть

### status

```ts
status(): Promise<Status>
```

Состояние узла и его представление о сети (значения условные):

```json
{
  "id": "12D3KooW…",              // ID этого узла
  "network": "12D3KooW…",         // ID сети
  "guest": false,                 // узел вошёл как гость (без сертификата)
  "open": false, "pow_bits": 24,  // открыта ли сеть для гостей
  "load": { "slots": 2, "busy": 0, "queued": 0, "capabilities": ["image/1", "tensor/1"] },
  "members": [
    { "id": "…", "roles": ["member"], "connected": true, "relayed": false, "rtt_ms": 0.4,
      "load": { "slots": 4, "busy": 1, "queued": 0, "capabilities": ["image/1", "tensor/1"] } },
    { "id": "…", "roles": ["guest"], "guest": true, "score": 0.12, "connected": true, … }
  ],
  "llm": [ { "id": "…", "device": "<имя GPU>", "speed": 20.5, "free_mem": 9000000000 } ]
}
```

Сумма `slots` узла и участников — это сколько задач сеть может считать
одновременно.

### trust

```ts
trust(): Promise<TrustPeer[]>
```

Что узел знает о пирах, с которыми работал. Полезно в открытой сети:

| Поле | Смысл |
|---|---|
| `id`, `member` | ID; `member` — узел с сертификатом (или этот узел) |
| `score` | репутация гостя от 0 до 1 (EigenTrust); у участников 1 |
| `served` / `consumed` | gas, потраченный этим узлом на пира / пиром на этот узел |
| `verified`, `failed`, `cheated` | подтверждённые результаты, сбои, пойманный обман |
| `lastCheated` | когда пир последний раз был пойман |

## Языковые модели

### importModel

```ts
importModel(options): Promise<{ model: string; config: ModelConfig; chat?: boolean }>
```

Загружает модель и делит её на блобы по слоям. ID модели — хеш манифеста,
поэтому он одинаков на любом узле для одних и тех же файлов, имени и `quant`.

| Параметр | Смысл |
|---|---|
| `checkpoint`, `tokenizer` | чекпойнт и токенизатор в формате llama2.c (`BlobSource`) |
| `hf` | или каталог модели Hugging Face (`config.json`, `tokenizer.json`, `*.safetensors`, по желанию `tokenizer_config.json`, `generation_config.json`). Поддерживаются Llama, Mistral, Qwen2/2.5, Qwen3 |
| `quant` | хранить веса как `'bf16'`, `'q8_0'` (вдвое меньше bf16, почти без потерь), `'q4_m'` или `'q4_0'` (около трети) |
| `name` | имя для манифеста |
| `signal` | отмена |

Узлам, которые будут считать стадии модели, нужен флаг `-llm`. Модели
Mistral и Llama 2 в формате HF сейчас не загрузятся: их токенизатор
(SentencePiece с byte fallback) не поддержан, импорт сообщит об этом.

### generate

```ts
generate(options: GenerateOptions): AsyncGenerator<GenerateEvent>
```

Генерирует текст моделью, разрезанной по слоям между узлами.

| Параметр | Смысл |
|---|---|
| `model` | ID из `importModel` |
| `prompt` | промпт |
| `chat` | обернуть промпт в формат чата модели (для instruct-моделей) |
| `maxTokens` | сколько токенов сгенерировать; по умолчанию до конца контекста |
| `temperature` | 0 (по умолчанию) — всегда самый вероятный токен |
| `seed` | зерно выборки при `temperature > 0`: тот же `seed` даёт тот же текст |
| `stages` | на сколько стадий делить слои; по умолчанию сеть раскладывает слои по скорости и памяти узлов сама |
| `signal` | отмена |

События: `{token, text}` по мере генерации, затем `{done: true, stats}`.

```js
for await (const e of mesh.generate({ model, prompt: 'Столица Франции?', chat: true, maxTokens: 50 })) {
  if (e.done) console.log(e.stats);
  else process.stdout.write(e.text);
}
```

`stats`: `promptTokens`, `tokens`, `elapsedMs`, `tokensPerSecond`,
`stages` (`{from, to, node, witness?}` — какие слои считал какой узел, и
свидетель для стадии на госте), `failovers` (стадии, переехавшие на другой
узел после сбоя), `caught` (пойманные расхождения у гостей). Если узел-стадия
упадёт посреди генерации, текст продолжится ровно таким, каким был бы без
сбоя. Выход из цикла останавливает генерацию.

## Типы

```ts
interface Limits {
  gas?: number;            // бюджет gas, по умолчанию 5e9
  timeoutMs?: number;      // время по часам, по умолчанию 10000
  memoryMB?: number;       // линейная память задачи, по умолчанию 64
  maxInputBytes?: number;  // по умолчанию 16 МиБ, но не меньше размера входа
  maxOutputBytes?: number; // по умолчанию 16 МиБ
}

interface Result {
  exitCode: number;        // 0 — успех
  output: Buffer;          // выход (пустой при outputAsBlob и большом выходе)
  outputBlob?: string;     // ID блоба, если выход больше 1 МиБ
  outputSize: number;      // размер выхода
  logs: string;            // лог задачи
  gasUsed: number;
  elapsedMs: number;       // время исполнения на узле
  node: string;            // ID узла, исполнившего задачу
  attempts: number;        // > 1, если исполнитель сбоил и задача ушла на другой
  replicas: Replica[];     // при replicas или проверке результата гостя
  checked: boolean;        // результат гостя подтверждён независимым запуском
}

interface Replica { node: string; agreed: boolean; error?: string }
```

Потолки лимитов задаёт политика узлов. Сейчас это 5·10⁹ gas, 10 с и
64 МиБ памяти на задачу, а размеры входа и выхода настраиваются флагами
демона (подробнее в [tasks.md](tasks.md#gas-и-лимиты)).

## Ошибки

Все ошибки — `TaskError` с полями `kind`, `message` и, если ответил узел,
`status` (HTTP-код).

| `kind` | Что случилось | Что делать |
|---|---|---|
| ошибки задачи: `out_of_gas`, `timeout`, `trap`, `memory_access`, `output_limit`, `invalid_module`, `forbidden_import`, `missing_entry`, `input_too_large`, `module_too_large` | задача не завершилась (см. [tasks.md](tasks.md#код-выхода-и-ошибки)); повторять бессмысленно | исправить задачу или поднять лимиты |
| `no_capacity` | нет узла, способного взять задачу: нет исполнителей, нет нужной capability, лимиты выше политики всех узлов, у гостя кончился кредит | текст объясняет причину |
| `not_enough_nodes` | разных узлов меньше, чем `replicas` | уменьшить `replicas` |
| `no_consensus` | реплики не дали большинства | серьёзный сигнал: узлы расходятся |
| `worker_crashed`, `worker_oom`, `internal`, `blob_unavailable`, `canceled` | сбой исполнителя после всех повторов | повторить позже |
| `bad_request` | неверный запрос: неизвестный модуль или блоб, неверные поля | исправить вызов |
| `not_found` | блоб не найден на узле | загрузить его |
| `generation_failed` | ошибка генерации текста | текст объясняет |
| `unavailable` | нет узла на сокете | запустить узел, проверить путь |

```js
try {
  await mesh.run({ module, input });
} catch (err) {
  if (err instanceof TaskError && err.kind === 'out_of_gas') { /* дать больше gas или разбить работу */ }
  else throw err;
}
```

## Отмена и таймауты

Все методы принимают `signal` (`AbortSignal`). Отмена `run` или `map`
закрывает соединение с узлом, и узел отменяет задачи на исполнителях.

```js
const res = await mesh.run({ module, input, signal: AbortSignal.timeout(30_000) });
```

`limits.timeoutMs` ограничивает время самой задачи на исполнителе, а
`signal` — всё ожидание клиента, включая очередь и повторы.
