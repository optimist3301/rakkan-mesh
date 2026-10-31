# @rakkan/sdk

Запуск WebAssembly-задач в сети Rakkan из Node.js (18+). Без зависимостей;
типы для TypeScript в `index.d.ts`. SDK общается с локальным узлом через его
API-сокет (`rakkan-node daemon -api rakkan.sock`), а узел распределяет задачи
по сети.

```js
import { connect } from '@rakkan/sdk';

const mesh = connect(); // $RAKKAN_SOCKET или ./rakkan.sock

// Одна задача: модуль загружается один раз и дальше передаётся по хешу.
const res = await mesh.run({ module: wasmBytes, input: 'hello', limits: { gas: 1e9, timeoutMs: 5000 } });
res.output; // Buffer
res.node;   // peer ID исполнившего узла
res.attempts; // > 1, если был сбой узла и задача ушла на другой

// Проверка результата: три разных узла должны вернуть одно и то же
// (вывод и gas побайтово), иначе принимается мнение большинства.
const checked = await mesh.run({ module: wasmBytes, input, replicas: 3 });
checked.replicas; // [{ node, agreed }]

// Много входов: задачи расходятся по сети, результаты приходят по мере готовности.
for await (const item of mesh.map({ module: wasmBytes, inputs, stopOnExitCode: 0 })) {
  if (item.error) console.error(item.index, item.error.kind);
  else console.log(item.index, item.exitCode, item.node);
}
// stopOnExitCode останавливает map на первой задаче с этим кодом выхода,
// выход из цикла (break) или AbortSignal отменяют оставшиеся задачи.
```

## Большие данные

Вход больше 256 КБ SDK сам загружает в хранилище узла как блоб и передаёт
ссылкой; выход больше 1 МиБ возвращается блобом и скачивается в `res.output`.
Блобы режутся на чанки по 1 МиБ с проверкой SHA-256 каждого; исполнитель
забирает вход по чанкам, при обрыве — докачивает.

```js
// Потоковая загрузка файла, без чтения в память.
const { id } = await mesh.putBlob({ path: 'photo.raw' });
const res = await mesh.run({ module, inputBlob: id, outputAsBlob: true });

// Потоковая выгрузка результата; start/end — диапазон байтов (докачка).
(await mesh.getBlob(res.outputBlob)).pipe(fs.createWriteStream('out.raw'));

// Map по блобам.
for await (const item of mesh.map({ module, inputBlobs: [id1, id2, id3] })) { ... }
```

Прикреплённые блобы задача читает кусками (`blob_read`), и исполнитель
скачивает только прочитанное — так один большой блоб (словарь, модель) служит
всем задачам map и скачивается один раз на узел:

```js
const { id: dict } = await mesh.putBlob({ path: 'dictionary.bin' });
for await (const item of mesh.map({ module, inputs, blobs: [dict] })) { ... }
```

## Изображения

Узлы с capability `image/1` (видно в `status()`) обрабатывают изображения
нативными kernels внутри изолированных воркеров: модуль задачи вызывает
`decode_input`, `fit`, `encode_output` и т. д. (см. ABI в корневом README).
Пример `examples/thumbnails.js` делает JPEG-миниатюры пачки фото по сети.

## Языковые модели

Модель в формате llama2.c (например, `stories15M.bin` и `tokenizer.bin` из
llama2.c/tinyllamas) делится по слоям между узлами, запущенными с
`rakkan-node daemon -llm`. Текст одинаков при любом разбиении, а если узел-стадия
упадёт посреди генерации, его слои переедут на другой узел и текст продолжится
так, как продолжился бы без сбоя.

```js
const { model } = await mesh.importModel({ checkpoint: { path: 'stories15M.bin' }, tokenizer: { path: 'tokenizer.bin' } });
for await (const e of mesh.generate({ model, prompt: 'Once upon a time', maxTokens: 200, stages: 3 })) {
  if (e.done) console.log(e.stats); // tokensPerSecond, stages, failovers
  else process.stdout.write(e.text);
}
```

Пример: `examples/story.js`.

Предел размера задаёт политика узлов: `rakkan-node daemon -max-input-mb 256
-max-output-mb 256`.

Ошибки задач — `TaskError` с полем `kind`: `out_of_gas`, `timeout`, `trap`,
`invalid_module`, `output_limit`, … (ошибки самой задачи, повторять бессмысленно),
`no_capacity`, `not_enough_nodes`, `no_consensus` (ошибки сети), `unavailable`
(нет узла на сокете).

Модуль задачи экспортирует `memory` и `run: () -> i32` и импортирует только
`rakkan.input_len/input_read/output_write/log` — см. ABI в корневом README.

Пример: `examples/search.js` ищет прообраз 32-битного хеша, разбивая диапазон
на части и параллельно перебирая их на всех узлах.

Тесты поднимают настоящий кластер из трёх демонов (нужен Go для сборки
`rakkan-node`, или укажите готовый бинарник в `RAKKAN_BIN`): `npm test`.
