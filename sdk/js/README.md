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
