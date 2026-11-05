# Rakkan Mesh: документация разработчика

Это руководство для тех, кто пишет программы, которые считают в сети Rakkan:
задачи, исполняемые на узлах, и клиентские программы, которые эти задачи
раздают и собирают результаты.

| Документ | Что внутри |
|---|---|
| [Задачи](tasks.md) | Как написать задачу: ABI WebAssembly, вход и выход, блобы, нативные kernels для изображений и тензоров, gas и лимиты, детерминизм, AssemblyScript, C, отладка |
| [JS SDK](sdk.md) | Справочник `@rakkan/sdk`: все методы, параметры, результаты, ошибки, отмена |
| [HTTP API](http-api.md) | Локальный API узла, на котором построен SDK: эндпоинты, форматы, коды ответов, примеры на `curl`. Нужен для клиентов на других языках |
| [Распределённые программы](distributed.md) | Как сеть исполняет задачи и как этим пользоваться: шаблоны (map/reduce, поиск с ранней остановкой, большие данные, проверка репликами), обработка ошибок, производительность, открытая сеть, языковые модели |

Как сеть устроена внутри, описывает [ARCHITECTURE.md](../ARCHITECTURE.md),
а запуск узлов — [инструкция для админа, узла и гостя](../Инструкция-для-чайников.md).

## Модель за одну минуту

```
 ваша программа ──(SDK, unix-сокет)──▶ локальный узел ──(сеть)──▶ узлы-исполнители
   (Node.js)                            rakkan-node daemon          песочницы WebAssembly
```

- **Задача** — модуль WebAssembly плюс вход (байты). Узел-исполнитель запускает
  функцию `run` модуля в песочнице. Задача читает вход, пишет выход и
  возвращает код выхода. Больше ей ничего не доступно: ни файлов, ни сети, ни
  часов, ни случайности.
- **Клиент** говорит только со своим локальным узлом, через unix-сокет
  `rakkan.sock`. Узел сам выбирает исполнителя, передаёт ему модуль (один раз,
  дальше по хешу) и вход, а при сбое исполнителя повторяет задачу на другом.
- **Детерминизм.** Одна и та же задача с одним и тем же входом даёт на любом
  узле один и тот же результат, байт в байт, и тратит одинаковый gas. На этом
  стоит проверка результатов репликами и защита от недобросовестных узлов.
- **Большие данные** (фото, архивы, веса моделей) не пересылаются в каждой
  задаче, а хранятся как **блобы**: файлы, адресуемые по хешу содержимого.
  Задача получает блоб на вход или читает куски прикреплённых блобов.
- **Распределённая программа** — это клиент, который режет работу на много
  независимых задач, раздаёт их одним вызовом `map` и собирает результаты.

## Быстрый старт

Нужны запущенный узел (см. [инструкцию](../Инструкция-для-чайников.md)) и
Node.js 18+.

**1. Задача** на AssemblyScript, `upper.ts`: переводит текст в верхний регистр.

```ts
import { inputString, outputString, log } from "@rakkan/as/assembly";

export function run(): i32 {        // 0 — успех
  const text = inputString();
  log("got " + text.length.toString() + " chars");
  outputString(text.toUpperCase());
  return 0;
}
```

```sh
npm install --save-dev assemblyscript@0.28.20 /путь/к/rakkan-mesh/sdk/as
npx asc upper.ts --outFile upper.wasm --runtime stub \
    --use abort=~lib/@rakkan/as/assembly/index/abort -O3
```

Проверка без сети:

```sh
printf 'hello, mesh' | rakkan-node run -input - upper.wasm
# HELLO, MESHgot 11 chars
# exit=0 gas=3006 elapsed=2.854ms
```

**2. Клиент** на Node.js, `client.js`:

```sh
npm install /путь/к/rakkan-mesh/sdk/js
```

```js
import { readFileSync } from 'node:fs';
import { connect } from '@rakkan/sdk';

const mesh = connect({ socket: '/путь/к/rakkan.sock' });
const module = readFileSync('upper.wasm');

// Одна задача.
const res = await mesh.run({ module, input: 'привет' });
console.log(res.output.toString(), 'на узле', res.node, 'gas', res.gasUsed);

// Много задач сразу: они расходятся по всем свободным узлам.
for await (const item of mesh.map({ module, inputs: ['a', 'b', 'c'] })) {
  if (item.error) console.error(item.index, item.error.kind);
  else console.log(item.index, item.output.toString());
}
```

Дальше: [как писать задачи](tasks.md) и [как строить из них
распределённые программы](distributed.md).

## Примеры в репозитории

| Пример | Что показывает |
|---|---|
| `sdk/js/examples/pi.js` + `sdk/as/examples/montecarlo.ts` | map/reduce: оценка π методом Монте-Карло, детерминированный генератор случайных чисел |
| `sdk/js/examples/search.js` | поиск прообраза хеша с ранней остановкой на первой находке |
| `sdk/js/examples/wordcount.js` + `sdk/as/examples/wordcount.ts` | большой файл как блоб, каждая задача читает свой диапазон |
| `sdk/js/examples/thumbnails.js` | обработка фото нативными kernels изображений |
| `sdk/js/examples/inference.js` + `sdk/as/examples/mlp.ts` | пакетный инференс нейросети, веса в прикреплённом блобе, tensor kernels |
| `sdk/js/examples/chat.js`, `story.js` | языковые модели, разрезанные по узлам |

## JSON-конфиг узла

Всё, что нужно демону, можно держать в одном файле: настройки, приватный ключ узла и сертификат сети.

```sh
rakkan-node config init -network <ID сети> -key node.key -cert node.cert -out node.json
rakkan-node daemon -config node.json
```

```json
{
  "network": "12D3Koo...",
  "private_key": "<base64 ключа узла>",
  "certificate": { "body": "...", "sig": "..." },
  "bootstrap": ["/ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo..."],
  "workers": 4,
  "status": "1m"
}
```

- `private_key` и `certificate` заменяют файлы `-key` и `-cert`. Без `certificate` узел входит гостем.
- Остальные поля называются как флаги `daemon`: строки, числа, булевы значения; длительности — строкой (`"30s"`), списки адресов — массивом. Неизвестное поле — ошибка.
- Флаг в командной строке сильнее файла, в том числе `-key` и `-cert`.
- Файл содержит приватный ключ, поэтому должен быть доступен только владельцу (`chmod 600`); иначе демон не запустится. `config init` создаёт его с правами 0600 и не перезаписывает существующий.

## Сборка и релизы

```sh
make                          # ./rakkan-node для этой машины
make test                     # go vet + go test
make dist VERSION=0.005-alpha # архивы и SHA256SUMS в dist/ для linux, macOS и Windows (amd64, arm64)
rakkan-node version
```

cgo не нужен, поэтому все платформы собираются с любой машины. Релиз делает GitHub Actions: тег `v*` (например `v0.005-alpha`) запускает `.github/workflows/release.yml`, который прогоняет тесты, собирает архивы и публикует релиз; теги с `alpha`, `beta` или `rc` помечаются как пре-релиз. Изоляция воркеров (seccomp, namespaces, cgroup) работает только на Linux.

## Установщики

Установка для текущего пользователя, без прав администратора. Скрипт показывает логотип, спрашивает язык, сеть (**публичная открытая** `12D3KooWJRPbdQ6Q8TXSctKQ8B4ZYHtKArqbKKReeQdx4iojLx7s` или **своя**), способ входа (гостем или по сертификату), тип узла (рабочий, клиент, ретранслятор, LLM), число задач, диск, адреса известных узлов и автозапуск. Скачанный архив проверяется по `SHA256SUMS`. Результат — один JSON-конфиг с ключом и сертификатом (см. выше), права только владельцу.

```sh
# Linux и macOS
curl -fsSL https://raw.githubusercontent.com/optimist3301/rakkan-mesh/main/scripts/install.sh -o install.sh
bash install.sh                  # --yes: без вопросов; --uninstall [--purge]: удалить
```

```powershell
# Windows (PowerShell)
Invoke-WebRequest https://raw.githubusercontent.com/optimist3301/rakkan-mesh/main/scripts/install.ps1 -OutFile install.ps1
.\install.ps1                    # -Yes: без вопросов; -Uninstall [-Purge]: удалить
```

Автозапуск: systemd `--user` на Linux, LaunchAgent на macOS, задача планировщика при входе на Windows. Строгая изоляция задач (namespaces + seccomp) есть только на Linux, и только если ядро разрешает user namespaces; иначе установщик ставит `"isolation": "process"`. На macOS и Windows всегда `process`.

На вопросы можно ответить заранее переменными `RAKKAN_NETWORK` (`public` или ID), `RAKKAN_NODE_TYPE`, `RAKKAN_CERT`, `RAKKAN_BOOTSTRAP`, `RAKKAN_AUTOSTART`, `RAKKAN_LANG` и другими (список в начале скриптов); вместе с `--yes` это даёт установку без участия человека.
