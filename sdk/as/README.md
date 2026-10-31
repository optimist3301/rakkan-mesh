# @rakkan/as

Задачи для сети Rakkan на AssemblyScript — TypeScript-подобном языке,
компилирующемся в WebAssembly.

```ts
import { input, outputString, log, blobs, Image } from "@rakkan/as";

export function run(): i32 {        // 0 — успех
  const text = String.UTF8.decode(input().buffer);
  log("got " + text.length.toString() + " chars");
  outputString(text.toUpperCase());
  return 0;
}
```

Сборка: `npm install && npm run build` (`build.js`) — собирает `examples/` и
`test/` в `build/`. Задаче нужны два флага компилятора, их передаёт `build.js`:

- `--runtime stub` — без сборщика мусора: задачи короткие, а освобождённая
  память всё равно уходит вместе с процессом. Следствие: не выделяйте память
  в горячих циклах (см. `blobs.readInto`). Для долгих задач с большим числом
  временных объектов подойдёт `--runtime incremental`;
- `--use abort=assembly/index/abort` — стандартный `abort` компилятора
  импортируется из `env`, которого в песочнице нет; привязки логируют
  сообщение и завершают задачу trap. Сообщение (`abort: Index out of range at
  ~lib/array.ts:114:42`) приходит пользователю в тексте ошибки.

## API (`assembly/index.ts`)

| | |
|---|---|
| `input()`, `inputString()` | вход задачи |
| `output(bytes)`, `outputString(s)` | дописать в выход |
| `log(message)` | строка в диагностический лог (видна в результате и в ошибке) |
| `readU32/readU64(buf, off)`, `u32Bytes(v)` | little-endian помощники |
| `blobs.count/size/read/readInto` | прикреплённые блобы, чтение кусками; узел скачивает только прочитанное |
| `Image.decodeInput/decodeBlob/decode/create` | изображения через нативные kernels (capability `image/1`) |
| `img.fit/resize/crop/grayscale/blur` | операции — новое изображение или `null` |
| `img.readPixels/writePixels` | RGBA8-буфер для своих алгоритмов |
| `img.outputJPEG(q)/outputPNG()` | закодировать сразу в выход |
| `Tensor.fromF32/fromI8/fromBlob` | тензоры через нативные kernels (capability `tensor/1`); веса из блоба |
| `t.matmul/matmulT/add/scale/relu/softmax/quantize` | операции — новый тензор; ошибка формы прерывает задачу |
| `t.argmax()/toF32()` | результаты в память задачи |

## Примеры

- `examples/search.ts` — поиск прообраза хеша (тот же, что `search.wasm`,
  собранный вручную; Go-тест сверяет их результаты);
- `examples/sepia.ts` — декодирование и масштабирование kernels, попиксельная
  сепия на AssemblyScript, JPEG в выход;
- `examples/wordcount.ts` — подсчёт слова в диапазоне большого прикреплённого
  блоба; запускается как map по диапазонам (`sdk/js/examples/wordcount.js`);
- `examples/mlp.ts` — двухслойный перцептрон, веса в прикреплённом блобе,
  матрицы в tensor kernels; батчи по сети — `sdk/js/examples/inference.js`;
- `examples/matmul-wasm.ts` / `matmul-kernel.ts` — одно умножение матриц на
  чистом WebAssembly и через kernel: результаты побитово равны, kernel в ~8 раз
  быстрее.

Собранные модули лежат в `build/` и используются тестами Go
(`internal/sandbox/as_test.go`) и JS SDK.
