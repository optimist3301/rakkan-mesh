// Asks a chat model from Hugging Face a question, with the model split
// across the mesh by layers.
//
//   node examples/chat.js <model dir> ["question"] [stages]
//
// The directory is a model as downloaded from the Hub: config.json,
// tokenizer.json, tokenizer_config.json, generation_config.json and the
// *.safetensors files. Llama, Mistral, Qwen2/2.5 and Qwen3 models load, for
// example Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen3-0.6B. Members started
// with `rakkan-node daemon -llm` serve the stages.

import { connect } from '@rakkan/sdk';

const [dir, question = 'Why is the sky blue? Answer briefly.', stages] = process.argv.slice(2);
if (!dir) {
  console.error('usage: node examples/chat.js <model dir> ["question"] [stages]');
  process.exit(2);
}
const mesh = connect();
const { model, config } = await mesh.importModel({ hf: dir });
console.error(`model ${model.slice(0, 12)}: ${config.n_layers} layers, dim ${config.dim}, ${config.dtype}`);
await new Promise((r) => setTimeout(r, 2500)); // let stage servers advertise

for await (const e of mesh.generate({ model, prompt: question, chat: true, maxTokens: 300, stages: stages ? Number(stages) : undefined })) {
  if (!e.done) {
    process.stdout.write(e.text);
    continue;
  }
  const s = e.stats;
  console.error(`\n\n${s.tokens} tokens in ${(s.elapsedMs / 1000).toFixed(2)}s (${s.tokensPerSecond.toFixed(1)} tokens/s)`);
  for (const st of s.stages) console.error(`  layers ${st.from}..${st.to - 1} on ${st.node.slice(-8)}`);
}
