// Tells a story with a language model split across the mesh by layers.
//
//   node examples/story.js <checkpoint.bin> <tokenizer.bin> ["prompt"] [stages]
//
// The checkpoint is in llama2.c's format, e.g. stories15M.bin and
// tokenizer.bin from Karpathy's llama2.c / tinyllamas. Members started with
// `rakkan-node daemon -llm` serve the stages.

import { connect } from '@rakkan/sdk';

const [checkpoint, tokenizer, prompt = 'Once upon a time', stages] = process.argv.slice(2);
if (!checkpoint || !tokenizer) {
  console.error('usage: node examples/story.js <checkpoint.bin> <tokenizer.bin> ["prompt"] [stages]');
  process.exit(2);
}
const mesh = connect();
const { model, config } = await mesh.importModel({ checkpoint: { path: checkpoint }, tokenizer: { path: tokenizer } });
console.error(`model ${model.slice(0, 12)}: ${config.n_layers} layers, dim ${config.dim}`);
await new Promise((r) => setTimeout(r, 2500)); // let stage servers advertise

process.stdout.write(prompt);
for await (const e of mesh.generate({ model, prompt, maxTokens: 200, stages: stages ? Number(stages) : undefined })) {
  if (!e.done) {
    process.stdout.write(e.text);
    continue;
  }
  const s = e.stats;
  console.error(`\n\n${s.tokens} tokens in ${(s.elapsedMs / 1000).toFixed(2)}s (${s.tokensPerSecond.toFixed(1)} tokens/s), failovers: ${s.failovers}`);
  for (const st of s.stages) console.error(`  layers ${st.from}..${st.to - 1} on ${st.node.slice(-8)}`);
}
