// Fails on purpose: an out-of-bounds access must reach the task's log
// through the bindings' abort and trap.
import { input } from "../assembly/index";

export function run(): i32 {
  const a = new Array<i32>(2);
  return a[input().length + 5];
}
