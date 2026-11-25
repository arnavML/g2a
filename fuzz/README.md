# vfuzz

A fuzz harness for the runtime's parsers and the shader translator. It is a standalone crate and not a workspace member, so it never affects the main build. It needs no game files.

```
cd fuzz
cargo run --release -- <target> [iterations] [seed]
cargo run --release -- show <file>     # decode a saved program
```

The release profile keeps overflow checks and debug assertions on. A broken invariant therefore shows up as a panic, not as wrong output.

| target | input |
|---|---|
| `decode` | random GFX10 instruction words, through the decoder |
| `ir` | random programs, through `ir::build` and `structure` |
| `dxbc`, `native` | random DXBC containers and console shader blobs |
| `pe` | random PE images and import libraries |
| `xma` | random XMA packets |
| `sgd5` | random SGD5 records and shader packs |
| `emit` | random compute programs, through the whole translator (IR, structure, resources, IO, DXIL) |
| `emit2` | shader-like compute programs (VALU work, compares, selects, lane branches, stores), through the whole translator. About a third of these are translated fully. |

A panic saves the input as `crash-<target>-<iteration>.bin` in the current directory, and the summary lists where it panicked. On Windows, `emit` and `emit2` also check each translated shader with dxil.dll (`VICTORY_DXIL`, or the SDK's). A rejected shader is saved as `invalid-<n>.bin`, and the summary lists the validator's first error. Run it from a scratch directory, so that saved inputs stay outside the repository.
