# Parser fuzz targets

This standalone crate is excluded from the engine workspace. It exercises public
CPU APIs without a model, a GPU, an HTTP listener or tool execution. Use the
repository's pinned nightly, `cargo-fuzz`, and a working C++ compiler/linker.
`just lab-fuzz` runs the crate's tests and every target below for 60 s on the box.

```sh
cargo install cargo-fuzz --locked
cargo fuzz build
cargo test --manifest-path fuzz/Cargo.toml
cargo clippy --manifest-path fuzz/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path fuzz/Cargo.toml -- --check

for target in jinja_parse jinja_render glmxml qwenxml dsml hermes; do
  cargo fuzz run "$target" "fuzz/corpus/$target" -- \
    -max_total_time=60 -max_len=4096 -dict=fuzz/parsers.dict
done
```

`jinja_parse` accepts valid UTF-8 and parses arbitrary template syntax, including
malformed syntax. A `TemplateError` is an ordinary result, and so is a template
nested past the parser's limit (`MAX_NESTING` in `crates/jinja/src/lib.rs`); a
process panic or a stack overflow is a finding. The Unicode tag-closer seed covers
unclosed markup containing multibyte characters. The parser's byte scan compares
ASCII delimiters as bytes, so an intermediate byte offset need not be a UTF-8
character boundary.

`jinja_render` uses a leading byte to choose one of six bounded template programs;
the remaining bytes are a JSON object used as the context. It checks deterministic
rendering with both supported float styles. The fixed programs avoid arbitrary
`range` bounds and template recursion while varying data, types, missing values,
scoping, string operations and JSON output.

The four scanner targets compare whole-message, character-sized and varying-sized
UTF-8 chunks, with and without a tool schema. Accepted messages must produce the
same content and calls. Malformed messages must agree on acceptance, while error
wording may quote the buffer available at the point of failure. Calls must have
dense indices and JSON object arguments. The harness collects parsed calls only;
it never invokes their names.

The serve gates `hw_*_parse_does_not_depend_on_chunking` in
`crates/serve/tests/{glmxml,qwenxml,hermes,dsml}.rs` pin the same chunking
property on fixed messages from the real templates. These targets hold it for
arbitrary input, so a finding here is a new case for those gates.

## Checking that the instrument detects a defect

In a disposable worktree, replace `self.buf.push_str(piece)` in
`crates/serve/src/glmxml.rs` with `self.buf = piece.to_owned()`. This discards a
partial tag on each push. Running the GLM target on the checked-in seeds with
`-runs=100 -seed=20261012` fails at `character chunks changed the parse`. Restore
the parser before ordinary validation. This mutation changes production stream
state, rather than adding an unconditional failure to the fuzz target.

For `jinja_render`, replace `m.iter()` in the `"items"` arm of `call_method` in
`crates/jinja/src/lib.rs` with `m.iter().collect::<std::collections::HashMap<_, _>>().into_iter()`.
Each render then orders the object's items by a fresh random hash seed. Running the
render target on the checked-in seeds with `-runs=100 -seed=20261012` fails at
`rendering leaked state between requests`. Restore the renderer before ordinary
validation.

A bounded run is evidence of the inputs it exercised, not a proof that a parser
is free of defects. Keep generated corpora and artifacts outside commits.
