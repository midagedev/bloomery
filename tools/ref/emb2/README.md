# EmbeddingGemma reference sets

Both encoder versions use mainline llama.cpp at `6868a52cb337450658006ad2f9281c2eb63f1666`.
`build-embedding-ref <architecture>` and `dump-ref-embedding <architecture> [steps...]`
share the linker, dumper and set writer. Model profiles own the vocabulary, context,
files, digests, available steps and optional image projector.

For v1, with the pinned mainline tree already built:

```sh
just build-emb1-ref
just dump-ref-tokenizer llama
just dump-ref-emb1
```

`build-emb1-ref` selects `gemma-embedding`; `build-emb2-ref` selects
`gemma-embedding2`. The v2 recipe names and step arguments remain available.
Both use `tools/ref/emb2/dump_emb2.cpp` and the same mainline build.
Set `LCPP` to that tree on another machine.

V1 writes `ref_gemma_embedding_text`, `ref_gemma_embedding_text.cpu` and
`ref_gemma_embedding_server`. Their families are `text-gemma-embedding`,
`text-gemma-embedding-cpu` and `server-gemma-embedding`. They pin the v1 model
file, its SHA256, architecture, oracle commit, set kind and device. The text-only
profile has no tower step or projector requirement.

In a v1 text set, `embd` holds the API's token rows after the two learned dense
projections, equal to the tap `result_embd`. `result_norm` holds the RMS-normalized
rows before that dense head. `pooled` holds the API's mean-pooled vector after
the dense head. The taps also include every `l_out-<layer>`. `tap_effect` compares
`result_embd_pooled` (stored as `result_embd`) with the clean pass in both versions.
The dense projections follow pooling in the pooled pass and operate on individual
rows when pooling is NONE.

The shared tokenizer cases precede `cases-emb1.txt`. The extra file includes
paired distant word substitutions and an input of 1500 words. In the pair whose
ids differ at token 512, compare the first-layer rows at query 256 (distance 256,
inside the window) and query 255 (distance 257, outside). This inspects the
symmetric window before later layers propagate information from farther away.
The server set contains raw and normalized vectors; its MRL rows are derived
and marked `DERIVED`, as in the v2 set.

The tokenizer set includes the five corpus snapshots as well as the fixed cases.
Its default profile's `IK` tree must contain `src/`, `docs/`, `README.md` and
`github-data/`, as for the v2 tokenizer set. `LCPP` selects the tokenizer executable;
`IK` selects those corpus texts.

## Decoded video frames (EmbeddingGemma 2)

`just dump-ref-emb2 video video.cpu` writes `ref_gemma_embedding2_video`
and its CPU twin. Both require the pinned text model, BF16 projector and mainline
build already present. No video decoder is used. The shared frame list in
`crates/refset/src/arch/gemma_embedding2/video-frames.txt` supplies one ordered
sequence, with `--image-max-tokens 140 --video-fps 1
--video-timestamp-interval 0`. The pinned helper's `Video:` text precedes the
frames. Each frame is a typed image content part. The sets
record the frame count, each frame's file and SHA256 in order, and the total
prompt position count. `video/raw` and `video/embd` are the single sequence's
raw and normalized pooled vectors, not an average of separately pooled frames.

`vision::arch::gemma4v::prepare_video` takes 1 to 32 already sampled RGB frames.
It preserves order and uses the dynamic 70 to 140 token canvas for every frame.
A canvas whose alignment exceeds the budget is refused, not truncated. The image
fixture `strip-1000x1` exercises that refusal in the pure test and remains in the
image gate at its larger image budget.
The encoder gate feeds those frames through one reused tower and one
`Encoder::pass_media` call, checking identities, positions and both vectors.
The existing `embedding model` named reserve remains
`bloomery_gpu_gemma_embedding2::card_bytes(text, ctx) + projector.card.total()` bytes.
The projector term counts one tower's weights and activation buffers at the
configured image limit, not one tower per frame. The encoder term counts the
combined sequence at `ctx`, including the input rows and pass buffers. Video
adds 0 device bytes to this reserve at the same limits: each frame reuses the
tower, and its projected BF16 rows stay on the host until `pass_media` combines
them. The host rows take `2 * text_width * sum(frame_tokens)` bytes,
at most `8,960 * text_width` bytes for 32 frames of 140 tokens [derived]. The assembled
text, delimiters and frame positions must still fit `ctx`; no second reserve
or expert budget is introduced. HTTP video inputs and frame sampling are not
part of this interface.
