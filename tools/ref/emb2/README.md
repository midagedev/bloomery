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
or expert budget is introduced.

### Served sampled frames

Embedding clients can send already sampled one-FPS frames to `/v1/embeddings`,
`/embeddings` or `/embedding`. Start the server with `--embd-model` and
`--embd-mmproj`. The request uses the existing image embedding content array:
a first text part whose entire text is `Video:`, followed by 1 to 32
`image_url` parts. Each URL holds a base64 PNG or JPEG. No URLs are fetched.

```json
{"input":{"content":[
  {"type":"text","text":"Video:"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
]}}
```

The prefix selects the projector's video preparation rule. Each supplied frame
represents one second; the server does not resample the original clip. Prepare
its samples with `gemma4v::prepare_clip` before sending them. The server prepares
the sampled frames with `gemma4v::prepare_video`, preserves frame order and image
keys, and assembles one BOS/EOS pair around the complete sequence. Text after the prefix,
empty video sequences, more than 32 frames and inputs past the embedding
context are refused. A frame whose video canvas exceeds the loaded projector's
`--embd-image-max-tokens` capacity returns 400 before GPU dispatch. The video
rule does not enlarge the tower or its named reserve. Image inputs without this
prefix keep their image budget.

This is the same request that `tools/ref/emb2/server_vectors.py:286-295` sends
to the pinned llama-server for `--frames`. The image shape is parsed by
`serve::media::parse_part` and assembled by `serve::embeddings::input`.
The pinned llama.cpp
[`server-context.cpp:6059-6065`](https://github.com/ggml-org/llama.cpp/blob/6868a52cb337450658006ad2f9281c2eb63f1666/tools/server/server-context.cpp#L6059-L6065)
routes such content arrays through `tokenize_oai_content_array`.

The serve gate's `--video-clips` mode reads `video_oracle.py`'s `clips.json` and
reference pairs. Before starting the server, it checks the loaded model and
projector files against their pinned SHA256 values. Before sending each content
array, it checks its frame files against both reference sets. It compares prompt
positions, raw vectors and normalized vectors with the unchanged CPU-twin-derived
bands:

```sh
BLOOMERY_BOX_LOCAL=1 just gate-gpu-gemma-embedding2-serve \
  --video-clips target/video-clips/clips.json
```

The recipe builds both binaries. The gate launches and stops its own server
and uses the existing named reserve, not a second video seat.

### Out of scope

Encoded `input_video` and `video_url` parts are refused with a named 400 error.
The pinned llama.cpp
[`server-common.cpp:1228-1242`](https://github.com/ggml-org/llama.cpp/blob/6868a52cb337450658006ad2f9281c2eb63f1666/tools/server/server-common.cpp#L1228-L1242)
accepts those parts with a `data` or `url` field and decodes them through an
FFmpeg subprocess.
Bloomery does not install or invoke a decoder. Supporting those parts needs a
separate decision about the subprocess, its resource bounds and its runtime
dependency. The sampled-frame request above works on both servers without it.

### Constant-rate decoded clips

`vision::arch::gemma4v::prepare_clip` accepts RGB frames in presentation order,
source FPS as a positive numerator and denominator, and the tower's patch size.
The first source timestamp is zero. The clip ends one source interval after the
last frame. The result contains the selected source indices and their prepared
frames; each prepared frame carries its image span length.

At the one-FPS checkpoint setting, source timestamps round to the nearest
integer second, with half seconds rounded up. Each output tick uses the last
source frame whose rounded timestamp is at most that tick. The EOF timestamp
rounds the same way and is excluded. Low-FPS input can repeat a source frame.
For example, a one-second 24-FPS clip selects index 11, and a seven-second
30-FPS clip selects indices 14, 44, 74, 104, 134, 164 and 194. Sampling every
24th or 30th source frame starting at zero gives different pixels.

The 32-frame limit belongs to bloomery's decoded-frame interface. It is not a
maximum in the pinned llama.cpp video helper. A 40-second clip at 30 FPS yields
40 sampled frames and is refused here. A clip shorter than half a second can
yield no output and is also refused. A 32-second clip is accepted. Sampling
does not change the existing per-frame resize rule or the assembled-context
limit.

The pinned helper adds `Video:` once before the first frame. With timestamp
interval zero it adds no timestamp text or inter-frame separator. Gemma4's
`<|image>` and `<image|>` still surround each frame's projected rows. Each frame
uses the existing aspect-preserving dynamic 70 to 140-token canvas.

Source contract: llama.cpp
[`mtmd-helper.cpp`](https://github.com/ggml-org/llama.cpp/blob/6868a52cb337450658006ad2f9281c2eb63f1666/tools/mtmd/mtmd-helper.cpp),
`mtmd_helper_video::start_ffmpeg`, `read_next_frame` and `read_next`.
The helper delegates sampling to an external `fps=1.000000` filter and does
not pin that executable's version. The constant-rate rule above follows
[`vf_fps.c` in FFmpeg n7.1](https://github.com/FFmpeg/FFmpeg/blob/n7.1/libavfilter/vf_fps.c),
`read_frame`, `write_frame` and `update_eof_pts`, using its default nearest
rounding and EOF behavior. No FFmpeg dependency or decoding subprocess is
added. Variable-rate input, nonzero initial timestamps and decoder-specific
EOF timestamps need a separate timestamp-bearing contract.

`video_clips.py OUT` prepares decoded RGB streams and selected PNG frames for
three synthetic constant-rate clips (1 second at 24 FPS, 7 seconds at 30 FPS,
and 16 seconds at 24 FPS). Its JSON and TSV manifests record source FPS,
geometry, selected indices and raw-stream digests.

On a Linux CUDA host with the model profile's files and pinned llama-server:

```sh
python3 tools/ref/emb2/video_oracle.py target/video-clips --oracle "$LCPP"
cargo oxide build --arch sm_86 -- --release -p bloomery-gpu-gates \
  --features gemma-embedding2,vision,deepseek41 --bin gate_gemma_embedding2_e2e
BLOOMERY_MODEL=gemma-embedding2 BLOOMERY_BOX_LOCAL=1 bash tools/box.sh \
  'bash tools/gpu-gate.sh gate_gemma_embedding2_e2e --video-clips target/video-clips/clips.tsv'
```

Choose the build architecture for the card. The collector refuses an existing
output directory, checks the model digests and oracle revision, and runs the
CUDA and CPU servers sequentially. The collector uses the CUDA driver's visible
device list for the CUDA set's card header, including ordinal, UUID and MIG
selections in `CUDA_VISIBLE_DEVICES`. A missing driver, a CUDA error or an empty
visible list stops collection. Each clip gets its own video reference pair.
The gate requires three uniquely named fixture rows and exactly one matching
video record in each reference set, including the CPU twin. The gate reads the
complete decoded RGB stream, calls `prepare_clip`, checks source indices and selected pixels against those sent to llama-server, and
compares position counts and pooled vectors with the existing CPU-twin bands.
It prints each ordered span. Every frame within one fixture has the same
geometry, so the server's total position count also checks its uniform span
length after accounting for the prefix and delimiters.

These commands exercise decoded-frame assembly, not HTTP video decoding or an
external FPS filter. The expected source indices follow the source contract
above; the oracle receives the corresponding PNGs without a video decoder.
