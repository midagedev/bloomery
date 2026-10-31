# Serving

How `bloomery-serve` behaves: the llama-server API and flags it shares, the seats' own behaviour, and where it ends
today. Install and the one-command start are in the [README](../README.md).

## llama-server compatibility

Your OpenAI client works unchanged. bloomery speaks llama-server's HTTP API and reads the same files:

- **The API**: `/v1/chat/completions`, `/completion`, streaming, tool calls, `reasoning_content`,
  `cache_prompt`, `/apply-template`, `/tokenize`, `/detokenize`, `/slots`, `/metrics`, `/v1/models`, `/props`. OpenAI SDKs, curl,
  and llama-server tutorials apply as they are.
- **Compressed request bodies**: a body sent with `Content-Encoding: gzip` (or `x-gzip`) or `deflate` (a zlib
  stream) is read as the bytes it decodes to. The 64 MiB body limit holds for the bytes sent and again for the decoded
  body. A body that does not decode, or carries bytes past its stream, is a 400 `invalid_request_error`. Any other
  coding (`br`, `zstd`), or a list of two or more, is a 415 whose `Accept-Encoding` header names `gzip, deflate`, sent
  before the body when the client waits for `100 Continue`. A request with no body is read as sent.
- **Model lookup**: `GET /v1/models/{model}` (and `/models/{model}`) answers the one `/v1/models` entry of that name.
  URL names are percent-decoded once with literal `+` preserved; invalid encoding is a 400 `invalid_request_error`.
  JSON names in `/api/show` are matched literally.
  A name the server does not serve is a 404 `not_found_error` naming the models it does, and a route it does not have
  a 404 that starts with the method and path (`File Not Found: GET /path`). For Ollama clients, `GET /api/tags`
  answers the `/v1/models` list (its `models` array is Ollama's shape) and `POST /api/show` with `{"model": NAME}`
  answers the chat model's template, its file's format, architecture and quant (`details`), the served context as
  `<arch>.context_length` (`model_info`) and its `capabilities`: `completion`, `tools` when the template's tool-call
  markup is one the server parses, `vision` with an image encoder; the embedding model answers `embedding`. A decision
  model's server answers `/api/tags` and the lookup but not `/api/show`. Ollama's other routes are not served.
- **One stderr line a request.** When a generation request ends — on `/completion`, `/v1/completions`,
  `/v1/chat/completions`, `/v1/messages` or `/v1/responses` — the server prints one line on stderr:
  `bloomery-serve: request <task id> <endpoint path>: prompt <N> ids (<C> reused) in <S> s, <R> ids/s; generated <G> ids in <S> s, <R> ids/s; stop <reason>; ctx <n_past>/<n_ctx>`.
  The seconds carry two decimals, the rates one, and a rate over no time reads `-`. `<reason>` is the generation's own
  cause: `eos`, `stop` (a stop string), `length`, `cancel` (the client went away) or `error`. While one request
  generates past every 30 s, a shorter line prints on the same clock — `request <task id> <endpoint path>:
  generating, <G> ids in <S> s, <R> ids/s; ctx <n_past>/<n_ctx>` — so a long generation is never silent. A request
  of several choices (`n` above 1) prints one line a choice, its index after the task id. The counts are the numbers
  `/metrics` books for the request; between the lines, a running request shows in `/slots` (`n_past`, `n_decoded`)
  and `/metrics` (`predicted_tokens_seconds`).
- **Where a round's time goes.** `/metrics` splits the decode rounds by `width`, the slots one round stepped or
  passed: `rounds_total`, `round_seconds_total` (the round's wall time on the engine thread, the engine's calls and
  the server's work between them), `round_card_wait_seconds_total` (the part the engine waited for the card's work)
  and `round_ple_wait_seconds_total` (the part a Qwen3.8 call waited for its PLE table rows read from NVMe); the rest
  is host time. `card_wait_seconds_total` and `ple_wait_seconds_total` count the same waits over every call, prompts
  and the prompt cache's work included. An engine that counts no waits reports 0. `spec_decode_num_draft_tokens_by_depth_total` and `spec_decode_num_accepted_tokens_by_depth_total`
  count a drafted pass's ids by `depth` (the id's place in its draft) under `source`, the drafter's kind (`/props`'
  `engine.draft.kind`). They sum to `spec_decode_num_draft_tokens_total` and
  `spec_decode_num_accepted_tokens_total`, and like them are booked when a request ends.
- **The Anthropic Messages API**: `POST /v1/messages` and `/v1/messages/count_tokens`, as llama-server serves
  them. Point Claude Code or the Anthropic SDK at `ANTHROPIC_BASE_URL=http://localhost:8080`.
- **The OpenAI text completion API**: `POST /v1/completions` runs the `/completion` path under OpenAI's shape —
  `object: "text_completion"`, `choices` with `finish_reason` `stop`/`length` (or `reasoning_loop`, below), `usage`,
  and a stream of chunks ending in `data: [DONE]`. One prompt a request: a `prompt` array of several prompts, `echo`,
  a non-empty `suffix` and `best_of > 1` are refused by name, as is `logprobs`.
- **Several choices**: `n` on `/v1/chat/completions` and `/v1/completions` asks for that many choices of the one
  prompt, each answered with its `index`, in a stream too. Choice `i` samples with the request's `seed` plus `i`, so
  choice 0 is the answer `n` 1 gives and a seeded request repeats; a greedy request answers the same text `n` times.
  The choices queue together, so `n` is at most the slots plus the queue depth (64 with the defaults): under
  `--parallel N` up to N run at once, and on one slot they run one after another. `usage` counts the prompt once and
  every choice's completion tokens; `timings` are the first choice's. Other routes refuse `n` above 1.
- **The OpenAI Responses API**: `POST /v1/responses` and `/v1/responses/input_tokens` on the chat path. A response
  is kept unless the request sends `store: false`: `GET /v1/responses/{id}` reads it, `DELETE` drops it, and
  `previous_response_id` continues its conversation (its input and output, not its `instructions`). `background: true`
  answers `queued` at once; poll `GET` until the response finishes, or end it with `POST /v1/responses/{id}/cancel`.
  The server keeps at most 1,024 responses and 256 MiB of JSON text, including conversations and running workers'
  input items. This is a serialized size bound; JSON value trees allocate additional memory. The oldest finished
  response goes first. A background request gets 503 with `Retry-After` while the running ones fill either bound;
  an oversized background result stays pollable as `failed` with a named error. All responses are lost when the
  server stops. A `namespace` tool (Codex's `multi_agent_v1`) flattens: each function inside it serves as a function
  tool under its bare name, and its calls answer with the namespace as the `function_call` item's `namespace` field.
- **Chat token counts**: `POST /chat/completions/input_tokens` (and its `/v1/` spelling) answers
  `{"input_tokens": N}` with the same chat request's prompt ids — the chat template's render, an image as its span;
  `max_tokens` is not needed.
- **Special-token spellings in client text are text.** llama-server tokenizes the chat template's render with special
  tokens parsed (`tools/server/server-context.cpp`, `tokenize_input_prompts(…, true, true, …)`), so a message that
  spells `<|im_start|>` or `<｜User｜>` writes a turn marker there. This server keeps what a client sends as text: a
  special-token spelling in a user, system or tool message, a message's role or name, a tool call's name or arguments,
  a tool's description or schema, or a `chat_template_kwargs` value tokenizes as its characters, and the tokens the
  template writes stay tokens. An assistant message's content and reasoning are the model's own past output and still
  carry the special ids it printed. The spellings (and the beginning or end of one a client's text stops at, which
  the template's next characters could finish) are marked before the render and the marks cut out after it
  (`crates/serve/src/guard.rs`), so a template sees the text it was sent; a template that slices a marked text through
  its mark is a 400 naming it, and an image or video placeholder typed into a message keeps its own refusal. Not
  covered: a template that builds a spelling from its own pieces around a client string (`'<' ~ key ~ '>'`, as
  Nemotron's does for the extra keys of a tool parameter's schema). A `/completion` prompt is the client's own
  tokenization and still parses special tokens.
- **Inspecting chat prompt ids**: `POST /apply-template` accepts the chat request's `messages`, tools and template
  options and normally answers `{"prompt": "..."}`. Set `"return_prompt_ids": true` to add `"prompt_ids": [...]`,
  using the same guarded tokenization as chat: a user's `<think>` is ordinary text, while template-written turn
  markers stay special tokens. Tokenizing the returned `prompt` with `/tokenize` is not equivalent, because that
  route parses special-token spellings in the text. An absent, false or null flag keeps the text-only response;
  another type is a named 400. Media placeholders are unexpanded in both fields: these ids precede image/video span
  expansion, not the full multimodal engine prompt.
- **Embeddings**: `POST /v1/embeddings` (OpenAI's shape: `encoding_format` `float` or `base64`, `dimensions`
  for a width the model cuts its vectors to) and `/embedding` or `/embeddings` (llama-server's own array) read
  `input`, or `content`, as llama-server does: a string, a list of ids, a list of strings and ids, or a list of
  these, each one input; `embd_normalize` is `-1` (none), `0` (onto the int16 range), `1` (taxicab), `2`
  (euclidean, the default) or a higher p-norm. They answer from the embedding model the server holds beside its
  chat model (`--embd-model FILE` or `--embd-hf <repo>[:<quant>]`), and with none a 501
  `not_supported_error`. The `gemma-embedding2` feature serves both EmbeddingGemma versions (Q8_0)
  on a separate thread and card stream. EmbeddingGemma v1 (`gemma-embedding`, 300M) takes up to
  2048 tokens, uses SentencePiece, and applies its two dense projections after mean pooling.
  EmbeddingGemma 2 takes up to 8192 tokens and projects each token before mean pooling.
  Both serve 128, 256 and 512 MRL cuts and the full output width, cut before normalization.
  A file whose output width is below 128 is refused at open. Its seat is the smallest card outside the chat plan, else a
  named `embedding model` reserve on the stage card, subtracted from the chat expert budget.
  `--mmproj` and `--embd-model` on the same off-plan card do not count each other's bytes; the second open
  fails by name if the card is short.
  The response `model` is the embedding file name, also listed separately in `/v1/models` with
  `meta.n_embd`, `meta.n_ctx` and `meta.prompt_names`. `/props` adds `embedding_model` with the side model's `alias` and
  `path` when one is loaded; `model_alias` and `model_path` still name the chat model.
  On `/v1/embeddings`, a `model` of another type than a string or a name
  other than the embedding model's is a 400 `invalid_request_error` and runs no pass; `null` is unset.
  `/embedding` and `/embeddings` ignore `model`.
  `prompt_name` selects an exact, case-sensitive name from the embedding model's
  `config_sentence_transformers.json` `prompts` map, such as `query` or `document`.
  The server reads this file at startup beside `--embd-model`, or from the `--embd-hf` repository
  (falling back to its model card's single `base_model`). It prepends the configured bytes to each
  text input before tokenization. No space or separator is added. Without `prompt_name`, input is
  unchanged, even if the config sets `default_prompt_name`. A missing config provides no names;
  a malformed config refuses startup. Unknown names, non-string names (including `null`) and
  token-ID or media inputs with `prompt_name` return a 400 `invalid_request_error`.
  For example, `{"input":"Where is Paris?","prompt_name":"query"}` uses the model's query prefix.
  An input past the model's context is a 400
  `exceed_context_size_error` naming both lengths. llama-server checks its physical batch first and answers an input
  past it with a 500; this server has no batch limit there. An object with a `content` array takes text and image
  parts as described below. An object with `prompt_string` is a 400.
- **Tool calls on every generative model**: DeepSeek-V4.1-Flash (DSML), GLM-5.3-Flash (its XML), Qwen3-30B
  (Hermes JSON), Qwen3.6-35B and Qwen3.8-Flash-Next (the `<function=…><parameter=…>` markup, a value on lines of
  its own) and MiMo-V2.6 (the same tags, a value bare between them: `<parameter=KEY>VALUE</parameter>`) parse into
  `tool_calls` and `tool_use` blocks, so agent clients work against each of them. Which of the two framings a
  template uses is read from the call it renders, not from its name, and a `tool_choice` that forces a call writes
  it in that framing. A generation the token limit cuts inside a call is an answer, as llama-server answers it:
  200 with `finish_reason` `length` (`stop_reason` `max_tokens`), the call's text as `content` while its name is not
  whole and, on the chat route, the call with the arguments read so far once it is; a generation that ends on its own
  inside a call is a 500 naming the call.
  DeepSeek-V4.1 writes its DSML tags with a space after the token (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`,
  `<｜DSML｜ parameter>`), V4 without (`<｜DSML｜tool_calls>`). The V4.1 GGUFs carry V4's template, so a
  `general.architecture` `deepseek41` file is served with that template respelled as V4.1's (the template llama.cpp
  ships for it), and one line at start says so: the tools header teaches the tags the model writes, the parser
  reads them, and an earlier call re-renders in the history as the model wrote it, so the slot's cached prefix
  survives a tool turn. A `deepseek4` file keeps V4's tags; a parser reads the one spelling its template spells.
  A template given by `--chat-template-file` (the V4.1, Qwen3, Qwen3.8, GLM, MiMo and Nemotron seats take it) is read the same
  way, so a V4-spelled file on the V4.1 seat is respelled too.
  The template file is read once at startup. Changes after that read take effect on the next start.
  A streamed `/v1/chat/completions` sends a call as the model writes it, in OpenAI's shape: the call's first delta carries
  `index`, `id`, `type` and `function.name`, and `function.arguments` fragments follow that concatenate to the
  `arguments` the non-streamed answer carries. A text parameter's value is released as it arrives, JSON-escaped, up to
  a tail that could still become a closing tag; a number or JSON value is released when it is complete. Markup that
  stops being the call after its name went out (a DSML parameter given twice, text that is not a parameter, a
  generation that ends inside the call) ends the stream with an error event naming the call: the non-streamed answer
  keeps DSML's raw text as content, and a stream cannot take its fragments back. `/v1/messages` and `/v1/responses`
  streams send a call when it closes.
  Hermes treats a complete markup `<tool_call>` opener at EOF as an unterminated call, including immediately after a
  previous call closed. Only a proper prefix of that opener remains text at EOF; neither decision depends on
  stream chunk boundaries.
  A think tag or a Qwen3-Coder, Hermes, GLM or Solar call tag held as one special token is markup
  only where the model generated that token. The same characters written in ordinary tokens (a quoted
  `</think>`, a copied `<tool_call>` example) stay reasoning or content, streamed or not, and the
  `reasoning_budget` and `reasoning_loop` span below ends at the same place.
  A call opened by a special token needs a special closing token too; an ordinary-token closing tag leaves
  the call open and the request ends with a named markup error if generation ends there.
  `bash tools/ref/serve-control-markers.sh URL OUTPUT_DIR` saves the copied-call and quoted-think reproductions
  and a real tool-call request, whole and streamed, against a running model-backed server. The whole replies
  include token logprobs so a diagnosis can distinguish a special id from ordinary tokens spelling its text.
- **Refused by name** (a 400 naming the field): `samplers` in an order other than llama-server's default list for the
  stages that are on, or naming a sampler this server lacks; typical_p, mirostat, `dry_*`, `xtc_*`, `top_n_sigma` and
  `dynatemp_*` when a request turns them on; `n_predict` or `max_tokens` outside -1..=i32::MAX (-2 does not mean "no
  limit"). `/tokenize` refuses a missing or non-string `content` by name.
- **`timings`** carry llama-server's fields and one of ours, `cache_ms`: the prompt cache's work before the
  prompt (the slot's state saved, a cached state put back, the cut), which `prompt_ms` does not count.
- **`reasoning_budget`**, per request on `/v1/chat/completions` and `/completion`: llama-server's
  `--reasoning-budget` — the think span's budget in generated ids taken while the span is open, `0` closing it at
  once, `-1` or absent unrestricted, and silently ignored when the template already closed the span.
  A Qwen3-Coder `<tool_call>` ends the span too, so the budget never forces a close inside its arguments; a
  finite budget beside a forced `tool_choice` is a 400 (structured output, below).
- **`reasoning_loop`**, per request on every generating route (`/completion`, `/v1/completions`,
  `/v1/chat/completions`, `/v1/responses`, `/v1/messages`): `true` ends a generation whose open think span
  repeats itself. The span's last ids repeat a cycle of up to 1,024 ids when each of them equals the id one cycle
  before it, over at least 256 ids and one whole cycle (so two copies of the cycle stand). The generation then ends
  with the reasoning so far and no answer: `finish_reason` `reasoning_loop` (`stop_reason` on `/v1/messages`,
  `stop_type` on `/completion`), and on `/v1/responses` `status` `incomplete` with `incomplete_details.reason`
  `reasoning_loop`. Absent, `null` or `false` turns it off; any other type is a 400 naming the field. Text after
  the span closes is never read, and like `reasoning_budget` the field is ignored when the template already
  closed the span. On `/completion`, read `stop_type` for the cause: a reasoning loop leaves
  `stopped_eos`, `stopped_word` and `stopped_limit` all `false`. Structured output may enable the guard too:
  a JSON document's program closes an open think span before it starts, and a forced call's reasoning is
  guarded until the model closes the span. One id costs one compare per cycle length, at most 1,024.
- **`developer` messages** reach the chat template as `system`, the mapping llama-server applies before it — every
  template but GPT-OSS's, whose `<|channel|>` marker in the template source keeps the role. A template with no
  branch for the role would drop the message, and OpenAI clients send one every turn.
- **`reasoning_effort: "none"`** turns thinking off before the template, as llama-server does: the template reads
  `enable_thinking: false` (`chat_template_kwargs` sets it too — a bool, a quoted string is a 400 naming it, and
  `"none"` wins over a kwargs `true`), no `reasoning_effort` reaches the template, and the whole output lands in
  `content`. A template with no thinking switch but an effort header (GLM-5.3's: its generation prompt always opens
  `<think>`) has nothing to turn off, so there `"none"` is sent as `reasoning_effort: "low"`, the template's lowest
  effort, replacing a kwargs `reasoning_effort`; the server prints one line at start saying so. `/props` names which
  case the template is in: `thinking_off` is `"switch"`, `"low_effort"` or `"neither"` (neither knob changes the
  prompt, so `"none"` is `enable_thinking: false` as above).
- **`reasoning_effort`** takes OpenAI's words (its SDK's `ReasoningEffort`: `none`, `minimal`, `low`, `medium`,
  `high`, `xhigh`) and `max` (V4.1's maximum-effort header), on `/v1/chat/completions`, `/apply-template` and the
  Responses API's `reasoning.effort`; any other word, the empty string included, is a 400 naming the field, the value and
  the accepted words. Every word but `none` asks for reasoning and reaches the template as it came. A template whose thinking is
  off until `enable_thinking` is set (V4.1's; `/props` `thinking_off` `"switch"`, found by comparing its renders at
  start) gets `enable_thinking: true` from it, so `reasoning_content` is split out; `max` also adds that template's
  maximum-effort header, and `low` to `high` are the template's default effort. A `chat_template_kwargs`
  `enable_thinking` in the same request wins either way. A template that thinks by default is asked for nothing it
  was not already doing. Unlike llama-server, which turns thinking on for any template that has the switch,
  this server keeps the template's own default when a request names neither field (BUILD.md: thinking is off for
  V4.1).
- **The files**: the GGUF uploads as downloaded, the same quantizations, the chat template read from the file,
  and a tokenizer bit-identical to `llama-tokenize`. Qwen3.8 also serves NVIDIA's NVFP4 safetensors checkpoint as
  downloaded (The seats, below).
- **The flags**: `-m`/`--model-file`, `--hf <repo>[@<rev>][:<quant>]` (download, a large file over up to 16 ranges at
  once, each resumed where it stopped and asked again after a cut or a busy hub's `Retry-After`; the sha256 check
  read behind the download; never fetched twice; the repo's `main` branch, or with `@<rev>` the branch, tag or commit `rev`, whose listing, files and cache
  directory `<owner>/<name>@<rev>` are that revision's, offline too; it also fetches the model's MTP draft when the repo publishes it, as unsloth's Qwen3.8-Flash-Next GGUFs
  do under `MTP/`, landing it beside the set, where the run picks it up), `--parallel/-np`, `--queue-depth`,
  `--cache-ram/-cram` (MiB; `0` turns the prompt cache off; llama-server's `-1`, no limit, is refused by name: the
  cache is host memory beside the load), `--ctx/--ctx-size` (unset, every generative seat and the decide seat take one slot at the model's limit and, where
  that plan is short of the card — or of the host, where the body keeps no tier past it, or of a unified pool no share
  of which between the card and the host fits, or that cannot also hold a checkpoint folder's load reading its parts
  one at a time — the largest context under it whose plan holds, a multiple of 1,024 down to 4,096;
  one `ctx` line names the rule, `target` or `fitted`. On Qwen3.8, where the adaptive residency runs with experts on the card
  (`BLOOMERY_RESIDENCY` unset on plan (a) or (b′): two routed experts a layer; a set `mid-p<P>-s<S>` word: P + S + 1), the
  plan holds only if it also leaves every layer of the card that many, so the context never takes all the card's room; where not even 4,096 positions leave them the fit is the plain one, the
  residency resolves `off`, and a `ctx experts-floor=unmet` line says why. A set value is the total the slots split, and
  one that does not fit is refused, never cut. PIN(2026-10-14): `generate::choose_ctx` over `ctx_ask::choose`),
  `--cache-type-k f16|q8_0` (the qwen3
  and qwen38 seats; llama-server's spelling, also the `BLOOMERY_QWEN3_KV` lever; q8_0 nearly doubles the auto
  context on the same card for the qwen3 seat. On Qwen3.8 FP4-Local at 16,896 positions, the attention store is
  773,279,360 B against f16's 967,921,280 B; the index keys stay f16, and the default context and slot count did
  not change there. Unset is f16, except a Qwen3.8 checkpoint folder,
  whose seat runs q8_0 (FP4-Local's Spark default; `--cache-type-k f16` keeps f16); `--cache-type-v` does not
  exist, both planes quantize together),
  `-ub/--ubatch-size N` (the qwen3 seat and the decide seat, on a file with no routed expert, whose only load is the
  whole one: that many tokens a prompt ubatch, or a refusal by name with the bytes it needs and a smaller size as
  the remedy; unset, the load takes the largest of 4096, 2048, 1024 and 512 the card holds at the asked context and
  slots, and one `ubatch` line names the sizes it tried; a routed file takes `BLOOMERY_QWEN3_UBATCH` and refuses
  the flag. The qwen38 seat takes it too: the most positions one prompt ubatch runs, 1 to 4096, over
  `BLOOMERY_QWEN3_UBATCH`, else 4096, the same size in the plan and the load, and the `load` line's `ubatch=`),
  `--host/--port`, `--alias`, `--mmproj <file>` and `--no-mmproj` (image input, below). The flags are checked
  before `--hf` downloads anything and before the driver is asked: with `--model` or `--head`, the selected seat's
  `--help` and refusal of a flag, value or lever come first. A generative start with llama-server placement flags
  resolves their overrides and checks its levers, including `--levers`, after device discovery. Without `--model`
  or `--head`, the model file picks the seat. What every seat refuses ends the start there. What only some seats
  refuse (`--prefill steps`, which only the glm and mimo2 seats take, or `--parallel 2` on Nemotron) makes the start
  read the `--hf` repo's file list and its model's architecture (the first GGUF shard's header, or a checkpoint's
  `config.json`, and no weights) and end with the refusal of the seat that serves that model, still before the
  download. Under `HF_HUB_OFFLINE`, or with no network, that read is skipped. The picked seat uses the same checks
  after the download.
- **The model's name** (`/v1/models` `id`, the `model` of every answer, `/props` `model_alias`) is `--alias`, else the
  GGUF's `general.name`, else the seat's own default. llama-server's default is the model file's name; this server
  names the model as its file does, which stays the same when a file is renamed or split into shards. A file whose
  `general.name` is a placeholder (a requantization script's) is named with `--alias`.
  The qwen3 and decide seats refuse `--alias`.
- **No placement flag needed**: leave `--place` out on every model and the load picks for you — the largest
  visible card, a split onto the CPU when the file does not fit it, and on GLM-5.3 the second card as an expert
  tier when its plan puts enough experts there to pay for it (the GLM row of the
  [A6000 + 3090 table](../README.md#numbers)). The plan line names what was picked.
- **llama-server's placement flags**: `-dev`, `-sm`, `-mg`, `-ts`, `-ngl`, `--cpu-moe`, `-fit`, `-devd` and
  `--list-devices` place the model as they do there where bloomery has the thing the flag names, and are refused by
  name where it has not. [Placement](#placement) lists each flag and what the load plans for it.
- **Clef-Flash's `/v1/systemone`** follows llama.cpp's decision server wire (`model` optional, `/v1/models`,
  501 for video, and for an image when the seat has no encoder, upstream's `confidence` formula).
- **`/v1/systemone` on a generative seat**: the same route, request and body as the decision models', answered by
  the seated general model. The request is read by the decision models' own reader (the same refusals), asked as
  the schema text in the system message and the state alone in the user message, and the reply is held to a JSON
  object (one key a question; a choice a string of its options, a score an integer level, a noul a boolean) by the
  structured-output path of `/v1/chat/completions`, schema jump included. Each question's scores are the logits
  of the token its options start with, read where the model picks (the first token that tells the options apart),
  and the body is written by the formulas the decision models' own are: `choice` is the model's pick, and
  `probabilities`, `confidence` and a score's `score` come from the softmax of those logits over the options
  alone. `usage.output_tokens` counts the generated ids and `timings.cache_n` the prompt ids a cache kept. A
  request may add `cache_prompt` (a boolean, default `true`). A choice whose options one token does not tell
  apart (`ab`, `ac` and `d` start with the same token) is asked by label: the model is shown `A = ab`, `B = ac`,
  `C = d` in the option order and picks a letter, whose logits are the options' scores, and the body names the
  options, never a letter. A choice of more such options than there are single-token letters (26) is a 400 naming
  the question. `/props` lists the route in `routes`, and the 404 names it.
- **`--api-key`/`--api-key-file`**, llama-server's authentication: a comma-separated list (a quoted item may
  hold commas) and a file (one key per line, `#` lines comments; read once at start, so a pipe such as
  `/dev/stdin` serves, and a later edit takes a restart), both flags adding to one set, carried by
  `Authorization: Bearer <key>` or `X-Api-Key: <key>`. `/health`, `/v1/health` and `OPTIONS` stay public;
  every other route — generation, `/props`, `/slots`, `/shutdown` — answers llama-server's 401
  (`Invalid API Key`) without a valid one, and a flag that names no key refuses to start rather than serve
  unlocked. The server binds `127.0.0.1` by default; `--host 0.0.0.0` without a key serves anyone who can
  reach the port.
  The key check runs on the headers, before `100 Continue` or body ingestion.
  A denied request gets 401 and the connection closes without draining its body.
  A file key cannot contain whitespace; blank or whitespace-only lines are skipped.
  A key file line can narrow its key with fields after it, separated by whitespace
  (`sk-app scope=inference inflight=2 rate=30/m burst=5`); a key with none reaches every route unlimited:
  - `scope=` takes `inference` (generation, token counts, the tokenizer, embeddings, the stored responses,
    `/v1/systemone`), `read` (`GET /props`, `/slots`, `/metrics`), or `admin` (every route, the default),
    comma-separated; both of the first two reach the model list and lookups (`GET /v1/models`,
    `/v1/models/{model}`, `/api/tags`, `POST /api/show`). A route outside the scope answers 403
    (`permission_error`) and closes the connection like the 401. Slot actions, `/residency/reset` and
    `/shutdown` need `admin`.
  - `inflight=N` is the most requests the key may have answered at once. A request holds its place until its
    answer is written and the engine has left its generation, also after a cancel or a disconnect; a
    background response holds it until both its generation and its initial HTTP answer end. A call on a
    stored response (`GET`, `DELETE`, `/cancel`) takes none.
  - `rate=N/s` or `rate=N/m` is a bucket of `burst=B` requests (`N` unless set) that refills `N` a second
    or a minute; each request the key makes past `/health` takes one.
  - A request past either limit answers 429 (`rate_limit_error`) with `Retry-After` in whole seconds and
    starts nothing; the connection stays open. Each key counts apart, so one key never uses another's room.
  - A field the line does not take, or names twice, refuses the start, naming the file, the line and the
    field's place, never the key. A key given twice with different fields does too.
  Public health and `OPTIONS` requests are answered from their headers; a declared
  unread body closes that connection rather than becoming another request.
- **Stopping the server**: `SIGINT`/`SIGTERM` stop it as llama-server's do — no new request is admitted, the
  engine thread ends between its calls, and the process exits 0 after one stderr line naming the cause; a second
  signal terminates at once. `POST /shutdown` does the same over HTTP, accepted only from a loopback peer (a
  machine on the network cannot stop your server): a WSL2 user cannot safely signal a GPU process — the host has
  locked up when one was signalled — so the route stops the server without a signal.

Where it ends today: one model a server, sm_86+ GPUs, and the seats' own defaults where llama-server has none
(see [Limits](#limits)).

## Settings

A start needs a model and nothing else: every flag and lever below has a default, and the start lines (`ctx`,
`parallel`, `ubatch`, `plan`, `setup`) name what the start chose. This section lists what each seat takes, which
setting wins when two name the same thing, and what a request's fields do.

### Flags by seat

`bloomery-serve` reads its own flags first: `--model <seat>`, the model (`-m`/`--model-file` or `--hf`), `--head`
and `--head-config` (the decide seat), the embedding side model (`--embd-*`, the generative seats),
llama-server's placement flags ([Placement](#placement); not on the decide seat), `--config`
([The settings file](#the-settings-file)), `--levers`, `--list-devices` and `--version`. The seat parses every other
flag. Every generative seat takes this common set: `--host`, `--port`,
`--ctx` (`--ctx-size`, `-c`), `--parallel` (`-np`), `--slot-ctx`, `--place`, `--plan`, `--queue-depth`, `--chat-template-file`,
`--api-key`, `--api-key-file`, `--alias` and `--slot-save-path`, and the qwen3 seat refuses the last two. The table
gives what differs. A flag the seat named by `--model` does not take is refused by name with the seat's usage, before `--hf`
downloads anything; `bloomery-serve --model <seat> --help` prints that usage.

| Seat | `--parallel` | `--cache-ram` | `-ub`, `--ubatch-size` | `--mmproj` ¹ | Its own flags |
|---|---|---|---|---|---|
| `ds41` | N | yes | no | yes | |
| `glm` | N | yes | no | yes | `--prefill batch\|steps` |
| `qwen38` | N or `auto` | yes | 1 to 4096 | yes ² | `--cache-type-k f16\|q8_0` |
| `qwen3` | N | no | a file with no routed expert ³ | yes | `--cache-type-k f16\|q8_0` |
| `mimo2` | N | yes | no | no | `--prefill batch\|steps` |
| `minimaxm3` | N | yes | no | no | |
| `flint` | N | no | no | no | |
| `musespark` | 1 | yes | no | no | |
| `solar2`, `nemotron`, `raon1` | 1 | no | no | no | |
| `decide` | no (one request at a time) | no | yes | Clef only | `--head`, `--head-config`; it takes only `--host`, `--port`, `--ctx`, `-ub`, the key flags and the image flags |

`N` is any count, 1 by default; `1` is the seat's one slot, and `--parallel` above 1 is refused by name. `auto`
counts the slots from memory ("Concurrent requests" in [The seats](#the-seats)); every other seat refuses `--parallel auto`
by name. `--mmproj`
stands for the image flags together (`--no-mmproj`, `--image-min-tokens`, `--image-max-tokens`; Image input, below).
`--park-ram` is refused by name on every seat that reads it (ds41, glm, qwen3, qwen38): their slots are resident
sequences. The short spellings `-c`, `-np`, `-ub`, `-a`, `-mm` and `-cram` are the long flags
([The environment](#the-environment)).

¹ A build with the `vision` feature; the release has it, and another build refuses the image flags by name.
² An image load drafts by the same rule as a text load (`BLOOMERY_DRAFT`), and the MTP draft reads a prompt's image positions too. It refuses `BLOOMERY_PREFILL=steps` by name.
³ A file with routed experts takes `BLOOMERY_QWEN3_UBATCH` and refuses the flag by name.

`-ub` on the qwen38 seat trades card memory for prompt speed. Each position of the size is about 0.81 MB of the
plan's card bytes on the FP4-Local folder on a GB10 [derived: the `need` record's card bytes at 4096 less those at 48,
3,289,846,032 B, over 4,048], so `-ub 48` frees about 3.3 GB beside the default 4096, and a smaller size runs a long prompt in more, shorter passes,
which is slower. Lower it only to fit a context or a slot count that the default does not fit; `--plan` prints the
card bytes of each size.

`--plan` prints the plan records and exits before any weight is read onto a card. It is not an offline check: it
needs the model file (`--hf` downloads it first), the driver, and the cards' free bytes, and the qwen38 seat opens a
CUDA context on its stage card first, as a load does. Run it on the machine that will serve.

### The settings file

`bloomery-serve --config FILE` reads a start's flags from one JSON object, so a long start line can live in a file
that people and services share. Each key is a flag's name without its dashes, and its value is the flag's:

```json
{
  "model-file": "/models/Qwen3.8-Flash-Next-FP4-Local",
  "ctx-size": 65536,
  "parallel": 2,
  "host": "0.0.0.0",
  "api-key-file": "/etc/bloomery/keys",
  "no-mmproj": true
}
```

`bloomery-serve --config FILE` with this file starts as `bloomery-serve -m /models/Qwen3.8-Flash-Next-FP4-Local
--ctx-size 65536 --parallel 2 --host 0.0.0.0 --api-key-file /etc/bloomery/keys --no-mmproj` does.

- **Values.** A string or a number is the flag's value. `true` gives a switch; a switch that is off is left out. A
  list gives the flag once for each value (`"api-key": ["k1", "k2"]`). A path is read as on the command line, from
  the directory the server starts in.
- **Keys.** Every flag that sets something has a key, its long name: `ctx-size` for `-c`, `device` for `-dev`, `model`
  for `--model`. A key the seat does not take is refused by the seat with the flag's name before anything is
  fetched, as the flag would be.
- **The command line replaces the file.** A flag on the command line, in any of its spellings, replaces the file's
  entry for it: `-c 4096` replaces `"ctx-size"`. So does a flag that sets the same thing another way: `-m` or `--hf`
  replaces `"model-file"`, `--embd-hf` replaces `"embd-model"`, `--no-mmproj` replaces `"mmproj"` and
  `--embd-no-mmproj` replaces `"embd-mmproj"` (and the reverse of each), `-ncmoe` or `-ot` replaces `"cpu-moe"`, and
  `--place` replaces `"device"`, `"split-mode"`, `"tensor-split"` and `"main-gpu"` while `-dev`, `-sm`, `-ts` or `-mg`
  replaces `"place"`. A catalog variant (`"model": "glm:Q8_0"`) is also a model source; a bare seat word
  (`"model": "glm"`) keeps its separate meaning.
- **One line on stderr for each entry** says what it was read as, or which flag of the command line replaced it. The
  line prints the value only for `ctx-size`, `ubatch-size`, `cache-ram`, `parallel`, `alias`, `host`, `port`,
  `api-key-file`, `slot-save-path`, `chat-template-file` and the image flags; for any other key, `api-key` and a
  misspelled `api_key` among them, it names the flag and leaves the value out.
- **The file is read before `--help`**, because it can name the seat whose usage `--help` prints: `--help` beside a
  file that cannot be read or is refused ends with that refusal.
- **Refused by name**, with the file's path, before the model is read: a file that is not one JSON object, a key
  named twice, a key written with its dashes, a key that asks the start to do something rather than set it (`plan`,
  `levers`, `help`, `version`, `list-devices`, `records-schema`, `recommend`, `dry-run`, `json`, `config`), `false`,
  `null`, an object or an empty list as a value, a value for a switch, `true` for a flag that takes a value, and
  `--config` given twice.

### Which setting wins

One value a setting, chosen in this order:

1. **The command line.** A flag replaces its `LLAMA_ARG_*` variable, and one line on stderr says so. A flag that
   stands for a `BLOOMERY_*` lever replaces the lever's value: `-ub` replaces `BLOOMERY_QWEN3_UBATCH`,
   `--cache-type-k` replaces `BLOOMERY_QWEN3_KV`, and the placement flags replace the levers they set
   ([Placement](#placement)).
2. **The settings file** (`--config`, [above](#the-settings-file)). A file's flag counts as the command line's against
   the variables and levers below, and the line that says so names it as the command line's.
3. **llama-server's variables.** A `LLAMA_ARG_*` variable is read as its flag ([The environment](#the-environment)),
   so it also replaces the lever: `LLAMA_ARG_UBATCH=48` beside `BLOOMERY_QWEN3_UBATCH=64` runs 48.
4. **The levers.** A `BLOOMERY_*` variable that no flag or `LLAMA_ARG_*` variable names.
5. **The seat's default.**

A command-line `--api-key` or `--api-key-file` replaces the settings file's entire list for that same flag.
Two cases are not ranked. The keys add: `LLAMA_API_KEY`, `--api-key` and `--api-key-file` together make one set. Two
sources of one thing are refused by name, never picked between: the model named twice (`-m` beside `--hf`,
`LLAMA_ARG_MODEL` beside `LLAMA_ARG_HF_REPO`, or `BLOOMERY_REF_MODEL` naming another file), `--place` beside a
placement flag, and the routed experts' place named twice. `<bin> --levers` prints each lever, whether the binary
(on `bloomery-serve`, the seat) applies it, and its value. The levers themselves are the rows of
[`crates/levers/src/registry.rs`](../crates/levers/src/registry.rs) ([`BUILD.md`](BUILD.md#levers)).

### Request fields

A request field is read, refused or ignored:

- **Read.** Validation is field-specific. Sampling fields such as `temperature` reject wrong types and non-finite
  values; `n_predict` and `repeat_last_n` enforce llama-server's count limits. Their `null` values are unset.
- **Refused.** A field the server cannot honor is a 400 naming it, never a 200 that drops it: `n` above 1 outside
  `/v1/chat/completions` and `/v1/completions` (and past the slots plus the queue depth on them), a non-empty
  `grammar`, probabilities on a route that renders none (`/completion` takes `n_probs`,
  `/v1/chat/completions` takes `logprobs` and `top_logprobs`), and the fields under "Refused by name" above.
  `/v1/responses` also refuses `conversation`, an `item_reference` item and `background: true` with `store: false`
  or `stream: true`.
- **`logit_bias`.** OpenAI's object from token ids to a number in -100 to 100, which is added to that id's logit
  before every pick, on `/completion`, `/v1/completions`, `/v1/chat/completions` and `/v1/responses` (also `/responses`):
  -100 keeps an id out in practice, +100 makes it the pick. It applies to greedy, sampled and structured-output requests,
  with or without the MTP draft (each pass draws from its verified rows with the bias added, as a step does, so the ids are
  the same). Under `ignore_eos` the stop ids stay at -inf whatever their bias. A structured-output request adds the
  bias before its schema's mask: the bias reorders the ids the schema allows and lets in no other. `logprobs` report
  the model's own row, before the bias. llama-server's other forms (an array of pairs, `false`, a token's text), a key
  that is not an id of the vocabulary, a value out of range and one id named twice are 400s naming the field;
  `generation_settings` reports the bias as llama-server does (`[{"bias": -100.0, "token": 15}]`).
- **Ignored.** A field the server does not know is ignored, as llama-server ignores it. So a misspelled field is no
  error: `"temprature": 0` answers 200 and samples at the default temperature.
- **Errors.** An error answers llama-server's object, `{"error": {"code": 400, "message": "...", "type":
  "invalid_request_error"}}`. Routes that answer this object add OpenAI's `param` when the error is about one request
  field: its top-level name, which a client can read instead of parsing the message. A fault inside a field names the
  field that holds it: `messages` for `messages[2].content`, `response_format` for a keyword its schema refuses.
  `/v1/responses` names its own fields (`input`, `text`, `reasoning`) where the chat request it runs as names others.
  An error about no single field has no `param`: a body that is not JSON, a prompt past the context, a full queue.
  The Anthropic Messages API keeps its own error object, which has no `param`.

Unset, a generation samples at llama-server's defaults: `temperature` 0.8, `top_k` 40, `top_p` 0.95, `min_p` 0.05,
`repeat_penalty` 1, `repeat_last_n` 64, no token limit but the context (`n_predict` -1) and `cache_prompt` true. A
structured-output request's `temperature` defaults to 0, so it decodes greedily unless it asks to sample (Limits,
below).

## Placement

`bloomery-serve` places a model with the flags llama-server places it with. Name a card and the plan is made on that
card. Name nothing and the load picks (see "No placement flag needed" above). The flags that name cards or set a lever print a line on stderr saying what they were read as, and a flag
bloomery cannot honor ends the start with a refusal that names it; none is ignored. llama-server's `LLAMA_ARG_*`
variables for these flags are read too ([The environment](#the-environment)), and so are the flags on the other
launchers of ours ([The other launchers](#the-other-launchers)).

`bloomery-serve --list-devices` prints the devices as llama-server does and exits, before any model is read:

    Available devices:
      CUDA0: NVIDIA RTX A6000 (… MiB, … MiB free)
      CUDA1: NVIDIA GeForce RTX 3090 (… MiB, … MiB free)

The first figure is the bytes a plan may use of the device, the second what it had free at the start. `CUDA<N>` is
the device's CUDA ordinal in this process, which `-dev`, `-mg`, `-ts` and `-devd` count by.

| Flag | What bloomery does |
|---|---|
| `-dev CUDA0,CUDA1` | The cards the plan may use; default every visible card. `none` is refused: no seat runs on the CPU alone. |
| `-sm layer` (default) | Every listed card takes part: the largest runs every layer and the head, the others hold experts under the host tier. bloomery does not split layers; one line says so. |
| `-sm none -mg N` | The one card at index N of the list (`-mg` default 0). |
| `-sm row`, `-sm tensor` | Refused: bloomery has no row split, and its tensor parallel is `--place tp` over `BLOOMERY_TP_NODES`, whose ranks are other nodes. |
| `-ts 3,1` | The cards that stay: a zero or missing share leaves the card out, as there. With two or more cards left the shares are also each card's share of the routed experts: `BLOOMERY_CARD_BUDGET` is set to shares by CUDA ordinal in whole millionths of the total (`CUDA0=750000x,CUDA1=250000x`), and the plan turns them into a budget per card from its own terms: each card plans with its floor (dense tensors, cache, context, scratch, reserves, margin) and its part of the routed experts' bytes, the parts in the proportion of the shares and as many in all as the card whose room is tightest for its share holds (so the ratio is kept, never bent), the host tier taking the rest. The plan line prints the resolved bytes. A share that rounds to no millionth is refused with its terms; so is, by the plan, a card whose floor passes its bytes, and shares on a seat whose plan puts no routed expert on a card. One card left, or `-sm none`, sets nothing. |
| `-mg N` under `-sm layer` | Refused unless N is 0: llama-server names no card by it there, and the stage here is the largest listed card. |
| `-ngl` | `all`, `auto`, a negative count, or a count of at least the file's layers + 1 (llama-server counts the output layer as one more): as the load is. A smaller count is refused: every layer runs on the stage card. |
| `--cpu-moe`, `-ncmoe N`, `-ot <experts>=CPU` | The qwen38 seat sets `BLOOMERY_QWEN38_EXPERTS=host`: every routed expert is planned on the host tier. `-ncmoe` counts only as every layer (N of at least the file's layers; 0 is nothing). `-ot` reads only the all-experts patterns (`exps`, `ffn_.*_exps`, `.ffn_.*_exps.`, `\.ffn_.*_exps\.`, `--cpu-moe`'s own regex) to `CPU`. Every other seat is refused: its plan puts on each card the experts the card's budget holds and leaves the rest to the host. |
| `-fit on` (default) | The plan always fits. `-fit off` is refused: there is no unfitted load. Size the load with `-dev` or `-sm none -mg N` (the card), `-fitt` (the margin kept free on it) and `--ctx` (the context). |
| `-fitc N` | N must be 4096, the least context an unset `--ctx` steps down to on every seat; pin a context with `--ctx`. |
| `-fitt MiB[,MiB…]` | The margin kept free on each card, as llama-server reads it: one value is every device's, a list one per listed device (`-dev`'s order, else CUDA order). Sets `BLOOMERY_CARD_MARGIN` (one value for every card, or `CUDA<N>=<bytes>` for the cards the flags name, or for every visible card when they name none): the card keeps that many bytes free past its context and scratch, which the expert rule never plans into. It is the plan's own margin (1024 MiB unless set, which is also llama-server's default, so `-fitt 1024` changes nothing) replaced, and no budget: the card's usable bytes stay whole, and `BLOOMERY_CARD_BUDGET` still sets a cap beside it. With no card named the load picks the cards as it does unset, each with its margin, and may take the second card as an expert tier where the model's rule does. One line says so. Refused by name: a list that is neither one value nor one per listed device, and a margin that leaves a card no usable byte. |
| `-devd CUDA1` | Sets `BLOOMERY_DSPARK_CARD=cuda1`, the card the V4.1 seat's DSpark draft (`BLOOMERY_DRAFT=dspark`) opens on. Unset, the draft sits on the tier card under `bp`, else on the smallest visible card (below). Another seat is refused. |

A lever a flag sets (`BLOOMERY_QWEN38_EXPERTS`, `BLOOMERY_DSPARK_CARD`, `BLOOMERY_CARD_BUDGET`, `BLOOMERY_CARD_MARGIN`) replaces the environment's value, as a
command-line flag replaces its `LLAMA_ARG_*` variable in llama-server. Giving a flag twice is refused. The cards
the flags name become the `--place` word the seat parses (`a` and `bp` where the cards are theirs, else `cuda1` or
`cuda1+cuda0`, the stage first), so a seat's own refusal of that word reaches you as a `--place` refusal after the
line that says what the flags were read as: a seat whose body serves no expert tier card, or fewer than the listed
cards, refuses the tier by name (the qwen38 seat serves one).

Six command lines (`model.gguf` stands for your file; the cards are the A6000 as CUDA0 and the 3090 as CUDA1):

    bloomery-serve -m model.gguf -sm none -mg 1

One card. The plan runs on CUDA1 alone (`--place cuda1`): that card runs every layer and the head and holds the
experts its budget allows, the host tier holds the rest, and CUDA0 is outside the plan. `-dev CUDA1` is the same.

    bloomery-serve -m model.gguf -sm none -mg 1 -fitt 4096

One card and its memory. The plan runs on CUDA1 with 4096 MiB kept free: the card's margin is that many bytes
(`BLOOMERY_CARD_MARGIN=4294967296`, and one line prints it), its usable bytes stay whole, and the host tier holds the
experts that no longer fit. `-dev CUDA0,CUDA1 -fitt 2048,4096` keeps 2048 MiB free on CUDA0 and 4096 MiB on CUDA1, and
`-fitt 4096` alone keeps 4096 MiB free on every visible card and leaves the choice of cards to the load.

The same margin without llama's flags: `BLOOMERY_CARD_MARGIN=4G` keeps 4 GiB free on every card, and
`BLOOMERY_CARD_MARGIN=CUDA0=2G,CUDA1=4G` gives each card named its own (a card it does not name keeps the plan's own).
A device the census does not list, a card named twice, shares, a list beside tier nodes (whose cards no ordinal of this
process names) and a margin that leaves a card no usable byte are refused by name.

A budget is a different thing, a cap on the bytes a card plans with: `BLOOMERY_CARD_BUDGET=38G` plans every card with at
most 38 GiB, and `BLOOMERY_CARD_BUDGET=CUDA0=38G,CUDA1=20G` gives each card named its own (a card it does not name keeps
its own bytes). It sizes the cards; `--place` (or the load's own pick) still chooses them. A device the census does not
list, a card named twice, and a budget below what a card needs with no expert on it are refused by name. A list of
shares, `BLOOMERY_CARD_BUDGET=CUDA0=3x,CUDA1=1x` (what `-ts 3,1` sets), splits the routed experts instead, as above.

    bloomery-serve -m model.gguf -dev CUDA0,CUDA1

All cards (the default of `-sm layer` with `-dev` naming both). The plan is `bp`: CUDA0, the larger, runs every layer
and the head and CUDA1 is its expert tier under the host tier. The order of the list does not matter; the stage is
the largest. The seat refuses the tier by name when its body serves none.

    bloomery-serve -m model.gguf -dev CUDA1,CUDA0 -ts 1,0

Two named devices and a split. The list is CUDA1 then CUDA0; `-ts` indexes that list, so the share of 1 is CUDA1's and
the 0 leaves CUDA0 out: the plan runs on CUDA1 alone. With `-ts 3,1` both stay, the plan is `bp`, and the cards hold the
routed experts in the proportion 3:1 (`BLOOMERY_CARD_BUDGET=CUDA0=750000x,CUDA1=250000x`, one line prints it): the
stage keeps its dense tensors and its cache beside its 3 parts, the tier its 1, and the host tier the experts the
tightest card cannot hold in that proportion.

    bloomery-serve --model qwen38 -m model.gguf --cpu-moe

Experts on the CPU, on the qwen38 seat: every routed expert is planned on the host tier (the
`BLOOMERY_QWEN38_EXPERTS=host` arm). The same flag on another seat is refused by name.

    BLOOMERY_DRAFT=dspark BLOOMERY_DSPARK_MODEL=draft.gguf bloomery-serve -m model.gguf -devd CUDA1

A draft on a named device, on the V4.1 seat: the draft opens on CUDA1 (`BLOOMERY_DSPARK_CARD=cuda1`). The plan sets the
draft's bytes aside on the card it opens on, before any expert there, so that card is on the plan: a card of the
placement at its place, any other visible device as a card that holds the draft and no expert. Without `-devd` (or
`BLOOMERY_DSPARK_CARD`, which also takes a card name) the draft sits:

- under `bp`, on the tier card: the stage keeps the experts it holds without a draft and the tier holds fewer;
- elsewhere, on the smallest visible card. When that is the stage (one visible card, or `gate`), the stage holds fewer
  experts. Under `a` beside a smaller card, that card joins the plan to hold the draft and no expert, and the stage
  keeps its experts.

A device that is not visible, a card name two devices carry and a list are refused by name.

**The older spelling.** `--place a` (the largest visible card), `--place gate` (the 3090's bytes alone), `--place bp`
(the next-largest card as the largest's expert tier) and a list by card name or CUDA index (`0+1`, `cuda1+cuda0`, the
stage first) keep their meaning. `--place` together with any flag above is refused by name.

### The environment

llama-server reads a variable for each of these flags, and so does every launcher here: the variable's value is read
as the flag with that value, under llama-server's name for it.

| Flag | Variable |
|---|---|
| `-dev` | `LLAMA_ARG_DEVICE` |
| `-sm` | `LLAMA_ARG_SPLIT_MODE` |
| `-ts` | `LLAMA_ARG_TENSOR_SPLIT` |
| `-mg` | `LLAMA_ARG_MAIN_GPU` |
| `-ngl` | `LLAMA_ARG_N_GPU_LAYERS` |
| `--cpu-moe` | `LLAMA_ARG_CPU_MOE` |
| `-ncmoe` | `LLAMA_ARG_N_CPU_MOE` |
| `-ot` | `LLAMA_ARG_OVERRIDE_TENSOR` |
| `-fit` | `LLAMA_ARG_FIT` |
| `-fitt` | `LLAMA_ARG_FIT_TARGET` |
| `-fitc` | `LLAMA_ARG_FIT_CTX` |

The names are the `.set_env(...)` of each flag in llama.cpp's `common/arg.cpp` (commit f5027fb). `-devd` has no
variable there, so it has none here. `LLAMA_ARG_DEVICE=CUDA1 LLAMA_ARG_N_GPU_LAYERS=99 bloomery-serve -m model.gguf`
is `-dev CUDA1 -ngl 99`.

- **The command line wins.** A flag on the command line replaces its variable, as in llama-server, and one line says
  so. `--place` replaces the variables that name cards (`LLAMA_ARG_DEVICE`, `LLAMA_ARG_SPLIT_MODE`,
  `LLAMA_ARG_TENSOR_SPLIT`, `LLAMA_ARG_MAIN_GPU`), also with a line; the other variables are read beside it.
  `--place` beside a placement flag on the command line is refused, as above.
- **Refused by name, as the flag is.** A variable whose flag would be refused ends the start under the variable's
  name with the flag's reason, for instance `LLAMA_ARG_SPLIT_MODE=row`.
- **`LLAMA_ARG_CPU_MOE` is a switch**: on for `on`, `enabled`, `true`, `1`, off for `off`, `disabled`, `false`, `0`.
  Any other value is refused, where llama-server ignores it. The routed experts' place is named once: `--cpu-moe`,
  `-ncmoe`, `-ot` and their variables are one place, and a second source of it is refused.
- **The other flags a server takes under llama's spelling** are read from their variables too, on the
  launchers whose seats take them (`bloomery-serve` and the single-seat servers): the variable's value is carried as
  the flag, unless the command line gives it in any of its spellings, and one line says which.

  | Flag | Variable |
  |---|---|
  | `-c`, `--ctx-size` | `LLAMA_ARG_CTX_SIZE` |
  | `-ub`, `--ubatch-size` | `LLAMA_ARG_UBATCH` |
  | `-cram`, `--cache-ram` | `LLAMA_ARG_CACHE_RAM` |
  | `-mm`, `--mmproj` | `LLAMA_ARG_MMPROJ` |
  | `--image-min-tokens`, `--image-max-tokens` | `LLAMA_ARG_IMAGE_MIN_TOKENS`, `LLAMA_ARG_IMAGE_MAX_TOKENS` |
  | `-np`, `--parallel` | `LLAMA_ARG_N_PARALLEL` |
  | `-a`, `--alias` | `LLAMA_ARG_ALIAS` |
  | `--host`, `--port` | `LLAMA_ARG_HOST`, `LLAMA_ARG_PORT` |
  | `--api-key` | `LLAMA_API_KEY` |
  | `--api-key-file` | `LLAMA_ARG_API_KEY_FILE` |
  | `--slot-save-path` | `LLAMA_ARG_SLOT_SAVE_PATH` |
  | `--chat-template-file` | `LLAMA_ARG_CHAT_TEMPLATE_FILE` |
  | `-m` (the model file), `-hf` (`--hf`) | `LLAMA_ARG_MODEL`, `LLAMA_ARG_HF_REPO` (`bloomery-serve` only; both set is refused) |

  The names are the same `.set_env(...)` lines of `common/arg.cpp` (commit f5027fb). A seat that takes none of a flag
  refuses it by its name after the line that says which variable it came from. The decision seat of `bloomery-serve`
  reads them too, for the flags it takes, and names each placement variable that is set (`LLAMA_ARG_DEVICE` and the
  rest of the table above) as one it does not act on: that seat places no card.
- **The keys add.** `LLAMA_API_KEY` is `--api-key` (the one name in the table with no `LLAMA_ARG_` prefix, line 3542)
  and `LLAMA_ARG_API_KEY_FILE` is `--api-key-file`. llama-server reads the variables before the command line and adds
  every key it is given, so here too the variable's keys count beside the command line's: a key flag does not replace
  them. The line that says a variable was read never prints a key. A variable that names no key (`LLAMA_API_KEY=`) ends
  the start by name, as `--api-key ""` does, and does not leave the server open.
- **Short spellings.** The command line's `-c`, `-ub`, `-np`, `-a`, `-mm` and `-cram` are respelled as `--ctx-size`,
  `--ubatch-size`, `--parallel`, `--alias`, `--mmproj` and `--cache-ram` before a seat parses
  (`llama_launch::respell`); a seat that takes none of the flag refuses it by the long name.
- **`-cram -1` is refused by name.** llama-server reads a negative count as a prompt cache with no limit (`-cram`,
  line 1719: "-1 - no limit, 0 - disable"). The cache here is host memory beside the load's host set, so it always has
  a budget: MiB, `0` for none, or unset for the default (half of what the host leaves past the load, at most
  llama-server's 8192 MiB). A negative count is refused with that, as `--cache-ram`, `-cram` or `LLAMA_ARG_CACHE_RAM`.
- **A variable bloomery does not read has a line saying so.** `LLAMA_ARG_FLASH_ATTN`, `LLAMA_ARG_THREADS` and the rest
  of llama-server's names that no flag of ours stands for are named, one line each, never silently dropped; so is a
  server variable on a launcher that takes none (the generate launchers), `LLAMA_API_KEY` among them.

### The other launchers

`bloomery-serve-ds41`, `bloomery-serve-qwen38`, `bloomery-chat`, `generate_ds41`, `generate_glm5next` and
`generate_qwen3moe` read the same flags and variables the same way, before any lever of theirs
(`bloomery_gpu_gates::llama_launch`), and `--list-devices` prints the devices on each. Their model is
`$BLOOMERY_REF_MODEL` as before, which `-ngl N` compares with. What a launcher has no thing for is refused by name:

| Launcher | `--cpu-moe`, `-ncmoe`, `-ot` | `-devd` |
|---|---|---|
| `bloomery-serve` (by seat) | the qwen38 seat | the V4.1 seat |
| `bloomery-serve-ds41`, `generate_ds41` | refused | accepted |
| `bloomery-serve-qwen38` | accepted | refused |
| `generate_qwen3moe` | on a qwen4exp file; refused on any other | refused |
| `generate_glm5next`, `bloomery-chat` | refused | refused |

`-ts`'s shares set a budget only on a load that names its card: on `generate_qwen3moe` it is refused (by the lever's
name) unless a flag that names cards (`-dev`, `-sm`, `-ts`, `-mg`) gives the load a `--place`, which `-ts` does when two
cards stay. `-fitt`'s margin is read wherever a placement is resolved, on every launcher. Each launcher parses the
`--place` word the cards come to, so a stage its body does not serve is refused there, as before.

`generate_glm5next` and `generate_qwen3moe` read their own flags by name and refuse any other argument by name, with the
flags they have; before, a misspelled flag ran the default. The other launchers read their flags in one pass that
refuses an argument it has no flag for, by name, with the usage (`bloomery-chat` and `generate_ds41` in their own parse,
`bloomery-serve` and the single-seat servers in the seat's).

## The seats

The qwen38 seat serves a Qwen3.8 safetensors checkpoint folder as downloaded, with no conversion step, in a build with
the `checkpoint` feature (the release's has it; any other build refuses the folder by name): `-m` names the
folder that holds `model.safetensors.index.json`, and `--hf <repo>` names a repo that holds such a checkpoint and no
GGUF set (a `:<quant>` on that repo is refused by name, as the checkpoint is the repo's one model). Every part keeps the
format the checkpoint stores it in. Checked with `nvidia/Qwen3.8-Flash-Next-NVFP4` on a DGX Spark's GB10, where it plans
the same bytes and gives the same greedy ids as the GGUF split made from it. The load reads the PLE table from the
folder on the NVMe tier. The NVFP4 folder's routed experts stay where the folder stores them for the host and the
residency machine, so its plan splits them between the card and the host as a GGUF's and `BLOOMERY_RESIDENCY` runs
the machine over them. A plan that pages them to the NVMe expert tier reads each matrix's codes and block scales into
the tier's RAM arena from whichever shards hold them, and moves the scales to follow the codes, so the host and the
residency machine read a paged expert as they read one in place. The FP4-Local folder's experts carry compressed block scales the
host does not read: its load holds every tensor on the stage card, so a plan that puts a tensor elsewhere and a set
`BLOOMERY_RESIDENCY` other than `off` are refused by name. The folder's MTP layer uses the same draft and verification
path as a shared GGUF draft, under the same `BLOOMERY_DRAFT` rule (Speculative decoding, below): `mtp` selects it,
`off` disables it, and unset selects it where the rule would select a GGUF draft. The draft borrows the target's
embedding and vocabulary head; its experts and dense projections keep their stored formats, so the folder is the one
download a drafted server needs. A folder whose MTP layer the draft does not read still serves its target: `mtp` is
refused with the layer's refusal, and unset turns the draft off with it. Before its address the seat prints one
`setup` line: the draft (`mtp`, `lookup` or `off`) and, drafting, whether its layer comes from a checkpoint folder or a draft file,
the slots and the context each serves, and the settings that change them.

The nemotron seat serves Nemotron 3 Nano 4B (`--model nemotron -m PATH`), Nemotron 3.5 Bolt 74B-A9B and
Nemotron-3.5-Lightning 30B-A3B, one sequence: `--parallel` above 1, a placement of more than one card or with a
tier card, and a draft lever are refused by name. The prompt path is the body's batched one: a prompt of several ids
runs in passes of many rows, one Mamba-2 pass a layer over the pass's rows. A Mamba-2 state cannot step back: a
request keeps what the slot holds only when its prompt extends the whole held sequence, and any other prompt resets
the slot and is read again. The seat runs no prompt cache; a snapshot or a resume is refused by name. A Bolt or
Lightning load the card cannot hold whole keeps the rest of its routed experts on the host, and a host need past the
host's room pages the overflow on the NVMe tier, as "When the host cannot hold its experts either" below says.

The decide seat answers SystemOne requests, one at a time. Clef-Flash also reads images (Image input, below); lev reads text states only. It serves two decision models: Clef-Flash (a joint schema head) and lev (`--hf ggml-org/lev-GGUF:Q4_K_M`, a label readout; its file's `qwen35.decision.type` picks it, and its request is rendered by the file's own `systemone` template):

```sh
curl -s http://127.0.0.1:8080/v1/systemone -d '{"state": "User: what is the weather in Seoul tomorrow? Tools available: web_search, calculator.",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
    "criteria": {"web_search": "Look something up online", "calculator": "Do arithmetic", "none": "Answer directly"}}}}'
```

**Two layouts of the Clef-Flash GGUF.** The file's architecture picks the head's source. llama.cpp's `clef` layout
carries the joint head inside the file (bartowski's current upload), so `bloomery-serve --hf
bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M` needs no head flag. The earlier `qwen35` layout takes the release's head
file: `--head joint_head.safetensors`, its `joint_head_config.json` beside it. A `--head` beside a `clef` file names the
head instead of the file's own.

**Concurrent requests.** Every generative seat but nemotron, raon1, solar2 and musespark (one slot each, [Flags by seat](#flags-by-seat)) takes `--parallel N` (unset, one slot; a set
`--ctx-size` with no `--parallel` serves one slot at the whole of it — the flag is one request's context, as
llama-server reads it, and the startup line says so). A count you set is never lowered. The N slots are resident sequences
inside the one model, switched by pointer exchange: nothing parks, and each round advances every busy slot by one
token or one drafted pass. Where the body runs several slots' rows as one pass, the round reads the weights once for
all of them: on a whole-card Qwen3-30B or Qwen3.6, on V4.1 and MiMo-V2.6-Flash (two slots' rows a pass), and on GLM-5.3 and Qwen3.8, where a greedy
or sampled request's drafted verify window rides the same pass (two windows a pass; with the draft off, the plain rows).
On Qwen3.8 the lookup draft's windows ride one pass the same way, two rows a slot. Which loads verify several slots'
windows together is the body's fact, not the draft's: the body must keep part of a slot's verified rows at the commit.
Qwen3.8 also batches plain steps on a drafting load, with every slot's draft brought up to date in one draft pass
before the round's pass. A round that leaves slot 0 out runs as one pass too and keeps slot 0's waiting hidden rows
intact. A
placed Qwen3-30B or Qwen3.6 (`--place a`), Qwen3.6 and V4.1 with the
lookup draft, Qwen3.8 under `--place bp` and MiniMax-M3 step their slots in turn, a select and a step a slot each round. At a
fixed expert placement each request answers the tokens of its run alone on the same server; under adaptive residency
the placement follows every stream's passes (see [Limits](#limits)). Under more slots than one the context splits as
llama-server splits it with `-np N` and no `-kvu`: a set `--ctx-size` is the total and each slot holds `total / N`
rows, on V4.1 the same way; with no context named, each slot holds the default's choice (the largest context the plan of N slots holds, on Qwen3.8 one that also leaves the residency its experts, above) and the total is that times N.
`--parallel 1` keeps one sequence with the whole context.

`--slot-ctx S` sets one request's context apart from the count: `--parallel 4 --slot-ctx 32768` serves four slots of
32,768 positions each, the same load as `--parallel 4 --ctx-size 131072`. Alone it serves one slot of `S`, and beside
`--ctx-size` it is refused by name, since one would name the total and the other a slot's share. On the command
line it replaces `LLAMA_ARG_CTX_SIZE` and a settings file's `ctx-size`, as `--ctx-size` does. Every seat that takes
`--parallel` takes it.

`--parallel auto` (the qwen38 seat) picks the count from memory: the most slots, up to 8, whose plan holds one
request's context. That context is `--slot-ctx` when set. Unset, it is the context one slot's default takes, when
that is the model's limit; where one slot already has to fit under the limit, auto serves one slot. The placement and
the draft are chosen at one slot first, so auto counts the slots that load holds. A `--slot-ctx` that not even one
slot holds is refused by name, as `--parallel 1 --slot-ctx S` would be. `--parallel auto` beside `--ctx-size` (or its
variable or file entry) is refused by name, and under `BLOOMERY_ROUTE_TRACE` it serves one slot. The `parallel` line names the count it chose
with `from=auto`. The default stays one slot; auto is never chosen for you.
The plan counts every slot, so the slots' caches together never pass what it holds, and `--park-ram` is refused by
name. A new request's prompt runs in one call between rounds, and the other streams wait for it. On Qwen3.8, unless the
MTP draft serves the slots or `--mmproj` loads an image encoder, a prompt with more than 4,096 ids past its cached
prefix runs instead as calls that end at the multiples of 4,096 positions, one a round between the other streams'
rounds, so a stream waits for one such call at a time. The calls leave the bits and the cached prefixes of the one
call.

**Small cards.** Qwen3.6 and Qwen3-30B at `Q4_K_M` (19-21 GB) run whole on a 24 GB card. On a 12-16 GB card
the load sees that the file does not fit the card's free bytes, sends the routed experts to the CPU, and says so
on its plan line; no flag is needed (`--place a` asks for that split on any card).

**A plan that fits the card you have.** The load reads each card's free bytes and the host's available RAM
before anything uploads; the expert share sizes to them, and a plan that cannot fit is refused by name with
every term (dense, KV, context, scratch, margin, and what the load holds beside the plan) and the processes holding
the card — before minutes of loading, not after. Inside a container the host's RAM is the smaller of `MemAvailable`
and the cgroup's own memory limit (`docker run --memory`, a systemd scope's `MemoryMax`), read the same way.

**Why a Qwen layer stays on the host.** Under the card expert rule, each
printed placement plan is followed by `host only layer=… tensor=… why=…`
records for routed tensors whose kernel contract excludes their layer from
the card. The rule is the actual body's, including the tensor's gate/up/down
role and matching gate/up types. Causes keep file order and are not deduplicated
by layer. A byte-budget limit alone, a dense model with no routed tensors, or an
explicit all-host expert choice emits no such diagnosis. These records use the
plan's stream: stderr in the server and stdout in the generation CLI.

**When the host cannot hold its experts either.** A Qwen3.8, GLM-5.3, V4.1, MiniMax-M3, Muse Spark, Solar-Open2,
Flint or Nemotron-H MoE plan whose host need passes the host's available RAM pages the overflow on the NVMe tier and
serves it from a RAM arena filled from the file. MiMo's refuses that room by name: its host run attaches no tier.
Who computes the arena's experts decides whether the tokens equal the all-RAM load's:
- On an anonymous arena (`BLOOMERY_NVTIER_MEM=anon`, the default on a card with its own memory) the host computes
  them, and the tokens equal the all-RAM load's.
- On a pinned arena (`pinned`, the default on an integrated GPU such as DGX Spark's GB10) GLM-5.3's decode steps
  compute them on the card from the arena's slots and add their sum to the host's. The card quantizes the
  activation as q8_1 (an f32 scale per 128 values, where the host's K-quant dots take Q8_2X4 with a bf16 scale per
  32) and sums in its own order, so the tokens can part from the all-RAM load's after some steps. A prompt's batch,
  a load whose residency machine runs, and the other families stay with the host. The `nvtier` record's `card_leg`
  says `on` or why the card does not compute them.

NVMe arena eviction options are documented in the [lever registry](../crates/levers/src/registry.rs).

The load prints an `nvtier` record (the arena's budget, the paged bytes and its
counters; a load that attached no tier prints none). Limits of a paged load, as the code has them:
- Qwen3.8 steps one column on a paged plan. With no `--parallel` it serves one slot and the MTP draft stays off; a set
  `--parallel` is served as asked, and one the room's floor cannot hold is refused by name. A set `BLOOMERY_DRAFT=mtp`
  runs. Under `--place bp` each slot steps alone, so a set count stays and only an unset draft goes off.
- Qwen3.8's prompt cache default is 0 on a plan the tier splits, and its `cache ram=…` line names the arena beside
  the load's own host need, so the two anonymous heaps cannot outrun `MemAvailable` into the OOM killer. GLM-5.3's
  default counts the arena inside the host need. `--cache-ram` sets the budget past both.
- Unset, adaptive residency is off on a paged plan: the Qwen3.8 seat's unset word resolves `off` there, and the
  `residency unset` record says so. A `BLOOMERY_RESIDENCY` word set on such a plan is served as given — the
  residency machine's victim fills read through the NVMe tier's reader while the step's ids stay pinned in the arena
  — so a paged Qwen3.8 seat serves with `BLOOMERY_RESIDENCY` set.
- GLM-5.3 has no refusal of its own for slots or the draft. Its NextN draft layer's routed experts are host
  experts too, so the trunk's room is the host room less them.
- V4.1 refuses a paged plan under an r8 pair that reads a sidecar, by name (`BLOOMERY_R8=off` reads the source and
  gives the same bits). The release uses no sidecar.

**Speculative decoding.** `BLOOMERY_DRAFT=mtp` drafts with the model's own MTP head (Qwen3.8 and GLM-5.3, on by
default in their servers); on the qwen3 seat, `BLOOMERY_DRAFT=mtp` drafts a `qwen35moe` or `qwen35` file's
own next-token layer on a whole-card load of one sequence, and a media load, a placed load and `--parallel`
above 1 refuse it by name; a `qwen3moe` file keeps the plain path, `mtp` refused by name (no such layer,
no verify pass). `BLOOMERY_DRAFT=dspark` drafts V4.1 with DeepSeek's DSpark head. The draft sits on the tier card
under `--place bp`, else on the smallest visible card: the stage itself, or a second card that joins the plan to hold
the draft and no expert; `-devd` names it (Placement, above). On the qwen3 seat, `BLOOMERY_DRAFT=dspark` with
`BLOOMERY_DSPARK_MODEL` naming an HF draft directory serves a whole-card `qwen35moe` target from one slot: RedHat's
Qwen3.6 speculator carries its own reduced head and id map, and MiniCPM-V 4.7's full-vocabulary draft borrows the
target's Q6_K `output.weight` in place and its `token_embd.weight` rows, so neither file is copied. A draft whose
ids, mask or taps do not fit the target is refused by name before a weight is read; a placed load and
`--parallel > 1` are refused by name.
The draft fits a target by its facts, not by the file's name: a fine-tune that carries Qwen3.6's shapes (Sage-A1.5,
Ornith-1.5) takes the same draft, and a reduced draft borrows nothing from the target, so a target whose own head
is not a Q6_K projection serves it too. On the same seat, `BLOOMERY_DRAFT=lookup` serves the n-gram lookup draft
on the same whole-card verifying load, one context a slot; a `qwen3moe` file has no multi-row pass and refuses it
by name.
Unset, the MTP draft yields to the context: it goes off when the plan with it would leave a slot half the positions
of the plan without it or fewer, a set `--ctx-size`/`--ctx` included, a `draft yield` record naming its
card bytes and the positions a slot gets either way, and the context line that follows is the plain plan's.
`BLOOMERY_DRAFT=mtp` keeps it on whatever the context. On Qwen3.8 the MTP draft runs on any slot count, set or unset.
The drafts run on greedy requests (`"temperature": 0`), and the MTP drafts also on sampled ones
(the default, temperature 0.8 as in llama-server), on ones that ban an id (`ignore_eos`) or bias ids (`logit_bias`),
on ones that ask for probabilities (`n_probs`, `logprobs`) and on structured output, alone or beside other running
requests: each kept id is the request's own pick from its verified row (its sampler's draw or the argmax, with each
biased id's bias added, every banned id's logit at -inf and, under a schema, every id it does not allow masked out),
as llama.cpp's speculative decoding takes it, so a request of one seed takes the same ids alone and beside others.
Its probabilities are read from the same verified rows, so they are the values its steps would report. A forced
call's program waits for the model's own close of its think span stepping one token at a time. A pass reads each
verified row back whole only when its request needs the whole row: one whose pick reads at most the row's 64 best
logits (its sampler's `top_k` after penalties that only lower a logit, one more for each id the penalty window or a
negative `logit_bias` or `ignore_eos` can lower, or the argmax of a greedy request) reads those 64 ids and logits a
row from the card instead of the vocabulary's floats. Probabilities, a schema, `top_k` 0, and a penalty or a bias
that can raise a logit read the whole row.
A request under DSpark decodes one token a step.

**Adaptive residency.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and swap routed experts
between the card and the host between steps; a prompt call streams its hottest host experts onto the card, so
decode starts warm (on by default, with `--place` unset, `a` or `bp`). On Qwen3.8 a call that arrives once the
decode has routed experts of its own puts the pool back where it found it at its end, so a long prompt does not
reset the placement its reply decodes from; a fresh server's calls keep it.

**Prompt cache (V4.1, Qwen3.8, GLM-5.3, MiMo, MiniMax-M3, Muse Spark).** `--cache-ram` (MiB; 0 turns it off) keeps saved sequence states in host RAM.
By default the budget is the lesser of 8 GiB and half the host RAM the plan leaves, with every slot's checkpoints and
the residency's churn pool counted in; the checkpoint budget follows the slot context. A request that continues an
earlier conversation (its prompt holds the whole earlier prompt) restores that state even after another request used
the slot. A request that shares only part of an earlier prompt, such as the same system prompt with another question,
restores too: a saved state keeps checkpoints at 512·2^i positions below its end and the last one, and a shared
prefix restores at the largest of them inside it. On Qwen3.8 the card's own checkpoints are 4,096 positions
apart, so there a shared prefix shorter than 4,096 tokens does not restore. Saving runs off the request path. On
MiniMax-M3 and Muse Spark a saved state is the K and V rows of the held positions (and MiniMax's index keys), with no
checkpoints: a request that shares any prefix with a saved state restores that prefix, and only those rows go back to
the card. The Qwen3 seats keep only the longest prefix the live slot holds; they run no saved prompt cache, and a
save is refused by name. For Qwen3.5 and Qwen3.6, when the vocabulary and chat template both carry
`<|im_start|>`, the seat cuts prompt calls at message starts, so a shared system prompt stays in the live slot
for another question. Reuse still requires that slot to hold the shared prefix; another conversation can replace it. `timings.cache_ms` and `/metrics`'
`prompt_cache_seconds_total` count the cache's work.

**Image input (V4.1, Qwen3.8, Qwen3.6, MiniCPM-V 4.7, Sage-A1.5, GLM-5.3, Clef).** These seats read images when they have a vision encoder: `--mmproj <file>`,
llama-server's flag, naming the model's encoder GGUF. On V4.1 the file is `mmproj-DeepSeek-V4.1-Flash-BF16.gguf`.
Under `--hf` with no `--mmproj`, every image seat takes the repo's own projector beside the model, unless `--no-mmproj`
says none. The V4.1, qwen38, qwen3 and GLM-5.3 seats load it only when the text model keeps the same card experts and
context with the encoder as without it, also when the encoder sits on a card off the plan; otherwise one vision line gives the projector's bytes and cost, and the seat serves text. Qwen3-30B-A3B has no encoder, and `--mmproj` there is refused by name. A file of
another encoder is refused by name. Images travel inside the request as PNG or JPEG bytes: an OpenAI `image_url`
part holding a `data:` URL, or an Anthropic `image` block with a base64 source; any other URL is refused, not
fetched. The encoder loads once at start and stays resident: on a card the load's plan leaves off it, or, when every
card is on the plan, as a named reserve whose bytes the plan counts on the stage card; the `vision` line at start
names the card and the bytes (V4.1's is 1.31 GB of weights and activations). On V4.1 a request's image spans ride
the prompt cache like its ids: the same image sent again keeps its span. Without an encoder the server serves no image, and `/props`
answers `vision: false`. A generative seat answers a request with an image with a 400 (`image input is not supported:
this server reads text only`). The Clef decide seat answers it with a 501 and llama.cpp's hint (`image input is not
supported - hint: if this is unexpected, you may need to provide the mmproj`). Lev takes no image: `--mmproj` beside
it is refused, and a request with an image is a 501.

Inline PNG/JPEG input shares a fixed 64 MiB decoded-RGB budget across a chat
request's new images and across every image of a batched embedding input. An
image whose key is resident in the prepared-image cache skips decoding and
preparation and draws on none of that budget; its file is still loaded and
hashed (the hash is the key). Evicted images are decoded and prepared again and
consume the decode budget like new ones. The cache is one per server, beside its
encoder, and retains at most 512 MiB, the least recently used entries first out;
how many images that is depends on the encoder and the image size (a 1280×720
screenshot prepares to about 5.6 MB on V4.1 and about 11.3 MB on the Qwen3-VL
and GLM encoders [derived], about 90 and 45 such screenshots). `/metrics` counts
its hits and misses (`prepared_image_hits_total`, `prepared_image_misses_total`)
and its retained bytes and images on a server with an encoder. Image headers
are checked before allocating decoder output,
and all of a request's new images are decoded within the budget before model
preparation starts; PNG temporary output must fit the remaining budget too. A
request whose new images together pass the budget is a named 400, not a large
allocation followed by a context refusal. Trusted offline image constructors are
not subject to this serving policy.

The budget bounds retained RGB and the cache prepared patches, not total process
memory: compressed request data, decoder internals, temporary conversion buffers
and model preparation are additional. Encoded-video planning and decoding keep
their own limits.

**Video input.** A model with a video rule reads a clip when `ffmpeg` and `ffprobe` (5.1 or newer) are on `PATH`; the
server looks once at start, prints one `video:` line, and answers `/props` with `modalities.video` true. A clip is an
`input_video` part (`{"data": <base64>, "format": "mp4"}`) or a `video_url` part holding a `data:video/*;base64,` URL;
any other URL is refused, not fetched, and so is a request body over 64 MiB. The server writes the clip to a private (0600) temp file and
reads it only as a file in one of these containers: MP4/MOV, Matroska/WebM, AVI, FLV, MPEG-PS, MPEG-TS and Ogg (a
playlist or any other container is refused, so a clip never makes ffmpeg open another file). It probes it (a clip whose container states no frame count is decoded once to count its video stream's frames), plans the
frames by the model's rule (rate, frame bounds, group size, token cap) under the context the request's text leaves (an
embeddings input's images and audio too), and has ffmpeg decode only the planned frames. Without ffmpeg, a clip is a 400 that names the
fix. EmbeddingGemma 2's embeddings route takes the same parts and still takes decoded frames (a first text part
`Video:`, then `image_url` frames).

**The engines behind the server.** The release ships `bloomery-serve` alone. A source build also has
`generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (streams text) and the standalone V4.1 and Qwen3.8
servers `bloomery-serve-ds41` and `bloomery-serve-qwen38` ([`BUILD.md`](BUILD.md)). Every seat and flag is its
binary's `--help`.

## Starting under a broker

A placed load prints a `need` record beside its `plan` record (a generator on stdout, a server and `bloomery-chat` on
stderr; a seat with a `--plan` flag prints both before it exits). It states what the load asks of the machine:

    need bytes=… pinned=… anon=… cache=… locked=… unified=… os=… cache_ram=… lock=false cards=[A6000:…] threads=… cpus=[…] caller=… allowed=[…]

- `bytes` is the host need the load checks against `MemAvailable` (the plan's host terms, the churn pool and what the
  load keeps beside them, less the OS reserve `os`). `pinned`, `anon`, `cache`, `locked` and `unified` split it by what
  can take a page back and sum to it: driver-pinned memory (ring shadows, the tier's rows, a pinned arena), anonymous
  pages (the NVMe tier's arena), file pages the kernel may reclaim, file pages `mlock`ed under `BLOOMERY_HOST_LOCK`
  (in place of the reclaimable ones), and an integrated device's share of its pool. `cache_ram` is the prompt cache of a
  server, beside `bytes`.
- `cards` is the bytes the plan puts on each card. `threads` counts the worker pool's participants, `cpus` the cpu each
  worker is pinned to, `caller` the cpu `BLOOMERY_PIN_MAIN` pins the dispatching thread to, and `allowed` every cpu the
  process may run on. The pool places its threads inside that mask (`sched_getaffinity`: `taskset`, a container's cpuset,
  a unit's `AllowedCPUs`) and says so in one line when the mask cuts the machine's cpus. The caller and the helper
  threads the engine spawns are not pinned by the pool: they float over `allowed`.

`BLOOMERY_NEED_HOOK=<executable>` calls a program once per placed load, after the plan and before a card is opened or a
byte read in, with that line on its standard input. Exit 0 goes on. Exit 75 ends the process with 75, naming the hook's
last line of output (the load has opened no card of its own, so a broker can queue the start and ask again). Any other status, or a
hook that cannot start, ends the load with a named error. A hook may wait as long as it likes; unset, nothing runs. A
logger is a hook: `cat >> need.log` as a script. The hook is a program named by an absolute path (a relative name is
refused at start). Every load that prints a `need` asks the hook before it opens its card, a whole model on the one
card of a placed plan (`--place`) included. A load that opens a whole model on one card with no plan (`--place` unset
and the model fits) states no need and consults no hook; a set hook is named on stderr as not asked.

## Limits

- **sm_86+** GPUs and NVIDIA driver R580+ (CUDA 13); the x86-64 archive carries sm_86 PTX, compiled for the card
  at the first start, and x86-64-v3 host code; its sm_120a twin carries sm_120a PTX and is the installer's pick on
  a machine whose every GPU is compute capability 12.0 (the RTX 50 series). The GB10 archive (Linux aarch64) carries its sm_121a code compiled for
  a DGX Spark, with that code's PTX for a driver that refuses it;
  another aarch64 machine builds from source with `BLOOMERY_ARCH` setting the CUDA arch
  ([`contrib/dgx-spark.md`](contrib/dgx-spark.md)). The AVX2 kernels stay x86-64 only, and a quant type without an
  aarch64 kernel runs through a scalar mirror there.
- GLM-5.3 holds at most 16,384 positions a slot (`ORACLE_POSITIONS`): a `--ctx-size` whose share a slot passes it
  is refused at start, and a longer prompt with `exceed_context_size_error`. On a 12–16 GB card the split Qwen3 load's
  unset context fits to the largest the plan holds; the old rule's sitting there measured 2,048 positions a slot at
  `--parallel 2`, and that card has not been remeasured. Agent clients with long system prompts want the context to
  one request there (`--parallel 1`, or a set `--ctx-size` with no `--parallel`).
- One model a server; one expert tier card at most (`--place bp`).
- Concurrent streams run as one pass on a whole-card Qwen3-30B or Qwen3.6, on V4.1, MiMo-V2.6-Flash, GLM-5.3 and Qwen3.8; a placed
  Qwen3-30B or Qwen3.6 steps in turn. A new request's prompt runs whole while the other
  streams wait (on Qwen3.8 without the MTP draft on the slots or `--mmproj`, a call of 4,096 positions at a time, above). The decide seat serves one request at a time, and the nemotron, raon1, solar2 and musespark seats one
  slot each; DSpark drafts never rejoin — `--parallel > 1` with
  `BLOOMERY_DRAFT=dspark` is refused by name.
- With adaptive residency on, a request's tokens follow the placement its passes ran on, and the placement follows
  the history of passes (the requests before it and the streams beside it): the same history gives the same tokens
  bit for bit, another history can reword an answer at a near tie, since an expert on the card and the same expert
  on the host round their activations differently. `POST /residency/reset` returns to the load's placement;
  `BLOOMERY_RESIDENCY=off` gives repeatable tokens at residency's cost in speed.
- A placed load's plan counts every slot's cache, so `--parallel 2` puts fewer experts on the card than
  `--parallel 1` (8 fewer on V4.1), and the two servers' answers can differ at a near tie, with residency on or off.
- Clef takes text states and, with an encoder, images; it reads backbone weights of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and F32 (not the IQ
  types, Q2_K, Q4_0 or Q4_1). A head inside a `clef` file is read as the file stores it (bartowski's: Q8_0 matrices,
  which move a probability by under 3e-4 against the release's bf16 head on the eight reference requests).
- V4.1 decode is bound by host memory bandwidth in each step's expert part, and by the card's own serial work
  before it; a long V4.1 prompt (P = 4096) is bound by the card, since the prompt call streams host experts in.
- Structured output is served for `response_format` of type `json_schema` or `json_object` and for `tool_choice`
  `"required"` or a named function (Anthropic `any` and `tool`, Responses `text.format` `json_schema` or
  `json_object`), on every generative seat. The schema subset is an object (every property written, in the schema's
  order), an array of `items`, `anyOf`, a `$ref` into the schema itself (`#/$defs/...`, written out where it stands;
  a recursive one, or one that writes out deeper than the 128 nested JSON levels the request parser reads, is
  refused), string, integer and boolean enums, `const`, booleans, `null`, integers, numbers, and free strings with an optional
  `maxLength`. `json_object` is any JSON object with at least one member (an empty object or array is served only
  nested). The JSON is compact with one space after `:` and `,`, and a free string takes no escape. `temperature` defaults to 0, and the request then decodes greedily (the largest logit among
  the ids the schema allows). A positive `temperature` or a penalty draws with the request's sampler from the row with
  every id the schema does not allow masked out, so `top_k`, `top_p`, `min_p` and `seed` apply as on free text.
  Arrays, numbers and `json_object` have no length bound; reaching the token or context limit can leave incomplete JSON with a `length` finish reason.
- A prompt that leaves the think span open (thinking on) is read by what the program writes. A JSON `response_format`
  closes the span at once and writes the document after it. A `tool_choice` that forces a call lets the model reason
  freely, with the request's sampling and `logit_bias`, until it generates the span's close itself, and the call's
  program starts after that close; only the close tag ends the span there (a call tag inside it is reasoning, on
  every route and in the stream). Its reasoning is in `reasoning_content`, or in the text on `/completion` and
  `/v1/completions`, which read an open span in the prompt for a forced call too. The token or context limit, a
  `stop` word and `reasoning_loop` end such a generation as any other, with no call; a model that ends its
  generation before it closes the span is a 500 naming the missing call, and so is a close spelled in ordinary
  tokens whose last token also writes text past it, except the parser's ASCII whitespace. Incomplete UTF-8
  bytes still held by the decoder at that close are refused before the close is emitted, even at the token limit;
  a matched stop word discards them as usual. A prompt that closed the span, or leaves it to the model, starts
  the program at once.
- Two arms follow the schema. The step arm is one `next` an id. The jump arm is the default: the fixed text that opens
  the answer rides the prompt's call, and each later run of ids the schema fixes (forced text, the rest of a chosen
  option) is fed as one call that also steps its last id, so the next decision is read from that call's row. Free
  strings step token by token on both arms. Both arms pick the largest allowed logit at every decision of a greedy
  request, ties to the lowest id, and draw with the request's sampler otherwise; either way a run of fixed ids costs the
  engine one call instead of one step each. Nothing is fed after the last decision: the closing text and the stop are
  written with no call, the stop being the vocabulary's end of sequence, and the prompt cache holds the answer up to its
  last decision. The step arm picks its stop from a row, so the two arms' token lists can differ in that last id alone;
  the text and the finish reason are the same. A run under two ids is stepped, and a request that asks for `logprobs`
  or sets a penalty takes the step arm, so its penalties count every fixed id before a draw. Where options of an
  `anyOf`, an array's close and its next item, or a JSON value's kinds begin with the same ids, the arms follow them
  side by side, and an id is fixed only when every one of them fixes it.
  `BLOOMERY_SCHEMAJUMP=off` is the way back to the step arm until the jump arm's gain is measured on each seat. A prompt
  call can round a row differently from the steps it replaces (a Qwen3.8 tier load runs a prompt call on the batch path;
  the Qwen3 seats' prompt call of more than eight ids runs the wide walk), so a decision inside a rounding of a tie can
  differ between the arms. A request's `timings.constrained` counts its arm's work after the prompt: `spans` and
  `span_ids` (the runs fed and their ids), `decisions`, `free_steps`, `next_calls`, `prefill_calls`, `next_after_calls`
  (the calls that fed a run and stepped it), `lead` (the ids the prompt's call carried) and `tail` (the ids taken after
  the last decision with no call). With `--parallel`, a run, and a prompt call with its lead, are calls the engine
  thread makes on a seat's prompt quantum, one call a round beside other busy slots, never inside a step.
- Refused by name: any other schema keyword (`oneOf`, `allOf`, `pattern`, `minLength`, `minItems`, `minimum`,
  `minProperties`, a list of types, ...), a `$ref` that points outside the schema or is met again inside the schema it
  points to (a recursive schema), a schema that writes out to more than 4,096 schemas, a schema or a request's tools
  together that write out past 64 MiB (the request body's limit, paths included), a program that tokenises more than
  16 MiB of fixed text (an `anyOf` tokenises the text on each side of it once per option), a forced tool's
  `parameters` with a root `anyOf` or `$ref` (they are read whole by the same rules, so a `required` name no property
  has is a 400 too), an `anyOf` with a text option in a tool argument that a Qwen, GLM, Solar or DSML markup writes
  raw, a top-level `json_schema`, `grammar`, a JSON `response_format` beside a forced call, `ignore_eos` and `reasoning_budget`. An enum
  whose options tokenise so that one is a prefix of another (`[1, 10]` on a vocabulary that splits digits) is a 400, and
  so is a constant, enum option or property name whose text tokenises to a stop id.
- A `<` never reaches a tool call's markup as data. A free JSON string inside a tool call (Hermes' arguments, or an
  object that a Qwen, GLM or DSML parameter holds) takes no token with a `<`, and so does one in a `response_format` reply that comes with
  tools (`tool_choice` other than `none`), since the reply is scanned for a call. A free string in a `response_format`
  reply with no tools may hold one, so code and HTML come out whole. A `<` in a JSON constant, enum option or property name is
  written as `\u003c` in both, which reads back as the same string. A string written raw in a tool call's XML markup is
  one line (Qwen) or holds no `<` (GLM, DSML), since the value ends at the text after it, and a raw constant or enum
  option with a `<` is a 400.
- A pinned toolchain with a pinned cuda-oxide revision ([`BUILD.md`](BUILD.md)).

In progress: serving on N cards for every model; Clef from the IQ and Q2_K files; DeepSeek-V4-Flash-0731.

## EmbeddingGemma 2 images

A side seat serves the `gemma-embedding2` encoder beside a generative model. Build `bloomery-serve` with
`--features glm5next,clef,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,vision` through `cargo oxide`, and add these flags
to the chat model's command:

```text
--embd-model embeddinggemma-2-Q8_0.gguf
--embd-mmproj mmproj-embeddinggemma-2-BF16.gguf
--embd-ctx 8192
```

`--embd-cache-bytes N` enables an in-memory LRU cache with a shared byte budget for full-width pooled vectors
and image tower outputs. It is off by default (`0`). Entries include their retained values and metadata, excluding
container spare capacity and allocator overhead. Oversized results are not retained. A vector hit skips the encoder; an image with different
surrounding text can reuse its tower rows. Cuts and normalization are applied after the vector lookup.
Audio inputs do not use the pooled-vector cache.

`--embd-hf <repo>[:<quant>]` selects the text file and the row's BF16 projector from that repository.
`--embd-no-mmproj` disables that automatic projector. A local `--embd-model` loads images only with an explicit
`--embd-mmproj`. The projector's vision matrices must be BF16; a Q8_0 projector is refused by tensor name.
The F32 patch convolution and learned position tables retain their file precision. With no activation keys,
the tower follows mainline's Quick GELU; `clip.use_gelu=true` selects its tanh GELU and SiLU is refused.
With `clip.audio.projector_type=gemma4a`, the EmbeddingGemma 2 side seat also loads the audio tower on its card
and adds its weight and scratch bytes to the side seat's byte count.

The encoder and tower share the embedding side seat's card and thread, with streams of their own. Before the chat placement
is made, their weight and activation bytes are counted together. On a card outside the chat plan this costs the
plan no bytes; otherwise the plan holds them in its named `embedding model` reserve, reducing the chat expert
budget. The side context bounds the complete input, including image rows and delimiter text.

Send one object per input inside OpenAI's `input` array:

```json
{
  "input": [
    {
      "content": [
        {"type": "text", "text": "A description of this picture:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "dimensions": 256
}
```

An image URL is a base64 `data:` URL for PNG or JPEG, as on the chat path; HTTP URLs are refused. Multiple image
parts remain in input order. Each image is resized with mtmd's aspect-preserving dynamic canvas, bicubic filter
and centered black padding. The default budget is 70 to 1120 pooled image tokens. Set
`--embd-image-min-tokens` and `--embd-image-max-tokens` to change it independently of the chat model's image flags.
A canvas exceeding the reserved token budget or learned position table is a request error before a tower call.

`<|image>` and `<image|>` are tokenized around each image's rows with the server's vocabulary rule,
included in the single bidirectional pass and included in mean pooling, as in the pinned mainline server.
Only text embeddings receive the text encoder's `sqrt(hidden)` scale: image rows enter already projected.
The entire input runs once; image chunks are not decoded separately. A request without a loaded image tower,
with an unsupported image part or beyond the side context is refused with a 400.

The reference families in `crates/refset/src/arch/gemma_embedding2` own the projector path, its digest and
llama.cpp build pin. `just gate-gpu-gemma4v-tower` reads the mainline card and CPU tower sets and every required
layer tensor, comparing the device to the card with each band derived from the CPU twin.
`just gate-gpu-gemma-embedding2-e2e` also runs image-only and `mix_free-448x448` text-first inputs through
`pass_media`, using `serve::embeddings::assemble` for the same part ordering and token spans as the HTTP route.
It compares raw and normalized vectors with the server and its CPU twin (`just dump-ref-emb2 server.cpu`).

To compare a running HTTP server directly with saved llama-server vectors:

```sh
python3 tools/ref/emb2/server_vectors.py compare --url http://127.0.0.1:8080 \
  --set "$BLOOMERY_DATA/ref_gemma_embedding2_server" \
  --image free-448x448=/path/to/free-448x448.png --mix-image free-448x448
```

Use the exact image bytes named by the reference set. Add `--ids-dir "$BLOOMERY_DATA/ref-tokenizer-gemma4"`
and `--case NAME` for text cases. Each selected `raw` and `embd` row prints cosine and maximum absolute
difference. Exit 0 means the comparison completed; it applies no numerical acceptance band.
