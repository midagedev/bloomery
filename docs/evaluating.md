# Evaluating bloomery

A checklist for a person or a coding agent (Claude Code, Codex) who wants to run it and judge it fairly:

1. **Check the machine.** `nvidia-smi` (card, driver, free memory), `ldd --version` (glibc 2.34+), `free -g` (host
   RAM). Find the model's row under [Models and hardware](models.md#what-each-model-needs): the card it needs, the
   host RAM, the file size.
2. **Use a release binary** ([Install](../README.md#install)). Building from source needs a pinned nightly and our
   cuda-oxide fork ([`BUILD.md`](BUILD.md)); a release binary is the same code.
3. **Start it with `--hf`**, for example
   `bloomery-serve --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL --port 8080`. The first start downloads the files
   (and the MTP draft where the repo has one). The first start also compiles the kernels for the card once. The plan
   line in the log names the cards and how the experts are split. Wait until `GET /health` answers 200.
4. **Pick the cards with `CUDA_VISIBLE_DEVICES`.** With `--place` unset the load takes the largest visible card and
   adds the host for the experts that do not fit it; on GLM-5.3 it also adds the next card as an expert tier when
   the plan finds that pays.
5. **Warm up first.** Send one request and discard its numbers. Adaptive residency then moves the experts the model
   calls most onto the card, so the first requests run slower than later ones. `BLOOMERY_RESIDENCY=off` gives
   repeatable tokens at some speed.
6. **Read the speed from the response's `timings`**, not from wall time that includes the download or the load:
   `predicted_per_second` (decode), `prompt_per_second` (prefill), `draft_n` and `draft_n_accepted` (the MTP draft).
7. **Compare with llama-server on equal terms**: the same GGUF file, the same `--ctx-size`, the same `--parallel`
   (`-np`), and the same request body (messages, `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`,
   `chat_template_kwargs`). Time one request alone for single-stream speed, and two at once for the total.
8. **Use the model card's sampling.** Qwen3.8 without thinking: `"temperature": 0.7, "top_p": 0.8, "top_k": 20,
   "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}`. With thinking on, the answer carries
   `reasoning_content`.
9. **Agents.** Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8080`. OpenAI clients: base URL
   `http://localhost:8080/v1`. Tool calls work on every generative model.
10. **When something fails**, the error names the cause (a plan that does not fit lists every term and the
    processes holding the card). Open an issue with `bloomery-serve --version`, `nvidia-smi`, the command and the log.

`AGENTS.md`, the `justfile`'s `gate-*` recipes and `tools/` are the maintainers' development contract. They run on
the maintainers' workstation with its cards and reference dumps, and they are not a test suite for an evaluator.

## Help wanted: hardware we have not run

We test on Ampere cards (RTX 3090, RTX A6000, RTX 3060) and one AMD CPU. If you have an Ada or Blackwell card (RTX
40xx or 50xx, L40S, RTX PRO 6000), an Intel host, or a multi-card machine, please run the checklist above and file a
[hardware report](https://github.com/midagedev/bloomery/issues/new?template=hardware-report.md). A report that it
worked helps as much as one that it failed.
