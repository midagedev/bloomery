//! EmbeddingGemma 2 beside Qwen3-30B-A3B-Instruct-2507 Q4_K_M: the server vectors,
//! the reserve's expert cost and unified-pool term, greedy tokens alone and during
//! embedding calls, and named refusals. One visible card keeps the side seat on the stage.
//! The plain 12 GiB plan leaves host experts; adding the reserve to the side run's
//! budget restores that split. At the unchanged budget the side run must seat fewer experts.
//! The same file beside the nemotron_h fixture: at one context the stage card's need grows by
//! the side seat's reserve, and `/v1/embeddings` answers.

#[cfg(feature = "gemma-embedding2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_serve", gate::run())
}

#[cfg(feature = "gemma-embedding2")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate uses the seat's planner, not its loaders"
)]
mod q3place;

#[cfg(feature = "gemma-embedding2")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Duration;

    use bloomery_gpu::arch::qwen3moe::KvQ8;
    use bloomery_gpu_gates::beside::{self, Beside};
    use bloomery_gpu_gates::generate::{Place, SEAT_STEPS};
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Served, check, curl, ids_of, json_of, metric, plan_run, serve_cmd, server_log, tokenized,
    };
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, gpu_census, ref_model_path,
        ref_tensor_logical_in,
    };
    use gguf::Split;
    use model::placement::{Plan, PlanLevers, UnifiedPool};
    use refset::arch::gemma_embedding2::{
        IDS_SET, MMPROJ_BF16, MMPROJ_BF16_SHA256, MODEL, MODEL_SHA256, REL_FLOOR, SERVER, TEXT_CPU,
        TWIN, VIDEO, VIDEO_CPU, VIDEO_CPU_SET, VIDEO_PREFIX, VIDEO_SET,
    };
    use refset::emb2::Emb2Set;
    use serde_json::{Value, json};

    use crate::q3place::PlaceQ3;

    const CTX: usize = 4096;
    const BUDGET: u64 = 12 * 1024 * 1024 * 1024;
    const N_PREDICT: usize = 512;
    const PROMPT: &str = "The capital of France is";

    struct Case {
        name: String,
        text: String,
        tokens: u64,
    }

    fn cases(man: &RefManifest, ids_set: &str) -> Result<Vec<Case>, GateError> {
        let texts = refset::tokenizer::TokenizerSet::read(&data_dir().join(ids_set), &SERVER)?;
        texts.check_texts()?;
        let mut cases = Vec::new();
        for line in &man.header.other {
            let Some(rest) = line.strip_prefix("# tokenize\t") else {
                continue;
            };
            let fields: Vec<_> = rest.split('\t').collect();
            let [name, "equal", count] = fields[..] else {
                return Err(format!("server set: malformed tokenization identity {line:?}").into());
            };
            let tokens = count
                .strip_prefix("prompt_tokens ")
                .ok_or_else(|| format!("server set: missing prompt_tokens in {line:?}"))?
                .parse()?;
            cases.push(Case {
                name: name.to_owned(),
                text: String::from_utf8(
                    texts
                        .texts
                        .iter()
                        .find(|text| text.name == name)
                        .ok_or_else(|| format!("server set: no tokenizer text {name}"))?
                        .read()?,
                )?,
                tokens,
            });
        }
        if cases.is_empty() {
            return Err("server set: no text cases".into());
        }
        cases.sort_by_key(|c| c.tokens);
        Ok(cases)
    }

    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    fn vector(v: &Value) -> Result<Vec<f32>, GateError> {
        v["data"][0]["embedding"]
            .as_array()
            .ok_or("no embedding array")?
            .iter()
            .map(|x| {
                let x = x.as_f64().ok_or("an embedding value is not numeric")? as f32;
                if !x.is_finite() {
                    return Err("an embedding value is not finite".into());
                }
                Ok(x)
            })
            .collect()
    }

    fn norm(v: &[f32]) -> f64 {
        v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt()
    }

    fn distances(a: &[f32], b: &[f32]) -> (f64, f64) {
        if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|x| !x.is_finite()) {
            return (f64::INFINITY, f64::INFINITY);
        }
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let cosine = (1.0 - dot / (norm(a) * norm(b))).abs();
        let max = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max);
        (cosine, max)
    }

    fn within(ours: &[f32], want: &[f32], raw: &[f32], twin: &[f32], normalized: bool) -> bool {
        let delta: Vec<_> = raw.iter().zip(twin).map(|(&x, &y)| x - y).collect();
        // The encoder gate's pooled floor, before normalization: the bands below derive from it,
        // not from measured serving distances.
        let relative = REL_FLOOR.max(TWIN * norm(&delta) / norm(raw));
        let absolute = relative * norm(raw);
        // Cutting can reduce the norm. Normalization magnifies an absolute error E by at most
        // 2 E / (norm - E); cosine distance between unit vectors is half their squared L2 distance.
        let cut_norm = norm(&raw[..want.len()]);
        let unit = 2.0 * absolute / (cut_norm - absolute);
        let max_band = if normalized { unit } else { absolute };
        let cosine_band = unit * unit / 2.0;
        let (cosine, max) = distances(ours, want);
        println!(
            "cosine {cosine:.6e} band {cosine_band:.6e}; max_abs {max:.6e} band {max_band:.6e} [derived bands]"
        );
        relative.is_finite()
            && cut_norm > absolute
            && cosine.is_finite()
            && max.is_finite()
            && cosine <= cosine_band
            && max <= max_band
    }

    fn embedding(url: &dyn Fn(&str) -> String, body: Value) -> Result<Value, GateError> {
        let (status, text) = curl(&url("/v1/embeddings"), Some(&body), false)?;
        json_of("/v1/embeddings", status, &text)
    }

    fn vectors(
        url: &dyn Fn(&str) -> String,
        cases: &[Case],
        server: &RefManifest,
        cpu: &RefManifest,
        model: &str,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let name = Path::new(model)
            .file_name()
            .ok_or("embedding file name")?
            .to_string_lossy();
        for (index, c) in cases.iter().enumerate() {
            let raw = tap(server, &format!("{}/raw", c.name))?;
            let twin = tap(cpu, &format!("{}/pooled", c.name))?;
            if raw.len() != 768 || twin.len() != raw.len() {
                return Err(format!("{}: reference vector widths are not 768", c.name).into());
            }
            for (width, normalized, row) in [
                (768, false, "raw"),
                (768, true, "embd"),
                (128, true, "mrl128"),
                (256, true, "mrl256"),
                (512, true, "mrl512"),
            ] {
                if index != 0 && width != 768 {
                    continue;
                }
                let body = json!({"input": c.text, "model": name,
                    "dimensions": width, "embd_normalize": if normalized { 2 } else { -1 }});
                let v = embedding(url, body)?;
                let ours = vector(&v)?;
                let want = tap(server, &format!("{}/{row}", c.name))?;
                if want.len() != width {
                    return Err(format!("{}: {row} width", c.name).into());
                }
                check(
                    &mut ok,
                    &format!("vectors_{}_{row}", c.name),
                    v["model"] == name.as_ref()
                        && v["usage"]["prompt_tokens"] == c.tokens
                        && within(&ours, &want, &raw, &twin, normalized),
                );
            }
        }
        Ok(ok)
    }

    fn cache_images(embedding: &str) -> Result<bool, GateError> {
        use bloomery_gpu_gates::side_seat::{Flags, Side};
        use serve::embed::Embed;
        use serve::embed_cache::{Value as Cached, image_key};

        let flags: Vec<String> = [
            "--embd-model",
            embedding,
            "--embd-mmproj",
            MMPROJ_BF16,
            "--embd-image-min-tokens",
            "1",
            "--embd-image-max-tokens",
            "70",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let flags = Flags::take(&flags)?.0;
        let side = flags.read()?.ok_or("no image side seat")?;
        let seat = side.seat(Place::A)?;
        let mut model = Side::start(Some((side, seat.clone())))?.ok_or("no encoder")?;
        let media = model.media().ok_or("no image media model")?;
        let tokenizer = model.tokenizer();
        let file = serve::media::load(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg==",
        )?;
        let prepared = std::sync::Arc::new(serve::media::prepare(&file, media.as_ref())?);
        let (begin, end) = media
            .embedding_delimiters()
            .ok_or("no embedding image delimiters")?;
        use serve::embeddings::InputPart;
        let image = || InputPart::Media {
            begin,
            end,
            positions: prepared.span_len,
            kind: "image",
        };
        let mut prompts = Vec::new();
        let mut baseline = Vec::new();
        for parts in [
            vec![InputPart::Text("Describe the coastal map."), image()],
            vec![
                image(),
                InputPart::Text("Retrieve a photograph of the harbor."),
            ],
        ] {
            let assembled = serve::embeddings::assemble(
                parts.into_iter().map(Ok),
                tokenizer.as_ref(),
                model.ctx(),
            )?;
            let at = assembled.spans[0].start;
            let prompt = serve::media::Prompt {
                held: assembled.ids.into(),
                feeds: vec![serve::media::MediaFeed {
                    at,
                    key: file.key,
                    prepared: prepared.clone(),
                }],
            };
            baseline.push(model.embed_media(&prompt.held.ids, &prompt.feeds, &[])?);
            prompts.push(prompt);
        }
        drop(model);
        let mut side = flags.read()?.ok_or("no image side seat")?;
        side.cache_bytes = 64 * 1024 * 1024;
        let mut model = Side::start(Some((side, seat)))?.ok_or("no cached encoder")?;
        let cache = model.cache().ok_or("image cache is disabled")?;
        let mut ok = true;
        check(
            &mut ok,
            "cache_image_position_changed",
            prompts[0].feeds[0].at != prompts[1].feeds[0].at
                && image_key(&prompts[0].feeds[0]) == image_key(&prompts[1].feeds[0]),
        );
        for (index, prompt) in prompts.iter().enumerate() {
            let ours = model.embed_media(&prompt.held.ids, &prompt.feeds, &[])?;
            check(
                &mut ok,
                &format!("cache_image_{index}_bits"),
                bloomery_gpu_gates::bits_equal(&ours, &baseline[index]),
            );
        }
        let prompt = &prompts[1];
        let key = image_key(&prompt.feeds[0]).ok_or("image has no cache key")?;
        let Some(Cached::Image(rows)) = cache.lock().map_err(|_| "cache lock poisoned")?.get(&key)
        else {
            return Err("image tower rows were not cached".into());
        };
        // Direct side calls bypass pooled vectors, so only a tower hit can consume these rows.
        cache
            .lock()
            .map_err(|_| "cache lock poisoned")?
            .insert(key, Cached::Image(vec![0; rows.len()]));
        let changed = model.embed_media(&prompt.held.ids, &prompt.feeds, &[])?;
        check(
            &mut ok,
            "cache_image_rows_consumed",
            !bloomery_gpu_gates::bits_equal(&changed, &baseline[1]),
        );
        cache
            .lock()
            .map_err(|_| "cache lock poisoned")?
            .insert(key, Cached::Image(rows));
        let restored = model.embed_media(&prompt.held.ids, &prompt.feeds, &[])?;
        check(
            &mut ok,
            "cache_image_restored_bits",
            bloomery_gpu_gates::bits_equal(&restored, &baseline[1]),
        );
        Ok(ok)
    }

    fn videos(url: &dyn Fn(&str) -> String, manifest: &Path) -> Result<bool, GateError> {
        let root = manifest.parent().ok_or("video manifest needs a parent")?;
        let clips: Value = serde_json::from_str(&std::fs::read_to_string(manifest)?)?;
        let clips = clips.as_array().ok_or("video manifest must be an array")?;
        if clips.len() != 3 {
            return Err("video gate requires three oracle clips".into());
        }
        let mut ok = true;
        let mut names = std::collections::HashSet::new();
        for clip in clips {
            let name = clip["name"].as_str().ok_or("clip needs a name")?;
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err("invalid clip name".into());
            }
            if !names.insert(name) {
                return Err(format!("{name}: duplicate video clip").into());
            }
            let card = Emb2Set::open(&root.join(name).join(VIDEO_SET), &VIDEO)?.man;
            let cpu = Emb2Set::open(&root.join(name).join(VIDEO_CPU_SET), &VIDEO_CPU)?.man;
            let images = clip["images"]
                .as_array()
                .ok_or("clip needs selected images")?;
            let mut content = vec![json!({"type":"text", "text":VIDEO_PREFIX})];
            for (index, image) in images.iter().enumerate() {
                let path = image.as_str().ok_or("frame path must be a string")?;
                let identity = format!(
                    "# image\tframe_{index}\t{path}\t{}",
                    gguf::digest::sha256_file(Path::new(path))?
                );
                if !card.header.other.contains(&identity) || !cpu.header.other.contains(&identity) {
                    return Err(format!("{name}: frame {index} differs from the oracle").into());
                }
                let data = serve::media::encode_base64(&std::fs::read(path)?);
                content.push(json!({"type":"image_url", "image_url":{"url":format!("data:image/png;base64,{data}")}}));
            }
            let raw = tap(&card, "video/raw")?;
            let twin = tap(&cpu, "video/raw")?;
            if raw.len() != 768 || twin.len() != raw.len() {
                return Err(format!("{name}: reference vector widths are not 768").into());
            }
            for (normalized, row) in [(false, "raw"), (true, "embd")] {
                let body = json!({"input":{"content":content}, "embd_normalize":if normalized {2} else {-1}});
                let v = embedding(url, body)?;
                let ours = vector(&v)?;
                let want = tap(&card, &format!("video/{row}"))?;
                let tokens = v["usage"]["prompt_tokens"]
                    .as_u64()
                    .ok_or("missing prompt_tokens")?;
                let identity = format!(
                    "# video\tvideo\tframes {}\tprompt_tokens {tokens}",
                    images.len()
                );
                println!(
                    "video {name}/{row}: frames {} positions {tokens}",
                    images.len()
                );
                check(
                    &mut ok,
                    &format!("video_{name}_{row}"),
                    want.len() == 768
                        && card.header.other.contains(&identity)
                        && cpu.header.other.contains(&identity)
                        && within(&ours, &want, &raw, &twin, normalized),
                );
            }
        }
        // A clip that is not a video is a named 400 from the decoder (ffprobe's refusal, or the
        // absent ffmpeg's, which names the decoded-frames route).
        for (ty, part) in [
            (
                "input_video",
                json!({"type":"input_video","input_video":{"data":"AAAA","format":"mp4"}}),
            ),
            (
                "video_url",
                json!({"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}),
            ),
        ] {
            let body = json!({"input":{"content":[part]}});
            let (status, text) = curl(&url("/v1/embeddings"), Some(&body), false)?;
            let value: Value = serde_json::from_str(&text)?;
            check(
                &mut ok,
                &format!("{ty}_refused"),
                status == 400
                    && value["error"]["type"] == "invalid_request_error"
                    && value["error"]["message"].as_str().is_some_and(|message| {
                        message.starts_with("ffprobe: ")
                            || message.contains("decoded one-FPS image_url frames")
                    }),
            );
        }
        Ok(ok)
    }

    fn cmd(budget: u64) -> Result<Command, GateError> {
        let mut cmd = serve_cmd()?;
        cmd.env(bloomery_levers::CARD_BUDGET, budget.to_string())
            .env(bloomery_levers::DRAFT, "off");
        Ok(cmd)
    }

    fn args<'a>(model: &'a str, side: bool, embedding: &'a str) -> Vec<&'a str> {
        let mut args = vec![
            "-m",
            model,
            "--port",
            "0",
            "--place",
            "a",
            "--ctx",
            "4096",
            "--parallel",
            "1",
        ];
        if side {
            args.extend(["--embd-model", embedding]);
        }
        args
    }

    fn log(path: &Path) -> Result<Log, GateError> {
        server_log(path, &[&record::PLAN38, &record::EMBEDDING_SEAT])
    }

    fn split(log: &Log) -> Result<(u64, u64), GateError> {
        let plan = log.first(&record::PLAN38)?.ok_or("missing plan record")?;
        Ok((plan.u64("card_experts")?, plan.u64("host_experts")?))
    }

    fn plan(
        model: &str,
        embedding: &str,
        dir: &Path,
        side: bool,
        budget: u64,
    ) -> Result<Log, GateError> {
        std::fs::create_dir_all(dir)?;
        let err = dir.join("plan.err");
        let (status, _) = plan_run(cmd(budget)?, &args(model, side, embedding), &err)?;
        if !status.success() {
            return Err(format!("plan failed ({status}): {}", err.display()).into());
        }
        log(&err)
    }

    fn pool(model: &Path, reserve: u64) -> Result<bool, GateError> {
        let split = Split::open(model)?;
        let census = gpu_census::census()?;
        // The plain plan is planned, not edited: a plan's card totals carry its selected reserves.
        let place = |embedding: Option<u64>| {
            PlaceQ3::qwen3_on(
                &split,
                Place::A,
                &census,
                CTX,
                KvQ8::F16,
                Beside {
                    image: None,
                    embedding,
                },
            )
        };
        let levers = |budget: u64| PlanLevers {
            card_budget_bytes: Some(budget).into(),
        };
        let (q, q0) = (place(Some(reserve))?, place(None)?);
        let p = q.plan(CTX, &levers(BUDGET + reserve))?;
        let p0 = q0.plan(CTX, &levers(BUDGET))?;
        // Exercise the pool formula on discrete cards too, keeping the real plans' rows and totals.
        let pooled = |plan: &Plan| {
            let mut held = plan.machine.clone();
            if held.nodes[0].unified.is_none() {
                held.nodes[0].unified = Some(UnifiedPool {
                    bytes: held.host().usable_bytes,
                });
            }
            held
        };
        let (held, plain) = (pooled(&p), pooled(&p0));
        let with = Plan {
            machine: &held,
            ..p.clone()
        }
        .pool_need()
        .ok_or("no pool term")?;
        let without = Plan {
            machine: &plain,
            ..p0.clone()
        }
        .pool_need()
        .ok_or("no pool term")?;
        println!(
            "pool cards with {} B, without {} B, reserve {reserve} B",
            with.cards, without.cards
        );
        Ok(beside::reserve_in(p.machine).embedding == Some(reserve)
            && with.cards.checked_sub(without.cards) == Some(reserve)
            && with.host == without.host)
    }

    fn chat(url: &dyn Fn(&str) -> String, ids: &[u32]) -> Result<Vec<u32>, GateError> {
        let body = json!({"prompt": ids, "n_predict": N_PREDICT, "temperature": 0,
            "ignore_eos": true, "return_tokens": true, "cache_prompt": false});
        let (status, text) = curl(&url("/completion"), Some(&body), false)?;
        ids_of(&json_of("/completion", status, &text)?["tokens"])
    }

    fn refusal(
        url: &dyn Fn(&str) -> String,
        body: Value,
        status: u16,
        kind: &str,
    ) -> Result<bool, GateError> {
        let (got, text) = curl(&url("/v1/embeddings"), Some(&body), false)?;
        let v: Value = serde_json::from_str(&text)?;
        Ok(got == status && v["error"]["type"] == kind && v["error"]["message"].is_string())
    }

    fn concurrent(
        url: &dyn Fn(&str) -> String,
        ids: &[u32],
        c: &Case,
        want: &[u32],
        server: &RefManifest,
        cpu: &RefManifest,
    ) -> Result<bool, GateError> {
        let completion_url = url("/completion");
        let ids = ids.to_vec();
        let raw = tap(server, &format!("{}/raw", c.name))?;
        let twin = tap(cpu, &format!("{}/pooled", c.name))?;
        let oracle = tap(server, &format!("{}/embd", c.name))?;
        let start = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
        std::thread::scope(|scope| {
            let generating =
                scope.spawn(|| chat(&|_| completion_url.clone(), &ids).map_err(|e| e.to_string()));
            let mut active = false;
            for _ in 0..200 {
                if generating.is_finished() {
                    break;
                }
                if metric(url, "n_decode_total")?.unwrap_or(start) > start {
                    active = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let mut decode_steps = 0.0_f64;
            for _ in 0..32 {
                if !active || generating.is_finished() {
                    break;
                }
                let before = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
                let v = embedding(url, json!({"input": c.text}))?;
                let after = metric(url, "n_decode_total")?.ok_or("missing n_decode_total")?;
                if !within(&vector(&v)?, &oracle, &raw, &twin, true) {
                    return Err("concurrent embedding is outside the server vector band".into());
                }
                decode_steps = decode_steps.max(after - before);
                if decode_steps >= 2.0 {
                    break;
                }
            }
            let got = generating
                .join()
                .map_err(|_| "chat request panicked")?
                .map_err(|e| format!("chat request: {e}"))?;
            println!(
                "concurrent decode observed {decode_steps} steps during one {}-token embedding; {} greedy tokens",
                c.tokens,
                got.len()
            );
            Ok(decode_steps >= 2.0 && got == want)
        })
    }

    fn audio_run(dir: &Path, model: &str) -> Result<(), GateError> {
        use refset::arch::gemma_embedding2::{AUDIO_SERVER, AUDIO_SERVER_CPU};
        let server = Emb2Set::open(&AUDIO_SERVER.path(AUDIO_SERVER.sets[0]), &AUDIO_SERVER)?.man;
        let cpu = Emb2Set::open(
            &AUDIO_SERVER_CPU.path(AUDIO_SERVER_CPU.sets[0]),
            &AUDIO_SERVER_CPU,
        )?
        .man;
        let mut argv = args(model, true, MODEL);
        argv.extend(["--embd-mmproj", MMPROJ_BF16]);
        let image = bloomery_gpu_gates::vision_seat::Projector::open(
            Path::new(MMPROJ_BF16),
            &bloomery_gpu_gates::vision_seat::Flags::default(),
            &["gemma4v"],
        )?
        .card
        .total();
        let text = bloomery_gpu_gemma_embedding2::card_bytes(&Split::open(MODEL)?, 8192)?;
        let audio = bloomery_gpu_gemma4a::arch::gemma4a::card_bytes();
        let reserve = text + image + audio;
        let mut ok = true;
        check(&mut ok, "audio_reserve_bytes", audio == 1_347_032_320);
        let err = dir.join("audio-plan.err");
        let (status, _) = plan_run(cmd(BUDGET)?, &argv, &err)?;
        if !status.success() {
            return Err(format!("audio plan failed: {status}").into());
        }
        let planned = log(&err)?;
        let seat = planned
            .first(&record::EMBEDDING_SEAT)?
            .ok_or("no embedding seat record")?;
        check(
            &mut ok,
            "audio_named_reserve",
            seat.u64("reserve")? == reserve
                && seat.u64("bytes")? == reserve
                && pool(Path::new(model), reserve)?,
        );
        println!("audio reserve {reserve} B = text {text} + image {image} + audio {audio}");
        let mut served = Served::spawn_cmd(cmd(BUDGET + reserve)?, &argv, dir)?;
        let addr = served.address(&dir.join("server.err"), 600, Duration::from_secs(1))?;
        let url = |path: &str| format!("http://{addr}{path}");
        let mut count = 0;
        let mut mixed = false;
        for line in &server.header.other {
            let Some(rest) = line.strip_prefix("# audio\t") else {
                continue;
            };
            if !cpu.header.other.contains(line) {
                return Err("audio CPU twin input identity differs".into());
            }
            let fields: Vec<_> = rest.split('\t').collect();
            let [name, path, digest, tokens, prefix] = fields[..] else {
                return Err("malformed audio identity".into());
            };
            if gguf::digest::sha256_file(Path::new(path))? != digest {
                return Err(format!("{name}: PCM sha256 differs").into());
            }
            let bytes = std::fs::read(path)?;
            let audio = json!({"type":"input_audio","input_audio":{"format":"f32","sample_rate":16000,"channels":1,"data":serve::media::encode_base64(&bytes)}});
            let mut parts = Vec::new();
            let prefix: Value = serde_json::from_str(prefix)?;
            if !prefix.is_null() {
                let text = prefix
                    .as_str()
                    .ok_or("audio prefix must be a string or null")?;
                parts.push(json!({"type":"text","text":text}));
            }
            parts.push(audio);
            if name == "aud_sine" {
                mixed = true;
                let image = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg=="}});
                let changed_image = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAC0lEQVR4nGP4DwYAFPIF+6QNfF4AAAAASUVORK5CYII="}});
                let audio = parts.last().ok_or("no audio part")?;
                let changed_pcm = vec![0; bytes.len()];
                let changed_audio = json!({"type":"input_audio","input_audio":{"format":"f32","sample_rate":16000,"channels":1,"data":serve::media::encode_base64(&changed_pcm)}});
                let image_only = embedding(&url, json!({"input":{"content":[image]}}))?;
                let audio_only = embedding(&url, json!({"input":{"content":[audio]}}))?;
                let empty = embedding(&url, json!({"input":""}))?;
                let count_of = |v: &Value| {
                    v["usage"]["prompt_tokens"]
                        .as_u64()
                        .ok_or("no prompt count")
                };
                let positions = count_of(&image_only)? + count_of(&audio_only)? - count_of(&empty)?;
                for (order, content, changed, changed_audio) in [
                    (
                        "image_audio",
                        json!([image, audio]),
                        json!([changed_image, audio]),
                        json!([image, changed_audio]),
                    ),
                    (
                        "audio_image",
                        json!([audio, image]),
                        json!([audio, changed_image]),
                        json!([changed_audio, image]),
                    ),
                ] {
                    let body = json!({"input":{"content":content}});
                    let first = embedding(&url, body.clone())?;
                    let again = embedding(&url, body)?;
                    let rows = vector(&first)?;
                    check(
                        &mut ok,
                        &format!("mixed_{order}_vector_and_count"),
                        rows.len() == 768
                            && norm(&rows) > 0.0
                            && rows == vector(&again)?
                            && count_of(&first)? == positions
                            && count_of(&again)? == positions,
                    );
                    let changed = embedding(&url, json!({"input":{"content":changed}}))?;
                    check(
                        &mut ok,
                        &format!("mixed_{order}_image_content"),
                        vector(&changed)? != rows && count_of(&changed)? == positions,
                    );
                    let changed = embedding(&url, json!({"input":{"content":changed_audio}}))?;
                    check(
                        &mut ok,
                        &format!("mixed_{order}_audio_content"),
                        vector(&changed)? != rows && count_of(&changed)? == positions,
                    );
                }
            }
            let raw = tap(&server, &format!("{name}/raw"))?;
            let twin = tap(&cpu, &format!("{name}/raw"))?;
            if raw.len() != 768 || twin.len() != 768 {
                return Err(format!("{name}: audio reference width is not 768").into());
            }
            for (row, normalized) in [("raw", false), ("embd", true)] {
                let v = embedding(
                    &url,
                    json!({"input":[{"content":parts}],"embd_normalize":if normalized {2} else {-1}}),
                )?;
                let ours = vector(&v)?;
                let want = tap(&server, &format!("{name}/{row}"))?;
                if want.len() != 768 {
                    return Err(format!("{name}: {row} width is not 768").into());
                }
                let n: u64 = tokens
                    .strip_prefix("prompt_tokens ")
                    .ok_or("no audio prompt count")?
                    .parse()?;
                check(
                    &mut ok,
                    &format!("audio_{name}_{row}"),
                    ours.len() == 768
                        && v["usage"]["prompt_tokens"] == n
                        && within(&ours, &want, &raw, &twin, normalized),
                );
                count += 1;
            }
        }
        check(&mut ok, "audio_cases_present", count >= 4 && mixed);
        let (status, text) = curl(
            &url("/v1/embeddings"),
            Some(
                &json!({"input":[{"content":[{"type":"input_audio","input_audio":{"format":"wav","data":"encoded"}}]}]}),
            ),
            false,
        )?;
        check(
            &mut ok,
            "encoded_audio_refused",
            status == 400 && text.contains("encoded audio is not supported"),
        );
        println!("audio server stopped {}", served.stop()?);
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The nemotron_h fixture beside EmbeddingGemma 2, on the one card this gate already requires.
    /// The fixture is dense, so two plans at [`SEAT_STEPS`]'s floor differ by the side seat's
    /// stage reserve, and `/v1/embeddings` answers with that file's name and a finite vector of
    /// its width. raon1 has no fixture file.
    fn nemotron_embeddings(
        dir: &Path,
        embedding: &str,
        arch: &str,
        embd_ctx: u64,
    ) -> Result<bool, GateError> {
        let fixture = refset::fixture::first_shard("nemotron_h")?;
        let ctx = SEAT_STEPS.floor.to_string();
        let plain = nemotron_plan(
            &fixture,
            embedding,
            &ctx,
            &dir.join("nemotron-plain"),
            false,
        )?;
        let reserved = nemotron_plan(&fixture, embedding, &ctx, &dir.join("nemotron-side"), true)?;
        let seat = reserved
            .first(&record::EMBEDDING_SEAT)?
            .ok_or("nemotron plan: no embedding_seat record")?;
        let reserve = seat.u64("reserve")?;
        let plain_b = stage_bytes(&plain)?;
        let side_b = stage_bytes(&reserved)?;
        println!(
            "nemotron need plain {plain_b} B, with the side seat {side_b} B, reserve {reserve} B"
        );
        let counted = reserve > 0
            && seat.word("seat")? == "stage-reserve"
            && seat.word("arch")? == arch
            && reserve == seat.u64("bytes")?
            && seat.u64("ctx")? == embd_ctx
            && plain
                .first(&record::PLAN)?
                .ok_or("nemotron plan: missing plan record")?
                .u64("ctx_max")?
                == SEAT_STEPS.floor
            && reserved
                .first(&record::PLAN)?
                .ok_or("nemotron plan: missing plan record")?
                .u64("ctx_max")?
                == SEAT_STEPS.floor
            && side_b
                == plain_b
                    .checked_add(reserve)
                    .ok_or("nemotron need bytes overflow")?;
        let live = nemotron_live(dir, &fixture, embedding, &ctx)?;
        Ok(counted && live)
    }

    fn nemotron_args<'a>(
        model: &'a str,
        embedding: &'a str,
        ctx: &'a str,
        side: bool,
    ) -> Vec<&'a str> {
        let mut args = vec![
            "--model",
            "nemotron",
            "-m",
            model,
            "--port",
            "0",
            "--place",
            "a",
            "--ctx",
            ctx,
            "--parallel",
            "1",
        ];
        if side {
            args.extend(["--embd-model", embedding]);
        }
        args
    }

    fn nemotron_plan(
        model: &str,
        embedding: &str,
        ctx: &str,
        dir: &Path,
        side: bool,
    ) -> Result<Log, GateError> {
        std::fs::create_dir_all(dir)?;
        let err = dir.join("plan.err");
        let (status, _) = plan_run(
            serve_cmd()?,
            &nemotron_args(model, embedding, ctx, side),
            &err,
        )?;
        if !status.success() {
            return Err(format!("nemotron plan failed ({status}): {}", err.display()).into());
        }
        server_log(
            &err,
            &[&record::PLAN, &record::NEED, &record::EMBEDDING_SEAT],
        )
    }

    fn stage_bytes(log: &Log) -> Result<u64, GateError> {
        let need = log
            .first(&record::NEED)?
            .ok_or("nemotron plan: missing need record")?;
        let cards = need.csv("cards")?;
        let [one] = cards[..] else {
            return Err(format!("nemotron plan: need names {} cards", cards.len()).into());
        };
        let (_, bytes) = one
            .rsplit_once(':')
            .ok_or("nemotron plan: need cards is not name:bytes")?;
        Ok(bytes.parse()?)
    }

    fn nemotron_live(
        dir: &Path,
        model: &str,
        embedding: &str,
        ctx: &str,
    ) -> Result<bool, GateError> {
        let work = dir.join("nemotron-live");
        std::fs::create_dir_all(&work)?;
        let mut served = Served::spawn_cmd(
            serve_cmd()?,
            &nemotron_args(model, embedding, ctx, true),
            &work,
        )?;
        let addr = served.address(&work.join("server.err"), 600, Duration::from_secs(1))?;
        let (status, text) = curl(
            &format!("http://{addr}/v1/embeddings"),
            Some(&json!({"input":"a"})),
            false,
        )?;
        println!("nemotron /v1/embeddings HTTP {status}");
        let live = if status == 200 {
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| format!("nemotron /v1/embeddings: {e}: {text}"))?;
            let rows = vector(&v)?;
            let name = Path::new(embedding)
                .file_name()
                .ok_or("embedding file name")?
                .to_string_lossy();
            rows.len() == 768 && v["model"] == name.as_ref()
        } else {
            false
        };
        println!("nemotron server stopped {}", served.stop()?);
        Ok(live)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
        record::at_main(
            "gate_gemma_embedding2_serve",
            record::GATE_GEMMA_EMBEDDING2_SERVE,
        );
        let mut args_it = std::env::args().skip(1);
        let mut dir = PathBuf::from("target/embedding2-serve-gate");
        let mut v1 = false;
        let mut clips = None;
        let mut audio = false;
        let mut images = false;
        while let Some(a) = args_it.next() {
            match a.as_str() {
                "--v1" if !v1 => v1 = true,
                "--audio" if !audio => audio = true,
                "--cache-images" if !images => images = true,
                "--dir" => dir = args_it.next().ok_or("--dir needs DIR")?.into(),
                "--video-clips" if clips.is_none() => {
                    clips = Some(PathBuf::from(args_it.next().ok_or("--video-clips needs clips.json")?));
                }
                _ => return Err(format!("unknown or duplicate argument {a}; takes --v1, --audio, --cache-images, --dir DIR and --video-clips clips.json").into()),
            }
        }
        if [v1, clips.is_some(), audio, images]
            .into_iter()
            .filter(|v| *v)
            .count()
            > 1
        {
            return Err(
                "--v1, --audio, --cache-images and --video-clips are mutually exclusive".into(),
            );
        }
        use refset::arch::gemma_embedding as emb1;
        let (embedding, digest, server_family, cpu_family, ids_set, arch, ctx) = if v1 {
            (
                emb1::MODEL,
                emb1::MODEL_SHA256,
                &emb1::SERVER,
                &emb1::TEXT_CPU,
                emb1::TOKENIZER.sets[0],
                emb1::ARCH,
                model::arch::gemma_embedding::hparams::CARD_CTX,
            )
        } else {
            (
                MODEL,
                MODEL_SHA256,
                &SERVER,
                &TEXT_CPU,
                IDS_SET,
                "gemma-embedding2",
                8192,
            )
        };
        std::fs::create_dir_all(&dir)?;
        if gpu_census::census()?.len() != 1 {
            return Err(
                "serve gate needs one visible card, so the side seat reserves the stage card"
                    .into(),
            );
        }
        if gguf::digest::sha256_file(Path::new(embedding))? != digest {
            return Err("embedding model sha256 differs from the server set".into());
        }
        if images {
            if gguf::digest::sha256_file(Path::new(MMPROJ_BF16))? != MMPROJ_BF16_SHA256 {
                return Err("embedding projector sha256 differs from the oracle".into());
            }
            return if cache_images(embedding)? {
                Ok(())
            } else {
                Err(checks_failed())
            };
        }
        let model_path = ref_model_path()?;
        let model = model_path.to_str().ok_or("chat model path is not UTF-8")?;
        if Split::open(model)?.architecture() != Some("qwen3moe") {
            return Err("serve gate needs Qwen3-30B-A3B-Instruct-2507 Q4_K_M".into());
        }
        if (audio || clips.is_some())
            && gguf::digest::sha256_file(Path::new(MMPROJ_BF16))? != MMPROJ_BF16_SHA256
        {
            return Err("embedding projector sha256 differs from the oracle".into());
        }
        if let Some(manifest) = clips {
            let mut launch = args(model, true, embedding);
            launch.extend(["--embd-mmproj", MMPROJ_BF16, "--embd-image-min-tokens", "1"]);
            let mut served = Served::spawn_cmd(cmd(BUDGET)?, &launch, &dir)?;
            let addr = served.address(&dir.join("server.err"), 600, Duration::from_secs(1))?;
            let url = |path: &str| format!("http://{addr}{path}");
            let ok = videos(&url, &manifest)?;
            println!("video server stopped {}", served.stop()?);
            return if ok { Ok(()) } else { Err(checks_failed()) };
        }
        if audio {
            return audio_run(&dir, model);
        }
        let server = Emb2Set::open(&server_family.path(server_family.sets[0]), server_family)?.man;
        let cpu = Emb2Set::open(&cpu_family.path(cpu_family.sets[0]), cpu_family)?.man;
        let cases = cases(&server, ids_set)?;
        let longest = cases.last().ok_or("server set: no text cases")?;
        let plain = plan(model, embedding, &dir.join("plan-plain"), false, BUDGET)?;
        let reserved = plan(model, embedding, &dir.join("plan-reserved"), true, BUDGET)?;
        let seat = reserved
            .first(&record::EMBEDDING_SEAT)?
            .ok_or("no embedding_seat record")?;
        let reserve = seat.u64("reserve")?;
        let compensated = plan(
            model,
            embedding,
            &dir.join("plan-compensated"),
            true,
            BUDGET + reserve,
        )?;
        let (base_card, base_host) = split(&plain)?;
        let (less_card, more_host) = split(&reserved)?;
        println!(
            "plain split {base_card}/{base_host}, reserved {less_card}/{more_host}, reserve {reserve} B"
        );
        let mut ok = true;
        check(
            &mut ok,
            "plan_expert_cost",
            base_card > 0
                && base_host > 0
                && reserve > 0
                && less_card < base_card
                && more_host > base_host,
        );
        check(
            &mut ok,
            "plan_same_split",
            split(&plain)? == split(&compensated)?,
        );
        check(
            &mut ok,
            "reserve_record",
            seat.word("arch")? == arch
                && reserve == seat.u64("bytes")?
                && seat.u64("ctx")? == ctx as u64
                && reserve
                    == bloomery_gpu_gemma_embedding2::card_bytes(&Split::open(embedding)?, ctx)?,
        );
        check(
            &mut ok,
            "pool_cards_count_reserve",
            pool(&model_path, reserve)?,
        );

        let mut baseline = Vec::new();
        for (side, budget, arm) in [(false, BUDGET, "plain"), (true, BUDGET + reserve, "side")] {
            let work = dir.join(arm);
            std::fs::create_dir_all(&work)?;
            let err = work.join("server.err");
            let mut served = Served::spawn_cmd(cmd(budget)?, &args(model, side, embedding), &work)?;
            let addr = served.address(&err, 600, Duration::from_secs(1))?;
            let url = |path: &str| format!("http://{addr}{path}");
            let ids = tokenized(&url, PROMPT)?;
            let tokens = chat(&url, &ids)?;
            std::fs::write(work.join("greedy.json"), serde_json::to_string(&tokens)?)?;
            check(
                &mut ok,
                &format!("{arm}_greedy_count"),
                tokens.len() == N_PREDICT,
            );
            check(
                &mut ok,
                &format!("{arm}_loaded_split"),
                split(&log(&err)?)? == split(&plain)?,
            );
            if !side {
                baseline = tokens;
                check(
                    &mut ok,
                    "no_side_501",
                    refusal(&url, json!({"input":"a"}), 501, "not_supported_error")?,
                );
            } else {
                check(&mut ok, "idle_greedy_same", tokens == baseline);
                ok &= vectors(&url, &cases, &server, &cpu, embedding)?;
                let (status, text) = curl(&url("/v1/models"), None, false)?;
                let models = json_of("/v1/models", status, &text)?;
                let chat_name = models["data"][0]["id"].as_str().ok_or("no chat model id")?;
                check(
                    &mut ok,
                    "embedding_model_listed",
                    models["data"][1]["id"]
                        == Path::new(embedding)
                            .file_name()
                            .ok_or("embedding file name")?
                            .to_string_lossy()
                            .as_ref()
                        && models["data"][1]["meta"]["n_embd"] == 768
                        && models["data"][1]["meta"]["n_ctx"] == ctx,
                );
                check(
                    &mut ok,
                    "chat_model_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "model": chat_name}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "unknown_model_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "model":"not-a-served-model"}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "dimensions_refused",
                    refusal(
                        &url,
                        json!({"input":"a", "dimensions":129}),
                        400,
                        "invalid_request_error",
                    )?,
                );
                check(
                    &mut ok,
                    "context_refused",
                    refusal(
                        &url,
                        json!({"input":vec![1;ctx + 1]}),
                        400,
                        "exceed_context_size_error",
                    )?,
                );
                check(
                    &mut ok,
                    "concurrent_greedy_same",
                    concurrent(&url, &ids, longest, &baseline, &server, &cpu)?,
                );
            }
            println!("{arm} server stopped {}", served.stop()?);
        }
        if !v1 {
            ok &= cache_images(embedding)?;
        }
        let err = dir.join("embedding-as-chat.err");
        let (status, lines) = plan_run(cmd(BUDGET)?, &["-m", embedding], &err)?;
        let text = lines.join("\n");
        check(
            &mut ok,
            "embedding_as_chat_refused",
            !status.success() && text.contains("embedding model") && text.contains("--embd-model"),
        );
        if !v1 {
            check(
                &mut ok,
                "nemotron_embeddings",
                nemotron_embeddings(&dir, embedding, arch, ctx as u64)?,
            );
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
