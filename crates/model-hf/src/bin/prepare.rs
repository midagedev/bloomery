//! prepare — a Hugging Face checkpoint folder as a GGUF split the server opens as it opens any
//! (`gguf::Split`). The folder's quantization config places each tensor in a format
//! (`hf::quant_config`), the model's map names it, says how its role reads it and moves it into
//! the program's layout (`model_hf::arch::qwen35moe::Checkpoint`, qwen4exp's and Qwen3.6's reader,
//! `model_hf::arch::qwen3moe::Checkpoint`, Qwen3-MoE's, or `model_hf::arch::kimi_k3::Checkpoint`,
//! Kimi K3's and K3.1's), and one table writes each format as
//! its GGUF type (`hf::modelopt::normalize`). The map is the one the config's `model_type` names,
//! among the maps `model_hf::arch::with_split_map` reads.
//!
//!     prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]
//!             [--drop <glob>]... [--only <glob>]... [--arch <word>]
//!             [--follow <list> [--follow-stall <seconds>] [--follow-poll <seconds>]] [--room <bytes>]
//!
//! writes `<out prefix>-00001-of-0000N.gguf` and on, each shard at most `n` GiB (48 by default)
//! unless one tensor with its scale and file layout is larger, the metadata in the first.
//! `--drop` leaves out the tensors whose GGUF name a glob matches (`*` any run of characters, `?`
//! one) and `--only` keeps those and no others; each flag may repeat, the two do not combine, and
//! a pattern that matches no tensor is refused by name. A split of a selection is checked by
//! opening it and holding its tensors to the plan's, where a whole split is read as the program
//! reads a model. `--arch <word>` writes the header under another of the map's architecture
//! words: a K3.1 folder as `kimi-k3` takes `--drop 'blk.*.indexer.*'` and has no indexer keys.
//! A quantized tensor keeps its stored format (NVFP4, MXFP8, FP8), and a BF16 matrix stays BF16,
//! the head and the token embedding among them; the server reads each through its BF16 site.
//! BF16 and F32 expert stacks are refused by name. An F32 matrix the map reads as Q8_0 is
//! refused unless `BLOOMERY_PREPARE_Q8=1` writes each single BF16 or F32 matrix read as Q8_0 as
//! ggml's reference Q8_0 instead, a lossy rewrite whose cost its lever row states; `--keep-bf16`
//! asks for the stored form and is refused beside it. Qwen3.8's decode reads about 9.27 GiB
//! of weights a token from the stored split where the Q8_0 one reads 5.73, the head alone
//! 0.55 GiB more
//! (1,271,398,400 bytes of BF16 against 34 bytes a 32 values). The output shard paths must
//! not exist. A private temporary directory holds the entire split until it passes the
//! server's reader; the first shard is published last. A persistent `<out prefix>.prepare.lock`
//! file serializes writers of that prefix. The disk must hold the split and a tenth more, or the
//! `--room <bytes>` a caller states when it frees the input behind the writer (`--follow` with a
//! deleter of the shards the writer has closed): the plan then refuses by name below that many
//! free bytes instead.
//! One line per tensor, a summary line, and exit status 1 with the error on any failure. Before
//! any shard is written, the tokenizer serving builds from the header is checked to split text
//! and stop generation where Hugging Face's does.
//!
//! `--follow <list>` writes the split behind the download that is filling the folder. The list
//! is the download's own, `path<TAB>bytes` a line (`model_hf::follow`); a shard is complete at
//! its listed bytes or with a `<shard>.done` sidecar beside it. The plan comes from the config
//! and the index (`FolderMap::stored`) before any shard is there, so the config, the index and
//! the files beside the shards (the tokenizer's) must be complete first: put them first in the
//! download. Each shard that completes is opened and its header held to the config's table, and
//! an output shard is written once every input shard it reads is complete, in whatever order
//! they finish. A tensor is made whole in memory and streamed to its output shard one at a
//! time: the largest, a layer's stacked routed experts, is one allocation of its length, so the
//! peak is that tensor and the page cache of the shard being read. Output shards are staged in
//! `<out prefix>.follow/`, each renamed into place and marked `<shard>.done` once its bytes are
//! synced: a rerun with the same arguments skips every marked shard, removes a partial one, and
//! refuses a stage another plan made. After the last shard the split is checked and published as
//! a whole. A wait refuses by name once no listed byte has arrived for `--follow-stall`
//! seconds (600 by default), polling every `--follow-poll` seconds (10 by default).
//!
//!     prepare <folder> --plan
//!
//! prints the plan of any folder a map reads (`model_hf::folder::open` picks the map by the
//! config's `model_type`) and writes nothing; it reads the headers and the tokenizer files only.
//! The plan is the tensors and bytes by where a load reads them and by type, the folder tensors
//! the map skips by reason, and the size of the vocabulary serving checked against Hugging Face's.
//! It is the stored form, which `--keep-bf16` names too; it takes no `--shard-gib` and refuses
//! `BLOOMERY_PREPARE_Q8=1`, whose rewrite it does not plan.

use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_levers::PREPARE_Q8;
use gguf::Value;
use gguf::write::{Layout, Writer};
use hf::modelopt::{self, Normal, Use};
use model_hf::arch::{self, MapUse};
use model_hf::folder::{self, FolderMap, Make, Planned};
use model_hf::follow::Follow;
use model_hf::select::Select;

const USAGE: &str = concat!(
    "usage: prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]\n",
    "                [--drop <glob>]... [--only <glob>]... [--arch <word>]\n",
    "                [--follow <list> [--follow-stall <seconds>] [--follow-poll <seconds>]]\n",
    "                [--room <bytes>]\n",
    "       prepare <folder> --plan"
);

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let ran = bloomery_levers::at_main(&[PREPARE_Q8])
        .map_err(Into::into)
        .and_then(|levers| {
            let args: Vec<String> = std::env::args().skip(1).collect();
            run(&args, levers.prepare_q8())
        });
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("prepare: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The split of the folder `args` name, or with `--plan` its plan; `q8` (`BLOOMERY_PREPARE_Q8`)
/// writes its BF16 matrices the map reads as Q8_0, and its F32 matrices, in Q8_0.
fn run(args: &[String], q8: bool) -> Res<()> {
    run_in(args, q8, 1 << 30)
}

/// [`run`] with the byte count of the `--shard-gib` unit: a test's shards are kilobytes.
fn run_in(args: &[String], q8: bool, unit: u64) -> Res<()> {
    let (mut shard_gib, mut keep_bf16, mut plan, mut paths) = (None, false, false, vec![]);
    let (mut drop, mut only, mut arch) = (vec![], vec![], None);
    let (mut follow, mut stall, mut poll) = (None, None, None);
    let mut room = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shard-gib" => shard_gib = Some(it.next().ok_or(USAGE)?.parse::<u64>()?),
            "--keep-bf16" => keep_bf16 = true,
            "--plan" => plan = true,
            "--drop" => drop.push(it.next().ok_or(USAGE)?.clone()),
            "--only" => only.push(it.next().ok_or(USAGE)?.clone()),
            "--arch" => {
                if arch.replace(it.next().ok_or(USAGE)?.clone()).is_some() {
                    return Err("--arch given twice".into());
                }
            }
            "--follow" => {
                if follow.replace(it.next().ok_or(USAGE)?.clone()).is_some() {
                    return Err("--follow given twice".into());
                }
            }
            "--follow-stall" => stall = Some(seconds("--follow-stall", it.next().ok_or(USAGE)?)?),
            "--follow-poll" => poll = Some(seconds("--follow-poll", it.next().ok_or(USAGE)?)?),
            "--room" => {
                let v = it.next().ok_or(USAGE)?;
                let bytes = v.parse::<u64>().map_err(|e| format!("--room {v}: {e}"))?;
                if room.replace(bytes).is_some() {
                    return Err("--room given twice".into());
                }
            }
            _ => paths.push(a),
        }
    }
    if plan {
        let [dir] = paths[..] else {
            return Err(USAGE.into());
        };
        if shard_gib.is_some()
            || q8
            || !drop.is_empty()
            || !only.is_empty()
            || arch.is_some()
            || follow.is_some()
            || stall.is_some()
            || poll.is_some()
            || room.is_some()
        {
            return Err(format!(
                "--plan prints the folder's stored form and writes no shard: it takes no \
                 --shard-gib, --drop, --only, --arch, --follow or --room, and {PREPARE_Q8}=1 asks for a \
                 Q8_0 rewrite it does not plan"
            )
            .into());
        }
        return print_plan(Path::new(dir));
    }
    let options = folder::Options {
        arch,
        select: Select::of(drop, only)?,
    };
    let shard_gib = shard_gib.unwrap_or(48);
    let [dir, prefix] = paths[..] else {
        return Err(USAGE.into());
    };
    if follow.is_none() && (stall.is_some() || poll.is_some()) {
        return Err(
            "--follow-stall and --follow-poll bound a wait for the download --follow \
                    reads: give --follow <list>"
                .into(),
        );
    }
    let follow = follow.map(|list| FollowFlags {
        list: PathBuf::from(list),
        stall: stall.unwrap_or(Duration::from_secs(600)),
        poll: poll.unwrap_or(Duration::from_secs(10)),
    });
    if keep_bf16 && q8 {
        return Err(format!(
            "--keep-bf16 keeps the BF16 matrices {PREPARE_Q8}=1 asks to write as Q8_0: set one"
        )
        .into());
    }
    let limit = u64::checked_mul(shard_gib, unit)
        .filter(|&l| l > 0)
        .ok_or_else(|| {
            format!("--shard-gib {shard_gib}: not a shard size of 1 GiB or more in u64 bytes")
        })?;
    let dir = Path::new(dir);
    let model_type = folder::model_type(dir)?;
    arch::with_split_map(
        &model_type,
        WriteSplit {
            dir,
            prefix,
            limit,
            q8,
            options: &options,
            follow: follow.as_ref(),
            room,
        },
    )
    .ok_or_else(|| {
        format!(
            "{}: model_type {model_type:?} has no split writer; prepare writes {:?}",
            dir.display(),
            arch::split_model_types()
        )
    })?
}

/// The split of a folder, written by whichever map reads it.
struct WriteSplit<'a> {
    dir: &'a Path,
    prefix: &'a str,
    /// The shard size limit in bytes.
    limit: u64,
    /// Whether a BF16 or F32 matrix is rewritten as Q8_0.
    q8: bool,
    /// The architecture word and the tensors the split holds.
    options: &'a folder::Options,
    /// The download to follow, when the folder's shards are still arriving.
    follow: Option<&'a FollowFlags>,
    /// The free bytes the output directory must hold at the plan, when the caller states one.
    room: Option<u64>,
}

/// `--follow` and its two bounds.
struct FollowFlags {
    list: PathBuf,
    stall: Duration,
    poll: Duration,
}

/// A number of seconds above zero, for the flag `flag`.
fn seconds(flag: &str, text: &str) -> Res<Duration> {
    let secs: f64 = text
        .parse()
        .map_err(|_| format!("{flag} {text:?}: not a number of seconds"))?;
    if secs <= 0.0 {
        return Err(format!("{flag} {text:?}: not above zero seconds").into());
    }
    Duration::try_from_secs_f64(secs)
        .map_err(|e| format!("{flag} {text:?}: not a number of seconds: {e}").into())
}

impl MapUse for WriteSplit<'_> {
    type Out = Res<()>;

    fn with<M: FolderMap>(self) -> Res<()> {
        write_as::<M>(&self)
    }
}

/// The split of the folder `job.dir`, read by the map `M`, written at `job.prefix`.
fn write_as<M: FolderMap>(job: &WriteSplit<'_>) -> Res<()> {
    let WriteSplit {
        dir,
        prefix,
        limit,
        q8,
        options,
        follow,
        room,
    } = *job;
    let whole = options.select == Select::All;
    let Some(flags) = follow else {
        let folder::Read {
            checkpoint,
            folder,
            plan,
            planned,
            kvs,
            ..
        } = folder::read_with::<M>(dir, q8, options)?;
        let split = lay_out(planned, &kvs, limit, prefix, &plan, room)?;
        let get = |n: &str| folder.raw(n);
        return write_split(
            prefix,
            split.layouts.clone(),
            |no, w| write_tensors(w, &split.shards[no], &checkpoint, &plan, get),
            |first| check_split(first, &split.held, whole),
        );
    };

    let mut run = Follow::new(dir, &flags.list, flags.stall, flags.poll)?;
    run.say(|line| println!("prepare: {line}"));
    let stage = PathBuf::from(format!("{prefix}.follow"));
    let folder::Staged {
        checkpoint,
        plan,
        planned,
        kvs,
        ..
    } = run.plan::<M>(&stage.join("shell"), q8, options)?;
    let split = lay_out(planned, &kvs, limit, prefix, &plan, room)?;
    let needs = split
        .shards
        .iter()
        .map(|shard| run.needs(&plan, shard))
        .collect::<Result<Vec<_>, _>>()?;
    let staging = Staging {
        prefix,
        dir: &stage,
        key: plan_key(&split, limit, q8, options),
    };
    write_follow(
        &staging,
        &split,
        &needs,
        &mut run,
        |f, no, w| write_tensors(w, &split.shards[no], &checkpoint, &plan, |n| f.raw(n)),
        |first| check_split(first, &split.held, whole),
    )
}

/// A split planned: its shards' tensors and layouts, and the names of every tensor it holds.
struct SplitPlan<S> {
    shards: Vec<Vec<Planned<S>>>,
    layouts: Vec<Layout>,
    held: BTreeSet<String>,
}

/// The shards `planned` fills to `limit` bytes, checked against the free space of `prefix`'s
/// directory (the split and a tenth more, or `room` bytes when the caller states the floor it
/// keeps free behind the writer), with the summary line.
fn lay_out<S>(
    planned: Vec<Planned<S>>,
    kvs: &[(String, Value)],
    limit: u64,
    prefix: &str,
    plan: &hf::quant_config::Plan,
    room: Option<u64>,
) -> Res<SplitPlan<S>> {
    let held: BTreeSet<String> = planned.iter().map(|p| p.decl.name.clone()).collect();
    let tensors = planned.len();
    let (shards, layouts): (Vec<_>, Vec<_>) = plan_shards(planned, kvs, limit)?.into_iter().unzip();
    let count = shards.len();
    let total = layouts.iter().try_fold(0u64, |sum, layout| {
        sum.checked_add(layout.file_len())
            .ok_or("the split's file sizes overflow u64")
    })?;
    let out_dir = Path::new(prefix)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let free = model::fileio::free_bytes(out_dir)
        .map_err(|e| format!("statvfs {}: {e}", out_dir.display()))?;
    let needed = match room {
        Some(room) => room,
        None => total
            .checked_add(total / 10)
            .ok_or("the split's file sizes and free-space margin overflow u64")?,
    };
    if free < needed {
        return Err(match room {
            Some(room) => format!(
                "{}: {free} bytes free; --room asks for {room} (the split is {total})",
                out_dir.display()
            ),
            None => format!(
                "{}: {free} bytes free; the split needs {total} and a tenth more (--room <bytes> \
                 states a lower floor when the input is freed behind the writer)",
                out_dir.display()
            ),
        }
        .into());
    }
    println!(
        "prepare: {} tensors, {total} bytes, {count} shards; plan refused {} and left {} input scales",
        tensors,
        plan.refused.len(),
        plan.input_scales.len()
    );
    Ok(SplitPlan {
        shards,
        layouts,
        held,
    })
}

/// The tensors of one output shard, each made of the folder's through `get` and written.
fn write_tensors<'a, M: FolderMap>(
    w: &mut Writer<BufWriter<File>>,
    shard: &[Planned<M::Step>],
    checkpoint: &M,
    plan: &hf::quant_config::Plan,
    get: impl Fn(&str) -> Result<safetensors::Raw<'a>, safetensors::Error> + Copy,
) -> Res<()> {
    // A tensor's scales reach the `.scale` that follows it, in the same shard.
    let mut tensor_scales: HashMap<String, Vec<f32>> = HashMap::default();
    for p in shard {
        let t = Instant::now();
        let name = &p.decl.name;
        match &p.make {
            Make::Scale { of } => {
                let s = tensor_scales
                    .remove(of)
                    .ok_or_else(|| format!("{name}: {of} gave no scale"))?;
                w.tensor(
                    name,
                    &s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                )?;
            }
            Make::Constant(c) => {
                w.tensor(name, &c.bytes())?;
            }
            Make::Tensor(m) if m.role == Use::Rows => {
                // A row table's parts, one at a time, all with the table's one scale.
                let (mut failed, mut scale) = (None, vec![]);
                let parts = m.parts(plan, get).map_while(|n| match n {
                    Ok(n) => {
                        scale = n.scale;
                        Some(n.bytes)
                    }
                    Err(e) => {
                        failed = Some(e);
                        None
                    }
                });
                let written = w.tensor_parts(name, parts);
                if let Some(e) = failed {
                    return Err(e.into());
                }
                written?;
                tensor_scales.insert(name.clone(), scale);
            }
            Make::Tensor(m) => {
                let mut n: Normal = modelopt::normalize(get, &m.placed(plan)?, m.role)?;
                if let Some(moves) = checkpoint.mover(name, &m.steps) {
                    moves(&mut n)?;
                }
                if n.type_id != p.decl.type_id {
                    return Err(
                        format!("{name}: type {}, declared {}", n.type_id, p.decl.type_id).into(),
                    );
                }
                w.tensor(name, &n.bytes)?;
                if !n.scale.is_empty() {
                    tensor_scales.insert(name.clone(), n.scale);
                }
            }
        }
        println!(
            "prepare: {name} type={} bytes={} secs={:.3}",
            p.decl.type_id,
            p.decl.nbytes,
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// The split opening at `first` holds the tensors the plan holds; a whole split also opens as
/// the server opens it, its hyperparameters and layers whole.
fn check_split(first: &Path, held: &BTreeSet<String>, whole: bool) -> Res<()> {
    let split = gguf::Split::open(first)?;
    let tensors: BTreeSet<String> = split.iter_tensors().map(|(_, t)| t.name.clone()).collect();
    if tensors != *held || split.tensor_count() != held.len() {
        return Err(format!(
            "the split opens with {} tensors, where the plan holds {}",
            split.tensor_count(),
            held.len()
        )
        .into());
    }
    if !whole {
        println!(
            "prepare: the split opens with the {} tensors of its selection",
            split.tensor_count()
        );
        return Ok(());
    }
    let read = model::arch::spec(&split)?;
    println!(
        "prepare: the split opens: {} layers, {} tensors",
        read.spec.layers.len(),
        split.tensor_count()
    );
    Ok(())
}

/// What a staged split must equal for a rerun to resume it: the plan's shards, their layouts and
/// tensors, and the arguments that shaped them.
fn plan_key<S>(split: &SplitPlan<S>, limit: u64, q8: bool, options: &folder::Options) -> String {
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |bytes: &[u8]| {
        for b in bytes {
            fnv = (fnv ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (shard, layout) in split.shards.iter().zip(&split.layouts) {
        mix(&layout.file_len().to_le_bytes());
        for p in shard {
            mix(p.decl.name.as_bytes());
            mix(&p.decl.type_id.to_le_bytes());
            mix(&p.decl.nbytes.to_le_bytes());
            p.decl.dims.iter().for_each(|d| mix(&d.to_le_bytes()));
        }
    }
    format!(
        "prepare-follow-1 shards={} tensors={} fnv={fnv:016x} limit={limit} q8={q8} arch={:?} select={:?}",
        split.shards.len(),
        split.held.len(),
        options.arch,
        options.select
    )
}

/// The paths of the `count` shards of the split at `prefix`.
fn shard_paths(prefix: &str, count: usize) -> Vec<PathBuf> {
    (1..=count)
        .map(|no| PathBuf::from(format!("{prefix}-{no:05}-of-{count:05}.gguf")))
        .collect()
}

/// The persistent lock file of the output at `prefix`, held, and its path.
fn lock_output(prefix: &str) -> Res<(File, String)> {
    let lock_path = format!("{prefix}.prepare.lock");
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("{lock_path}: open lock: {e}"))?;
    match lock.try_lock() {
        Ok(()) => Ok((lock, lock_path)),
        Err(std::fs::TryLockError::WouldBlock) => {
            Err(format!("{prefix}: prepare output is busy").into())
        }
        Err(std::fs::TryLockError::Error(e)) => Err(format!("{lock_path}: lock: {e}").into()),
    }
}

/// Where a followed split is staged: the output prefix, the stage directory and the key of the
/// plan it holds the shards of.
struct Staging<'a> {
    prefix: &'a str,
    dir: &'a Path,
    key: String,
}

/// The split written shard by shard as the input shards complete ([`Follow::run`]), each staged
/// under `stage` and marked done, then checked and published as `write_split` does. Any failure
/// keeps the marked shards: a rerun with the same plan resumes from them.
fn write_follow<S>(
    staging: &Staging<'_>,
    split: &SplitPlan<S>,
    needs: &[BTreeSet<String>],
    run: &mut Follow,
    mut write: impl FnMut(&Follow, usize, &mut Writer<BufWriter<File>>) -> Res<()>,
    validate: impl FnOnce(&Path) -> Res<()>,
) -> Res<()> {
    let Staging {
        prefix,
        dir: stage,
        key,
    } = staging;
    let (lock, lock_path) = lock_output(prefix)?;
    let count = split.layouts.len();
    let paths = shard_paths(prefix, count);
    let file = |no: usize| {
        paths[no]
            .file_name()
            .expect("a shard has a filename")
            .to_owned()
    };
    let marker = |no: usize| {
        let mut name = file(no);
        name.push(".done");
        stage.join(name)
    };
    let result = (|| -> Res<()> {
        for path in &paths {
            match std::fs::symlink_metadata(path) {
                Ok(_) => return Err(format!("{}: output already exists", path.display()).into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("{}: stat: {e}", path.display()).into()),
            }
        }
        let key_path = stage.join("plan.key");
        // The plan has made the stage for its shell; a stage with a key holds a plan's shards.
        std::fs::create_dir_all(stage)?;
        match std::fs::read_to_string(&key_path) {
            Ok(have) if have == *key => {
                for entry in std::fs::read_dir(stage)? {
                    let path = entry?.path();
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default();
                    let stray = name.ends_with(".part")
                        || (name.ends_with(".gguf")
                            && !stage.join(format!("{name}.done")).exists());
                    if stray {
                        println!(
                            "prepare: removing the partial staged shard {}",
                            path.display()
                        );
                        std::fs::remove_file(&path)?;
                    }
                }
            }
            Ok(have) => {
                return Err(format!(
                    "{}: staged by another plan ({have:?}), this one is {key:?}; remove it to \
                     start over",
                    stage.display()
                )
                .into());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let held = std::fs::read_dir(stage)?
                    .filter_map(|e| e.ok()?.file_name().into_string().ok())
                    .find(|n| n.ends_with(".gguf") || n.ends_with(".done"));
                if let Some(name) = held {
                    return Err(format!(
                        "{}: holds {name} and no plan.key; remove it to start over",
                        stage.display()
                    )
                    .into());
                }
                std::fs::write(&key_path, key)?;
            }
            Err(e) => return Err(format!("{}: {e}", key_path.display()).into()),
        }
        let done: Vec<bool> = (0..count).map(|no| marker(no).exists()).collect();
        for no in (0..count).filter(|&no| done[no]) {
            let at = stage.join(file(no));
            let len = std::fs::metadata(&at)
                .map_err(|e| format!("{}: marked done: {e}", at.display()))?
                .len();
            if len != split.layouts[no].file_len() {
                return Err(format!(
                    "{}: marked done at {len} bytes, where the plan's shard is {}",
                    at.display(),
                    split.layouts[no].file_len()
                )
                .into());
            }
            println!(
                "prepare: {} is staged and marked done: skipped",
                at.display()
            );
        }
        run.run(needs, done, |f, no| {
            let t = Instant::now();
            let part = stage.join(format!("{}.part", file(no).to_string_lossy()));
            let wrote = (|| -> Res<()> {
                let mut w = Writer::new(
                    BufWriter::new(File::create_new(&part)?),
                    split.layouts[no].clone(),
                )?;
                write(f, no, &mut w)?;
                w.finish()?
                    .into_inner()
                    .map_err(|e| e.into_error())?
                    .sync_all()?;
                std::fs::rename(&part, stage.join(file(no)))?;
                File::create_new(marker(no))?.sync_all()?;
                Ok(())
            })();
            if let Err(e) = wrote {
                let _ = std::fs::remove_file(&part);
                return Err(folder::Error::Refused(format!(
                    "output shard {} of {count}: {e}",
                    no + 1
                )));
            }
            println!(
                "prepare: staged output shard {} of {count}, {} bytes, {:.1} s",
                no + 1,
                split.layouts[no].file_len(),
                t.elapsed().as_secs_f64()
            );
            Ok(())
        })?;
        validate(&stage.join(file(0)))?;
        publish(stage, &paths)
    })();
    let mut cleanup = vec![];
    match &result {
        Ok(()) => {
            if let Err(e) = std::fs::remove_dir_all(stage) {
                cleanup.push(format!("{}: remove the stage: {e}", stage.display()));
            }
        }
        Err(_) => {
            let marked = (0..count).filter(|&no| marker(no).exists()).count();
            println!(
                "prepare: {marked} of {count} output shards stay staged in {}; a rerun with the \
                 same arguments resumes from them",
                stage.display()
            );
        }
    }
    if let Err(e) = lock.unlock() {
        cleanup.push(format!("{lock_path}: unlock: {e}"));
    }
    if cleanup.is_empty() {
        result
    } else {
        let cleanup = cleanup.join("; ");
        match result {
            Ok(()) => Err(format!("split published; cleanup failed: {cleanup}").into()),
            Err(e) => Err(format!("{e}; cleanup failed: {cleanup}").into()),
        }
    }
}

/// The staged shards of `paths` (by file name under `stage`) moved to their paths. The first
/// shard is the entry point, so its siblings are published before it and it appears last; a
/// shard that cannot be published puts back the ones that were, and a path that exists is never
/// replaced.
fn publish(stage: &Path, paths: &[PathBuf]) -> Res<()> {
    let staged = |path: &Path| stage.join(path.file_name().expect("a shard has a filename"));
    let mut published: Vec<&PathBuf> = vec![];
    for path in paths.iter().rev() {
        if let Err(e) = model::fileio::rename_noreplace(&staged(path), path) {
            let restore: Vec<String> = published
                .iter()
                .filter_map(|p| {
                    std::fs::rename(p, staged(p))
                        .err()
                        .map(|e| format!("{}: take back: {e}", p.display()))
                })
                .collect();
            let restore = if restore.is_empty() {
                String::new()
            } else {
                format!("; {}", restore.join("; "))
            };
            return Err(format!("{}: publish: {e}{restore}", path.display()).into());
        }
        published.push(path);
    }
    for path in paths {
        println!("prepare: wrote {}", path.display());
    }
    Ok(())
}

fn write_split(
    prefix: &str,
    layouts: Vec<Layout>,
    mut write: impl FnMut(usize, &mut Writer<BufWriter<File>>) -> Res<()>,
    validate: impl FnOnce(&Path) -> Res<()>,
) -> Res<()> {
    let (lock, lock_path) = lock_output(prefix)?;
    let count = layouts.len();
    let paths = shard_paths(prefix, count);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stage = PathBuf::from(format!("{prefix}.part.{}.{nonce}", std::process::id()));
    let mut staged = false;
    let result = (|| {
        for path in &paths {
            match std::fs::symlink_metadata(path) {
                Ok(_) => return Err(format!("{}: output already exists", path.display()).into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("{}: stat: {e}", path.display()).into()),
            }
        }
        std::fs::create_dir(&stage)?;
        staged = true;
        for (no, layout) in layouts.into_iter().enumerate() {
            let part = stage.join(paths[no].file_name().expect("a shard has a filename"));
            let mut w = Writer::new(BufWriter::new(File::create_new(&part)?), layout)?;
            write(no, &mut w)?;
            w.finish()?;
        }
        validate(&stage.join(paths[0].file_name().expect("a shard has a filename")))?;
        publish(&stage, &paths)
    })();
    let mut cleanup = vec![];
    if staged && let Err(e) = std::fs::remove_dir_all(&stage) {
        cleanup.push(format!("{}: remove temporary split: {e}", stage.display()));
    }
    if let Err(e) = lock.unlock() {
        cleanup.push(format!("{lock_path}: unlock: {e}"));
    }
    if cleanup.is_empty() {
        result
    } else {
        let cleanup = cleanup.join("; ");
        match result {
            Ok(()) => Err(format!("split published; cleanup failed: {cleanup}").into()),
            Err(e) => Err(format!("{e}; cleanup failed: {cleanup}").into()),
        }
    }
}

/// The plan of the folder `dir`, whichever map reads it: headers and tokenizer, no tensor bytes.
fn print_plan(dir: &Path) -> Res<()> {
    let opened = model_hf::folder::open(dir)?;
    for line in model_hf::folder::plan_lines(&opened.header, &opened.skipped) {
        println!("prepare: plan {line}");
    }
    println!(
        "prepare: plan tokenizer: {} tokens, its specials and stop set as Hugging Face's",
        opened.tokenizer.n_vocab()
    );
    Ok(())
}

fn plan_shards<S>(
    planned: Vec<Planned<S>>,
    kvs: &[(String, Value)],
    limit: u64,
) -> Res<Vec<(Vec<Planned<S>>, Layout)>> {
    let tensors32 =
        i32::try_from(planned.len()).map_err(|_| "too many tensors for a GGUF split")?;
    let shard_keys = |no: u16, count: u16| {
        let mut keys = if no == 0 { kvs.to_vec() } else { vec![] };
        keys.extend([
            ("split.no".to_string(), Value::U16(no)),
            ("split.count".to_string(), Value::U16(count)),
            ("split.tensors.count".to_string(), Value::I32(tensors32)),
        ]);
        keys
    };
    let mut shards: Vec<Vec<Planned<S>>> = vec![vec![]];
    let mut pending = planned.into_iter().peekable();
    while let Some(p) = pending.next() {
        let mut group = vec![p];
        if pending
            .peek()
            .is_some_and(|p| matches!(p.make, Make::Scale { .. }))
        {
            group.push(pending.next().expect("the scale was just inspected"));
        }
        let no = shards.len() - 1;
        let no16 = u16::try_from(no).map_err(|_| "too many shards for a GGUF split")?;
        let last = &mut shards[no];
        let declarations = last.iter().chain(&group).map(|p| p.decl.clone()).collect();
        // split.count has the same U16 width for every final count.
        let candidate = Layout::new(&shard_keys(no16, 1), declarations)?;
        if !last.is_empty() && candidate.file_len() > limit {
            shards.push(group);
        } else {
            last.extend(group);
        }
    }
    let count16 = u16::try_from(shards.len()).map_err(|_| "too many shards for a GGUF split")?;
    shards
        .into_iter()
        .enumerate()
        .map(|(no, shard)| {
            let keys = shard_keys(u16::try_from(no)?, count16);
            let layout = Layout::new(&keys, shard.iter().map(|p| p.decl.clone()).collect())?;
            Ok((shard, layout))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use gguf::write::TensorDecl;
    use gguf::{GENERAL_ARCHITECTURE, GgmlType};
    use hf::quant_config::{Config, Format};
    use model_hf::arch::qwen35moe::Checkpoint;
    use model_hf::arch::qwen35moe::folder::{Keep, plan_tensors, served_as_hf};
    use safetensors::Dtype;
    fn split_fixture(count: usize) -> Vec<super::Layout> {
        (0..count)
            .map(|no| {
                super::Layout::new(
                    &[
                        ("split.no".into(), gguf::Value::U16(no as u16)),
                        ("split.count".into(), gguf::Value::U16(count as u16)),
                        ("split.tensors.count".into(), gguf::Value::I32(count as i32)),
                    ],
                    vec![TensorDecl {
                        name: format!("t{no}"),
                        dims: vec![2],
                        type_id: 0,
                        nbytes: 8,
                    }],
                )
                .unwrap()
            })
            .collect()
    }

    fn split_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("prepare-{tag}-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn split_failure_preserves_existing_shards() {
        use super::*;
        let dir = split_dir("failure");
        let prefix = dir.join("model");
        let prefix = prefix.to_str().unwrap();
        write_split(
            prefix,
            split_fixture(2),
            |no, w| {
                w.tensor(&format!("t{no}"), &[1; 8])?;
                Ok(())
            },
            |first| {
                gguf::Split::open(first)?;
                Ok(())
            },
        )
        .unwrap();
        let paths: Vec<_> = (1..=2)
            .map(|no| format!("{prefix}-{no:05}-of-00002.gguf"))
            .collect();
        let before: Vec<_> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
        let result = write_split(
            prefix,
            split_fixture(2),
            |no, w| {
                if no == 1 {
                    return Err("later shard failed".into());
                }
                w.tensor(&format!("t{no}"), &[2; 8])?;
                Ok(())
            },
            |_| Ok(()),
        );
        assert!(result.is_err());
        for (path, before) in paths.iter().zip(before) {
            assert_eq!(std::fs::read(path).unwrap(), before, "{path}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn split_concurrent_writers_cannot_change_a_successful_output() {
        use super::*;
        let dir = split_dir("concurrent");
        let prefix = dir.join("model");
        let prefix = prefix.to_str().unwrap();
        let path = format!("{prefix}-00001-of-00001.gguf");
        let mut other = None;
        let first = write_split(
            prefix,
            split_fixture(1),
            |_, w| {
                let second = write_split(
                    prefix,
                    split_fixture(1),
                    |_, w| {
                        w.tensor("t0", &[2; 8])?;
                        Ok(())
                    },
                    |_| Ok(()),
                );
                other = Some(second.map(|()| std::fs::read(&path).unwrap()));
                w.tensor("t0", &[1; 8])?;
                Ok(())
            },
            |_| Ok(()),
        );
        match other.unwrap() {
            Ok(bytes) => assert_eq!(std::fs::read(&path).unwrap(), bytes),
            Err(e) => {
                assert!(e.to_string().contains("busy"), "{e}");
                first.unwrap();
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn split_is_complete_and_validated_before_its_first_shard_appears() {
        use super::*;
        for count in [1, 2, 4] {
            let dir = split_dir(&format!("complete-{count}"));
            let prefix = dir.join("model");
            let prefix = prefix.to_str().unwrap();
            let first = format!("{prefix}-00001-of-{count:05}.gguf");
            write_split(
                prefix,
                split_fixture(count),
                |no, w| {
                    assert!(!Path::new(&first).exists());
                    w.tensor(&format!("t{no}"), &[no as u8; 8])?;
                    Ok(())
                },
                |staged| {
                    assert!(!Path::new(&first).exists());
                    let split = gguf::Split::open(staged)?;
                    assert_eq!(split.shard_count(), count);
                    assert_eq!(split.tensor_count(), count);
                    Ok(())
                },
            )
            .unwrap();
            let split = gguf::Split::open(&first).unwrap();
            assert_eq!(split.shard_count(), count);
            for (no, layout) in split_fixture(count).into_iter().enumerate() {
                let mut expected = Writer::new(vec![], layout).unwrap();
                expected.tensor(&format!("t{no}"), &[no as u8; 8]).unwrap();
                assert_eq!(
                    std::fs::read(split.shard_path(no).unwrap()).unwrap(),
                    expected.finish().unwrap()
                );
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn split_write_validation_and_publication_errors_clean_only_their_own_files() {
        use super::*;
        for (case, failed_shard, invalid, conflict) in [
            ("write-first", Some(0), false, None),
            ("write-later", Some(1), false, None),
            ("validate", None, true, None),
            ("publish-first", None, false, Some(1)),
            ("publish-last", None, false, Some(0)),
        ] {
            let dir = split_dir(case);
            let prefix = dir.join("model");
            let prefix = prefix.to_str().unwrap();
            let paths: Vec<_> = (1..=2)
                .map(|no| format!("{prefix}-{no:05}-of-00002.gguf"))
                .collect();
            let result = write_split(
                prefix,
                split_fixture(2),
                |no, w| {
                    if failed_shard == Some(no) {
                        return Err("injected write failure".into());
                    }
                    w.tensor(&format!("t{no}"), &[1; 8])?;
                    Ok(())
                },
                |first| {
                    gguf::Split::open(first)?;
                    if invalid {
                        return Err("injected validation failure".into());
                    }
                    if let Some(no) = conflict {
                        std::fs::write(&paths[no], b"external file")?;
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{case}");
            for (no, path) in paths.iter().enumerate() {
                if conflict == Some(no) {
                    assert_eq!(std::fs::read(path).unwrap(), b"external file", "{case}");
                } else {
                    assert!(!Path::new(path).exists(), "{case}: {path}");
                }
            }
            for entry in std::fs::read_dir(&dir).unwrap() {
                let name = entry.unwrap().file_name();
                assert!(
                    !name.to_string_lossy().contains(".part"),
                    "{case}: {name:?}"
                );
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn shard_budget_counts_scales_metadata_and_padding() {
        use super::*;

        let config: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
                "/../hf/tests/fixtures/quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz"))[..],
        )).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let shape = [
            config["text_config"]["vocab_size"].as_u64().unwrap() as usize,
            config["text_config"]["hidden_size"].as_u64().unwrap() as usize,
        ];
        let kvs = vec![
            (
                GENERAL_ARCHITECTURE.into(),
                Value::String("qwen4exp".into()),
            ),
            ("general.description".into(), Value::String("x".repeat(33))),
        ];
        for (limit, lengths) in [
            (640, vec![480, 352]),
            (704, vec![704]),
            (479, vec![480, 352]),
        ] {
            let mapped = checkpoint
                .map([
                    ("lm_head.weight", shape.as_slice()),
                    ("model.language_model.embed_tokens.weight", shape.as_slice()),
                ])
                .unwrap();
            assert_eq!(mapped.len(), 2);
            let mut planned = vec![];
            for (stem, m) in ["a", "b"].into_iter().zip(mapped.into_values()) {
                let name = format!("{stem}.weight");
                planned.push(Planned {
                    decl: TensorDecl {
                        name: name.clone(),
                        dims: vec![32],
                        type_id: 0,
                        nbytes: 128,
                    },
                    make: Make::Tensor(m),
                });
                planned.push(Planned {
                    decl: TensorDecl {
                        name: format!("{stem}.scale"),
                        dims: vec![1],
                        type_id: 0,
                        nbytes: 4,
                    },
                    make: Make::Scale { of: name },
                });
            }
            let shards = plan_shards(planned, &kvs, limit).unwrap();
            let mut written = vec![];
            for (shard, layout) in shards {
                let file_len = layout.file_len();
                let mut writer = Writer::new(vec![], layout).unwrap();
                for p in &shard {
                    writer
                        .tensor(&p.decl.name, &vec![0; p.decl.nbytes as usize])
                        .unwrap();
                }
                let bytes = writer.finish().unwrap();
                assert_eq!(bytes.len() as u64, file_len);
                eprintln!("budget={limit} file_len={file_len} tensors={}", shard.len());
                assert!(
                    file_len <= limit || shard.len() == 2,
                    "{file_len} > {limit}: {} tensors",
                    shard.len()
                );
                written.push(file_len);
            }
            assert_eq!(written, lengths, "budget={limit}");
        }
    }

    /// The HF contract is exact, including when the legacy vocabulary guesses extra roles.
    #[test]
    fn the_served_tokenizer_splits_and_stops_as_hf() {
        use super::*;

        let stop_key = "tokenizer.bloomery.eog_token_ids";
        for (second, added, stop, explicit, refused) in [
            ("b", vec![2], vec![2], false, None),
            (
                "</s>",
                vec![2],
                vec![2],
                false,
                Some("splits text on tokens [1, 2]"),
            ),
            ("b", vec![2], vec![2, 0], false, Some("stops at token 0")),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![2],
                false,
                Some("serving stops at token 1"),
            ),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![1, 2],
                false,
                Some("control/rendering at token 1"),
            ),
            (
                "<|channel|>",
                vec![1, 2],
                vec![2],
                false,
                Some("control/rendering at token 1"),
            ),
            ("", vec![2], vec![2], true, Some("tokens[1] \"\"")),
            ("▁", vec![2], vec![2], true, Some("tokens[1] \"▁\"")),
            ("Ġx", vec![1, 2], vec![2], true, Some("tokens[1] \"Ġx\"")),
            ("▁x", vec![1, 2], vec![2], true, None),
            ("<|endoftext|>", vec![1, 2], vec![2], true, None),
            ("<|fim_pad|>", vec![1, 2], vec![2], true, None),
            ("<|repo_name|>", vec![1, 2], vec![2], true, None),
            ("<|file_sep|>", vec![1, 2], vec![2], true, None),
            ("<|channel|>", vec![1, 2], vec![2], true, None),
        ] {
            let second_type =
                if second == "<|channel|>" || (second == "<|endoftext|>" && stop == [2]) {
                    3
                } else if added.contains(&1) {
                    4
                } else {
                    1
                };
            let mut keys = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        ["a", second, "<|im_end|>", "aa"]
                            .map(|t| Value::String(t.into()))
                            .to_vec(),
                    ),
                ),
                (
                    "tokenizer.ggml.token_type".into(),
                    Value::Array([1, second_type, 3, 1].map(Value::I32).to_vec()),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    Value::Array(vec![Value::String("a a".into())]),
                ),
                ("tokenizer.ggml.eos_token_id".into(), Value::U32(2)),
                ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
            ];
            if explicit {
                keys.push((
                    stop_key.into(),
                    Value::Array(stop.iter().copied().map(Value::U32).collect()),
                ));
            }
            let vocab = hf::vocab::Keys {
                keys: keys.clone(),
                added,
                stop,
            };
            match (served_as_hf(&keys, &vocab).map(|_| ()), refused) {
                (Ok(()), None) => {}
                (Err(e), Some(what)) if e.to_string().contains(what) => {}
                (other, _) => panic!("{second} {:?}: {other:?}", vocab.stop),
            }
            if explicit && refused.is_none() {
                for (key, value, error) in [
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0), Value::U32(2)]),
                        "serving stops at token 0",
                    ),
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0)]),
                        "stops at token 2",
                    ),
                    (
                        "tokenizer.ggml.add_bos_token",
                        Value::Bool(true),
                        "add_bos_token",
                    ),
                    (
                        "tokenizer.ggml.add_eos_token",
                        Value::Bool(true),
                        "add_eos_token",
                    ),
                    ("tokenizer.ggml.bos_token_id", Value::U32(1), "bos_token_id"),
                    ("tokenizer.ggml.eos_token_id", Value::U32(0), "eos_token_id"),
                    (
                        "tokenizer.ggml.padding_token_id",
                        Value::U32(1),
                        "padding_token_id",
                    ),
                    (
                        "tokenizer.ggml.unknown_token_id",
                        Value::U32(1),
                        "unknown_token_id",
                    ),
                    (
                        "tokenizer.ggml.seperator_token_id",
                        Value::U32(1),
                        "seperator_token_id",
                    ),
                ] {
                    let mut changed = keys.clone();
                    changed.retain(|(k, _)| k != key);
                    changed.push((key.into(), value));
                    let err = served_as_hf(&changed, &vocab).err().unwrap();
                    assert!(err.to_string().contains(error), "{second} {key}: {err}");
                }
                // The EOS id is asked whatever the flags say.
                let mut absent = hf::vocab::Keys {
                    keys: keys.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                absent
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.eos_token_id");
                let err = served_as_hf(&absent.keys, &absent).err().unwrap();
                assert!(
                    err.to_string()
                        .contains("serving requires tokenizer.ggml.eos_token_id"),
                    "{err}"
                );
                // PIN(2026-10-28): a BOS id is asked only where the vocabulary adds a BOS. It was
                // asked always, and the Qwen3-Coder folder, which names none, was refused.
                let mut absent = hf::vocab::Keys {
                    keys: keys.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                absent
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.bos_token_id");
                let served = served_as_hf(&absent.keys, &absent).unwrap();
                assert_eq!(served.specials().bos, None, "{second}");
                absent
                    .keys
                    .push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                let err = served_as_hf(&absent.keys, &absent).err().unwrap();
                assert!(
                    err.to_string()
                        .contains("missing metadata key tokenizer.ggml.bos_token_id"),
                    "{err}"
                );
                let mut both = keys.clone();
                let mut expected = hf::vocab::Keys {
                    keys: both.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                both.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                assert!(
                    served_as_hf(&both, &expected)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token")
                );
                expected.keys = both.clone();
                assert!(served_as_hf(&both, &expected).is_ok());
                // A flag of HF's that is not a bool is refused, not read as off.
                expected
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.add_bos_token");
                expected
                    .keys
                    .push(("tokenizer.ggml.add_bos_token".into(), Value::U32(1)));
                assert!(
                    served_as_hf(&both, &expected)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token is U32(1), not a bool")
                );
            }
        }
    }

    /// `--plan` takes the folder alone: another path, a shard size and the Q8_0 rewrite are refused
    /// before any folder is read, a folder with no config is refused by its name, and the write
    /// form's own refusals stand.
    #[test]
    fn plan_takes_a_folder_alone() {
        use super::*;

        let refused = |args: &[&str], q8| {
            let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            run(&args, q8).unwrap_err().to_string()
        };
        for args in [&["--plan"][..], &["--plan", "a", "b"], &["a"]] {
            let e = refused(args, false);
            assert!(e.contains("usage: prepare <folder> <out prefix>"), "{e}");
            assert!(e.contains("prepare <folder> --plan"), "{e}");
        }
        let e = refused(&["--plan", "--shard-gib", "1", "a"], false);
        assert!(e.contains("takes no --shard-gib"), "{e}");
        let e = refused(&["--plan", "a"], true);
        assert!(e.contains(PREPARE_Q8), "{e}");
        let e = refused(&["--plan", "/nonexistent-folder"], false);
        assert!(e.contains("/nonexistent-folder/config.json"), "{e}");
        let e = refused(&["a", "b", "--keep-bf16"], true);
        assert!(e.contains("--keep-bf16 keeps the BF16 matrices"), "{e}");
    }

    /// A folder whose `model_type` no writer reads is refused by name, with the types the writers
    /// read.
    #[test]
    fn a_model_type_no_writer_reads_is_refused() {
        use super::*;

        let dir = split_dir("no-writer");
        std::fs::write(dir.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
        let args = [
            dir.display().to_string(),
            dir.join("out").display().to_string(),
        ];
        let e = run(&args, false).unwrap_err().to_string();
        assert!(
            e.contains("model_type \"llama\" has no split writer"),
            "{e}"
        );
        assert!(e.contains("qwen3_moe") && e.contains("qwen3_5_moe"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Serving's tokenizer, built from a Qwen3.8 folder's files as `prepare` builds it, encodes
    /// every text of the corpus to the ids Hugging Face's `tokenizers` gives
    /// (`hf/tests/fixtures/hf_encode.py`, which pins the `tokenizer.json` it ran on).
    #[test]
    #[ignore = "needs a Qwen3.8 folder's config.json and tokenizer files at $BLOOMERY_QWEN38_HF"]
    fn hw_the_served_tokenizer_encodes_the_corpus_as_hf() {
        use super::*;
        use sha2::{Digest, Sha256};

        let dir = PathBuf::from(
            std::env::var_os("BLOOMERY_QWEN38_HF").expect("$BLOOMERY_QWEN38_HF names the folder"),
        );
        let path = format!(
            "{}/../hf/tests/fixtures/qwen3.8-hf-encode.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let corpus: serde_json::Value =
            serde_json::from_reader(flate2::read::GzDecoder::new(File::open(&path).unwrap()))
                .unwrap();
        let file = std::fs::read(dir.join("tokenizer.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&file)),
            corpus["sha256"].as_str().unwrap(),
            "{}: not the tokenizer.json of {}",
            dir.display(),
            corpus["source"]
        );
        let config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap());
        let checkpoint = Checkpoint::parse(&config.unwrap()).unwrap();
        let vocab = hf::vocab::keys(&dir, &checkpoint.vocab).unwrap();
        let served = served_as_hf(&vocab.keys, &vocab).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() >= 2000, "{} cases in {path}", cases.len());
        let (mut ids, mut failures) = (0, Vec::new());
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let ours = served.encode(text, false, true);
            ids += want.len();
            if ours != want {
                failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
            }
        }
        println!(
            "HF-ENCODE cases={} ids={ids} differences={} tokenizers={} source={}",
            cases.len(),
            failures.len(),
            corpus["tokenizers"],
            corpus["source"]
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A shard size of no bytes, or of more bytes than a u64 holds, is refused before the folder
    /// is read.
    #[test]
    fn a_shard_size_out_of_range_is_refused() {
        for n in ["0", "17179869184", "17179869185"] {
            let args = ["no-folder", "out", "--shard-gib", n].map(String::from);
            let e = super::run(&args, false).unwrap_err().to_string();
            assert!(e.contains("--shard-gib"), "{n}: {e}");
        }
    }

    /// Every tensor of layers 0, 1 and 3 and the unlayered text tensors of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4 is declared in its stored format: each one unsloth's
    /// UD-Q4_K_XL holds in Q8_0, a BF16 matrix in the checkpoint (the head and the token
    /// embedding among them), is BF16, and Q8_0 only when `BLOOMERY_PREPARE_Q8` asks; every other
    /// tensor (the NVFP4 stacks, the PLE table's F8 rows, the gains, the scales) is the same
    /// either way. Router parts keep their stored F32 or BF16 type with either lever value.
    /// Other synthetic F32 matrices require the Q8_0 rewrite; BF16 and F32 routed stacks are
    /// refused with either lever value. Dropping either refusal fails these assertions.
    /// `--keep-bf16` asks for the stored form: taken alone, refused beside the lever.
    #[test]
    fn a_bf16_matrix_is_written_as_stored_unless_q8_is_asked() {
        use super::*;
        use model::arch::qwen35moe::names;
        use std::collections::BTreeMap;

        let gz = |name: &str| -> serde_json::Value {
            let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
            let file = File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            serde_json::from_reader(flate2::read::GzDecoder::new(file)).unwrap()
        };
        let config = gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz");
        let hf_quant =
            gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let quant = Config::parse(&config, Some(&hf_quant)).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let header = gz("checkpoint-headers/qwen3.8-flash-next-nvfp4-layers-0-1-3.tensors.json.gz");
        let entries: Vec<(String, Dtype, Vec<usize>)> = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let shape = e["shape"].as_array().unwrap();
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape.iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                )
            })
            .collect();
        let unsloth: Vec<(String, Vec<u64>, u32)> = serde_json::from_value(gz(
            "checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-layers-0-1-3.tensors.json.gz",
        ))
        .unwrap();
        let q8_0: Vec<&str> = unsloth
            .iter()
            .filter(|(_, _, ty)| *ty == GgmlType::Q8_0.as_u32())
            .map(|(n, _, _)| n.as_str())
            .collect();
        for name in [names::output(), names::token_embd(), names::attn_output(3)] {
            assert!(q8_0.contains(&name.as_str()), "{name}");
        }
        let types = |q8| -> BTreeMap<String, GgmlType> {
            let (_, planned) = plan_tensors(&checkpoint, &quant, &entries, q8, Keep::Made).unwrap();
            planned
                .into_iter()
                .map(|p| (p.decl.name, GgmlType::from_u32(p.decl.type_id)))
                .collect()
        };
        let (stored, rewritten) = (types(false), types(true));
        assert_eq!(
            stored.keys().collect::<Vec<_>>(),
            rewritten.keys().collect::<Vec<_>>()
        );
        for (name, ty) in &stored {
            let want = if q8_0.contains(&name.as_str()) {
                (GgmlType::BF16, GgmlType::Q8_0)
            } else {
                (*ty, *ty)
            };
            assert_eq!((*ty, rewritten[name]), want, "{name}");
        }
        for (router, shared) in [
            (Dtype::F32, Dtype::Bf16),
            (Dtype::Bf16, Dtype::F32),
            (Dtype::F32, Dtype::F32),
        ] {
            let mut mixed = entries.clone();
            let parts = [
                (
                    "model.language_model.layers.0.mlp.gate.weight",
                    names::ffn_gate_inp(0),
                    router,
                ),
                (
                    "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                    names::ffn_gate_inp_shexp(0),
                    shared,
                ),
            ];
            for (source, _, dtype) in &parts {
                mixed.iter_mut().find(|(n, _, _)| n == source).unwrap().1 = *dtype;
            }
            for q8 in [false, true] {
                let (_, planned) =
                    plan_tensors(&checkpoint, &quant, &mixed, q8, Keep::Made).unwrap();
                for (_, target, dtype) in &parts {
                    let p = planned.iter().find(|p| p.decl.name == *target).unwrap();
                    let want = if *dtype == Dtype::F32 {
                        GgmlType::F32
                    } else {
                        GgmlType::BF16
                    };
                    assert_eq!(p.decl.type_id, want.as_u32(), "{target}, q8={q8}");
                    assert_eq!(
                        p.decl.nbytes,
                        p.decl.dims.iter().product::<u64>() * want.type_size().unwrap(),
                        "{target}, q8={q8}"
                    );
                }
            }
        }
        for (target, source) in [
            (
                names::attn_output(3),
                "model.language_model.layers.3.self_attn.o_proj.weight",
            ),
            (names::output(), "lm_head.weight"),
            (
                names::token_embd(),
                "model.language_model.embed_tokens.weight",
            ),
        ] {
            let mut f32 = entries.clone();
            f32.iter_mut().find(|(n, _, _)| n == source).unwrap().1 = Dtype::F32;
            for q8 in [false, true] {
                let result = plan_tensors(&checkpoint, &quant, &f32, q8, Keep::Made);
                if !q8 {
                    let Err(e) = result else {
                        panic!("F32 matrix {target} must be refused");
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source) && e.contains(&target) && e.contains(PREPARE_Q8),
                        "{e}"
                    );
                } else {
                    let (_, planned) = result.unwrap();
                    let p = planned.iter().find(|p| p.decl.name == target).unwrap();
                    assert_eq!(p.decl.type_id, 8, "{target}");
                }
            }
        }
        let bf16: Vec<_> = entries
            .iter()
            .filter_map(|(n, d, s)| {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    n.ends_with(".weight")
                        .then(|| (n.clone(), Dtype::Bf16, Format::Nvfp4.logical(s)))
                } else {
                    Some((n.clone(), *d, s.clone()))
                }
            })
            .collect();
        let mut single = config;
        single["text_config"]["num_experts"] = 1.into();
        single["text_config"]["num_experts_per_tok"] = 1.into();
        let single = Checkpoint::parse(&single).unwrap();
        for dtype in [Dtype::Bf16, Dtype::F32] {
            let mut stack = bf16.clone();
            for (n, d, _) in &mut stack {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    *d = dtype;
                }
            }
            let source = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
            let one: Vec<_> = stack
                .iter()
                .filter(|(n, _, _)| {
                    n == source
                        || n == "lm_head.weight"
                        || n == "model.language_model.embed_tokens.weight"
                })
                .cloned()
                .collect();
            for (checkpoint, stack) in [(&checkpoint, &stack), (&single, &one)] {
                for q8 in [false, true] {
                    let Err(e) = plan_tensors(checkpoint, &quant, stack, q8, Keep::Made) else {
                        panic!(
                            "{dtype:?} expert stacks must be refused for q8={q8}, entries={}",
                            stack.len()
                        );
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source)
                            && e.contains(&names::ffn_gate_exps(0))
                            && e.contains("expert stacks are unsupported"),
                        "{e}"
                    );
                }
            }
        }
        for q8 in [false, true] {
            let args = ["no-folder", "out", "--keep-bf16"].map(String::from);
            let e = run(&args, q8).unwrap_err().to_string();
            assert_eq!(e.contains("--keep-bf16"), q8, "q8={q8}: {e}");
        }
    }
}

/// A toy Kimi folder, built as the integration tests build theirs.
#[cfg(test)]
#[path = "../../tests/common/kimi_toy.rs"]
mod kimi_toy;

#[cfg(test)]
mod follow_tests {
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::time::Duration;

    use super::{kimi_toy, run_in};

    /// One `--shard-gib` is a mebibyte here, so the toy's twenty megabytes make eleven shards.
    const UNIT: u64 = 1 << 20;

    /// The toy folder's shards in the order they land in the tests: layers 3, 1, 2, 0, then the
    /// head, which every output shard of the head's tensors waits on. The vision tower's shard,
    /// which no tensor reads, is the sixth.
    const LANDING: [&str; 5] = [
        "model-00004-of-00006.safetensors",
        "model-00002-of-00006.safetensors",
        "model-00003-of-00006.safetensors",
        "model-00001-of-00006.safetensors",
        "model-00005-of-00006.safetensors",
    ];

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("prepare-follow-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `prepare` run in process over `args`, its refusal as text.
    fn prepare(args: &[&str]) -> Result<(), String> {
        let args: Vec<String> = args.iter().map(ToString::to_string).collect();
        run_in(&args, false, UNIT).map_err(|e| e.to_string())
    }

    /// `prepare folder prefix` of every tensor in shards of 2 MiB, with `extra` flags. A whole
    /// Kimi split is refused until the program reads the type of its tensors, so a split of all of
    /// them is a selection (`--only '*'`), checked by its tensor set.
    fn split(folder: &Path, prefix: &Path, extra: &[&str]) -> Result<(), String> {
        let mut args = vec![folder.to_str().unwrap(), prefix.to_str().unwrap()];
        args.extend(["--only", "*", "--shard-gib", "2"]);
        args.extend(extra);
        prepare(&args)
    }

    /// [`split`] following the listing at `list`, bounded by `stall` seconds.
    fn follow(folder: &Path, prefix: &Path, list: &Path, stall: &str) -> Result<(), String> {
        split(
            folder,
            prefix,
            &[
                "--follow",
                list.to_str().unwrap(),
                "--follow-poll",
                "0.02",
                "--follow-stall",
                stall,
            ],
        )
    }

    /// The listing of a toy folder: every file beside the shards and every shard, `path<TAB>bytes`.
    fn list(dir: &Path, shards: &kimi_toy::Shards, out: &Path) {
        let mut text = String::new();
        for f in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tiktoken.model",
            "tokenization_kimi.py",
            "model.safetensors.index.json",
        ] {
            text += &format!("{f}\t{}\n", std::fs::metadata(dir.join(f)).unwrap().len());
        }
        for (file, tensors) in shards {
            text += &format!("{file}\t{}\n", kimi_toy::shard_bytes(tensors).len());
        }
        std::fs::write(out, text).unwrap();
    }

    /// The shards of a split at `prefix`, by their names after the prefix, with their bytes.
    fn written(prefix: &Path) -> Vec<(String, Vec<u8>)> {
        let (dir, stem) = (
            prefix.parent().unwrap(),
            prefix.file_name().unwrap().to_str().unwrap(),
        );
        let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?.strip_prefix(stem)?.to_string();
                name.ends_with(".gguf")
                    .then(|| (name, std::fs::read(&p).unwrap()))
            })
            .collect();
        out.sort();
        out
    }

    /// The split of the finished toy folder, written without a follow: what every follow must
    /// equal.
    fn reference() -> &'static Vec<(String, Vec<u8>)> {
        static REFERENCE: OnceLock<Vec<(String, Vec<u8>)>> = OnceLock::new();
        REFERENCE.get_or_init(|| {
            let root = scratch("reference");
            let folder = root.join("folder");
            kimi_toy::write_folder(&folder, true, None);
            split(&folder, &root.join("whole"), &[]).unwrap();
            let out = written(&root.join("whole"));
            std::fs::remove_dir_all(&root).unwrap();
            out
        })
    }

    /// The staged shards marked done under `stage`.
    fn marked(stage: &Path) -> Vec<String> {
        let Ok(dir) = std::fs::read_dir(stage) else {
            return vec![];
        };
        let mut names: Vec<String> = dir
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".gguf.done"))
            .collect();
        names.sort();
        names
    }

    /// A shard file arriving as a download does: its first half, a pause, the rest.
    fn deliver(dir: &Path, file: &str, bytes: &[u8], pause: Duration) {
        use std::io::Write;
        let at = dir.join(file);
        let (head, tail) = bytes.split_at(bytes.len() / 2);
        std::fs::write(&at, head).unwrap();
        std::thread::sleep(pause);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&at)
            .unwrap()
            .write_all(tail)
            .unwrap();
    }

    /// A toy K3.1 folder with its config, tokenizer files and index and none of its shards, and
    /// the listing of all of them at `root/list.tsv`.
    fn bare(root: &Path) -> (PathBuf, kimi_toy::Shards, PathBuf) {
        let folder = root.join("folder");
        let shards = kimi_toy::write_folder(&folder, true, Some(&[]));
        let list_path = root.join("list.tsv");
        list(&folder, &shards, &list_path);
        (folder, shards, list_path)
    }

    fn put(folder: &Path, shards: &kimi_toy::Shards, files: &[&str]) {
        for file in files {
            std::fs::write(folder.join(file), kimi_toy::shard_bytes(&shards[*file])).unwrap();
        }
    }

    /// The split `--follow` writes behind a download whose shards land out of order and in two
    /// halves is the split of the finished folder, byte for byte; output shards are staged while
    /// the last shard they need has not landed, and no stage is left.
    #[test]
    fn a_followed_split_is_the_finished_folders_split_whatever_order_its_shards_land_in() {
        let reference = reference();
        assert!(reference.len() >= 3, "{} output shards", reference.len());
        let root = scratch("order");
        let (folder, shards, list_path) = bare(&root);
        let (followed, stage) = (root.join("followed"), root.join("followed.follow"));
        let landing = {
            let (folder, stage) = (folder.clone(), stage.clone());
            std::thread::spawn(move || {
                let mut before_the_head = 0;
                for (n, file) in LANDING.iter().enumerate() {
                    if n == LANDING.len() - 1 {
                        // The head is the last shard a tensor reads: an output shard must be
                        // staged before it lands.
                        let until = std::time::Instant::now() + Duration::from_secs(30);
                        while marked(&stage).is_empty() && std::time::Instant::now() < until {
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        before_the_head = marked(&stage).len();
                    }
                    let bytes = kimi_toy::shard_bytes(&shards[*file]);
                    deliver(&folder, file, &bytes, Duration::from_millis(900));
                    std::thread::sleep(Duration::from_millis(100));
                }
                before_the_head
            })
        };
        // The landing takes longer than the stall bound and never goes that long without a byte.
        follow(&folder, &followed, &list_path, "3").unwrap();
        assert!(
            landing.join().unwrap() >= 1,
            "no output shard was staged before the head landed"
        );
        assert_eq!(&written(&followed), reference);
        assert!(
            !stage.exists(),
            "the stage is removed once the split is published"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A download that stops is refused by name once nothing has grown for the stall bound, with
    /// the output shards its landed shards complete staged; a stage of another plan is refused; a
    /// rerun of the same plan after the rest lands skips the staged shards, removes a partial one
    /// and an unmarked one, and writes the finished folder's split.
    #[test]
    fn a_stalled_follow_is_refused_by_name_and_its_rerun_resumes_from_the_staged_shards() {
        let reference = reference();
        let root = scratch("resume");
        let (folder, shards, list_path) = bare(&root);
        let (followed, stage) = (root.join("followed"), root.join("followed.follow"));
        put(&folder, &shards, &LANDING[..2]);

        let stalled = follow(&folder, &followed, &list_path, "1").unwrap_err();
        assert!(
            stalled.contains("no listed byte arrived for 1 s"),
            "{stalled}"
        );
        assert!(
            stalled.contains("model-00003-of-00006.safetensors"),
            "{stalled}"
        );
        let staged = marked(&stage);
        assert!(
            !staged.is_empty() && staged.len() < reference.len(),
            "{staged:?}"
        );
        let stamps: Vec<_> = staged
            .iter()
            .map(|d| {
                let shard = stage.join(d.strip_suffix(".done").unwrap());
                (
                    shard.clone(),
                    std::fs::metadata(&shard).unwrap().modified().unwrap(),
                )
            })
            .collect();
        assert!(
            written(&followed).is_empty(),
            "a split was published before it was whole"
        );

        // Another plan's stage is refused, and left as it is.
        let other = split(
            &folder,
            &followed,
            &[
                "--shard-gib",
                "3",
                "--follow",
                list_path.to_str().unwrap(),
                "--follow-stall",
                "1",
            ],
        )
        .unwrap_err();
        assert!(other.contains("staged by another plan"), "{other}");
        assert_eq!(marked(&stage), staged);

        // A partial shard and an unmarked one, as a crash leaves them.
        let names: Vec<String> = (1..=reference.len())
            .map(|n| format!("followed-{n:05}-of-{:05}.gguf", reference.len()))
            .collect();
        let todo: Vec<&String> = names
            .iter()
            .filter(|n| !staged.contains(&format!("{n}.done")))
            .collect();
        std::fs::write(stage.join(format!("{}.part", todo[0])), b"partial").unwrap();
        std::fs::write(stage.join(todo[1]), b"unmarked").unwrap();

        put(&folder, &shards, &LANDING[2..]);
        follow(&folder, &followed, &list_path, "60").unwrap();
        assert_eq!(&written(&followed), reference);
        for (shard, stamp) in stamps {
            let published = followed.with_file_name(shard.file_name().unwrap());
            assert_eq!(
                std::fs::metadata(published).unwrap().modified().unwrap(),
                stamp,
                "{} was written again",
                shard.display()
            );
        }
        assert!(!stage.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A shard whose header disagrees with the config is refused by name when it completes,
    /// before an output shard reads it: a tensor of another shape, one of another dtype, one the
    /// header lacks and one the config does not derive and the map does not skip.
    #[test]
    fn a_shard_whose_header_disagrees_with_the_config_is_refused_by_name() {
        let file = LANDING[3];
        let case = |tag: &str, edit: &dyn Fn(&mut kimi_toy::Rows) -> String| {
            let root = scratch(tag);
            let (folder, mut shards, _) = bare(&root);
            let name = edit(shards.get_mut(file).unwrap());
            list(&folder, &shards, &root.join("list.tsv"));
            put(&folder, &shards, &[file]);
            let err = follow(&folder, &root.join("out"), &root.join("list.tsv"), "5").unwrap_err();
            assert!(marked(&root.join("out.follow")).is_empty());
            std::fs::remove_dir_all(&root).unwrap();
            (name, err)
        };
        let (name, err) = case("shape", &|ts| {
            let t = ts
                .iter_mut()
                .find(|t| t.2.len() == 2 && t.2[0] != t.2[1])
                .unwrap();
            t.2.reverse();
            t.0.clone()
        });
        assert!(err.starts_with(&format!("{name}: {file} stores")), "{err}");
        assert!(err.contains("the config derives"), "{err}");
        let (name, err) = case("dtype", &|ts| {
            let t = ts
                .iter_mut()
                .find(|t| t.1 == safetensors::Dtype::Bf16)
                .unwrap();
            t.1 = safetensors::Dtype::F32;
            t.3 = vec![0; t.3.len() * 2];
            t.0.clone()
        });
        assert!(err.starts_with(&format!("{name}: {file} stores")), "{err}");
        let (name, err) = case("lacking", &|ts| ts.remove(0).0);
        assert_eq!(
            err,
            format!("{name}: the config derives it in {file}; its header does not hold it")
        );
        let (name, err) = case("extra", &|ts| {
            let name = "language_model.model.layers.0.surprise.weight".to_string();
            ts.push((name.clone(), safetensors::Dtype::Bf16, vec![2], vec![0; 4]));
            name
        });
        assert_eq!(
            err,
            format!(
                "{name}: {file} holds it; the config derives no such tensor and the map does not skip it"
            )
        );
    }

    /// What the plan reads is waited for and what is missing is named: a shard the listing gives
    /// no size is refused at once, and a plan file that never completes ends the wait.
    #[test]
    fn a_listing_without_a_shard_and_a_plan_file_that_never_lands_are_refused_by_name() {
        let root = scratch("plan");
        let (folder, shards, list_path) = bare(&root);
        let text = std::fs::read_to_string(&list_path).unwrap();
        let without: String = text
            .lines()
            .filter(|l| !l.starts_with(LANDING[2]))
            .map(|l| format!("{l}\n"))
            .collect();
        std::fs::write(&list_path, without).unwrap();
        let err = follow(&folder, &root.join("out"), &list_path, "5").unwrap_err();
        assert_eq!(
            err,
            format!("{}: the listing names no size for it", LANDING[2])
        );

        list(&folder, &shards, &list_path);
        std::fs::remove_file(folder.join("tiktoken.model")).unwrap();
        std::fs::remove_file(folder.join("tokenization_kimi.py")).unwrap();
        let err = follow(&folder, &root.join("out"), &list_path, "0.3").unwrap_err();
        assert!(
            err.contains("the plan waits on tiktoken.model, tokenization_kimi.py"),
            "{err}"
        );
        assert!(!root.join("out.follow").join("plan.key").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The flags' refusals, each before the plan is read: a bound without `--follow`, a number
    /// that is no number of seconds, `--follow` twice or with `--plan`, and a listing that is
    /// not there.
    #[test]
    fn the_follow_flags_are_refused_by_name() {
        let root = scratch("flags");
        let (folder, out) = (root.join("folder"), root.join("out"));
        let run = |extra: &[&str]| {
            let mut args = vec![folder.to_str().unwrap(), out.to_str().unwrap()];
            args.extend(extra);
            prepare(&args).unwrap_err()
        };
        assert!(
            run(&["--follow-stall", "5"]).contains("bound a wait for the download --follow reads")
        );
        assert!(run(&["--follow-poll", "5"]).contains("give --follow <list>"));
        assert!(run(&["--follow", "a", "--follow", "b"]).contains("--follow given twice"));
        for bad in ["0", "-1", "abc", "NaN", "inf"] {
            let e = run(&["--follow", "a", "--follow-poll", bad]);
            assert!(
                e.starts_with(&format!("--follow-poll {bad:?}: not ")),
                "{bad}: {e}"
            );
        }
        assert!(run(&["--follow", "a", "--follow-stall", "0"]).contains("not above zero seconds"));
        let e = prepare(&[folder.to_str().unwrap(), "--plan", "--follow", "a"]).unwrap_err();
        assert!(e.contains("--plan prints the folder's stored form"), "{e}");
        let missing = root.join("no-list.tsv");
        kimi_toy::write_folder(&folder, true, Some(&[]));
        let e = run(&["--follow", missing.to_str().unwrap()]);
        assert!(e.contains("no-list.tsv"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
