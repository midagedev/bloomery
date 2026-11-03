//! `generate_ds41 --place tp`: one rank of a tensor-parallel greedy run of an original V4.1
//! checkpoint folder, one process a node.
//!
//!     generate_ds41 --place tp --ranks FILE --rank R (--folder DIR | --store DIR)
//!                   [--fabric MANIFEST] [--depth D | --text T] [-n N] [--ctx C]
//!
//! Rank 0 holds the folder (`--folder`) and sends every other rank the byte ranges it reads
//! (`model_hf::arch::deepseek41::rank_reads`) that its store does not hold yet
//! (`bloomery_gpu::host::rank::dist`); each digest is the folder's `SHA256SUMS` line. Every other
//! rank keeps them in its store (`--store`) and opens the folder through a view of links, each
//! file's name to its sparse copy, after checking that its store covers every range it reads.
//! Every rank then opens the group (`RankSession::open`, the rank file and the fabric manifest
//! as `link_probe` reads them), agrees on the source, the headers this rank reads and the
//! build, loads its rank of the model (`body::open_rank`) and runs the group's greedy run
//! (`app::tp::greedy`): rank 0 feeds the `--depth` ids (the depth sequence), or the folder
//! tokenizer's ids of `--text`, as steps, then generates N tokens. Each rank prints a `tp run` record; rank 0 then prints the `tokens` and
//! their decode by the folder's tokenizer (`tp text`).
//!
//! Defaults: D 4096, N 128, C D + N.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use app::tp::greedy;
use bloomery_gpu::Gpu;
use bloomery_gpu::host::rank::dist::{self, ReadRange, SourceFile};
use bloomery_gpu::host::rank::{Bootstrap, Fingerprint, RankList, RankSession};
use bloomery_gpu_deepseek41::body::{self, OpenCfg, RankFrom};
use bloomery_gpu_deepseek41::chain::join;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_levers::Levers;
use gguf::digest::{Sha256, hex, unhex};
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::tp::{TpUnits, cut_of, rank_span};
use model::placement::HEAD_TENSOR;
use model_hf::arch::deepseek41::rank_reads;
use model_hf::folder::{self, Opened};

const USAGE: &str = "usage: generate_ds41 --place tp --ranks FILE --rank R (--folder DIR | --store \
                     DIR) [--fabric MANIFEST] [--depth D | --text T] [-n N] [--ctx C]";

/// How long a rank waits for the others at the distribution and at the group's open.
const WAIT: Duration = Duration::from_secs(600);

/// The version of the normalization the fingerprint names (`model_hf::arch::deepseek41`).
const FORM: u32 = 1;

struct Opts {
    ranks: RankList,
    rank: usize,
    folder: Option<PathBuf>,
    store: Option<PathBuf>,
    fabric: Option<PathBuf>,
    depth: usize,
    text: Option<String>,
    n: usize,
    ctx: Option<usize>,
}

fn opts(args: &[String]) -> Result<Opts, GateError> {
    let (mut ranks, mut rank, mut folder, mut store, mut fabric) = (None, None, None, None, None);
    let (mut depth, mut text, mut n, mut ctx) = (4096, None, 128, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
        match flag.as_str() {
            "--place" if v == "tp" => {}
            "--ranks" => ranks = Some(RankList::read(v.as_ref())?),
            "--rank" => rank = Some(v.parse::<usize>()?),
            "--folder" => folder = Some(PathBuf::from(v)),
            "--store" => store = Some(PathBuf::from(v)),
            "--fabric" => fabric = Some(PathBuf::from(v)),
            "--depth" => depth = v.parse()?,
            "--text" => text = Some(v.clone()),
            "-n" => n = v.parse()?,
            "--ctx" => ctx = Some(v.parse()?),
            _ => return Err(format!("{flag} {v}: not a --place tp flag; {USAGE}").into()),
        }
    }
    let (Some(ranks), Some(rank)) = (ranks, rank) else {
        return Err(format!("--ranks and --rank are required; {USAGE}").into());
    };
    if rank >= ranks.len() || depth == 0 {
        return Err(format!("--rank below the rank file's ranks and --depth >= 1; {USAGE}").into());
    }
    let held = if rank == 0 {
        folder.is_some() && store.is_none()
    } else {
        store.is_some() && folder.is_none()
    };
    if !held {
        return Err(format!("rank 0 takes --folder, every other rank --store; {USAGE}").into());
    }
    Ok(Opts {
        ranks,
        rank,
        folder,
        store,
        fabric,
        depth,
        text,
        n,
        ctx,
    })
}

/// The source files of the folder at `dir`: every file its `SHA256SUMS` names (a path inside
/// `dir`), in name order, with its digest and length, each checked against its digest once
/// (`dist::verify`, its record under `~/.cache/bloomery/verified`).
fn source_files(dir: &Path) -> Result<Vec<SourceFile>, GateError> {
    let sums = dir.join("SHA256SUMS");
    let text = std::fs::read_to_string(&sums).map_err(|e| format!("{}: {e}", sums.display()))?;
    let mut files = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let bad = || format!("{}: {line:?} is not `<sha256>  <name>`", sums.display());
        let (digest, name) = line.split_once(char::is_whitespace).ok_or_else(bad)?;
        let name = name
            .trim_start()
            .trim_start_matches('*')
            .trim_start_matches("./");
        if !dist::inside(name) {
            return Err(format!(
                "{}: {name:?} is not a path inside the folder",
                sums.display()
            )
            .into());
        }
        let digest = unhex(digest).ok_or_else(bad)?;
        let size = std::fs::metadata(dir.join(name))
            .map_err(|e| format!("{}: {e}", dir.join(name).display()))?
            .len();
        if files.insert(name.to_string(), (digest, size)).is_some() {
            return Err(format!("{}: {name:?} is listed twice", sums.display()).into());
        }
    }
    let files: Vec<SourceFile> = files
        .into_iter()
        .map(|(name, (digest, size))| SourceFile { name, digest, size })
        .collect();
    let home = std::env::var_os("HOME")
        .ok_or("HOME is not set: the head records its checked files under it")?;
    let abs = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut h = Sha256::new();
    h.update(abs.as_os_str().as_encoded_bytes());
    let record = PathBuf::from(home)
        .join(".cache/bloomery/verified")
        .join(format!("{}.tsv", &hex(&h.finish())[..16]));
    dist::verify(dir, &files, &record)?;
    Ok(files)
}

/// `reads` of the folder `root` as ranges of `files` (`dist::read_ranges`), the folder's
/// `SHA256SUMS` itself left out.
fn ranges(
    files: &[SourceFile],
    root: &Path,
    mut reads: Vec<(PathBuf, std::ops::Range<u64>)>,
) -> Result<Vec<ReadRange>, GateError> {
    let sums = root.join("SHA256SUMS");
    reads.retain(|(p, _)| *p != sums);
    Ok(dist::read_ranges(files, root, &reads)?)
}

/// A rank's folder: rank 0's own, every other rank's the view of its store, each file's name a
/// link to its sparse copy, opened once the store covers every range the rank reads.
fn rank_folder(o: &Opts, ranks: usize) -> Result<(Opened, Vec<SourceFile>), GateError> {
    let start = Instant::now();
    if let Some(dir) = &o.folder {
        let files = source_files(dir)?;
        let checked = start.elapsed().as_secs_f64();
        let opened = folder::open(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let reads = (1..ranks)
            .map(|r| ranges(&files, dir, rank_reads(&opened, r, ranks)?))
            .collect::<Result<Vec<_>, _>>()?;
        let planned = start.elapsed().as_secs_f64();
        let served = dist::serve(&o.ranks, WAIT, dir, &files, &|r| reads[r - 1].clone())?;
        for s in served {
            eprintln!("tp: rank {} sent {} B, kept {} B", s.rank, s.sent, s.kept);
        }
        eprintln!(
            "tp: the head checked its files in {checked:.1} s, planned the reads in {:.1} s and \
             sent them in {:.1} s",
            planned - checked,
            start.elapsed().as_secs_f64() - planned
        );
        return Ok((opened, files));
    }
    let store_dir = o.store.as_deref().ok_or(USAGE)?;
    let fetched = dist::fetch(&o.ranks, o.rank, WAIT, store_dir)?;
    let files: Vec<SourceFile> = fetched.files.iter().map(|(f, _, _)| f.clone()).collect();
    let view = store_dir.join(format!("view-{}", &hex(&source_digest(&files))[..16]));
    std::fs::create_dir_all(&view).map_err(|e| format!("{}: {e}", view.display()))?;
    for (f, sparse, _) in &fetched.files {
        // The link's target holds from the view's folder only when it is absolute.
        if !sparse.is_absolute() {
            return Err(
                format!("{}: a sparse copy's path is not absolute", sparse.display()).into(),
            );
        }
        // The link stays in the view: every part of the name is a plain one.
        let mut link = view.clone();
        for part in Path::new(&f.name).components() {
            let Component::Normal(part) = part else {
                return Err(format!("{:?} is not a name inside the folder", f.name).into());
            };
            link.push(part);
        }
        if let Some(parent) = link.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        match std::fs::read_link(&link) {
            Ok(to) if to == *sparse => continue,
            Ok(_) => std::fs::remove_file(&link).map_err(|e| format!("{}: {e}", link.display()))?,
            Err(_) if link.symlink_metadata().is_ok() => {
                return Err(format!("{}: a file the view does not own", link.display()).into());
            }
            Err(_) => {}
        }
        std::os::unix::fs::symlink(sparse, &link)
            .map_err(|e| format!("{}: {e}", link.display()))?;
    }
    let opened = folder::open(&view).map_err(|e| format!("{}: {e}", view.display()))?;
    for r in ranges(&files, &view, rank_reads(&opened, o.rank, ranks)?)? {
        if !fetched.files[r.file].2.holds(r.offset, r.len) {
            return Err(format!(
                "rank {}: the store holds no verified {} B at {} of {}",
                o.rank, r.len, r.offset, files[r.file].name
            )
            .into());
        }
    }
    eprintln!(
        "tp: rank {} received {} B and kept {} B in {:.1} s",
        o.rank,
        fetched.sent,
        fetched.kept,
        start.elapsed().as_secs_f64()
    );
    Ok((opened, files))
}

/// The SHA-256 over every source file's digest, in order.
fn source_digest(files: &[SourceFile]) -> [u8; 32] {
    let mut h = Sha256::new();
    for f in files {
        h.update(&f.digest);
    }
    h.finish()
}

/// The identity of the headers the group reads, the same on every rank that reads the same
/// header: each tensor's name, type and dims, and how the model's cut splits it among the ranks
/// (the dim and the unit it splits by, and the blocks a part keeps whole). Each rank's own part
/// follows from these and its rank, which the fingerprint names apart.
fn headers_digest(header: &gguf::Split, u: &TpUnits) -> [u8; 32] {
    let mut h = Sha256::new();
    for (_, t) in header.iter_tensors() {
        let cut = cut_of(&t.name, t.ty, &t.dims, u)
            .map(|c| c.map(|(c, blocks)| (c.dim, c.unit, blocks)))
            .map_err(|e| e.to_string());
        h.update(format!("{} {} {:?} {cut:?}\n", t.name, t.ty, t.dims).as_bytes());
    }
    h.finish()
}

pub fn run(args: &[String], levers: &Levers) -> Result<(), GateError> {
    let o = opts(args)?;
    let ranks = o.ranks.len();
    let cfg = OpenCfg::from_levers(levers)?;
    let (opened, files) = rank_folder(&o, ranks)?;
    let Opened {
        header,
        source,
        rows,
        tokenizer,
        ..
    } = opened;
    let hp = Hparams::read(&header).map_err(|e| e.to_string())?;
    let u = TpUnits::of(&hp);
    let fp = Fingerprint {
        source: source_digest(&files),
        headers: headers_digest(&header, &u),
        form: FORM,
        build: env!("CARGO_PKG_VERSION").to_string(),
        n: u16::try_from(ranks)?,
        rank: u16::try_from(o.rank)?,
    };
    let want_ctx = o.ctx.unwrap_or(o.depth + o.n);
    let how = Bootstrap {
        fabric: o.fabric.clone(),
        wait: WAIT,
    };
    let gpu = Gpu::new()?;
    let mut session = RankSession::open(
        &gpu,
        &o.ranks,
        o.rank,
        &how,
        &fp,
        &[("ctx", want_ctx as u64)],
        &join::shapes(&hp)?,
    )?;
    let ctx = session
        .limits()
        .iter()
        .find(|(k, _)| *k == "ctx")
        .map_or(want_ctx, |&(_, v)| v as usize);
    let (_, head) = header.find(HEAD_TENSOR).ok_or("the folder holds no head")?;
    let vocab_at = rank_span(&head.name, head.ty, &head.dims, &u, o.rank, ranks)?
        .map_or(0, |(_, at)| at.start);
    let sums = session.sums();
    let t_load = Instant::now();
    // A rank whose load fails tells the group, so every rank names it.
    let mut m = body::open_rank(
        gpu,
        RankFrom {
            file: header,
            source: &source,
            stored: Some(&rows),
        },
        (o.rank, ranks),
        sums,
        ctx,
        cfg.body,
    )
    .map_err(|e| session.fail(&format!("its load: {e}")))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let prompt = match (o.rank, &o.text) {
        (0, Some(text)) => tokenizer.encode(text, true, false),
        (0, None) => crate::drive::depth_ids(o.depth),
        _ => Vec::new(),
    };
    let (mut prompt_s, mut decode_s, mut prompted, mut decoded) = (0.0, 0.0, 0usize, 0usize);
    let tokens = greedy(
        &mut session,
        o.rank,
        u32::try_from(vocab_at)?,
        (&prompt, o.n),
        |round| {
            let t = Instant::now();
            let mut pick = 0;
            for &id in &round.tokens {
                pick = m.step(&[id])?;
            }
            let logits = m.logits().map(|l| (pick, l));
            match round.prompt {
                Some(_) => {
                    prompt_s += t.elapsed().as_secs_f64();
                    prompted += round.tokens.len();
                }
                None => {
                    decode_s += t.elapsed().as_secs_f64();
                    decoded += round.tokens.len();
                }
            }
            logits
        },
    );
    let tokens = match tokens {
        Ok(tokens) => tokens,
        Err(e) => {
            // A release that did not settle leaves a card waiting on the page: dropping the
            // model would synchronize that wait for good.
            if matches!(e, bloomery_gpu::host::rank::RankError::Unsettled { .. }) {
                std::mem::forget(m);
            }
            return Err(e.into());
        }
    };
    Record::new(&record::TP_RUN)
        .u("rank", o.rank)
        .u("ranks", ranks)
        .u("ctx", ctx)
        .f("load_s", load_s)
        .u("resident_bytes", m.resident_bytes())
        .u("prompt", prompted)
        .f("prompt_s", prompt_s)
        .u("decoded", decoded)
        .f(
            "decode_tps",
            decoded as f64 / decode_s.max(f64::MIN_POSITIVE),
        )
        .print();
    if o.rank == 0 {
        Record::new(&record::TOKENS).list("tokens", &tokens).print();
        Record::new(&record::TP_TEXT)
            .w("text", format!("{:?}", tokenizer.decode(&tokens)))
            .print();
    }
    Ok(())
}
