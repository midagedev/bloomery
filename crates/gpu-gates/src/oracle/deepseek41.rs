//! The deepseek41 (DeepSeek-V4.1-Flash) oracle: ik's dump of this model,
//! made on the CPU backend of our port. No CUDA set is made for this model.

use super::Oracle;
use model::arch::Arch;

/// The decode-step sets, each one token after a prefill run node by node
/// under the dumped schedule (the model profile's `ref_step_variant`,
/// `tools/ref/models/deepseek41.sh`): step4 at an even position, where no
/// csa group completes; d1 at 301 and d2 at 1,025, where one does, the
/// window mask 512 and 1,280 cells wide; d1 and d2 once more with the
/// indexer's scores and top-k as nodes of their own; d1n at 301 with the
/// file's top-k, where neither stream builds an indexer.
pub const STEP_SETS: &[&str] = &[
    "ref_deepseek41_step4_every_node",
    "ref_deepseek41_d1_every_node",
    "ref_deepseek41_d1_unfused_every_node",
    "ref_deepseek41_d1n_every_node",
    "ref_deepseek41_d2_every_node",
    "ref_deepseek41_d2_unfused_every_node",
];

pub static ORACLE: Oracle = Oracle {
    arch: Arch::Deepseek41,
    cuda_set: None,
    cpu_set: "ref_deepseek41",
    legacy_cuda_set: None,
    step_sets: STEP_SETS,
    taps: &[],
};

#[cfg(test)]
mod tests {
    use crate::oracle::{Set, for_arch};
    use crate::{RefManifest, RefRow, ref_ints_of_in, verdict};
    use model::arch::Arch;

    /// Rows of one kind read, with a logical twin, and failed; a row fails
    /// once however many of its checks do.
    #[derive(Default)]
    struct Tally {
        rows: u64,
        logical: u64,
        failed: u64,
    }

    /// `file` is in the set with `want` bytes. A tensor or input file gets
    /// this check only: its values are not read.
    fn sized(man: &RefManifest, file: &str, want: u64) -> Result<(), String> {
        let path = man.dir.join(file);
        match std::fs::metadata(&path) {
            Ok(m) if m.len() == want => Ok(()),
            Ok(m) => Err(format!(
                "{} is {} bytes, want {want}",
                path.display(),
                m.len()
            )),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// A tensor or input row: manifest bytes = 4 × count, and the plain file
    /// named by the file-name rule holds 4 × count bytes — its logical twin
    /// too when the row has one.
    fn read_f32_row(man: &RefManifest, row: &RefRow, t: &mut Tally, fails: &mut Vec<String>) {
        t.rows += 1;
        let want = 4 * row.count();
        let mut errs = Vec::new();
        if row.bytes != want {
            errs.push(format!("manifest bytes {} != 4 x count {want}", row.bytes));
        }
        errs.extend(sized(man, &row.file_name(), want).err());
        if row.logical == Some(1) {
            t.logical += 1;
            errs.extend(sized(man, &row.logical_file_name(), want).err());
        }
        if !errs.is_empty() {
            t.failed += 1;
            fails.push(format!(
                "{} {}/{}: {}",
                row.kind.as_str(),
                row.name,
                row.occurrence,
                errs.join("; ")
            ));
        }
    }

    /// The harness reads every row of the V4.1 oracle set, opened through
    /// the table: each `tensor` and `input` row's files (above), each `int`
    /// row through the integer reader — file, count, bytes, sum and absmax
    /// against the row — agreeing with the row it twins on type and count;
    /// `(kind, name, occurrence)` names one row; and the `# complete` trailer
    /// counts the node rows the manifest holds, so no row went missing.
    #[test]
    #[ignore = "hw: needs the V4.1 oracle set in $BLOOMERY_DATA on the box"]
    fn hw_ds41_oracle_reads_every_row() {
        let man = for_arch(Arch::Deepseek41)
            .and_then(|o| o.open(Set::Cpu))
            .unwrap_or_else(|e| panic!("hw_ds41_oracle: {e}"));
        let mut fails = Vec::new();
        let (mut tensor, mut input, mut int) =
            (Tally::default(), Tally::default(), Tally::default());
        for row in &man.tensors {
            read_f32_row(&man, row, &mut tensor, &mut fails);
        }
        for row in &man.inputs {
            read_f32_row(&man, row, &mut input, &mut fails);
        }

        let duplicates = man.duplicate_keys();
        for row in &man.ints {
            int.rows += 1;
            let checked = ref_ints_of_in(&man.dir, row)
                .map_err(|e| e.to_string())
                .and_then(|v| match man.find(row.of, &row.name, row.occurrence) {
                    None => Err("it twins a row the manifest does not hold".to_string()),
                    Some(t) if t.ty != row.ty || t.count() != v.len() as u64 => Err(format!(
                        "the row it twins is {} x {}, the twin {} x {}",
                        t.ty,
                        t.count(),
                        row.ty,
                        v.len()
                    )),
                    Some(_) => Ok(()),
                });
            if let Err(e) = checked {
                int.failed += 1;
                fails.push(format!("{row}: {e}"));
            }
        }
        let trailer_ok = man.complete == Some((man.tensors.len() as u64, man.skipped_nodes));

        for f in fails.iter().take(20) {
            println!("hw_ds41_oracle: FAIL {f}");
        }
        if fails.len() > 20 {
            println!("hw_ds41_oracle: ... and {} more", fails.len() - 20);
        }
        let pass = fails.is_empty() && trailer_ok && duplicates == 0;
        println!(
            "hw_ds41_oracle: {} (arch {}, build {}) — read tensor {} (+{} logical), input {} (+{} logical), \
             int {}, skip {} — failed tensor {}, input {}, int {}; trailer {:?} for {}/{}; duplicate keys {} — {}",
            man.dir.display(),
            man.arch.as_deref().unwrap_or("-"),
            man.build.as_deref().unwrap_or("-"),
            tensor.rows,
            tensor.logical,
            input.rows,
            input.logical,
            int.rows,
            man.skipped_nodes + man.skipped_inputs,
            tensor.failed,
            input.failed,
            int.failed,
            man.complete,
            man.tensors.len(),
            man.skipped_nodes,
            duplicates,
            verdict(pass)
        );
        assert!(
            pass,
            "hw_ds41_oracle: {} unreadable row(s), trailer ok {trailer_ok}, duplicate keys {duplicates}",
            fails.len()
        );
    }

    /// Every decode-step set of the table opens by name — its manifest
    /// parses and names this architecture — was written by the batch set's
    /// ik build (`# build`), and carries the step's position (`# decode_pos`).
    #[test]
    #[ignore = "hw: needs the V4.1 oracle sets in $BLOOMERY_DATA on the box"]
    fn hw_ds41_oracle_step_sets_share_the_batch_build() {
        let o = for_arch(Arch::Deepseek41).unwrap_or_else(|e| panic!("hw_ds41_oracle: {e}"));
        let batch = o
            .open(Set::Cpu)
            .unwrap_or_else(|e| panic!("hw_ds41_oracle: {e}"));
        let want = batch
            .build
            .as_deref()
            .unwrap_or_else(|| panic!("hw_ds41_oracle: {} has no # build", batch.dir.display()));
        let mut failed = 0usize;
        for &name in o.step_sets {
            let (pass, what) = match o.open_named(name) {
                Err(e) => (false, e.to_string()),
                Ok(step) => {
                    let build = step.build.as_deref();
                    let pos = step.header.decode_pos;
                    (
                        build == Some(want) && pos.is_some(),
                        format!(
                            "build {} (the batch set's {want}), decode_pos {}",
                            build.unwrap_or("-"),
                            pos.map_or_else(|| "-".to_string(), |p| p.to_string())
                        ),
                    )
                }
            };
            failed += usize::from(!pass);
            println!(
                "hw_ds41_oracle: step set {name}: {what} — {}",
                verdict(pass)
            );
        }
        let pass = failed == 0 && !o.step_sets.is_empty();
        println!(
            "hw_ds41_oracle: {} step sets, {failed} failed — {}",
            o.step_sets.len(),
            verdict(pass)
        );
        assert!(pass, "hw_ds41_oracle: {failed} step set(s) failed");
    }
}
