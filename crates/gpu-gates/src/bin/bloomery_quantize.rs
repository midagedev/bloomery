//! `bloomery-quantize` — llama-quantize's job in one pass, byte for byte: the routed-expert stacks quantized to
//! IQ1_S or IQ2_XXS on the card, every other tensor by the host's K-quant rules, the metadata and the shard layout as
//! llama-quantize writes them.
//!
//!     bloomery-quantize [options] <in first shard> <out> <ftype>
//!     bloomery-quantize --help | --version
//!
//! The plan, the type logic, the host rules and the writer are the `bloomery-quantize` crate's (`quantize`); this
//! binary is the card's [`Backend`] and the process around it. `--cpu` runs every rule on all cores and opens no
//! card; `--plan` opens none either. Exit: 0 done; 1 a refusal, named; 64 a usage error; 66 a missing input file.
//! `docs/quantize.md` is the user's page.

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::iq_quant::IqQuant;
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::iq_quant::IqKind;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use quantize::cli::Parsed;
#[cfg(feature = "gpu")]
use quantize::{Backend, BackendError, Error};
use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(feature = "gpu")]
    {
        real()
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("bloomery-quantize: built without the `gpu` feature");
        ExitCode::from(1)
    }
}

/// The plan over the input the command line names: a GGUF split, or under `hf` a checkpoint folder read as the split
/// `prepare` would write of it.
#[cfg(feature = "gpu")]
fn prepare(a: &quantize::cli::Args) -> quantize::Result<quantize::run::Prepared> {
    #[cfg(feature = "hf")]
    if let quantize::cli::Input::Hf(h) = &a.input {
        return quantize::run::prepare_from(a, model_hf::quantize_source::open(h)?);
    }
    quantize::run::prepare(a)
}

/// The card as a [`Backend`]: one quantizer a kind, the bound tensor's importance on the card.
#[cfg(feature = "gpu")]
struct CardBackend<'a> {
    gpu: &'a Gpu,
    ordinal: usize,
    quant: [Option<IqQuant>; 2],
    bound: Option<(usize, DeviceBuffer<f32>)>,
}

#[cfg(feature = "gpu")]
impl Backend for CardBackend<'_> {
    fn name(&self) -> String {
        format!("card {}", self.ordinal)
    }

    fn uses_host_pool(&self) -> bool {
        false
    }

    fn bind(&mut self, kind: IqKind, qw: &[f32]) -> Result<(), BackendError> {
        let slot = usize::from(kind == IqKind::Iq1S);
        if self.quant[slot].is_none() {
            self.quant[slot] = Some(IqQuant::new(self.gpu.context(), self.gpu.stream(), kind)?);
        }
        self.bound = Some((slot, DeviceBuffer::from_host(self.gpu.stream(), qw)?));
        Ok(())
    }

    fn quantize(
        &self,
        ty: GgmlType,
        src: &[u8],
        ne0: usize,
        row0: usize,
        rows_per_expert: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let (slot, qw) = self.bound.as_ref().ok_or("quantize before bind")?;
        let q = self.quant[*slot].as_ref().expect("bind made it");
        Ok(q.quantize(self.gpu.stream(), ty, src, ne0, row0, rows_per_expert, qw)?)
    }
}

#[cfg(feature = "gpu")]
fn real() -> ExitCode {
    if let Err(e) = bloomery_levers::at_main(&[]) {
        eprintln!("bloomery-quantize: {e}");
        return ExitCode::from(64);
    }
    match go() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bloomery-quantize: {e}");
            ExitCode::from(e.status())
        }
    }
}

#[cfg(feature = "gpu")]
fn go() -> Result<(), Error> {
    let a = match quantize::cli::parse(std::env::args().skip(1))? {
        Parsed::Help => {
            println!("{}", quantize::cli::USAGE);
            return Ok(());
        }
        Parsed::Version => {
            println!(
                "bloomery-quantize {} (commit {})",
                env!("CARGO_PKG_VERSION"),
                quantize::COMMIT
            );
            return Ok(());
        }
        Parsed::Run(a) => a,
    };
    let prep = prepare(&a)?;
    if a.plan {
        let free = os::free_bytes(&a.out_dir()).ok();
        print!(
            "{}",
            prep.plan.render(&a, free, quantize::host::pool_threads())
        );
        return Ok(());
    }
    // A refusal that needs no card comes before the card opens.
    quantize::run::check_room(&a, &prep.plan)?;
    let mut log = |l: &str| println!("{l}");
    let s = if a.cpu {
        quantize::run(&a, prep, &mut quantize::host::HostBackend::new(), &mut log)?
    } else {
        let ordinal = a.card.unwrap_or(0);
        let gpu = Gpu::open_ordinal(ordinal)
            .map_err(|e| Error::Refused(format!("card {ordinal}: {e}")))?;
        let mut b = CardBackend {
            gpu: &gpu,
            ordinal,
            quant: [None, None],
            bound: None,
        };
        quantize::run(&a, prep, &mut b, &mut log)?
    };
    println!(
        "bloomery-quantize: done in {:.0} s; {} tensors quantized in {:.0} s at {:.3e} weights/s",
        s.wall_s,
        s.quantized,
        s.quant_s,
        if s.quant_s > 0.0 {
            s.weights / s.quant_s
        } else {
            0.0
        }
    );
    Ok(())
}
