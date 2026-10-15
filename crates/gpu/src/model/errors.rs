//! Model error routing before a host refusal or device fault is read.

use crate::GpuError;
use crate::fault::Fault;
use crate::hybrid::{Refusal, name_refusal};

pub(super) fn name_host_refusal<T>(
    r: Result<T, GpuError>,
    take: impl FnOnce() -> Option<Refusal>,
    read: impl FnOnce() -> Result<Option<Fault>, GpuError>,
) -> Result<T, GpuError> {
    const WHAT: &str = "GpuModel::name_host_refusal";
    let e = match r {
        Err(e) => e,
        Ok(v) => return Ok(v),
    };
    let Some(refusal) = take() else {
        return Err(e);
    };
    // The release left the cards waiting: the stream behind the refusal does not drain.
    if matches!(&e, GpuError::Unsettled { .. }) {
        return Err(e);
    }
    match read() {
        Ok(word) => Err(name_refusal(&refusal, word)),
        Err(s) => Err(GpuError::shape(
            WHAT,
            format!(
                "{} refused the input of layer {} ({}) and the call failed ({e}), and reading \
                 the fault word behind it failed too ({s})",
                refusal.what, refusal.layer, refusal.detail
            ),
        )),
    }
}

pub(super) fn note_fault<T>(
    r: Result<T, GpuError>,
    behind: impl FnOnce(GpuError) -> GpuError,
) -> Result<T, GpuError> {
    match r {
        Ok(v) => Ok(v),
        Err(e @ (GpuError::Fault { .. } | GpuError::Unsettled { .. })) => Err(e),
        Err(e) => Err(behind(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::{name_host_refusal, note_fault};
    use crate::GpuError;
    use crate::hybrid::Refusal;
    use std::error::Error;

    #[test]
    fn an_unsettled_service_keeps_its_source_without_model_fault_reads() {
        let error = GpuError::Unsettled {
            what: "Hybrid::serve",
            source: Box::new(GpuError::protocol(
                "Hybrid::serve",
                "the generation ran ahead",
            )),
            detail: "the stage end left its done counter waiting".into(),
        };
        let text = error.to_string();
        let mut taken = false;
        let named = name_host_refusal::<()>(
            Err(error),
            || {
                taken = true;
                Some(Refusal {
                    what: "Hybrid::serve",
                    layer: 3,
                    detail: "planted refusal".into(),
                })
            },
            || panic!("an unsettled service must not read the fault word"),
        );
        assert!(
            taken,
            "the refusal is taken once, so a later call cannot name it"
        );
        let error = note_fault(named, |_| {
            panic!("an unsettled service must not enter fault readback")
        })
        .expect_err("the unsettled service stays an error");
        assert!(matches!(
            error.source().and_then(|s| s.downcast_ref::<GpuError>()),
            Some(GpuError::Protocol { .. })
        ));
        assert_eq!(error.to_string(), text);
    }
}
