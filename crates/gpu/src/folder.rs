//! What a model loaded from a checkpoint folder is to a body's open: the one owner of the choice
//! between a GGUF split and a folder, of the folder as a load reads it, and of what a folder's
//! load does not run.
//!
//! A body's open takes a [`Model`] (`impl Into<Model>`): a [`Split`], which the load reads as it
//! reads any GGUF, or a [`Folder`]. [`Model::into_origin`] makes the choice once: the header the
//! load plans by and an [`Origin`], whose [`Origin::bytes`] is what
//! [`GpuModel::load_placed_from`](crate::GpuModel::load_placed_from) takes for the card's
//! tensors. A body that is a folder's load adds only what its architecture forces: the part of
//! the folder it reads beyond the parts (`X`, [`Folder::with`]). Under a residency machine the
//! machine's source reads the experts where the folder stores them, through the located header
//! (`crate::host::swap_source::FileSwap`).

use std::sync::Mutex;

use gguf::Split;
use model::part::Source;
use model::placement::Plan;

use crate::GpuError;
use crate::host::swap::Residency;
use crate::model::CardBytes;
use crate::parts::{InPlace, PlacedSource};

/// What the errors name.
const WHAT: &str = "folder::Model::into_origin";

type CardParts = Box<dyn crate::model::PartsLoad>;

/// The model a load reads: a GGUF split, or a checkpoint folder ([`Folder`]). `X` is what the
/// folder's architecture carries beside the folder's parts.
pub enum Model<X = ()> {
    Split(Split),
    Folder(Folder<X>),
}

impl<X> Model<X> {
    /// The header the load plans by: the split's first shard's, or the folder's located header.
    #[must_use]
    pub fn header(&self) -> &Split {
        match self {
            Model::Split(file) => file,
            Model::Folder(f) => &f.header,
        }
    }

    /// The header a load of `plan` plans by and where the card's tensors come from, the choice a
    /// body's open takes once. A folder's load under `residency` stages its experts from where
    /// its located header says the folder stores them; a header that locates no file
    /// (`gguf::Split::header_only`) is refused by name under a residency machine, its host holding
    /// no expert to stage. Before any host reader, a folder's part source writes what its header
    /// locates in another form than the folder stores (`model::part::Source::make_in_place`)
    /// wherever the host reads in place: under a residency machine, or where `plan` holds such a
    /// tensor on the host or the NVMe tier.
    pub fn into_origin(
        self,
        residency: Residency,
        plan: &Plan<'_>,
    ) -> Result<(Split, Origin<X>), GpuError> {
        if let Model::Folder(f) = &self {
            if residency != Residency::Off && f.header.is_header_only() {
                return Err(GpuError::shape(
                    WHAT,
                    "a checkpoint folder whose header locates no file runs under \
                     BLOOMERY_RESIDENCY=off: the residency machine's source reads each \
                     expert where the folder stores it",
                ));
            }
            if residency != Residency::Off || InPlace::of(&f.header).read_off_cards(plan) {
                f.parts.make_in_place()?;
            }
        }
        Ok(self.into_whole())
    }

    /// The header and where the card's tensors come from for a load that holds the whole model on
    /// its card, whose host reads nothing in place.
    #[must_use]
    pub fn into_whole(self) -> (Split, Origin<X>) {
        match self {
            Model::Split(file) => (file, Origin::File),
            Model::Folder(f) => (
                f.header,
                Origin::Folder {
                    parts: f.parts,
                    extra: f.extra,
                },
            ),
        }
    }
}

impl<X> From<Split> for Model<X> {
    fn from(file: Split) -> Model<X> {
        Model::Split(file)
    }
}

impl<X> From<Folder<X>> for Model<X> {
    fn from(folder: Folder<X>) -> Model<X> {
        Model::Folder(folder)
    }
}

/// A checkpoint folder as a load reads it: its header (`gguf::Split::located`, or
/// `gguf::Split::header_only` for a folder whose host reads nothing where the folder stores it),
/// the header its plan is made from, whose in-place tensors the host reads where the folder
/// stores them ([`InPlace`]), and its card tensors through its part source
/// (`crate::parts::PlacedSource`). `X` is the architecture's own addition to the load, `()` for
/// none.
pub struct Folder<X = ()> {
    header: Split,
    parts: CardParts,
    extra: X,
}

impl<X> Folder<X> {
    /// The folder of header `header`, its card tensors read from `source`, with the
    /// architecture's `extra`. A load reads only its plan's tensors of the source, so the source
    /// may also hold another plan's (a draft's, which loads through [`Origin::bytes`] too).
    pub fn with<S: Source + Send + Sync + 'static>(
        header: Split,
        source: S,
        extra: X,
    ) -> Folder<X> {
        let in_place = InPlace::of(&header);
        Folder {
            header,
            parts: Box::new(PlacedSource {
                source,
                in_place,
                last: Mutex::default(),
            }),
            extra,
        }
    }
}

impl<X> Folder<X> {
    /// The least host bytes a load of card `card` of `plan` reads beside the plan's own terms: its
    /// parts one at a time, with the staging ring, the part source's read at no room (the one
    /// owner of these bytes, which [`crate::parts::least_reads`] asks of a source and header held
    /// apart). The load's own check counts its read-ahead (`HostNeed::read_ahead`) from the same
    /// parts at the room it reads, never under this.
    pub fn least_reads(&self, plan: &Plan<'_>, card: usize) -> Result<u64, GpuError> {
        self.parts.host_bytes(plan, card, 0)
    }
}

impl Folder {
    /// The folder of header `header`, its card tensors read from `source`.
    pub fn new<S: Source + Send + Sync + 'static>(header: Split, source: S) -> Folder {
        Folder::with(header, source, ())
    }
}

/// Where a load's card tensors come from ([`Model::into_origin`]).
pub enum Origin<X = ()> {
    /// The split's own mapped tensors.
    File,
    /// A folder's part source, and the architecture's `extra`.
    Folder { parts: CardParts, extra: X },
}

impl<X> Origin<X> {
    /// The card's bytes as [`GpuModel::load_placed_from`](crate::GpuModel::load_placed_from)
    /// reads them.
    #[must_use]
    pub fn bytes(&self) -> CardBytes<'_> {
        match self {
            Origin::File => CardBytes::File,
            Origin::Folder { parts, .. } => CardBytes::Parts(parts.as_ref()),
        }
    }

    /// The folder's architecture addition; `None` for a split.
    #[must_use]
    pub fn extra(&self) -> Option<&X> {
        match self {
            Origin::File => None,
            Origin::Folder { extra, .. } => Some(extra),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use gguf::write::{Layout, TensorDecl};
    use gguf::{GgmlType, Place, Value};
    use model::part::{Part, Planned, Take};
    use model::placement::{
        KvBytes, ModelTensor, ModelTensors, PlanLevers, Role, plan_host_routed, plan_with,
        workstation,
    };

    /// A part source that holds nothing, a load that read it refused by name, which counts the
    /// times a load asks it to write what the host reads in place.
    #[derive(Clone, Default)]
    struct Empty(Arc<AtomicUsize>);

    impl Source for Empty {
        type Error = std::fmt::Error;

        fn plan<E: std::fmt::Display>(
            &self,
            _: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
        ) -> Result<Vec<Planned>, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read_bytes(&self, _: &Planned) -> Result<u64, Self::Error> {
            Err(std::fmt::Error)
        }

        fn read(&self, _: &Planned) -> Result<Part<'_>, Self::Error> {
            Err(std::fmt::Error)
        }

        fn make_in_place(&self) -> Result<(), Self::Error> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct NoKv;

    impl KvBytes for NoKv {
        fn layer_bytes(&self, _: usize, _: u64) -> u64 {
            0
        }
    }

    /// The header's one tensor: the head, or (`routed`) one layer's routed stack of one expert.
    fn model_of(routed: bool) -> ModelTensors {
        let (layer, role, dims) = match routed {
            false => (None, Role::Head, vec![256, 16]),
            true => (Some(0), Role::RoutedExperts, vec![256, 16, 1]),
        };
        ModelTensors {
            tensors: vec![ModelTensor {
                name: "w.weight".into(),
                shard: 0,
                layer,
                role,
                ty: GgmlType::Q8_0,
                dims,
                file_bytes: 4352,
                gathered_rows: None,
            }],
            layers: 1,
            experts: u64::from(routed),
            experts_used: u64::from(routed),
            bf16_words: |_| false,
        }
    }

    /// `f` of a plan that holds the header's one tensor on an A6000 workstation's card.
    fn with_plan<R>(f: impl FnOnce(&Plan<'_>) -> R) -> R {
        let model = model_of(false);
        let machine = workstation::plan_on(workstation::A6000, 1);
        f(&plan_with(&model, &machine, 16, &NoKv, None).expect("a plan"))
    }

    fn layout() -> Layout {
        let kvs = [("toy.layers".to_string(), Value::U32(1))];
        let tensors = vec![TensorDecl {
            name: "w.weight".into(),
            dims: vec![256, 16],
            type_id: GgmlType::Q8_0.as_u32(),
            nbytes: 4352,
        }];
        Layout::new(&kvs, tensors).unwrap()
    }

    fn header() -> Split {
        Split::header_only(std::env::temp_dir().join("folder-toy.gguf"), &layout()).unwrap()
    }

    /// A load of a split reads the split's own tensors under any residency and carries no
    /// addition; a folder's reads its part source and carries its addition.
    #[test]
    fn a_load_reads_a_splits_own_tensors_or_a_folders_parts() {
        let mid = Residency::Mid {
            pinned: 1,
            spares: 1,
        };
        with_plan(|plan| {
            let (_, split) = Model::<u8>::from(header()).into_origin(mid, plan).unwrap();
            assert!(matches!(split.bytes(), CardBytes::File));
            assert!(split.extra().is_none());
            let (_, folder) = Model::from(Folder::with(header(), Empty::default(), 7u8))
                .into_origin(Residency::Off, plan)
                .unwrap();
            assert!(matches!(folder.bytes(), CardBytes::Parts(_)));
            assert_eq!(folder.extra(), Some(&7));
            let plain: Model = Folder::new(header(), Empty::default()).into();
            assert!(plain.into_origin(Residency::Off, plan).is_ok());
        });
    }

    // PIN(2026-10-22): folder drafts use the placed MTP load; only residency is refused here.
    /// A folder whose header locates no file runs no residency machine: it is refused by name,
    /// and a split's load under one passes.
    #[test]
    fn a_folder_load_refuses_residency() {
        let mid = Residency::Mid {
            pinned: 1,
            spares: 1,
        };
        with_plan(|plan| {
            let folder = || Model::<()>::from(Folder::new(header(), Empty::default()));
            let e = folder().into_origin(mid, plan).err().unwrap();
            assert!(e.to_string().contains("BLOOMERY_RESIDENCY=off"), "{e}");
            assert!(folder().into_origin(Residency::Off, plan).is_ok());
            assert!(Model::<()>::from(header()).into_origin(mid, plan).is_ok());
        });
    }

    /// A folder's part source writes what its located header holds in place before a load whose
    /// host reads there: under a residency machine, and under `off` where the plan holds such a
    /// tensor on the host (`parts::InPlace::read_off_cards`), not under `off` with every tensor on
    /// the card, and never for a header that locates no file or a whole-card load. Mutants: the
    /// write left out under a residency machine, or for an `off` plan that holds it on the host.
    #[test]
    fn a_folder_writes_what_its_host_reads_in_place_first() {
        let dir = std::env::temp_dir().join(format!("folder-in-place-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("w.bin");
        std::fs::write(&file, vec![0u8; 4352]).unwrap();
        let located = || {
            let place = Place::File {
                shard: 0,
                range: 0..4352,
            };
            Split::located(
                file.clone(),
                &layout(),
                std::slice::from_ref(&file),
                vec![place],
            )
            .unwrap()
        };
        let mid = Residency::Mid {
            pinned: 1,
            spares: 1,
        };
        with_plan(|plan| {
            for (header, residency, made) in [
                (located(), Residency::Off, 0),
                (located(), mid, 1),
                (header(), Residency::Off, 0),
            ] {
                let source = Empty::default();
                let model: Model = Folder::new(header, source.clone()).into();
                model.into_origin(residency, plan).unwrap();
                assert_eq!(source.0.load(Ordering::Relaxed), made, "{residency:?}");
            }
            let source = Empty::default();
            let _ = Model::from(Folder::new(located(), source.clone())).into_whole();
            assert_eq!(source.0.load(Ordering::Relaxed), 0);
        });
        let model = model_of(true);
        let machine = workstation::plan_on(workstation::A6000, 1);
        let host =
            plan_host_routed(&model, &machine, 16, &NoKv, &PlanLevers::default()).expect("a plan");
        assert!(
            InPlace::of(&located()).read_off_cards(&host),
            "the plan holds the stack on the host"
        );
        let source = Empty::default();
        let model: Model = Folder::new(located(), source.clone()).into();
        model.into_origin(Residency::Off, &host).unwrap();
        assert_eq!(source.0.load(Ordering::Relaxed), 1, "off, on the host");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
