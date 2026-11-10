//! What a Qwen3.8 load captures past its card reading ([`capture_rule38`]): the one rule the serve
//! seat's plans and its open read, and the serve gate's plans that restate the seat's.

use app::mtp::MtpBody;
use bloomery_gpu::arch::qwen3moe::{Body38, MTP_GRAPH_ROWS};
use bloomery_gpu::model::SlotRows;
use bloomery_gpu_gates::bind::Pass;
use model::placement::graphs::{CaptureArm, CaptureRule, DraftCaptures};

/// The rule of a Qwen3.8 load's captures past its card reading ([`CaptureRule`]), by the facts a
/// seat fixes before its first plan: whether the load leaves the body its pass of several slots
/// (`resident`: no expert tier card beside it) and the rows of the lookup draft a load of no draft
/// weights opens on every slot (`None`: such a load drafts nothing). A drafted load's slots each
/// capture the MTP draft's verify passes of 2 to its verify rows and its walks; a lookup's, its
/// verify pass of its rows; a round's pass holds a window's rows a slot (a row on a plain load),
/// within the body's pass, and runs as one pass of the busy rows where the body's verify does
/// ([`Pass::of_body`], the shape the seat's open runs).
pub fn capture_rule38(resident: bool, lookup_rows: Option<usize>) -> CaptureRule {
    let arm = |draft: Option<DraftCaptures>, window: Option<usize>| CaptureArm {
        draft,
        rows: window.unwrap_or(1),
        one_pass: Pass::of_body::<Body38>(2, resident, window) == Pass::One,
    };
    let verify = <Body38 as MtpBody>::VERIFY_ROWS;
    CaptureRule {
        drafted: arm(
            Some(DraftCaptures {
                verify_rows: verify,
                walks: MTP_GRAPH_ROWS + 1,
            }),
            Some(verify),
        ),
        undrafted: arm(
            lookup_rows.map(|rows| DraftCaptures {
                verify_rows: rows,
                walks: 0,
            }),
            lookup_rows,
        ),
        limits: (
            <Body38 as SlotRows>::MAX_ROWS,
            <Body38 as SlotRows>::MAX_SLOTS,
        ),
    }
}
