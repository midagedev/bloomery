use gguf::iq_quant::IqKind;
use quantize::Error;
use quantize::check::compare_row;

#[test]
fn one_flipped_byte_is_named() {
    let want = vec![7u8; 100];
    assert!(compare_row("t", IqKind::Iq1S, 5, 4, &want, &want).is_ok());
    let mut got = want.clone();
    got[57] ^= 0x10;
    let e = compare_row("blk.2.ffn_up_exps.weight", IqKind::Iq1S, 9, 4, &got, &want).unwrap_err();
    assert!(
        matches!(
            e,
            Error::Mismatch {
                expert: 2,
                row: 1,
                block: 1,
                byte: 7,
                got: 0x17,
                want: 0x07,
                ..
            }
        ),
        "{e}"
    );
    assert!(
        e.to_string()
            .contains("blk.2.ffn_up_exps.weight expert 2 row 1")
    );
    assert!(compare_row("t", IqKind::Iq1S, 0, 4, &got[..99], &want).is_err());
}
