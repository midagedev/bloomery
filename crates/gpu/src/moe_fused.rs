//! MoE block fused kernels (P8 prep): the six routed experts' gate·up·swiglu
//! in one launch (`sel`-indirect rows, the shape `q3k_gemv_sel` addresses),
//! and the combine of the six down outputs with the router weights, the
//! shared-expert output and the residual in one launch. Bit-identical to the
//! per-op path they replace; the gate is `gate_moe_fused`.
