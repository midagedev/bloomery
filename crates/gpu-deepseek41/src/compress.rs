//! Pooled compressed KV: DS4_COMP pools `ratio` consecutive tokens'
//! `wkv_c · x` under softmax(`wgate_c · x`) weights into one latent row,
//! then norm, rope and an f16 write into the compressed cache.
