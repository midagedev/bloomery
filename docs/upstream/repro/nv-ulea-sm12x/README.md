# ptxas 13.0 drops a pointer's high word on sm_120/sm_121 (`ULEA` without `ULEA.HI.X`)

Ledger row #55 (`docs/upstream/nvlabs-ledger.md`). `conv_res.ptx` is one entry of bloomery's GPU crate, as cuda-oxide
emits it for `sm_120a`. Its parameters are pointers plus `u32` indices; the kernel indexes a ring buffer with
`ptr + (zext u32 << 2)`.

## Reproduce

    ptxas -arch=sm_120a conv_res.ptx -o cr.cubin && cuobjdump -sass cr.cubin | grep -E 'ULEA|IADD.64 R22'

ptxas 13.0 (`cuda_13.0.r13.0/compiler.36424714_0`, V13.0.88):

    /*04b0*/  ULEA UR8, UR8, UR10, 0x2 ;
    /*0590*/  IADD.64 R22, R22, UR8 ;

The low word is computed with no carry out, and the high word of `UR8:UR9` is not formed from the pointer's high word.

ptxas 13.3 (`cuda_13.3.r13.3/compiler.38244171_0`), same PTX:

    /*04c0*/  ULEA UR8, UP0, UR7, UR13, 0x2 ;
    /*0550*/  ULEA.HI.X UR9, UR7, UR11, UR9, 0x2, UP0 ;
    /*05c0*/  IADD.64 R22, R22, UR8 ;

ptxas 13.1 and 13.2 match 13.3. `sm_86`, `sm_89`, `sm_90` and `sm_100a` are not affected.

## On a card

The R580 driver (580.178.04, `cuDriverGetVersion` 13000) compiles the same PTX the same way when it JITs it on a DGX
Spark (GB10, cc 12.1). compute-sanitizer:

    Invalid __global__ write of size 4 bytes at conv_res+0x1be0 ... Access to 0x1ffb0100

`0x1ffb0100` is the low 32 bits of the ring buffer's address. An RTX 5090 on driver 615.71.09 runs the same binary
with 0 memcheck errors.

## Notes

- This looks like the bug in NVIDIA/nccl#2445 (ptxas 12.8–13.0, sm_120). That thread treats it as fixed in newer
  toolkits. The R580 branch's JIT still carries it, and R580 is what DGX OS ships, so any PTX loaded through the driver
  on those cards can hit it.
- bloomery's workaround: for cc 12.x it ships cubins compiled by the toolkit's ptxas 13.3, scanned for the pattern
  (`tools/oxart-pack.sh`, `tools/ulea-scan.py`), and it refuses to hand PTX to a driver before CUDA 13.1.
