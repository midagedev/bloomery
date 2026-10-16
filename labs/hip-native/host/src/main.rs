mod selected;

use std::ffi::{CStr, CString, c_char, c_void};

unsafe extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipSetDevice(device: i32) -> i32;
    fn hipGetErrorString(error: i32) -> *const c_char;
    fn hipModuleLoad(module: *mut *mut c_void, path: *const c_char) -> i32;
    fn hipModuleGetFunction(
        function: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> i32;
    fn hipModuleUnload(module: *mut c_void) -> i32;
    fn hipStreamCreate(stream: *mut *mut c_void) -> i32;
    fn hipStreamSynchronize(stream: *mut c_void) -> i32;
    fn hipStreamDestroy(stream: *mut c_void) -> i32;
    fn hipMalloc(buffer: *mut *mut c_void, bytes: usize) -> i32;
    fn hipFree(buffer: *mut c_void) -> i32;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn hipModuleLaunchKernel(
        function: *mut c_void,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared: u32,
        stream: *mut c_void,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

fn checked(code: i32, op: &str) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: HIP returns a static NUL-terminated string for every error code.
    let text = unsafe { CStr::from_ptr(hipGetErrorString(code)) }.to_string_lossy();
    Err(format!("{op}: {code}: {text}"))
}

struct Context {
    module: *mut c_void,
    stream: *mut c_void,
    buffers: Vec<*mut c_void>,
}

impl Context {
    fn open(path: &str) -> Result<Self, String> {
        let path = CString::new(path).map_err(|e| e.to_string())?;
        let mut s = Self {
            module: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            buffers: Vec::new(),
        };
        // SAFETY: output pointer storage and the C string remain live; Drop frees partial initialization.
        unsafe {
            checked(hipInit(0), "hipInit")?;
            checked(hipSetDevice(0), "hipSetDevice")?;
            checked(hipModuleLoad(&mut s.module, path.as_ptr()), "hipModuleLoad")?;
            checked(hipStreamCreate(&mut s.stream), "hipStreamCreate")?;
        }
        Ok(s)
    }
    fn alloc(&mut self, bytes: usize) -> Result<*mut c_void, String> {
        if bytes == 0 {
            return Err("zero allocation refused".into());
        }
        let mut p = std::ptr::null_mut();
        // SAFETY: p receives the allocation and this owner frees it exactly once.
        unsafe {
            checked(hipMalloc(&mut p, bytes), "hipMalloc")?;
        }
        self.buffers.push(p);
        Ok(p)
    }
    fn function(&self, name: &str) -> Result<*mut c_void, String> {
        let name = CString::new(name).map_err(|e| e.to_string())?;
        let mut p = std::ptr::null_mut();
        // SAFETY: the module is live and the name is a valid C string; function lives until module unload.
        unsafe {
            checked(
                hipModuleGetFunction(&mut p, self.module, name.as_ptr()),
                "hipModuleGetFunction",
            )?;
        }
        Ok(p)
    }
    fn sync(&self) -> Result<(), String> {
        // SAFETY: this owner exclusively uses the live stream.
        unsafe { checked(hipStreamSynchronize(self.stream), "hipStreamSynchronize") }
    }
    fn launch(&self, f: *mut c_void, n: usize, args: &mut [*mut c_void]) -> Result<(), String> {
        self.launch_shape(f, n.div_ceil(64), 64, args)
    }
    fn launch_shape(
        &self,
        f: *mut c_void,
        blocks: usize,
        threads: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let blocks = u32::try_from(blocks).map_err(|e| e.to_string())?;
        // SAFETY: callers provide pointers to exact ABI argument storage and checked allocation sizes; synchronize before returning.
        unsafe {
            checked(
                hipModuleLaunchKernel(
                    f,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )?;
        }
        self.sync()
    }
    unsafe fn copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: i32,
    ) -> Result<(), String> {
        // SAFETY: caller ensures both allocations hold bytes and their lifetimes cover synchronization.
        unsafe {
            checked(
                hipMemcpyAsync(dst, src, bytes, kind, self.stream),
                "hipMemcpyAsync",
            )?;
        }
        self.sync()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: all handles belong to this thread and are released once after its stream completes.
        unsafe {
            if !self.stream.is_null() {
                hipStreamSynchronize(self.stream);
            }
            for &p in &self.buffers {
                hipFree(p);
            }
            if !self.stream.is_null() {
                hipStreamDestroy(self.stream);
            }
            if !self.module.is_null() {
                hipModuleUnload(self.module);
            }
        }
    }
}

fn shape(k: usize, rows: usize, cols: usize, weight_len: usize) -> Result<(usize, usize), String> {
    if k == 0 || !k.is_multiple_of(256) || rows == 0 || cols == 0 {
        return Err("positive dimensions, K multiple of 256 required".into());
    }
    for dim in [k, rows, cols] {
        u32::try_from(dim).map_err(|e| e.to_string())?;
    }
    let need = (k / 256)
        .checked_mul(144)
        .and_then(|x| x.checked_mul(rows))
        .ok_or("weight size overflow")?;
    if weight_len != need {
        return Err("weight byte count mismatch".into());
    }
    let nx = k
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(4))
        .ok_or("activation size overflow")?;
    let ny = rows
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(4))
        .ok_or("output size overflow")?;
    Ok((nx, ny))
}

fn fixture(k: usize, rows: usize) -> Vec<u8> {
    let mut out = vec![0; k / 256 * rows * 144];
    for (b, block) in out.as_chunks_mut::<144>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&0x1800_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x1000_u16.to_le_bytes());
        for (i, x) in block[4..].iter_mut().enumerate() {
            *x = ((i * 29 + b * 13 + 17) % 256) as u8;
        }
    }
    out
}

fn case(s: &mut Context, weights: &[u8], k: usize, rows: usize, cols: usize) -> Result<(), String> {
    let (nx, ny) = shape(k, rows, cols, weights.len())?;
    let x: Vec<f32> = (0..k * cols)
        .map(|i| ((i as f32 * 0.013).sin() * 0.7) + (i % 11) as f32 * 0.01)
        .collect();
    let mut y = vec![0.0_f32; rows * cols];
    let mut roundtrip = vec![0.0_f32; x.len()];
    let w = s.alloc(weights.len())?;
    let mut dx = s.alloc(nx)?;
    let mut dy = s.alloc(ny)?;
    let mut copy_out = s.alloc(nx)?;
    // SAFETY: buffers match validated byte lengths, host vectors remain live and every transfer synchronizes.
    unsafe {
        s.copy(w, weights.as_ptr().cast(), weights.len(), 1)?;
        s.copy(dx, x.as_ptr().cast(), nx, 1)?;
    }
    let cp = s.function("bloomery_hip_copy")?;
    let mut n = u32::try_from(x.len()).map_err(|e| e.to_string())?;
    let mut cp_args = [
        (&mut dx as *mut *mut c_void).cast(),
        (&mut copy_out as *mut *mut c_void).cast(),
        (&mut n as *mut u32).cast(),
    ];
    s.launch(cp, x.len(), &mut cp_args)?;
    // SAFETY: both copy buffers hold nx bytes and remain live through synchronization.
    unsafe {
        s.copy(roundtrip.as_mut_ptr().cast(), copy_out, nx, 2)?;
    }
    if !roundtrip
        .iter()
        .zip(&x)
        .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return Err("Rust GPU copy roundtrip differs".into());
    }
    let f = s.function("bloomery_hip_q4_matmul")?;
    let mut dw = w;
    let (mut kk, mut rr, mut cc) = (k as u32, rows as u32, cols as u32);
    let mut args = [
        (&mut dw as *mut *mut c_void).cast(),
        (&mut dx as *mut *mut c_void).cast(),
        (&mut dy as *mut *mut c_void).cast(),
        (&mut kk as *mut u32).cast(),
        (&mut rr as *mut u32).cast(),
        (&mut cc as *mut u32).cast(),
    ];
    let before = s.buffers.len();
    for _ in 0..3 {
        s.launch(f, y.len(), &mut args)?;
    }
    if s.buffers.len() != before {
        return Err("steady launch allocated a GPU buffer".into());
    }
    // SAFETY: y and dy hold ny bytes and remain live through synchronization.
    unsafe {
        s.copy(y.as_mut_ptr().cast(), dy, ny, 2)?;
    }
    let mut decoded = vec![0.0_f32; k];
    let mut error = 0.0_f64;
    let mut norm = 0.0_f64;
    let row_bytes = k / 256 * 144;
    for row in 0..rows {
        gguf::dequant_row(
            gguf::GgmlType::Q4_K,
            &weights[row * row_bytes..(row + 1) * row_bytes],
            &mut decoded,
        )
        .map_err(|e| e.to_string())?;
        for col in 0..cols {
            let reference: f64 = decoded
                .iter()
                .zip(&x[col * k..(col + 1) * k])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let got = f64::from(y[col * rows + row]);
            if !got.is_finite() || !reference.is_finite() {
                return Err("non-finite output or reference".into());
            }
            error += (got - reference).powi(2);
            norm += reference.powi(2);
        }
    }
    let nmse = error / norm.max(1e-30);
    if !nmse.is_finite() || nmse > 1e-6 {
        return Err(format!("NMSE {nmse} exceeds fixed 1e-6"));
    }
    println!(
        "{{\"operation\":\"q4_matmul\",\"k\":{k},\"rows\":{rows},\"cols\":{cols},\"nmse\":{nmse},\"copy_equal\":true,\"steady_host_buffer_allocations\":0}}"
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .ok_or("usage: bloomery-hip-spike DEVICE_ELF [Q4_K_FILE K ROWS]")?;
    if args.len() != 1 && args.len() != 4 {
        return Err("expected DEVICE_ELF [Q4_K_FILE K ROWS]".into());
    }
    let mut s = Context::open(path)?;
    for (k, rows, cols) in [(256, 7, 1), (512, 64, 1), (512, 64, 32)] {
        case(&mut s, &fixture(k, rows), k, rows, cols)?;
    }
    if args.len() == 4 {
        let weights = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
        let k = args[2].parse::<usize>().map_err(|e| e.to_string())?;
        let rows = args[3].parse::<usize>().map_err(|e| e.to_string())?;
        for cols in [1, 32] {
            case(&mut s, &weights, k, rows, cols)?;
        }
        selected::real(&mut s, &weights, k, rows)?;
    }
    dense_ffn(&mut s)?;
    selected::synthetic(&mut s)?;
    Ok(())
}

fn dense_ffn(s: &mut Context) -> Result<(), String> {
    const N: usize = 256;
    let x: Vec<f32> = (0..N)
        .map(|i| (i as f32 * 0.021).sin() * 0.4 - 0.2)
        .collect();
    let gamma: Vec<f32> = (0..N).map(|i| 0.8 + i as f32 * 0.001).collect();
    let mut wg = fixture(N, N);
    let mut wu = wg.clone();
    let mut wd = wg.clone();
    for (which, weights) in [&mut wg, &mut wu, &mut wd].iter_mut().enumerate() {
        for (i, block) in weights.as_chunks_mut::<144>().0.iter_mut().enumerate() {
            block[..2].copy_from_slice(&0x0c00_u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x0800_u16.to_le_bytes());
            for byte in &mut block[16..] {
                *byte = byte.wrapping_add((which * 19 + i % 13) as u8);
            }
        }
    }
    let mut dx = s.alloc(N * 4)?;
    let mut dgamma = s.alloc(N * 4)?;
    let mut norm = s.alloc(N * 4)?;
    let mut gate = s.alloc(N * 4)?;
    let mut up = s.alloc(N * 4)?;
    let mut hidden = s.alloc(N * 4)?;
    let mut out = s.alloc(N * 4)?;
    let mut w = [s.alloc(wg.len())?, s.alloc(wu.len())?, s.alloc(wd.len())?];
    // SAFETY: all host/device buffers have the exact lengths passed, and each copy synchronizes.
    unsafe {
        s.copy(dx, x.as_ptr().cast(), N * 4, 1)?;
        s.copy(dgamma, gamma.as_ptr().cast(), N * 4, 1)?;
        for (dst, source) in w.iter().zip([&wg, &wu, &wd]) {
            s.copy(*dst, source.as_ptr().cast(), source.len(), 1)?;
        }
    }
    let rms = s.function("bloomery_hip_rms_norm")?;
    let mm = s.function("bloomery_hip_q4_matmul")?;
    let act = s.function("bloomery_hip_swiglu")?;
    let add = s.function("bloomery_hip_residual")?;
    let (mut k, mut rows, mut cols, mut n) = (N as u32, N as u32, 1_u32, N as u32);
    let mut eps = 1e-6_f32;
    let before = s.buffers.len();
    for _ in 0..3 {
        let mut args = [
            (&mut dx as *mut *mut c_void).cast(),
            (&mut dgamma as *mut *mut c_void).cast(),
            (&mut norm as *mut *mut c_void).cast(),
            (&mut n as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        s.launch(rms, N, &mut args)?;
        for (matrix, y) in [(0, &mut gate), (1, &mut up)] {
            let mut args = [
                (&mut w[matrix] as *mut *mut c_void).cast(),
                (&mut norm as *mut *mut c_void).cast(),
                (y as *mut *mut c_void).cast(),
                (&mut k as *mut u32).cast(),
                (&mut rows as *mut u32).cast(),
                (&mut cols as *mut u32).cast(),
            ];
            s.launch(mm, N, &mut args)?;
        }
        let mut args = [
            (&mut gate as *mut *mut c_void).cast(),
            (&mut up as *mut *mut c_void).cast(),
            (&mut hidden as *mut *mut c_void).cast(),
            (&mut n as *mut u32).cast(),
        ];
        s.launch(act, N, &mut args)?;
        let mut args = [
            (&mut w[2] as *mut *mut c_void).cast(),
            (&mut hidden as *mut *mut c_void).cast(),
            (&mut out as *mut *mut c_void).cast(),
            (&mut k as *mut u32).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
        ];
        s.launch(mm, N, &mut args)?;
        let mut args = [
            (&mut dx as *mut *mut c_void).cast(),
            (&mut out as *mut *mut c_void).cast(),
            (&mut n as *mut u32).cast(),
        ];
        s.launch(add, N, &mut args)?;
    }
    if s.buffers.len() != before {
        return Err("FFN steady launch issued a buffer allocation".into());
    }
    let mut got = vec![0.0_f32; N];
    // SAFETY: got and out cover N*4 bytes; the copy synchronizes before got is read.
    unsafe {
        s.copy(got.as_mut_ptr().cast(), out, N * 4, 2)?;
    }
    let inverse = 1.0
        / (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / N as f64 + f64::from(eps)).sqrt();
    let normalized: Vec<f64> = x
        .iter()
        .zip(&gamma)
        .map(|(&v, &g)| f64::from(v) * inverse * f64::from(g))
        .collect();
    let matrix = |weights: &[u8], x: &[f64]| -> Result<Vec<f64>, String> {
        let mut row = vec![0.0_f32; N];
        let mut y = Vec::with_capacity(N);
        for data in weights.as_chunks::<144>().0 {
            gguf::dequant_row(gguf::GgmlType::Q4_K, data, &mut row).map_err(|e| e.to_string())?;
            y.push(row.iter().zip(x).map(|(&a, &b)| f64::from(a) * b).sum());
        }
        Ok(y)
    };
    let gate_ref = matrix(&wg, &normalized)?;
    let up_ref = matrix(&wu, &normalized)?;
    let hidden_ref: Vec<f64> = gate_ref
        .iter()
        .zip(&up_ref)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let reference = matrix(&wd, &hidden_ref)?;
    let mut error = 0.0;
    let mut energy = 0.0;
    for ((actual, reference), &residual) in got.iter().zip(reference).zip(&x) {
        let reference = reference + f64::from(residual);
        if !actual.is_finite() || !reference.is_finite() {
            return Err("FFN output or reference is nonfinite".into());
        }
        error += (f64::from(*actual) - reference).powi(2);
        energy += reference.powi(2);
    }
    let nmse = error / energy.max(1e-30);
    if !nmse.is_finite() || nmse > 1e-6 {
        return Err(format!("FFN NMSE {nmse} exceeds fixed 1e-6"));
    }
    println!(
        "{{\"operation\":\"dense_ffn\",\"width\":256,\"nmse\":{nmse},\"intermediate_host_copies\":0,\"steady_host_buffer_allocations\":0}}"
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bloomery-hip-spike: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::shape;
    #[test]
    fn refuses_invalid_shapes_and_truncated_weights() {
        for (k, r, c, w) in [
            (0, 1, 1, 0),
            (257, 1, 1, 144),
            (256, 0, 1, 0),
            (256, 1, 0, 144),
            (256, 1, 1, 143),
        ] {
            assert!(shape(k, r, c, w).is_err());
        }
        assert_eq!(shape(256, 7, 32, 1008).unwrap(), (32768, 896));
    }
}
