//! The MLX spike (R9, #230): the layer-kind seam (`Backend`) implemented through Apple's MLX, by its C API (mlx-c), for one kind: the dense
//! MLP block (its input norm, the gated product, the residual), checked against the reference. macOS with the `mlx` feature only. A
//! spike, not a backend: the residual stream crosses to the host at each block, and the other kinds are refused by `supports`.
use super::backend::{
    Arenas, Backend, Captured, Collective, DeviceId, DeviceInfo, Format, KindRef, LayerDesc, LayerImpl, Placement, StepCx, StepShape, Support, WeightSource,
};
use crate::ffi::{err, Result};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Raw {
    ctx: *mut c_void,
}

/// mlx/c/array.h's `mlx_dtype`: MLX_FLOAT32.
const FLOAT32: i32 = 10;

#[link(name = "mlxc")]
extern "C" {
    fn mlx_array_new() -> Raw;
    fn mlx_array_free(a: Raw) -> i32;
    fn mlx_array_new_data(data: *const c_void, shape: *const i32, dim: i32, dtype: i32) -> Raw;
    fn mlx_array_eval(a: Raw) -> i32;
    fn mlx_array_data_float32(a: Raw) -> *const f32;
    fn mlx_default_gpu_stream_new() -> Raw;
    fn mlx_stream_free(s: Raw) -> i32;
    fn mlx_matmul(r: *mut Raw, a: Raw, b: Raw, s: Raw) -> i32;
    fn mlx_transpose(r: *mut Raw, a: Raw, s: Raw) -> i32;
    fn mlx_sigmoid(r: *mut Raw, a: Raw, s: Raw) -> i32;
    fn mlx_multiply(r: *mut Raw, a: Raw, b: Raw, s: Raw) -> i32;
    fn mlx_add(r: *mut Raw, a: Raw, b: Raw, s: Raw) -> i32;
    fn mlx_fast_rms_norm(r: *mut Raw, x: Raw, w: Raw, eps: f32, s: Raw) -> i32;
}

/// An MLX array, freed when dropped. (MLX's arrays are reference-counted handles; a block's are used by one step at a time.)
struct Array(Raw);

// SAFETY: an array is a handle MLX counts references to; this module uses each from the one thread a step runs on at a time.
unsafe impl Send for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: an array this module made, freed once.
        unsafe { mlx_array_free(self.0) };
    }
}

impl Array {
    /// `data` (row-major, `shape`) copied into an f32 array.
    fn from(data: &[f32], shape: &[usize]) -> Result<Array> {
        if shape.iter().product::<usize>() != data.len() {
            return err(format!("mlx: {} values for the shape {shape:?}", data.len()));
        }
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        // SAFETY: data and dims are live for the call, which copies them.
        Ok(Array(unsafe { mlx_array_new_data(data.as_ptr() as *const c_void, dims.as_ptr(), dims.len() as i32, FLOAT32) }))
    }

    /// The array computed and its `n` values read back.
    fn read(&self, n: usize) -> Result<Vec<f32>> {
        // SAFETY: an array of this module's; eval returns once its values are there, and they are f32 (every array here is).
        unsafe {
            if mlx_array_eval(self.0) != 0 {
                return err("mlx: an array could not be computed");
            }
            let p = mlx_array_data_float32(self.0);
            if p.is_null() {
                return err("mlx: an array has no f32 values");
            }
            Ok(std::slice::from_raw_parts(p, n).to_vec())
        }
    }
}

/// An operation's result (`f` writes it), or its failure named.
fn op(what: &str, f: impl FnOnce(*mut Raw) -> i32) -> Result<Array> {
    // SAFETY: a new empty array, given to the operation to fill.
    let mut r = unsafe { mlx_array_new() };
    if f(&mut r) != 0 {
        // SAFETY: the array made above.
        unsafe { mlx_array_free(r) };
        return err(format!("mlx: {what} failed"));
    }
    Ok(Array(r))
}

/// A step's queue: MLX's default GPU stream.
pub struct Queue(Raw);

impl Drop for Queue {
    fn drop(&mut self) {
        // SAFETY: the stream `Mlx::queue` made, freed once.
        unsafe { mlx_stream_free(self.0) };
    }
}

/// The MLX spike's backend: the dense MLP block on the Mac's GPU, f32.
pub struct Mlx {
    devices: Vec<DeviceInfo>,
}

impl Default for Mlx {
    fn default() -> Self {
        Mlx { devices: vec![DeviceInfo { vendor: "apple".into(), arch: std::env::consts::ARCH.into(), memory: 0, unified: true }] }
    }
}

/// The dense MLP block as MLX runs it: x + down(silu(gate(h)) up(h)), h the RMS norm of x; the matrices kept transposed for x W^T.
struct MlpBlock {
    hidden: usize,
    eps: f32,
    norm: Array,
    gate_t: Array,
    up_t: Array,
    down_t: Array,
}

impl LayerImpl<Mlx> for MlpBlock {
    fn workspace(&self, _: &StepShape) -> u64 {
        0
    }

    fn capturable(&self) -> bool {
        false
    }

    fn run(&mut self, cx: &mut StepCx<Mlx>, x: &mut Vec<f32>) -> Result<()> {
        let (d, s) = (self.hidden, cx.queue.0);
        if x.len() % d != 0 {
            return err(format!("mlx: {} values, not rows of {d}", x.len()));
        }
        let xa = Array::from(x, &[x.len() / d, d])?;
        // SAFETY: every operand is a live array of this module's, on MLX's stream; each result is a new array.
        unsafe {
            let h = op("rms_norm", |r| mlx_fast_rms_norm(r, xa.0, self.norm.0, self.eps, s))?;
            let g = op("matmul", |r| mlx_matmul(r, h.0, self.gate_t.0, s))?;
            let u = op("matmul", |r| mlx_matmul(r, h.0, self.up_t.0, s))?;
            let sg = op("sigmoid", |r| mlx_sigmoid(r, g.0, s))?;
            let silu = op("multiply", |r| mlx_multiply(r, g.0, sg.0, s))?;
            let a = op("multiply", |r| mlx_multiply(r, silu.0, u.0, s))?;
            let o = op("matmul", |r| mlx_matmul(r, a.0, self.down_t.0, s))?;
            let y = op("add", |r| mlx_add(r, xa.0, o.0, s))?;
            *x = y.read(x.len())?;
        }
        Ok(())
    }
}

impl Backend for Mlx {
    type Queue = Queue;
    type Fence = ();
    type Tensor = Vec<f32>;

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// The dense MLP alone, of weights a source gives as f32 (the spike reads them so).
    fn supports(&self, kind: &KindRef, fmt: &Format) -> Support {
        match (kind, fmt) {
            (KindRef::Mlp, Format::F32 | Format::Bf16) => Support::Yes { rate: 1.0 },
            (KindRef::Mlp, f) => Support::No(format!("{f:?} weights: the MLX spike reads f32")),
            (k, _) => Support::No(format!("{k:?}: the MLX spike runs the dense MLP alone")),
        }
    }

    fn build(&self, layer: &LayerDesc, w: &dyn WeightSource, _: &Placement, _: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        let b = layer.block;
        let fmt = w.format("gate").unwrap_or(Format::F32);
        if let Support::No(why) = self.supports(&KindRef::of(b), &fmt) {
            return err(format!("layer {}: {why}", layer.index));
        }
        let p = super::seam::mlp_parts(layer, w, "the MLX spike")?;
        let (d, norm, gate, up, down) = (p.hidden, p.norm, p.gate, p.up, p.down);
        // SAFETY: a stream for the transposes, freed when this returns (the arrays it made are computed when first read).
        let q = Queue(unsafe { mlx_default_gpu_stream_new() });
        let t = |m: &super::backend::HostTensor| -> Result<Array> {
            let a = Array::from(&m.data, &m.shape)?;
            // SAFETY: a live array on a live stream.
            op("transpose", |r| unsafe { mlx_transpose(r, a.0, q.0) })
        };
        Ok(Box::new(MlpBlock { hidden: d, eps: p.eps, norm: Array::from(&norm.data, &[d])?, gate_t: t(&gate)?, up_t: t(&up)?, down_t: t(&down)? }))
    }

    fn queue(&self, _: DeviceId) -> Result<Queue> {
        // SAFETY: MLX's default GPU stream, a new handle to it.
        Ok(Queue(unsafe { mlx_default_gpu_stream_new() }))
    }

    fn capture(&self, _: &Queue, body: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>> {
        body()?;
        Ok(None)
    }

    fn collective(&self) -> Option<&dyn Collective> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R9's MLX spike: the dense MLP block through MLX gives the reference's stream (five tokens of a 256-wide model, and three of 1,024); the
    /// other kinds are refused.
    #[test]
    fn the_dense_mlp_through_mlx_is_the_references() {
        let m = Mlx::default();
        for (hidden, inter, tokens) in [(256, 704, 5), (1024, 2816, 3)] {
            let r = crate::runtime::seam::check_mlp(&m, hidden, inter, tokens).unwrap();
            assert!(r <= 1e-4, "{tokens} tokens of {hidden}: off by {r:.2e} of the reference's scale");
        }
        assert!(matches!(m.supports(&KindRef::Attention, &Format::F32), Support::No(_)));
        assert!(matches!(m.supports(&KindRef::Mlp, &Format::Fp8Block), Support::No(_)));
    }
}
