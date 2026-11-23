//! The CUDA spike (R9, #230): the layer-kind seam (`Backend`) implemented on an NVIDIA GPU for one kind, the dense MLP block, with plain f32
//! kernels (kernels/seam.cu), checked against the reference by `glyd-engine seamcheck`. A spike, not a backend: f32 weights as the source
//! gives them, the stream to and from the host at each block; the other kinds are refused by `supports`.
use super::backend::{
    Arenas, Backend, Captured, Collective, DeviceId, DeviceInfo, Format, KindRef, LayerDesc, LayerImpl, Placement, StepCx, StepShape, Support, WeightSource,
};
use crate::dev::{Buf, Dev};
use crate::ffi::{err, Result};
use crate::launch;

/// A step's queue: the GPU's context, its stream and the spike's kernels.
pub struct Queue {
    dev: Dev,
    rms: crate::kt::seam_rms,
    mm: crate::kt::seam_mm,
    swiglu: crate::kt::seam_swiglu,
    add: crate::kt::seam_add,
}

/// The CUDA spike's backend on GPU `ordinal`.
pub struct Cuda {
    devices: Vec<DeviceInfo>,
    ordinal: i32,
}

impl Cuda {
    pub fn new(ordinal: i32) -> Result<Cuda> {
        let g = Dev::probe(ordinal)?;
        Ok(Cuda { devices: vec![DeviceInfo { vendor: "nvidia".into(), arch: format!("sm_{}{}", g.cc.0, g.cc.1), memory: g.total, unified: false }], ordinal })
    }
}

/// The dense MLP block's weights on the GPU, f32.
struct MlpBlock {
    hidden: usize,
    inter: usize,
    eps: f32,
    norm: Buf,
    gate: Buf,
    up: Buf,
    down: Buf,
}

impl LayerImpl<Cuda> for MlpBlock {
    fn workspace(&self, s: &StepShape) -> u64 {
        (s.tokens * (2 * self.hidden + 3 * self.inter) * 4) as u64
    }

    fn capturable(&self) -> bool {
        false
    }

    fn run(&mut self, cx: &mut StepCx<Cuda>, x: &mut Vec<f32>) -> Result<()> {
        let (q, d, i) = (&*cx.queue, self.hidden, self.inter);
        if x.len() % d != 0 {
            return err(format!("cuda: {} values, not rows of {d}", x.len()));
        }
        let n = x.len() / d;
        let s = q.dev.stream;
        let xb = Buf::new(n * d * 4)?;
        xb.write(0, x)?;
        let (h, g, u, a, o) = (Buf::new(n * d * 4)?, Buf::new(n * i * 4)?, Buf::new(n * i * 4)?, Buf::new(n * i * 4)?, Buf::new(n * d * 4)?);
        let blocks = |len: usize| (len as u32).div_ceil(256);
        // SAFETY: every buffer is live and as long as its kernel's arguments say (n rows of d or i f32), in this queue's context; the
        // stream orders the launches and is synchronized before the stream is read back.
        unsafe {
            launch!(q.rms, [n as u32, 1, 1], [256, 1, 1], s; xb.ptr, self.norm.ptr, h.ptr, d as i32, self.eps)?;
            launch!(q.mm, [blocks(n * i), 1, 1], [256, 1, 1], s; h.ptr, self.gate.ptr, g.ptr, n as i32, d as i32, i as i32)?;
            launch!(q.mm, [blocks(n * i), 1, 1], [256, 1, 1], s; h.ptr, self.up.ptr, u.ptr, n as i32, d as i32, i as i32)?;
            launch!(q.swiglu, [blocks(n * i), 1, 1], [256, 1, 1], s; g.ptr, u.ptr, a.ptr, (n * i) as i32)?;
            launch!(q.mm, [blocks(n * d), 1, 1], [256, 1, 1], s; a.ptr, self.down.ptr, o.ptr, n as i32, i as i32, d as i32)?;
            launch!(q.add, [blocks(n * d), 1, 1], [256, 1, 1], s; xb.ptr, o.ptr, (n * d) as i32)?;
        }
        q.dev.sync()?;
        xb.read(0, x)
    }
}

impl Backend for Cuda {
    type Queue = Queue;
    type Fence = ();
    type Tensor = Vec<f32>;

    fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// The dense MLP alone, of weights a source gives as f32.
    fn supports(&self, kind: &KindRef, fmt: &Format) -> Support {
        match (kind, fmt) {
            (KindRef::Mlp, Format::F32 | Format::Bf16) => Support::Yes { rate: 1.0 },
            (KindRef::Mlp, f) => Support::No(format!("{f:?} weights: the CUDA spike reads f32")),
            (k, _) => Support::No(format!("{k:?}: the CUDA spike runs the dense MLP alone")),
        }
    }

    /// The block's weights onto the GPU of the queue made last on this thread (`queue` makes the GPU's context current).
    fn build(&self, layer: &LayerDesc, w: &dyn WeightSource, _: &Placement, _: &mut Arenas<Self>) -> Result<Box<dyn LayerImpl<Self>>> {
        if let Support::No(why) = self.supports(&KindRef::of(layer.block), &w.format("gate").unwrap_or(Format::F32)) {
            return err(format!("layer {}: {why}", layer.index));
        }
        let p = super::seam::mlp_parts(layer, w, "the CUDA spike")?;
        let up = |t: &super::backend::HostTensor| -> Result<Buf> {
            let b = Buf::new(t.data.len() * 4)?;
            b.write(0, &t.data)?;
            Ok(b)
        };
        Ok(Box::new(MlpBlock { hidden: p.hidden, inter: p.inter, eps: p.eps, norm: up(&p.norm)?, gate: up(&p.gate)?, up: up(&p.up)?, down: up(&p.down)? }))
    }

    fn queue(&self, _: DeviceId) -> Result<Queue> {
        let dev = Dev::new(self.ordinal)?;
        Ok(Queue { rms: dev.typed()?, mm: dev.typed()?, swiglu: dev.typed()?, add: dev.typed()?, dev })
    }

    fn capture(&self, _: &Queue, body: &mut dyn FnMut() -> Result<()>) -> Result<Option<Captured>> {
        body()?;
        Ok(None)
    }

    fn collective(&self) -> Option<&dyn Collective> {
        None
    }
}

/// `seamcheck [--gpu N]`: R9's CUDA spike on a GPU, the dense MLP block through the seam against the reference (and `supports`' refusals).
#[cfg(feature = "dev")]
pub fn check(a: &crate::args::Args) -> Result<()> {
    let b = Cuda::new(a.num("gpu", 0i32)?)?;
    let mut fails = 0;
    for (hidden, inter, tokens) in [(256, 704, 5), (1024, 2816, 3)] {
        let r = super::seam::check_mlp(&b, hidden, inter, tokens)?;
        let ok = r <= 1e-5;
        fails += !ok as usize;
        println!(
            "{} the dense MLP block through the seam on {} ({}): {tokens} tokens of {hidden} (inner {inter}), off by {r:.2e} of the reference's scale",
            if ok { "PASS" } else { "FAIL" },
            b.devices[0].arch,
            "CUDA"
        );
    }
    let refused =
        matches!(b.supports(&KindRef::Attention, &Format::F32), Support::No(_)) && matches!(b.supports(&KindRef::Mlp, &Format::Fp8Block), Support::No(_));
    println!("{} the kinds it does not run are refused by supports", if refused { "PASS" } else { "FAIL" });
    fails += !refused as usize;
    if fails > 0 {
        return err(format!("seamcheck: {fails} failed"));
    }
    println!("seamcheck: PASS");
    Ok(())
}
