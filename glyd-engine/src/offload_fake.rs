//! The expert cache on a driver of the host (tools/cuda_fake.cpp: `libcuda.so.1` on the host, "device" memory host memory, streams that run their work when the host makes them), for the development build's check of
//! what no unit test reaches without a GPU: the copies, the events, the host's pairs and the disk tier of src/offload.rs through `ExpertStore::run_groups` and `run`, as the engine calls them. The kernels are the one thing
//! the driver does not run: `body`, which the engine fills with their launches, reads the buffers they would read and checks them against the experts' bytes in the directory.
//!
//! The test runs the tiny Hummingbird model's layer (16 experts in two widths, 4 chosen a token) through two stores over the same directory, one with every expert in the computer's memory and one with a few of them
//! (the rest read from the packed directory), on random calls in each way the host serves a miss, with the planner's times held; and the streams' work is run as late as it can be (FAKE_CUDA_PROGRESS=0: at a synchronize
//! alone) and as it goes (a few operations at each query). In every call, every pair that has a place finds the bytes of its expert in it when the kernels would run, the sums the host sends up are the sums of its pairs
//! computed from the directory's bytes to the bit, and the two stores make the same plan, send the same places and the same sums.
//!
//! It runs where the driver of tools/cuda_fake.cpp is the one the loader finds (see the top of that file); anywhere else it does nothing (a computer with a GPU has a driver of its own).

use super::*;
use crate::cpu_experts::{Kernel, Z};
use crate::cpu_hb::experts_with;
use crate::dev::{Buf, Dev};
use crate::moe_hb::{index_layer, read_layer, HbDisk, HbHost, HbLayerHost, HbStack};
use std::path::Path;

/// A number source for the test (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// One store over the tiny layer, twice (the store's layers 0 and 1 are the two widths of one copy of it, 2 and 3 of another: a call is made on one and the hints and the predictor's go to the other): the groups' places on
/// the "device", and what a place holds.
struct Rig {
    store: Rc<ExpertStore>,
    // each store layer's places: the addresses of the codes' and of the scales and codebooks' (the first place's), and the bytes of a place in each
    places: [([u64; 2], [u64; 2]); 4],
    _stacks: Vec<HbStack>,
}

/// The store over the layer's groups (`host`: the experts' bytes; `disk`: the directory's reader and the memory's budget, where some of the experts are to stay there), `gpu` places for each group, in `mode` with the
/// planner's times held.
fn rig(host: &HbLayerHost, disk: Option<(&Arc<HbDisk>, &crate::moe_hb::HbIndex, HostMemory)>, gpu: [usize; 2], mode: Mode) -> Rig {
    let store = ExpertStore::new();
    store.set_offload(Offload { mode, threads: 2, ..Offload::default() });
    store.hold_costs();
    store.set_host_memory(disk.map(|d| d.2));
    let mut places = [([0; 2], [0; 2]); 4];
    let mut stacks = Vec::new();
    for (layer, place) in places.iter_mut().enumerate() {
        let g = layer % 2;
        let h = host.groups[g].as_ref().expect("both widths");
        let stack = HbStack::pooled(h.place, h.experts, gpu[g]).unwrap();
        let (dev, stride) = stack.places();
        *place = (dev, stride);
        match disk {
            None => store.add_layer(layer, h.experts, gpu[g], &[PackIn { bytes: [&h.data, &h.meta], stride, dev }], Some(HostHalf::Hb(h.place))).unwrap(),
            Some((d, ix, _)) => store
                .add_layer_on_disk(
                    layer,
                    h.experts,
                    gpu[g],
                    DiskIn { place: h.place, dev, ram: store.host_slots(h.experts, h.place.bytes()).unwrap(), read: d.reader(ix.members[g].clone()) },
                )
                .unwrap(),
        }
        stacks.push(stack);
    }
    Rig { store, places, _stacks: stacks }
}

/// The device buffers of a call.
struct Buffers {
    ids: Buf,
    wts: Buf,
    z: Buf,
    ways: [Buf; 2],
    sum_ids: Buf,
    part: Buf,
}

/// What a call showed the kernels: each group's places of the pairs, the ids of the sum (None: the router's), the host's sums (None: none).
#[derive(Debug, PartialEq)]
struct Shown {
    ways: [Vec<i64>; 2],
    sum_ids: Option<Vec<i64>>,
    part: Option<Vec<u32>>,
}

/// The bytes at a "device" address (host memory in the fake driver).
fn at<'a>(addr: u64, n: usize) -> &'a [u8] {
    // SAFETY: the fake driver's device memory is host memory; the address is of a live buffer of at least n bytes.
    unsafe { std::slice::from_raw_parts(addr as *const u8, n) }
}

/// Expert `e` of the layer in the directory's bytes (the truth): its group, number there, and its data and meta.
fn truth(host: &HbLayerHost, e: usize) -> (usize, &[u8], &[u8], &HbHost) {
    let (g, k) = (host.group_of[e] as usize, host.local_of[e] as usize);
    let h = host.groups[g].as_ref().unwrap();
    (g, &h.data[k * h.place.data..(k + 1) * h.place.data], &h.meta[k * h.place.meta..(k + 1) * h.place.meta], h)
}

/// One call of a layer, in the way `NhMoeHb::experts` makes it for a small call: `run_groups`, the kernels played by `body`. `ids`, `wts` and `z` are what the router and the rotation left. Returns what the kernels were shown.
fn call(rig: &Rig, dev: &Dev, host: &HbLayerHost, b: &Buffers, base: usize, (ids, wts, z): (&[i64], &[f32], &[f32]), tokens: usize) -> Shown {
    let pairs = ids.len();
    let hidden = z.len() / tokens;
    b.ids.write(0, ids).unwrap();
    b.wts.write(0, wts).unwrap();
    b.z.write(0, z).unwrap();
    let groups = Groups { layers: [Some(base), Some(base + 1)], group_of: &host.group_of, local_of: &host.local_of };
    let c = GroupsCall {
        ids: b.ids.ptr,
        wts: b.wts.ptr,
        z: b.z.ptr,
        ways: [b.ways[0].ptr, b.ways[1].ptr],
        sum_ids: b.sum_ids.ptr,
        part: b.part.ptr,
        tokens,
        pairs,
    };
    assert!(rig.store.serves_groups(&groups, tokens, pairs));
    let mut shown = Shown { ways: [vec![-1; pairs], vec![-1; pairs]], sum_ids: None, part: None };
    let mut body = |stage: Stage, g: usize, hy: &Hybrid| -> Result<()> {
        // (the kernels run behind the copies the engine's stream waited for: the host driver runs the stream's work up to here)
        dev.sync()?;
        match stage {
            Stage::Products => {
                let mut way = vec![0i64; pairs];
                b.ways[g].read(0, &mut way)?;
                for j in 0..pairs {
                    let (eg, data, meta, h) = truth(host, ids[j] as usize);
                    if way[j] >= 0 {
                        assert_eq!(eg, g, "pair {j} has a place in group {g}, its expert is in {eg}");
                        let (dev_at, stride) = rig.places[base + g];
                        let p = way[j] as u64;
                        assert!(
                            at(dev_at[0] + p * stride[0], h.place.data) == data && at(dev_at[1] + p * stride[1], h.place.meta) == meta,
                            "the place {p} of group {g} does not hold the bytes of expert {} (pair {j}) when the products would run",
                            ids[j]
                        );
                    }
                }
                assert!(hy.gpu == way.iter().any(|&w| w >= 0));
                shown.ways[g] = way;
            }
            Stage::Sum => {
                if hy.ids != 0 {
                    let mut s = vec![0i64; pairs];
                    b.sum_ids.read(0, &mut s)?;
                    shown.sum_ids = Some(s);
                }
                if hy.extra != 0 {
                    let mut p = vec![0f32; tokens * hidden];
                    b.part.read(0, &mut p)?;
                    shown.part = Some(p.iter().map(|x| x.to_bits()).collect());
                }
            }
        }
        Ok(())
    };
    // SAFETY: the buffers are live, of the sizes the call names, and `body` queues nothing.
    unsafe { rig.store.run_groups(&groups, dev, &c, &mut body) }.unwrap();
    dev.sync().unwrap();
    // the pairs the host kept are the ones that have no place; their sums, by token and in the pairs' order, are those of the experts' bytes in the directory
    let mine: Vec<usize> = (0..pairs).filter(|&j| shown.ways[0][j] < 0 && shown.ways[1][j] < 0).collect();
    assert_eq!(shown.part.is_some(), !mine.is_empty());
    if let (Some(part), Some(sum_ids)) = (&shown.part, &shown.sum_ids) {
        for j in 0..pairs {
            assert_eq!(sum_ids[j], if mine.contains(&j) { -1 } else { ids[j] }, "the sum's ids");
        }
        let k = pairs / tokens;
        let mut want = vec![0f32; tokens * hidden];
        for &j in &mine {
            let (_, data, meta, h) = truth(host, ids[j] as usize);
            let expert = h.place.expert(data, meta).unwrap();
            let mut row = vec![0f32; hidden];
            experts_with(Kernel::Scalar, Z::F32(z), &[crate::cpu_hb::HbPair { expert: &expert, token: j / k, weight: wts[j] }], &mut row, 1).unwrap();
            want[(j / k) * hidden..][..hidden].iter_mut().zip(&row).for_each(|(a, b)| *a += *b);
        }
        assert!(part.iter().zip(&want).all(|(&a, &b)| a == b.to_bits()), "the host's sums are not those of its pairs computed from the directory's bytes");
    }
    shown
}

/// A call of the layer in the other way, `run` (the copy path of a group: a prompt's chunk, or the mode `Copy`), for the group's layer `layer`: the pairs of the group's experts, in waves, each wave's places holding the bytes of
/// its experts when the products would run. Returns how many waves.
fn copy_call(rig: &Rig, dev: &Dev, host: &HbLayerHost, b: &Buffers, base: usize, g: usize, ids: &[i64]) -> usize {
    let pairs = ids.len();
    // the pairs' numbers in the group (-1: another group's), as `hb_group_ids` makes them
    let local: Vec<i64> = ids.iter().map(|&e| if host.group_of[e as usize] as usize == g { host.local_of[e as usize] as i64 } else { -1 }).collect();
    b.ids.write(0, &local).unwrap();
    let c = Call { ids: b.ids.ptr, wts: b.wts.ptr, z: b.z.ptr, way: b.ways[0].ptr, part: b.part.ptr, tokens: pairs / 4, pairs };
    let mut waves = 0;
    let mut body = |stage: Stage, _w: usize, n: usize, _hy: &Hybrid| -> Result<()> {
        if stage == Stage::Products {
            dev.sync()?;
            waves = n;
            let mut way = vec![0i64; pairs];
            b.ways[0].read(0, &mut way)?;
            for j in 0..pairs {
                if way[j] >= 0 {
                    let (eg, data, meta, h) = truth(host, ids[j] as usize);
                    assert_eq!(eg, g);
                    let (dev_at, stride) = rig.places[base + g];
                    let p = way[j] as u64;
                    assert!(
                        at(dev_at[0] + p * stride[0], h.place.data) == data && at(dev_at[1] + p * stride[1], h.place.meta) == meta,
                        "the place {p} of group {g} does not hold the bytes of expert {} when the wave's products would run",
                        ids[j]
                    );
                } else {
                    assert!(local[j] < 0 || waves > 1, "a pair of the group has no place in a call of one wave");
                }
            }
        }
        Ok(())
    };
    // SAFETY: as `call`.
    unsafe { rig.store.run(base + g, dev, &c, &mut body) }.unwrap();
    dev.sync().unwrap();
    waves
}

/// A call's choices: `tokens` tokens of 4 distinct experts out of 16 (most from a hot few), their weights, and the experts' input (bf16 values in fp32, as the rotation leaves them).
fn choices(r: &mut Rng, tokens: usize, hidden: usize) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
    let bf = |v: f32| f32::from_bits(v.to_bits() & 0xFFFF_0000);
    let mut ids = Vec::new();
    for _ in 0..tokens {
        let mut pick: Vec<i64> = Vec::new();
        while pick.len() < 4 {
            let e = if r.below(10) < 7 { r.below(5) } else { r.below(16) } as i64;
            if !pick.contains(&e) {
                pick.push(e);
            }
        }
        ids.extend(pick);
    }
    let wts = (0..tokens * 4).map(|_| 0.25 + (r.next() >> 40) as f32 / (1u64 << 24) as f32).collect();
    let z = (0..tokens * hidden).map(|_| bf((r.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)).collect();
    (ids, wts, z)
}

#[test]
fn fake_gpu_the_expert_cache_with_experts_on_the_disk_is_the_expert_cache_with_all_in_memory() {
    // (the driver of tools/cuda_fake.cpp is the one the loader found, or this does nothing: a computer with a GPU has a driver of its own)
    if Dev::probe(0).map_or(true, |g| g.name != "Fake GPU") {
        eprintln!("fake_gpu: the driver is not tools/cuda_fake.cpp's (see the top of that file): nothing done");
        return;
    }
    let dir = std::env::temp_dir().join(format!("glyd-engine-fake-gpu-{}", std::process::id()));
    let (out, c) = crate::hbpack::tests::packed_toy(Path::new(&dir), 1 << 30, 40);
    let ck = glyd_gpu::safetensors::Checkpoint::open(&out).unwrap();
    let (_, manifest) = crate::weights::v7_manifest(&out).unwrap().unwrap();
    let l = 1;
    let (host, ix) = (read_layer(&ck, &manifest, &c, l).unwrap(), index_layer(&manifest, &c, l).unwrap());
    let disk = Arc::new(HbDisk::open(&ck, &c, l, &ix).unwrap());
    let dev = Dev::new(0).unwrap();
    let hidden = c.moe.latent;
    let bufs = || Buffers {
        ids: Buf::new(8 * 64).unwrap(),
        wts: Buf::new(4 * 64).unwrap(),
        z: Buf::new(4 * 4 * hidden).unwrap(),
        ways: [Buf::new(8 * 64).unwrap(), Buf::new(8 * 64).unwrap()],
        sum_ids: Buf::new(8 * 64).unwrap(),
        part: Buf::new(4 * 4 * hidden).unwrap(),
    };
    let counts = ix.counts();
    // the memory for the experts: 0.46 of them (a few more than a call's choices of each width) and a share that leaves one slot to each width
    let total: u64 = 2 * (0..2).map(|g| counts[g] as u64 * host.groups[g].as_ref().unwrap().place.bytes()).sum::<u64>();
    for (progress, label) in [("0", "work runs only when the host makes it"), ("3", "work runs a few operations at each query")] {
        std::env::set_var("FAKE_CUDA_PROGRESS", progress);
        for mode in [Mode::Cpu, Mode::Split, Mode::Auto] {
            for share in [0.46f64, 0.15] {
                for gpu in [[2usize, 1], [5, 3]] {
                    let memory = HostMemory { bytes: (total as f64 * share) as u64, total };
                    let all = rig(&host, None, gpu, mode);
                    let few = rig(&host, Some((&disk, &ix, memory)), gpu, mode);
                    let (ba, bb) = (bufs(), bufs());
                    let mut r = Rng(0x5EED_0000 + (share * 100.0) as u64 + gpu[0] as u64);
                    let mut hosted = 0usize;
                    for round in 0..120 {
                        let tokens = 1 + r.below(3);
                        let (ids, wts, z) = choices(&mut r, tokens, hidden);
                        let (a, b) = (call(&all, &dev, &host, &ba, 0, (&ids, &wts, &z), tokens), call(&few, &dev, &host, &bb, 0, (&ids, &wts, &z), tokens));
                        assert_eq!(a, b, "{label}, {mode:?}, {share} of the experts in memory, {gpu:?} places, round {round}: the plan, the places or the host's sums differ with experts on the disk");
                        hosted += a.part.is_some() as usize;
                        // every so often the call is a prompt's chunk instead: the copy path of each group, in waves
                        if round % 15 == 7 {
                            let (ids, _, _) = choices(&mut r, 12, hidden);
                            for g in 0..2 {
                                let (wa, wb) = (copy_call(&all, &dev, &host, &ba, 0, g, &ids), copy_call(&few, &dev, &host, &bb, 0, g, &ids));
                                assert_eq!(wa, wb, "{label}, {mode:?}: the waves of a prompt's chunk");
                            }
                        }
                    }
                    let st = few.store.stats().unwrap();
                    assert!(
                        st.disk_reads > 0 && all.store.stats().unwrap().disk_reads == 0,
                        "{label}, {mode:?}, {share}: {} experts read from the disk",
                        st.disk_reads
                    );
                    if mode == Mode::Cpu {
                        assert!(hosted > 20, "{label}: the host computed pairs in only {hosted} of 120 calls");
                    }
                    eprintln!("fake_gpu: {label}, {mode:?}, {share} of the experts in memory, {gpu:?} places: ok ({} read from the disk, {} found in memory, {hosted} calls with pairs on the host)", st.disk_reads, st.ram_hits);
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
