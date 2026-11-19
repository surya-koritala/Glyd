use glyd_gpu::pack::{pack_tiered, tiered_bytes, unpack_tiered};
fn main(){
    let a:Vec<String>=std::env::args().collect();
    let b=std::fs::read(&a[1]).unwrap(); let rows:usize=a[2].parse().unwrap(); let cols:usize=a[3].parse().unwrap();
    let w:Vec<u16>=b.chunks_exact(2).map(|c|u16::from_le_bytes([c[0],c[1]])).collect();
    let t=std::time::Instant::now();
    let p=pack_tiered(&w,rows,cols);
    let n=tiered_bytes(&p);
    let mut o=vec![0u16;w.len()]; let ok=unpack_tiered(&p,&mut o)&&o==w;
    println!("{} {} {} {:.2}",n,b.len(),ok,t.elapsed().as_secs_f32());
}
