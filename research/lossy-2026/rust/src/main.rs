use glyd_lossy_study::{approximate,as_float,decode,encode};
use std::{env,fs};
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let a:Vec<String>=env::args().collect();
    if a.len()!=3 { return Err("usage: glyd-lossy-study INPUT.bf16 OUTPUT.json".into()); }
    let data=fs::read(&a[1])?;
    if data.is_empty() || data.len()%2!=0 { return Err("input must contain whole BF16 words".into()); }
    let words:Vec<u16>=data.chunks_exact(2).map(|b|u16::from_le_bytes([b[0],b[1]])).collect();
    let finite=words.iter().filter(|&&w|w&0x7f80!=0x7f80).count();
    let mut rows=Vec::new();
    for m in 0..=7 { for block in [256,1024] {
        let packed=encode(&words,m,block)?;
        let restored=decode(&packed)?;
        if restored.iter().zip(&words).any(|(&a,&b)|a!=approximate(b,m)) { return Err("decoder differs from rounded reference".into()); }
        let (mut energy,mut error,mut bias,mut max_abs)=(0.0f64,0.0f64,0.0f64,0.0f64);
        let mut changed=0usize;
        for (&w,&r) in words.iter().zip(&restored) {
            if w!=r { changed+=1; }
            if w&0x7f80==0x7f80 { continue; }
            let (x,y)=(as_float(w),as_float(r)); let d=y-x;
            energy+=x*x; error+=d*d; bias+=d; max_abs=max_abs.max(d.abs());
        }
        let nrmse=if energy>0.0 {(error/energy).sqrt()} else {0.0};
        let mean_error=if finite>0 {bias/finite as f64} else {0.0};
        rows.push(format!("{{\"fraction_bits\":{m},\"block_values\":{block},\"packed_bytes\":{},\"bits_per_weight\":{},\"reduction_vs_bf16\":{},\"changed_values\":{changed},\"finite_values\":{finite},\"nrmse\":{nrmse},\"mean_error\":{mean_error},\"max_absolute_error\":{max_abs},\"exact_after_rounding\":true}}",packed.len(),8.0*packed.len() as f64/words.len() as f64,1.0-packed.len() as f64/data.len() as f64));
    }}
    fs::write(&a[2],format!("{{\"values\":{},\"source_bytes\":{},\"scope\":\"weight-sample size and numerical error only; no model quality or GPU speed result\",\"variants\":[{}]}}\n",words.len(),data.len(),rows.join(",")))?;
    println!("{} values; 16 candidates verified against rounded reference",words.len());
    Ok(())
}
