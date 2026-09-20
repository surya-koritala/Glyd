fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let d = &d[..d.len().min(64 << 20)];
    let d = &d[..d.iter().rposition(|&b| b == b'\n').map_or(d.len(), |p| p + 1)];
    let img = glyd::record::transform(d).unwrap();
    let back = glyd::record::inverse(&img).unwrap();
    let i = back.iter().zip(d).position(|(a, b)| a != b).unwrap_or(back.len().min(d.len()));
    println!("lengths {} vs {}; first difference at {}", back.len(), d.len(), i);
    let lo = i.saturating_sub(60); let hi = (i + 60).min(d.len()).min(back.len());
    println!("orig: {:?}", String::from_utf8_lossy(&d[lo..hi]));
    println!("back: {:?}", String::from_utf8_lossy(&back[lo..hi]));
}
