use serde_json::{Value, json};
use starlings_core::profiles::ProfileAge;
use starlings_protocol::*;
const MODE: FixtureMode = FixtureMode::Simulation;
fn pool(id: &str, resident: u64, kv: u64, recurrent: u64, conv: u64) -> Value {
    json!({"pool_id":id,"resident_weight_bytes":resident.to_string(),"resident_overhead_bytes":"1","active_weight_bytes_per_token":(resident/2).to_string(),"kv_bytes_per_token":kv.to_string(),"recurrent_bytes_per_session":recurrent.to_string(),"convolution_bytes_per_session":conv.to_string(),"prefill_scratch_bytes":"3","decode_scratch_bytes":"5","staging_bytes":"7","reserve_bytes":"11"})
}
fn zero(id: &str) -> Value {
    let mut value = pool(id, 0, 0, 0, 0);
    for (key, v) in value.as_object_mut().unwrap() {
        if key != "pool_id" {
            *v = json!("0");
        }
    }
    value
}
pub fn fixture() -> (ModelManifest, ModelCostProfile) {
    let mut manifest: ModelManifest = decode(
        include_bytes!("../../../../examples/model-manifest.json"),
        MODE,
    )
    .unwrap();
    let mut full = manifest
        .layers
        .iter()
        .find(|l| l.layer_type == LayerType::FullAttention)
        .unwrap()
        .clone();
    full.index = BoundedU32::new(1).unwrap();
    manifest.layers = vec![manifest.layers[0].clone(), full];
    // The nonzero canonical digest lets the test also exercise default production identity checks.
    manifest.manifest_digest =
        canonical_digest(&serde_json::to_value(&manifest).unwrap(), "manifest_digest").unwrap();
    let timing = json!({"prefill_us":"13","decode_us_per_token":"17"});
    let layers:Vec<_>=manifest.layers.iter().map(|l|json!({"index":l.index,"layer_type":l.layer_type,"state_profile":l.state_profile,
        "pools":[pool("host",10,0,0,0),pool("device",100,if l.layer_type==LayerType::FullAttention {2}else{0},if l.layer_type==LayerType::LinearAttention {3}else{0},if l.layer_type==LayerType::LinearAttention {5}else{0})],"timing":timing})).collect();
    let components:Vec<_>=manifest.non_layer_components.iter().map(|c|json!({"name":c.name,"disposition":c.disposition,"owner_role":c.owner_role,"pools":if c.disposition==ComponentDisposition::Executed {vec![pool("host",20,0,0,0),pool("device",200,0,0,0)]} else {vec![zero("host"),zero("device")]},"timing":if c.disposition==ComponentDisposition::Executed {timing.clone()}else{Value::Null}})).collect();
    let digest = byte_digest(b"synthetic profile identity");
    let profile = json!({"kind":"ModelCostProfile","version":1,"fixture":true,
        "key":{"manifest_digest":manifest.manifest_digest,"execution_profile":manifest.execution_profile,"boundary_profile":manifest.required_boundary_profile,"backend_build_digest":digest,"runtime_config_digest":digest,"device_digest":digest,"pool_inventory_digest":digest,"workload":{"context_tokens":10,"prompt_tokens":6,"output_tokens":4,"prefill_chunk_tokens":3,"sessions":2}},
        "costs":{"status":"available","measurement":{"basis":"estimated","evidence_digest":digest,"observed_at":"2026-10-16T03:00:00Z","valid_for_ms":1000,"samples":0,"relative_error_bps":null},"pool_ids":["host","device"],"layers":layers,"components":components,"per_stage":[pool("host",30,0,0,0),pool("device",300,0,0,0)],"per_stage_timing":timing}});
    (
        manifest,
        decode(&serde_json::to_vec(&profile).unwrap(), MODE).unwrap(),
    )
}
pub fn age(n: u64) -> ProfileAge {
    let wall = time::OffsetDateTime::parse(
        "2026-10-16T03:00:00Z",
        &time::format_description::well_known::Rfc3339,
    )
    .unwrap()
    .unix_timestamp()
        * 1000;
    ProfileAge {
        captured_wall_ms: wall + 10,
        now_wall_ms: wall + 10 + n as i64,
        captured_at_ms: 100,
        now_ms: 100 + n,
    }
}
