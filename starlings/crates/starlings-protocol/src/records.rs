use crate::scalars::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRun {
    pub kind: String,
    pub schema_version: Version,
    pub fixture: bool,
    pub run_id: InvocationId,
    pub status: BenchmarkStatus,
    pub compute_commit: String,
    pub backend_build_digests: Vec<Digest>,
    pub manifest_digest: Digest,
    pub execution_profile: String,
    pub plan_digest: Digest,
    pub baseline_run_id: Option<InvocationId>,
    pub hardware_inventory_path: String,
    pub hardware_inventory_sha256: Digest,
    pub workload_sha256: Digest,
    pub network_profile: String,
    pub concurrency: BoundedU32<1, 256>,
    pub started_at: String,
    pub timing_source: String,
    pub definitions_version: String,
    pub warmup_requests_excluded: BoundedU32<0, 10000>,
    pub requests: Vec<BenchmarkRequest>,
    pub peak_memory_pools: Vec<BenchmarkMemory>,
    pub raw_logs_path: String,
    pub raw_logs_sha256: Digest,
    pub notes: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkStatus {
    Measured,
    Failed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkOutcome {
    Success,
    Oom,
    Cancelled,
    Timeout,
    Error,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRequest {
    pub request_id: RequestId,
    pub outcome: BenchmarkOutcome,
    pub prompt_tokens: BoundedU32<0, 2147483647>,
    pub output_tokens: BoundedU32<0, 2147483647>,
    pub ttft_ms: Option<f64>,
    pub duration_ms: f64,
    pub inter_token_ms: Vec<f64>,
    pub error_code: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkMemory {
    pub node_id: NodeId,
    pub pool_id: String,
    pub peak_bytes: Decimal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPool {
    pub pool_id: String,
    pub kind: MemoryPoolKind,
    pub total_bytes: Decimal,
    pub available_bytes: Decimal,
    pub reserved_bytes: Decimal,
    pub device_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Backend {
    pub backend_id: String,
    pub build_digest: Digest,
    pub architectures: Vec<String>,
    pub execution_profiles: Vec<String>,
    pub boundary_profiles: Vec<String>,
    pub task_types: Vec<BackendTaskTypes>,
    pub layer_sharding: bool,
    pub max_context_tokens: BoundedU32<1, 1048576>,
    pub max_sessions: BoundedU32<1, 256>,
    pub device_id: String,
    pub memory_pool_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeCapabilities {
    pub kind: String,
    pub protocol_version: Version,
    pub fixture: bool,
    pub node_id: NodeId,
    pub cluster_id: ClusterId,
    pub snapshot_id: SnapshotId,
    pub observed_at: String,
    pub sample_age_ms: BoundedU32<0, 60000>,
    pub state: NodeCapabilitiesState,
    pub os: NodeCapabilitiesOs,
    pub architecture: NodeCapabilitiesArchitecture,
    pub memory_pools: Vec<MemoryPool>,
    pub backends: Vec<Backend>,
    pub control_endpoint: String,
    pub tensor_endpoint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub index: BoundedU32<0, 7>,
    pub node_id: NodeId,
    pub device_id: String,
    pub backend_id: String,
    pub backend_build_digest: Digest,
    pub layer_start: BoundedU32<0, 4096>,
    pub layer_end_exclusive: BoundedU32<1, 4096>,
    pub owns_embedding: bool,
    pub owns_final_norm_and_head: bool,
    pub memory_pool_id: String,
    pub resident_bytes: Decimal,
    pub state_budget_bytes: Decimal,
    pub scratch_budget_bytes: Decimal,
    pub safety_reserve_bytes: Decimal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementPlan {
    pub kind: String,
    pub protocol_version: Version,
    pub fixture: bool,
    pub plan_id: PlanId,
    pub plan_digest: Digest,
    pub cluster_id: ClusterId,
    pub authority_epoch: Decimal,
    pub manifest_digest: Digest,
    pub mode: PlacementPlanMode,
    pub model_layer_count: BoundedU32<1, 4096>,
    pub execution_profile: String,
    pub boundary_profile: String,
    pub context_limit: BoundedU32<1, 1048576>,
    pub max_sessions: BoundedU32<1, 256>,
    pub stages: Vec<Stage>,
    pub created_at: String,
    pub expires_at: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lease {
    pub kind: String,
    pub protocol_version: Version,
    pub lease_id: LeaseId,
    pub cluster_id: ClusterId,
    pub plan_digest: Digest,
    pub authority_epoch: Decimal,
    pub owner_node_id: NodeId,
    pub issued_at: String,
    pub lifetime_ms: BoundedU32<1000, 60000>,
    pub max_sessions: BoundedU32<1, 256>,
    pub reserved_bytes: Decimal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub profile: String,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: BoundedU32<0, 2147483647>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    pub kind: String,
    pub protocol_version: Version,
    pub session_id: SessionId,
    pub generation_id: GenerationId,
    pub client_id: ClientId,
    pub lease_id: LeaseId,
    pub plan_digest: Digest,
    pub authority_epoch: Decimal,
    pub prompt_token_count: BoundedU32<1, 1048576>,
    pub max_output_tokens: BoundedU32<1, 1048576>,
    pub context_limit: BoundedU32<1, 1048576>,
    pub sampling: Sampling,
    pub state: SessionState,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireError {
    pub error: ErrorDetail,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub kind: String,
    pub manifest_version: Version,
    pub fixture: bool,
    pub manifest_digest: Digest,
    pub canonicalization: String,
    pub model_id: String,
    pub source: ModelSource,
    pub architecture: String,
    pub source_precision: String,
    pub execution_precision: String,
    pub execution_profile: String,
    pub is_lossy: bool,
    pub execution_scope: Vec<ModelManifestExecutionScope>,
    pub required_boundary_profile: String,
    pub config_sha256: Digest,
    pub tokenizer_sha256: Digest,
    pub template_sha256: Digest,
    pub assets: Vec<Asset>,
    pub layers: Vec<Layer>,
    pub non_layer_components: Vec<Component>,
    pub license: ModelLicense,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorHeader {
    pub kind: String,
    pub protocol_version: Version,
    pub lease_id: LeaseId,
    pub plan_digest: Digest,
    pub authority_epoch: Decimal,
    pub session_id: SessionId,
    pub generation_id: GenerationId,
    pub invocation_id: InvocationId,
    pub sequence: Decimal,
    pub source_stage: BoundedU32<0, 7>,
    pub destination_stage: BoundedU32<0, 7>,
    pub phase: TensorHeaderPhase,
    pub position_start: BoundedU32<0, 1048575>,
    pub token_count: BoundedU32<1, 1048576>,
    pub boundary_profile: String,
    pub tensor_index: BoundedU32<0, 7>,
    pub tensor_count: BoundedU32<1, 8>,
    pub tensor_role: TensorHeaderTensorRole,
    pub dtype: TensorHeaderDtype,
    pub shape: Vec<BoundedU32<1, 1048576>>,
    pub total_tensor_bytes: Decimal,
    pub chunk_offset_bytes: Decimal,
    pub chunk_bytes: Decimal,
    pub end_of_tensor: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: BoundedU32<1, 1048576>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<BoundedU32<0, 2147483647>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorDetail {
    pub code: ErrorDetailCode,
    pub message: String,
    pub request_id: RequestId,
    pub retryable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSource {
    pub repository: String,
    pub revision: String,
    pub created_at_evidence: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: String,
    pub role: AssetRole,
    pub sha256: Digest,
    pub size_bytes: Decimal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub index: BoundedU32<0, 4095>,
    #[serde(rename = "type")]
    pub layer_type: LayerType,
    pub resident_weight_bytes: Decimal,
    pub state_profile: String,
    pub asset_paths: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub name: String,
    pub disposition: ComponentDisposition,
    pub resident_weight_bytes: Decimal,
    pub owner_role: ComponentOwnerRole,
    pub asset_paths: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLicense {
    pub source_identifier: String,
    pub terms_url: String,
    pub redistribution_approved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: ChatMessageRole,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPoolKind {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "device")]
    Device,
    #[serde(rename = "unified")]
    Unified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendTaskTypes {
    #[serde(rename = "generate")]
    Generate,
    #[serde(rename = "embed")]
    Embed,
    #[serde(rename = "transcribe")]
    Transcribe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeCapabilitiesState {
    #[serde(rename = "ready")]
    Ready,
    #[serde(rename = "draining")]
    Draining,
    #[serde(rename = "unavailable")]
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeCapabilitiesOs {
    #[serde(rename = "linux")]
    Linux,
    #[serde(rename = "macos")]
    Macos,
    #[serde(rename = "windows")]
    Windows,
    #[serde(rename = "ios")]
    Ios,
    #[serde(rename = "android")]
    Android,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeCapabilitiesArchitecture {
    #[serde(rename = "x86_64")]
    X8664,
    #[serde(rename = "aarch64")]
    Aarch64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementPlanMode {
    #[serde(rename = "whole_request")]
    WholeRequest,
    #[serde(rename = "layer_pipeline")]
    LayerPipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    #[serde(rename = "queued")]
    Queued,
    #[serde(rename = "admitted")]
    Admitted,
    #[serde(rename = "prefilling")]
    Prefilling,
    #[serde(rename = "decoding")]
    Decoding,
    #[serde(rename = "cancelling")]
    Cancelling,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelManifestExecutionScope {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "vision")]
    Vision,
    #[serde(rename = "audio")]
    Audio,
    #[serde(rename = "mtp")]
    Mtp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorHeaderPhase {
    #[serde(rename = "prefill")]
    Prefill,
    #[serde(rename = "decode")]
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorHeaderTensorRole {
    #[serde(rename = "hidden")]
    Hidden,
    #[serde(rename = "residual")]
    Residual,
    #[serde(rename = "token_ids")]
    TokenIds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorHeaderDtype {
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "u32")]
    U32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorDetailCode {
    #[serde(rename = "INVALID_REQUEST")]
    InvalidRequest,
    #[serde(rename = "UNAUTHENTICATED")]
    Unauthenticated,
    #[serde(rename = "FORBIDDEN")]
    Forbidden,
    #[serde(rename = "UNSUPPORTED_MODEL")]
    UnsupportedModel,
    #[serde(rename = "UNSUPPORTED_PROFILE")]
    UnsupportedProfile,
    #[serde(rename = "UNSUPPORTED_FEATURE")]
    UnsupportedFeature,
    #[serde(rename = "STALE_CAPABILITY")]
    StaleCapability,
    #[serde(rename = "CAPACITY_UNAVAILABLE")]
    CapacityUnavailable,
    #[serde(rename = "DEADLINE_EXCEEDED")]
    DeadlineExceeded,
    #[serde(rename = "ARTIFACT_MISMATCH")]
    ArtifactMismatch,
    #[serde(rename = "PROTOCOL_MISMATCH")]
    ProtocolMismatch,
    #[serde(rename = "STALE_EPOCH")]
    StaleEpoch,
    #[serde(rename = "WORKER_LOST")]
    WorkerLost,
    #[serde(rename = "CANCELLED")]
    Cancelled,
    #[serde(rename = "INTERNAL")]
    Internal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetRole {
    #[serde(rename = "config")]
    Config,
    #[serde(rename = "tokenizer")]
    Tokenizer,
    #[serde(rename = "template")]
    Template,
    #[serde(rename = "weights")]
    Weights,
    #[serde(rename = "conversion_metadata")]
    ConversionMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    #[serde(rename = "full_attention")]
    FullAttention,
    #[serde(rename = "linear_attention")]
    LinearAttention,
    #[serde(rename = "other")]
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentDisposition {
    #[serde(rename = "executed")]
    Executed,
    #[serde(rename = "not_in_scope")]
    NotInScope,
    #[serde(rename = "unsupported")]
    Unsupported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentOwnerRole {
    #[serde(rename = "first")]
    First,
    #[serde(rename = "last")]
    Last,
    #[serde(rename = "none")]
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatMessageRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}
