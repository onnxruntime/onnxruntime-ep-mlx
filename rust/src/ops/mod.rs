//! Wave-1 op handler modules. Each op = translate handler + claim predicate + registry entry.

pub mod attention;
pub mod conv;
pub mod elementary_contrib;
pub mod elementwise;
mod host_scan;
pub mod image;
mod kv_cache;
pub(crate) mod lut4;
pub mod math;
pub mod matmul;
pub mod nonzero;
pub mod norm;
pub mod onnx_ml_linear;
pub mod onnx_ml_preprocess;
pub(crate) mod qdq;
pub mod quant;
pub mod reduction;
pub mod shape;
// signal/random/recurrent/ssm/misc/controlflow
pub mod controlflow;
pub mod misc;
pub mod random;
pub mod recurrent;
pub mod selective_scan;
pub mod signal;
pub mod ssm;
pub mod stragglers;
pub mod unique;
pub mod vision;
