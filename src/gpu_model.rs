use crate::gpu_tensor::GpuTensor;

pub struct GpuAdamState {
    pub m: GpuTensor,
    pub v: GpuTensor,
}

pub struct GpuParameter {
    pub value: GpuTensor,
    pub grad: GpuTensor,
    pub adam: GpuAdamState,
}

pub struct GpuAttention {
    pub w_q: GpuParameter,
    pub w_k: GpuParameter,
    pub w_v: GpuParameter,
    pub w_o: GpuParameter,
}

pub struct GpuFfn {
    pub w_gate: GpuParameter,
    pub w_up: GpuParameter,
    pub w_down: GpuParameter,
}

pub struct GpuTransformerBlock {
    pub gamma_1: GpuParameter,
    pub gamma_2: GpuParameter,
    pub attn: GpuAttention,
    pub ffn: GpuFfn,
}

pub struct GpuLanguageModel {
    pub embedding: GpuParameter,
    pub final_gamma: GpuParameter,
    pub lm_head: GpuParameter,
    pub blocks: Vec<GpuTransformerBlock>,
}
