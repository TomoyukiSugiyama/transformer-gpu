use std::collections::HashMap;

use crate::{
    checkpoint::{Checkpointable, WeightMap},
    gpu_context::GpuContext,
    gpu_tensor::GpuTensor,
};

const SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamWParams {
    pub len: u32,
    pub step: u32,
    pub pad0: u32,
    pub pad1: u32,

    pub beta1: f32,
    pub beta2: f32,
    pub lr: f32,
    pub eps: f32,

    pub weight_decay: f32,
    pub pad2: f32,
    pub pad3: f32,
    pub pad4: f32,
}

pub fn encode_adamw_step(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    bind_group: &wgpu::BindGroup,
    weight_len: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("adamw_pass"),
        timestamp_writes: None,
    });

    pass.set_pipeline(&ctx.adamw_pipeline);
    pass.set_bind_group(0, bind_group, &[]);

    pass.dispatch_workgroups(weight_len.div_ceil(SIZE), 1, 1);
}

pub fn create_adamw_bind_group(
    ctx: &GpuContext,
    grad: &GpuTensor,
    weight: &GpuTensor,
    m: &GpuTensor,
    v: &GpuTensor,
    dims: &wgpu::Buffer,
    label: Option<&str>,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label,
        layout: &ctx.adamw_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grad.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: m.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dims.as_entire_binding(),
            },
        ],
    })
}

pub struct AdamWParam {
    data: Vec<f32>, // パラメータ本体(W1, W2, b など)
    m: Vec<f32>,    // 一次モーメント
    v: Vec<f32>,    // 二次モーメント
}

impl AdamWParam {
    pub fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Self {
            data,
            m: vec![0.0f32; n],
            v: vec![0.0f32; n],
        }
    }

    pub fn step(
        &mut self,
        grad: &[f32],
        t: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
    ) {
        let t = t as f32;

        for i in 0..self.data.len() {
            let g = grad[i];
            // モーメント更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - beta2.powf(t));

            self.data[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * self.data[i]);
        }
    }
}

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_count: usize,
    params: HashMap<String, AdamWParam>,
    grad_scale: f32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self::new_with_wd(lr, 0.01)
    }

    pub fn new_with_wd(lr: f32, wd: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            step_count: 0,
            params: HashMap::new(),
            grad_scale: 1.0,
        }
    }

    pub fn set_wd(&mut self, wd: f32) {
        self.wd = wd;
    }

    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    pub fn set_grad_scale(&mut self, batch_size: usize) {
        self.grad_scale = 1.0 / batch_size as f32;
    }

    pub fn reset_grad_scale(&mut self) {
        self.grad_scale = 1.0;
    }

    pub fn increment_step(&mut self) {
        self.step_count += 1;
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }
    pub fn beta1(&self) -> f32 {
        self.beta1
    }
    pub fn beta2(&self) -> f32 {
        self.beta2
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }
    pub fn weight_decay(&self) -> f32 {
        self.wd
    }

    pub fn step(&mut self, param_id: &str, param: &mut Vec<f32>, grad: &[f32]) {
        assert_eq!(
            param.len(),
            grad.len(),
            "param/grad size mismatch: {param_id}"
        );

        let scale = self.grad_scale;
        let scaled_grad: Vec<f32> = grad.iter().map(|&g| g * scale).collect();

        let entry = self
            .params
            .entry(param_id.to_string())
            .or_insert_with(|| AdamWParam::new(param.clone()));

        entry.step(
            &scaled_grad,
            self.step_count,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.wd,
        );

        param.copy_from_slice(&entry.data);
    }
}

impl Checkpointable for AdamW {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("lr", self.lr.to_bits() as u64);
        map.insert_scalar("beta1", self.beta1.to_bits() as u64);
        map.insert_scalar("beta2", self.beta2.to_bits() as u64);
        map.insert_scalar("eps", self.eps.to_bits() as u64);
        map.insert_scalar("wd", self.wd.to_bits() as u64);
        map.insert_scalar("step_count", self.step_count as u64);
        map.insert_scalar("grad_scale", self.grad_scale.to_bits() as u64);
        for (k, param) in &self.params {
            map.insert_vector(&format!("m.{k}"), param.m.clone());
            map.insert_vector(&format!("v.{k}"), param.v.clone());
            map.insert_vector(&format!("data.{k}"), param.data.clone());
        }
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> std::io::Result<()> {
        self.lr = f32::from_bits(map.get_scalar("lr")? as u32);
        self.beta1 = f32::from_bits(map.get_scalar("beta1")? as u32);
        self.beta2 = f32::from_bits(map.get_scalar("beta2")? as u32);
        self.eps = f32::from_bits(map.get_scalar("eps")? as u32);
        self.wd = f32::from_bits(map.get_scalar("wd")? as u32);
        self.step_count = map.get_scalar("step_count")? as usize;
        self.grad_scale = f32::from_bits(map.get_scalar("grad_scale")? as u32);
        for key in map.vector_keys() {
            if let Some(k) = key.strip_prefix("data.") {
                let data = map.get_vector(key)?.clone();
                let n = data.len();
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![0.0; n],
                        m: vec![0.0; n],
                        v: vec![0.0; n],
                    });
                entry.data = data;
            } else if let Some(k) = key.strip_prefix("m.") {
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![],
                        m: vec![],
                        v: vec![],
                    });
                entry.m = map.get_vector(key)?.clone();
            } else if let Some(k) = key.strip_prefix("v.") {
                let entry = self
                    .params
                    .entry(k.to_string())
                    .or_insert_with(|| AdamWParam {
                        data: vec![],
                        m: vec![],
                        v: vec![],
                    });
                entry.v = map.get_vector(key)?.clone();
            }
        }

        for (k, p) in &self.params {
            if p.data.len() != p.m.len() || p.data.len() != p.v.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("AdamW state size mismatch for {k}"),
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use crate::gpu_tensor::read_f32_tensor;

    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut opt = AdamW::new(0.001);
        let mut param = vec![1.0f32, 2.0f32];
        let grad = vec![0.1f32, 0.2f32];

        opt.increment_step();
        opt.step("w", &mut param, &grad);

        // step 後にパラメータが減少していることを確認
        assert!(param[0] < 1.0, "param[0] should decrease");
        assert!(param[1] < 2.0, "param[1] should decrease");
    }

    #[test]
    fn test_adamw_grad_scale() {
        let mut opt1 = AdamW::new(0.001);
        let mut opt2 = AdamW::new(0.001);

        // バッチサイズ 4 で平均
        opt2.set_grad_scale(4);

        let mut p1 = vec![1.0f32];
        let mut p2 = vec![1.0f32];
        let grad_large = vec![4.0f32];
        // 4.0 / 4 = 1.0 と等価
        let grad_small = vec![1.0f32];

        opt1.increment_step();
        opt2.increment_step();
        opt1.step("w", &mut p1, &grad_large);
        opt2.step("w", &mut p2, &grad_small);

        // grad_scale=1/4 で 4.0 を渡すのは、1.0 をそのまま渡すのと等価
        assert!((p1[0] - p2[0]).abs() < 1e-6);
    }

    fn cpu_adamw_step_reference(
        weight: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        step: u32,
        beta1: f32,
        beta2: f32,
        lr: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        assert_eq!(weight.len(), grad.len());
        assert_eq!(weight.len(), m.len());
        assert_eq!(weight.len(), v.len());
        assert!(step >= 1);

        let step_f = step as f32;
        let beta1_correction = 1.0 - beta1.powf(step_f);
        let beta2_correction = 1.0 - beta2.powf(step_f);

        for i in 0..weight.len() {
            let g = grad[i];

            let new_m = beta1 * m[i] + (1.0 - beta1) * g;
            let new_v = beta2 * v[i] + (1.0 - beta2) * g * g;

            m[i] = new_m;
            v[i] = new_v;

            let m_hat = new_m / beta1_correction;
            let v_hat = new_v / beta2_correction;

            weight[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * weight[i]);
        }
    }

    fn assert_f32_slice_close(name: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");

        for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                got.is_finite(),
                "{name}[{i}] is not finite: got={got}, expected={want}"
            );

            assert!(
                want.is_finite(),
                "{name}[{i}] expected is not finite: got={got}, expected={want}"
            );

            let abs_error = (got - want).abs();
            let allowed_error = atol + rtol * want.abs();

            assert!(
                abs_error <= allowed_error,
                "{name}[{i}] mismatch:\n\
                 gpu      = {got:.9e}\n\
                 cpu      = {want:.9e}\n\
                 abs diff = {abs_error:.9e}\n\
                 allowed  = {allowed_error:.9e}",
            );
        }
    }

    fn make_values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as u32;
                let bits = x
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(seed.wrapping_mul(12_345))
                    .wrapping_add(12_345);

                let unit = (bits % 10_001) as f32 / 10_000.0;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn gpu_adamw_matches_cpu_for_three_steps() {
        let ctx = GpuContext::new();

        // 256の境界と、最後のpartial workgroupを検証する。
        let n = 257usize;

        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let lr = 3.0e-4_f32;
        let eps = 1.0e-8_f32;
        let weight_decay = 0.01_f32;

        let initial_weight = make_values(n, 100, 0.20);

        let gradients = [
            make_values(n, 101, 0.10),
            make_values(n, 102, 0.10),
            make_values(n, 103, 0.10),
        ];

        // ---------- CPU reference ----------
        let mut cpu_weight = initial_weight.clone();
        let mut cpu_m = vec![0.0_f32; n];
        let mut cpu_v = vec![0.0_f32; n];

        for (index, grad) in gradients.iter().enumerate() {
            cpu_adamw_step_reference(
                &mut cpu_weight,
                grad,
                &mut cpu_m,
                &mut cpu_v,
                (index + 1) as u32,
                beta1,
                beta2,
                lr,
                eps,
                weight_decay,
            );
        }

        // ---------- GPU buffers ----------
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let grad = GpuTensor::new_f32(
            &ctx.device,
            vec![n],
            usage,
            Some("adamw_test_grad".to_owned()),
        );

        let weight = GpuTensor::new_f32(
            &ctx.device,
            vec![n],
            usage,
            Some("adamw_test_weight".to_owned()),
        );

        let m = GpuTensor::new_f32(&ctx.device, vec![n], usage, Some("adamw_test_m".to_owned()));

        let v = GpuTensor::new_f32(&ctx.device, vec![n], usage, Some("adamw_test_v".to_owned()));

        // GPU bufferは初期化保証がないため、明示的に初期値を入れる。
        weight.write_f32(&ctx.queue, &initial_weight);
        m.write_f32(&ctx.queue, &vec![0.0_f32; n]);
        v.write_f32(&ctx.queue, &vec![0.0_f32; n]);

        let initial_params = AdamWParams {
            len: n as u32,
            step: 1,
            pad0: 0,
            pad1: 0,

            beta1,
            beta2,
            lr,
            eps,

            weight_decay,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };

        let params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adamw_test_params"),
                contents: bytemuck::bytes_of(&initial_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group = create_adamw_bind_group(
            &ctx,
            &grad,
            &weight,
            &m,
            &v,
            &params,
            Some("adamw_test_bind_group"),
        );

        // ---------- GPU three steps ----------
        for (index, grad_values) in gradients.iter().enumerate() {
            let step = (index + 1) as u32;

            grad.write_f32(&ctx.queue, grad_values);

            let step_params = AdamWParams {
                len: n as u32,
                step,
                pad0: 0,
                pad1: 0,

                beta1,
                beta2,
                lr,
                eps,

                weight_decay,
                pad2: 0.0,
                pad3: 0.0,
                pad4: 0.0,
            };

            ctx.queue
                .write_buffer(&params, 0, bytemuck::bytes_of(&step_params));

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("adamw_test_encoder"),
                });

            encode_adamw_step(&ctx, &mut encoder, &bind_group, n as u32);

            ctx.queue.submit([encoder.finish()]);
        }

        // read_f32_tensor内部でCOPY_SRC付きstorage bufferから
        // staging bufferへcopyし、map_asyncしてVec<f32>へ戻す前提。
        let gpu_weight = read_f32_tensor(&ctx, &weight);
        let gpu_m = read_f32_tensor(&ctx, &m);
        let gpu_v = read_f32_tensor(&ctx, &v);

        // m/vは値が小さいので絶対誤差を重視する。
        assert_f32_slice_close("weight", &gpu_weight, &cpu_weight, 2.0e-5, 2.0e-5);

        assert_f32_slice_close("m", &gpu_m, &cpu_m, 2.0e-6, 2.0e-6);

        assert_f32_slice_close("v", &gpu_v, &cpu_v, 2.0e-7, 2.0e-5);
    }
}
