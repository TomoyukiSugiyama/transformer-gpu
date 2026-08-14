use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const TILE: u32 = 16;

fn matmul(
    ctx: &GpuContext,
    a: &[f32],
    b: &[f32],
    m: u32,
    k: u32,
    n: u32,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let a_rows = if trans_a { k } else { m };
    let a_cols = if trans_a { m } else { k };

    let b_rows = if trans_b { n } else { k };
    let b_cols = if trans_b { k } else { n };
    let op_a_rows = if trans_a { a_cols } else { a_rows };
    let op_a_cols = if trans_a { a_rows } else { a_cols };

    let op_b_rows = if trans_b { b_cols } else { b_rows };
    let op_b_cols = if trans_b { b_rows } else { b_cols };

    assert_eq!(
        op_a_cols, op_b_rows,
        "inner dimension mismatch: op(A) ({}, {}), op(B) ({}, {})",
        op_a_rows, op_a_cols, op_b_rows, op_b_cols
    );
    let byte_size = (m * n * 4) as u64;
    let buf_a = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_b = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B"),
            contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 8] = [m, k, n, trans_a as u32, trans_b as u32, 0, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/matmul.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &module,
            entry_point: Some("matmul"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dims.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        pass.dispatch_workgroups(n.div_ceil(TILE), m.div_ceil(TILE), 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_c, 0, &buf_read, 0, byte_size);
    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

pub fn matmul_forward(ctx: &GpuContext, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
    matmul(ctx, a, b, m, k, n, false, false)
}

pub fn matmul_backward(
    ctx: &GpuContext,
    grad_output: &[f32], // dY: (m × n)
    a: &[f32],           // forward の X: (m × k)
    b: &[f32],           // forward の W: (k × n)
    m: u32,
    k: u32,
    n: u32,
) -> (Vec<f32>, Vec<f32>) {
    // dX = dY @ W^T
    let grad_a = matmul(ctx, grad_output, b, m, n, k, false, true);
    // dW = X^T @ dY
    let grad_b = matmul(ctx, a, grad_output, k, m, n, true, false);
    (grad_a, grad_b)
}

// CPU リファレンス
#[cfg(test)]
fn matmul_cpu(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let a_val = if trans_a { a[p * m + i] } else { a[i * k + p] };
                let b_val = if trans_b { b[j * k + p] } else { b[p * n + j] };
                c[i * n + j] += a_val * b_val;
            }
        }
    }
    c
}

#[cfg(test)]
pub fn matmul_forward_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    matmul_cpu(a, b, m, k, n, false, false)
}

#[cfg(test)]
pub fn matmul_backward_cpu(
    grad_output: &[f32], // dY: (m × n)
    a: &[f32],           // forward の X: (m × k)
    b: &[f32],           // forward の W: (k × n)
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    // dX = dY @ W^T
    let grad_a = matmul_cpu(grad_output, b, m, n, k, false, true);
    // dW = X^T @ dY
    let grad_b = matmul_cpu(a, grad_output, k, m, n, true, false);
    (grad_a, grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_utils::assert_close, util::random_f32};

    #[test]
    fn test_matmul_identity() {
        let ctx = GpuContext::new();
        let (m, k, n) = (4, 4, 4);
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let eye: Vec<f32> = (0..16)
            .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
            .collect();
        let gpu = matmul(&ctx, &a, &eye, m as u32, k as u32, n as u32, false, false);
        assert_close(&gpu, &a, 1e-4, 1e-5);
    }

    #[test]
    fn test_matmul_random() {
        let (m, k, n) = (64, 64, 64);
        let scale = 0.1f32;
        let a = random_f32(m * k, 42, scale);
        let b = random_f32(k * n, 43, scale);

        let cpu = matmul_cpu(&a, &b, m, k, n, false, false);
        let ctx = GpuContext::new();
        let gpu = matmul(&ctx, &a, &b, m as u32, k as u32, n as u32, false, false);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_trans_a() {
        let (m, k, n) = (3, 5, 7);
        let scale = 0.1f32;

        // op(A) = A^T: [m × k]
        // 元Aは [k × m]
        let a = random_f32(k * m, 42, scale);

        // Bは [k × n]
        let b = random_f32(k * n, 43, scale);

        let cpu = matmul_cpu(&a, &b, m, k, n, true, false);

        let ctx = GpuContext::new();

        let gpu = matmul(&ctx, &a, &b, m as u32, k as u32, n as u32, true, false);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_trans_b() {
        let (m, k, n) = (3, 5, 7);
        let scale = 0.1f32;

        // Aは [m × k]
        let a = random_f32(m * k, 42, scale);

        // op(B) = B^T: [k × n]
        // 元Bは [n × k]
        let b = random_f32(n * k, 43, scale);

        let cpu = matmul_cpu(&a, &b, m, k, n, false, true);

        let ctx = GpuContext::new();

        let gpu = matmul(&ctx, &a, &b, m as u32, k as u32, n as u32, false, true);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_trans_ab() {
        let (m, k, n) = (3, 5, 7);
        let scale = 0.1f32;

        // 元Aは [k × m]
        let a = random_f32(k * m, 42, scale);

        // 元Bは [n × k]
        let b = random_f32(n * k, 43, scale);

        let cpu = matmul_cpu(&a, &b, m, k, n, true, true);

        let ctx = GpuContext::new();

        let gpu = matmul(&ctx, &a, &b, m as u32, k as u32, n as u32, true, true);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_matmul_non_tile_boundary() {
        let (m, k, n) = (32usize, 100usize, 17usize);
        let scale = 0.1f32;
        let a = random_f32(m * k, 1, scale);
        let b = random_f32(k * n, 2, scale);

        let cpu = matmul_cpu(&a, &b, m, k, n, false, false);
        let ctx = GpuContext::new();
        let gpu = matmul(&ctx, &a, &b, m as u32, k as u32, n as u32, false, false);

        assert_close(&gpu, &cpu, 1e-4, 1e-6);
    }

    #[test]
    fn test_matmul_forward() {
        let (m, k, n) = (10, 11, 12);
        let scale = 0.1f32;
        let a = random_f32(m * k, 42, scale);
        let b = random_f32(k * n, 43, scale);

        let cpu = matmul_forward_cpu(&a, &b, m, k, n);
        let ctx = GpuContext::new();
        let gpu = matmul_forward(&ctx, &a, &b, m as u32, k as u32, n as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_matmul_backward() {
        let (m, k, n) = (10, 11, 12);
        let scale = 0.1f32;
        let dy = random_f32(m * n, 42, scale);
        let x = random_f32(m * k, 43, scale);
        let w = random_f32(k * n, 44, scale);

        let (cpu_dx, cpu_dw) = matmul_backward_cpu(&dy, &x, &w, m, k, n);
        let ctx = GpuContext::new();
        let (gpu_dx, gpu_dw) = matmul_backward(&ctx, &dy, &x, &w, m as u32, k as u32, n as u32);

        assert_close(&cpu_dx, &gpu_dx, 1e-4, 1e-5);
        assert_close(&cpu_dw, &gpu_dw, 1e-4, 1e-5);
    }
}
