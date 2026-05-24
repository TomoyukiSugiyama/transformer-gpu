use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const TILE: u32 = 16;

pub fn matmul_gpu(ctx: &GpuContext, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
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

    let dims_padded: [u32; 4] = [m, k, n, 0];
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

// CPU リファレンス
#[cfg(test)]
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
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
        let gpu = matmul_gpu(&ctx, &a, &eye, m as u32, k as u32, n as u32);
        assert_close(&gpu, &a, 1e-4, 1e-5);
    }

    #[test]
    fn test_matmul_random() {
        let (m, k, n) = (64, 64, 64);
        // ランダム入力（再現性のため固定シード）
        let a = random_f32(m * k, 42);
        let b = random_f32(k * n, 43);

        let cpu = matmul_cpu(&a, &b, m, k, n);
        let ctx = GpuContext::new();
        let gpu = matmul_gpu(&ctx, &a, &b, m as u32, k as u32, n as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_matmul_non_tile_boundary() {
        let (m, k, n) = (32usize, 100usize, 17usize);
        let a = random_f32(m * k, 1);
        let b = random_f32(k * n, 2);

        let cpu = matmul_cpu(&a, &b, m, k, n);
        let ctx = GpuContext::new();
        let gpu = matmul_gpu(&ctx, &a, &b, m as u32, k as u32, n as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-6);
    }
}
