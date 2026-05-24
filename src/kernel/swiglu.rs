use wgpu::util::DeviceExt;

use crate::{gpu_context::GpuContext, kernel::matmul::matmul_gpu};

const SIZE: u32 = 256;

pub fn swiglu_gpu(
    ctx: &GpuContext,
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    seq: u32,
    d_model: u32,
    d_ff: u32,
) -> Vec<f32> {
    let gate = matmul_gpu(ctx, x, w_gate, seq, d_model, d_ff);
    let up = matmul_gpu(ctx, x, w_up, seq, d_model, d_ff);
    let a = swiglu_elementwise(ctx, &gate, &up, seq * d_ff);
    let y = matmul_gpu(ctx, &a, w_down, seq, d_ff, d_model);
    y
}

fn swiglu_elementwise(ctx: &GpuContext, gate: &[f32], up: &[f32], size: u32) -> Vec<f32> {
    let byte_size = (size * 4) as u64;
    let buf_w_gate = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gate"),
            contents: bytemuck::cast_slice(gate),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_w_up = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("up"),
            contents: bytemuck::cast_slice(up),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [size as u32, 0, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/swiglu_elementwise.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swiglu_elementwise"),
            layout: None,
            module: &module,
            entry_point: Some("swiglu_elementwise"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_w_gate.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_w_up.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
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

        pass.dispatch_workgroups(size.div_ceil(SIZE), 1, 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_read, 0, byte_size);
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
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn elementwise_with<F: Fn(f32, f32) -> f32>(data: &[f32], other: &[f32], f: F) -> Vec<f32> {
    assert_eq!(
        data.len(),
        other.len(),
        "elementwise_with len mismatch: {:?} vs {:?}",
        data.len(),
        other.len()
    );
    let y: Vec<f32> = data
        .iter()
        .zip(other.iter())
        .map(|(&a, &b)| f(a, b))
        .collect();
    y
}

#[cfg(test)]
pub fn swiglu_cpu(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    seq: usize,
    d_model: usize,
    d_ff: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), (seq * d_model) as usize, "x must be seq×d_head");
    assert_eq!(
        w_gate.len(),
        (d_model * d_ff) as usize,
        "w_gate must be d_model×d_ff"
    );
    assert_eq!(
        w_up.len(),
        (d_model * d_ff) as usize,
        "w_up must be d_model×d_ff"
    );
    assert_eq!(
        w_down.len(),
        (d_ff * d_model) as usize,
        "w_down must be d_ff×d_model"
    );

    use crate::kernel::matmul::matmul_cpu;

    let gate = matmul_cpu(x, w_gate, seq, d_model, d_ff);
    let up = matmul_cpu(x, w_up, seq, d_model, d_ff);
    let a = elementwise_with(&gate, &up, |g, u| swish(g) * u);
    let y = matmul_cpu(&a, w_down, seq, d_ff, d_model);
    y
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::swiglu::{swiglu_cpu, swiglu_gpu},
        test_utils::{assert_close, random_f32},
    };

    #[test]
    fn test_swiglu() {
        let seq: usize = 1;
        let d_model: usize = 2;
        let d_ff = 4;
        let x: Vec<f32> = vec![1.0, 2.0];
        let w_gate: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w_up: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w_down: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let cpu = swiglu_cpu(&x, &w_gate, &w_up, &w_down, seq, d_model, d_ff);
        let ctx = GpuContext::new();
        let gpu = swiglu_gpu(
            &ctx,
            &x,
            &w_gate,
            &w_up,
            &w_down,
            seq as u32,
            d_model as u32,
            d_ff as u32,
        );

        // gate = x・w_gate
        // | 1.0 2.0 || 1.0 1.0 1.0 1.0 |
        //            | 1.0 1.0 1.0 1.0 |
        // =
        // | 3.0 3.0 3.0 3.0 |
        // up = x・w_up
        // | 3.0 3.0 3.0 3.0 |
        // a = swish(g) * u
        // swish = x / (1.0 + (-x).exp())
        // | 8.57316 8.57316 8.57316 8.57316|
        // y = a・w_down
        // | 8.57316 8.57316 8.57316 8.57316 || 1.0 1.0 |
        //                                    | 1.0 1.0 |
        //                                    | 1.0 1.0 |
        //                                    | 1.0 1.0 |
        // =
        // | 34.29264 34.29264 |
        let exp: Vec<f32> = vec![34.29264, 34.29264];
        cpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!(
                    (*c - *e).abs() < 1e-4,
                    "CPU index={} got={:.6} exp={:.6}",
                    i,
                    c,
                    e
                );
            });
        gpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!(
                    (*c - *e).abs() < 1e-4,
                    "GPU index={} got={:.6} exp={:.6}",
                    i,
                    c,
                    e
                );
            });
    }

    #[test]
    fn test_swiglu_random() {
        let seq: usize = 4;
        let d_model: usize = 64;
        let d_ff = 128;
        let x: Vec<f32> = random_f32(seq * d_model, 42);
        let w_gate: Vec<f32> = random_f32(d_model * d_ff, 43);
        let w_up: Vec<f32> = random_f32(d_model * d_ff, 44);
        let w_down: Vec<f32> = random_f32(d_ff * d_model, 45);

        let ctx = GpuContext::new();

        let cpu = swiglu_cpu(&x, &w_gate, &w_up, &w_down, seq, d_model, d_ff);
        let gpu = swiglu_gpu(
            &ctx,
            &x,
            &w_gate,
            &w_up,
            &w_down,
            seq as u32,
            d_model as u32,
            d_ff as u32,
        );

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
