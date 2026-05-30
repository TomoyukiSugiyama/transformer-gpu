use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const SIZE: u32 = 256;

pub fn swiglu_elementwise(ctx: &GpuContext, gate: &[f32], up: &[f32]) -> Vec<f32> {
    let size = gate.len() as u32;
    let byte_size = (size * 4) as u64;
    let buf_gate = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gate"),
            contents: bytemuck::cast_slice(gate),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_up = ctx
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
                resource: buf_gate.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_up.as_entire_binding(),
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

pub fn swiglu_elementwise_backward(
    ctx: &GpuContext,
    dy: &[f32],
    gate: &[f32],
    up: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dy_size = dy.len() as u32;
    let size = 2 * dy_size as u32;
    let byte_size = (size * 4) as u64;

    let buf_dy = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dy"),
            contents: bytemuck::cast_slice(dy),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_gate = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gate"),
            contents: bytemuck::cast_slice(gate),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_up = ctx
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

    let dims_padded: [u32; 4] = [dy_size as u32, 0, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx.device.create_shader_module(wgpu::include_wgsl!(
        "../shader/swiglu_elementwise_backward.wgsl"
    ));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swiglu_elementwise_backward"),
            layout: None,
            module: &module,
            entry_point: Some("swiglu_elementwise_backward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_dy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_gate.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_up.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
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

        pass.dispatch_workgroups(dy_size.div_ceil(SIZE), 1, 1);
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
    let out: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&data);

    let d_gate: Vec<f32> = out[..dy_size as usize].to_vec();
    let d_up: Vec<f32> = out[dy_size as usize..].to_vec();
    (d_gate, d_up)
}

// CPU リファレンス
#[cfg(test)]
pub fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
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
pub fn swiglu_elementwise_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    elementwise_with(gate, up, |g, u| swish(g) * u)
}

#[cfg(test)]
pub fn swiglu_elementwise_backward_cpu(
    dy: &[f32],
    gate: &[f32],
    up: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = dy.len();
    let mut d_gate = vec![0.0f32; n];
    let mut d_up = vec![0.0f32; n];

    for i in 0..n {
        let g = gate[i];
        let sig = 1.0 / (1.0 + (-g).exp());
        // swish(x) = x * sigmoid(x) = x / (1 - e^[-x])
        let swish_g = g * sig;
        // dswish(x)/dx = σ(x) + x*σ(x)(1-σ(x)) = σ(x)(1 + x(1-σ(x)))
        let swish_prime = sig * (1.0 + g * (1.0 - sig));

        // dup_i = dy_i * swish(gate_i)
        d_up[i] = dy[i] * swish_g;
        // dgate_i = dy_i * up_i * swish'(gate_i)
        d_gate[i] = dy[i] * up[i] * swish_prime;
    }

    (d_gate, d_up)
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::swiglu_elementwise::{
            swiglu_elementwise, swiglu_elementwise_backward, swiglu_elementwise_backward_cpu,
            swiglu_elementwise_cpu,
        },
        test_utils::assert_close,
    };

    #[test]
    fn test_swiglu_elementwise() {
        let gate = vec![1.0, 2.0];
        let up = vec![1.0, 2.0];

        let cpu = swiglu_elementwise_cpu(&gate, &up);
        let ctx = GpuContext::new();
        let gpu = swiglu_elementwise(&ctx, &gate, &up);

        // a = swish(g) * u
        // swish = x / (1.0 + (-x).exp())
        // exp(-1) = 0.36787
        // exp(-2) = 0.13533
        // swish = 1 / (1.0 + 0.36787) = 0.73106
        // swish = 2 / (1.0 + 0.13533) = 1.76160
        // a = [0.73106, 3.5232]

        let exp = vec![0.73106, 3.5232];
        cpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!((c - e).abs() < 1e-4, "index {i}: cpu={}, exp={}", c, e)
            });
        gpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!((g - e).abs() < 1e-4, "index {i}: gpu={}, exp={}", g, e)
            });

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_swiglu_elementwise_backward() {
        let dy = vec![1.0, 2.0];
        let gate = vec![1.0, 2.0];
        let up = vec![1.0, 2.0];

        let (cpu_dgate, cpu_dup) = swiglu_elementwise_backward_cpu(&dy, &gate, &up);
        let ctx = GpuContext::new();
        let (gpu_dgate, gpu_dup) = swiglu_elementwise_backward(&ctx, &dy, &gate, &up);

        // swish(x) = x * sigmoid(x) = x / (1 - e^[-x])
        // exp(-1) = 0.36787
        // exp(-2) = 0.13533
        // sigmoid(1) = 1 / (1.0 + 0.36787) = 0.73106
        // sigmoid(2) = 1 / (1.0 + 0.13533) = 0.88080
        // swish(1) = 1 / (1.0 + 0.36787) = 0.73106
        // swish(2) = 2 / (1.0 + 0.13533) = 1.76160

        // swish'(x) = σ(x) + x*σ(x)(1-σ(x)) = σ(x)(1 + x(1-σ(x)))
        // swish'(1) = 0.73106 * (1 + 1*(1-0.73106)) = 0.92767
        // swish'(2) = 0.88080 * (1 + 2*(1-0.88080)) = 1.09078

        // dup_i = dy_i * swish(gate_i)
        // dup[1] = 1 * 0.73106 = 0.73106
        // dup[2] = 2 * 1.76160 = 3.5232

        // dgate_i = dy_i * up_i * swish'(gate_i)
        // dgate[1] = 1 * 1 * 0.92767 = 0.92767
        // dgate[2] = 2 * 2 * 1.09078 = 4.36312

        let exp_dup = vec![0.73106, 3.5232];
        let exp_dgate = vec![0.92767, 4.36312];
        cpu_dup
            .iter()
            .zip(exp_dup.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!((c - e).abs() < 1e-4, "dup index {i}: cpu={}, exp={}", c, e)
            });
        cpu_dgate
            .iter()
            .zip(exp_dgate.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!(
                    (c - e).abs() < 1e-4,
                    "dgate index {i}: cpu={}, exp={}",
                    c,
                    e
                )
            });

        gpu_dup
            .iter()
            .zip(exp_dup.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!((g - e).abs() < 1e-4, "dup index {i}: gpu={}, exp={}", g, e)
            });
        gpu_dgate
            .iter()
            .zip(exp_dgate.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "dgate index {i}: gpu={}, exp={}",
                    g,
                    e
                )
            });

        // assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
