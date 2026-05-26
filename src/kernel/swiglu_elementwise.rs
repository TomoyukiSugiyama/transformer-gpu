use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const SIZE: u32 = 256;

pub fn swiglu_elementwise(ctx: &GpuContext, gate: &[f32], up: &[f32]) -> Vec<f32> {
    let size = gate.len() as u32;
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
