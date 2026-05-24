use wgpu::util::DeviceExt;

const WG: u32 = 256;

pub fn rms_norm_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &[f32],
    gamma: &[f32],
    eps: f32,
    d_model: u32,
) -> Vec<f32> {
    assert_eq!(
        x.len() as u32 % d_model,
        0,
        "x.len() must be divisible by d_model"
    );
    assert!(
        d_model <= WG,
        "d_model={} exceeds workgroup_size={}",
        d_model,
        WG
    );
    let seq = x.len() as u32 / d_model;
    let byte_size = (seq * d_model * 4) as u64;
    let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"),
        contents: bytemuck::cast_slice(&x),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamma"),
        contents: bytemuck::cast_slice(&gamma),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [d_model as u32, eps.to_bits(), 0, 0];
    let buf_dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims_padded),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(wgpu::include_wgsl!("../shader/rms_norm.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rms_norm"),
        layout: None,
        module: &module,
        entry_point: Some("rms_norm"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_gamma.as_entire_binding(),
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

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        pass.dispatch_workgroups(seq, 1, 1);
    }

    let buf_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_read, 0, byte_size);
    queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

// CPU リファレンス
#[cfg(test)]
fn rms_norm_cpu(x: &[f32], gamma: &[f32], eps: f32, d_model: usize) -> Vec<f32> {
    assert_eq!(x.len() % d_model, 0, "x.len() must be divisible by d_model");

    let seq = x.len() / d_model;
    let mut out = vec![0.0f32; x.len()];

    for row in 0..seq {
        let base = row * d_model;
        let row_x = &x[base..base + d_model];

        // 行ごとに RMS を計算
        let rms = (row_x.iter().map(|v| v * v).sum::<f32>() / d_model as f32 + eps).sqrt();

        for (i, (xi, gi)) in row_x.iter().zip(gamma.iter()).enumerate() {
            out[base + i] = xi / rms * gi;
        }
    }
    out
}

#[cfg(test)]
mod test {
    use crate::{
        kernel::rms_norm::{rms_norm_cpu, rms_norm_gpu},
        test_utils::{assert_close, gpu_context, random_f32},
    };

    #[test]
    fn test_rms_norm() {
        let x: Vec<f32> = vec![1.0, 2.0];
        let eps = 1.5;
        let gamma: Vec<f32> = vec![3.0, 3.0];
        let d_model = 2;
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let (device, queue) = gpu_context();
        let gpu = rms_norm_gpu(&device, &queue, &x, &gamma, eps, d_model as u32);

        // rms = √(sum(x^2)/d + ε)
        // √((1.0*1.0+2.0*2.0)/2 + 1.5) = 2.0
        // cpu = x / rms * γ
        // [1.0 / 2.0 * 3.0, 2.0 / 2.0 * 3.0]
        // = [1.5, 3.0]
        let exp: Vec<f32> = vec![1.5, 3.0];

        assert_eq!(cpu[0], exp[0]);
        assert_eq!(cpu[1], exp[1]);
        assert_eq!(gpu[0], exp[0]);
        assert_eq!(gpu[1], exp[1]);
    }

    #[test]
    fn test_rms_norm_random() {
        let seq = 4;
        let d_model = 64;
        let len = seq * d_model;
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let x = random_f32(len, 42);
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let (device, queue) = gpu_context();
        let gpu = rms_norm_gpu(&device, &queue, &x, &gamma, eps, d_model as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
