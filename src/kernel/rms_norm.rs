use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

pub fn rms_norm(ctx: &GpuContext, x: &[f32], gamma: &[f32], eps: f32, d_model: u32) -> Vec<f32> {
    assert_eq!(
        x.len() as u32 % d_model,
        0,
        "x.len() must be divisible by d_model"
    );
    let seq = x.len() as u32 / d_model;
    let byte_size = (seq * d_model * 4) as u64;
    let buf_x = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_gamma = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gamma"),
            contents: bytemuck::cast_slice(&gamma),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [d_model as u32, eps.to_bits(), 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/rms_norm.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm"),
            layout: None,
            module: &module,
            entry_point: Some("rms_norm"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        pass.dispatch_workgroups(seq, 1, 1);
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

pub fn rms_norm_backward(
    ctx: &GpuContext,
    dy: &[f32],    // (seq × d_model)
    x: &[f32],     // (seq × d_model)  forward の入力
    gamma: &[f32], // (d_model,)
    eps: f32,
    d_model: u32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        x.len() as u32 % d_model,
        0,
        "x.len() must be divisible by d_model"
    );
    let seq = x.len() as u32 / d_model;
    let byte_size = (seq * d_model * 4) as u64;
    let buf_dy = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dy"),
            contents: bytemuck::cast_slice(&dy),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_x = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_gamma = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gamma"),
            contents: bytemuck::cast_slice(&gamma),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_dx = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dx"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // let buf_dgamma_partial = ctx.device.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("dgamma_partial"),
    //     size: byte_size,
    //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    //     mapped_at_creation: false,
    // });

    let dims_padded: [u32; 4] = [d_model as u32, eps.to_bits(), 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/rms_norm_backward.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm_backward"),
            layout: None,
            module: &module,
            entry_point: Some("rms_norm_backward"),
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
                resource: buf_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_gamma.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dx.as_entire_binding(),
            },
            // wgpu::BindGroupEntry {
            //     binding: 4,
            //     resource: buf_dgamma_partial.as_entire_binding(),
            // },
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

        pass.dispatch_workgroups(seq, 1, 1);
    }

    let dx_buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    // let dgamma_partial_buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
    //     label: None,
    //     size: byte_size,
    //     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    //     mapped_at_creation: false,
    // });
    encoder.copy_buffer_to_buffer(&buf_dx, 0, &dx_buf_read, 0, byte_size);
    // encoder.copy_buffer_to_buffer(
    //     &buf_dgamma_partial,
    //     0,
    //     &dgamma_partial_buf_read,
    //     0,
    //     byte_size,
    // );

    ctx.queue.submit([encoder.finish()]);

    let dx_slice = dx_buf_read.slice(..);
    dx_slice.map_async(wgpu::MapMode::Read, |_| {});
    // let dgamma_partial_slice = dgamma_partial_buf_read.slice(..);
    // dgamma_partial_slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let dx_data = dx_slice.get_mapped_range();
    let dx = bytemuck::allocation::pod_collect_to_vec(&dx_data);
    // let dgamma_partial_data = dgamma_partial_slice.get_mapped_range();
    // let dgamma_partial: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&dgamma_partial_data);

    // let mut dgamma = vec![0.0; d_model as usize];

    // dgamma_partial.chunks(d_model as usize).for_each(|dg_row| {
    //     dg_row.iter().enumerate().for_each(|(i, dg_i)| {
    //         dgamma[i] += dg_i;
    //     });
    // });
    let mut dgamma = vec![0.0f32; d_model as usize];
    for row in 0..seq {
        let base = (row * d_model) as usize;
        let rms = (x[base..base + d_model as usize]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / d_model as f32
            + eps)
            .sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..d_model as usize {
            dgamma[i] += dy[base + i] * x[base + i] * inv_rms;
        }
    }

    (dx, dgamma)
}

// CPU リファレンス
#[cfg(test)]
pub fn rms_norm_cpu(x: &[f32], gamma: &[f32], eps: f32, d_model: usize) -> Vec<f32> {
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
pub fn rms_norm_backward_cpu(
    dy: &[f32],    // (seq × d_model)
    x: &[f32],     // (seq × d_model)  forward の入力
    gamma: &[f32], // (d_model,)
    eps: f32,
    d_model: usize,
) -> (Vec<f32>, Vec<f32>) {
    let seq = x.len() / d_model;

    let mut dx = vec![0.0f32; x.len()];
    let mut dgamma = vec![0.0f32; d_model];
    for row in 0..seq {
        let base = row * d_model;
        let row_x = &x[base..base + d_model];
        let row_dy = &dy[base..base + d_model];
        // rms = √(sum(x_j*x_j)/d+eps)
        let rms = (row_x.iter().map(|v| v * v).sum::<f32>() / d_model as f32 + eps).sqrt();

        // x_hat = x_i/rms
        let inv_rms = 1.0 / rms;
        let x_hat: Vec<f32> = row_x.iter().map(|v| v * inv_rms).collect();
        // dot = (sum_j(dy_j*gamma_j*x_hat)/d)
        let dot = row_dy
            .iter()
            .zip(gamma.iter())
            .zip(x_hat.iter())
            .map(|((dy_i, g_i), xh_i)| dy_i * g_i * xh_i)
            .sum::<f32>()
            / d_model as f32;

        for i in 0..d_model {
            // dx_i = gamma_i/rms*(dy_i-x_hat_i*dot)
            dx[base + i] = gamma[i] * inv_rms * (row_dy[i] - x_hat[i] * dot);
            // dgamma_i = sum_row(dy_i*x_hat_i)
            dgamma[i] += row_dy[i] * x_hat[i];
        }
    }

    (dx, dgamma)
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::rms_norm::{rms_norm, rms_norm_backward, rms_norm_backward_cpu, rms_norm_cpu},
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_rms_norm() {
        let x: Vec<f32> = vec![1.0, 2.0];
        let eps = 1.5;
        let gamma: Vec<f32> = vec![3.0, 3.0];
        let d_model = 2usize;
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let gpu = rms_norm(&ctx, &x, &gamma, eps, d_model as u32);

        // rms = √(sum(x^2)/d + ε)
        // √((1.0*1.0+2.0*2.0)/2 + 1.5) = 2.0
        // cpu = x / rms * γ
        // [1.0 / 2.0 * 3.0, 2.0 / 2.0 * 3.0]
        // = [1.5, 3.0]
        let exp: Vec<f32> = vec![1.5, 3.0];

        assert_eq!(cpu, exp);
        assert_eq!(gpu, exp);
    }

    #[test]
    fn test_rms_norm_backward() {
        let dy = vec![1.0, 2.0];
        let x: Vec<f32> = vec![1.0, 2.0];
        let eps = 1.5;
        let gamma: Vec<f32> = vec![3.0, 3.0];
        let d_model = 2usize;
        let (cpu_dx, cpu_dgamma) = rms_norm_backward_cpu(&dy, &x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let (gpu_dx, gpu_dgamma) = rms_norm_backward(&ctx, &dy, &x, &gamma, eps, d_model as u32);

        // rms = √(sum(x^2)/d + ε)
        // √((1.0*1.0+2.0*2.0)/2 + 1.5) = 2.0

        // x_hat = x_i/rms
        // [1.0/2.0, 2.0/2.0] = [0.5, 1.0]

        // dot = (sum_j(dy_j*gamma_j*x_hat)/d)
        // dot = (1.0*3.0*0.5 + 2.0*3.0*1.0)/2 = 3.75

        // dx_i = gamma_i/rms*(dy_i-x_hat_i*dot)
        // [3.0/2.0*(1.0-0.5*3.75), 3.0/2.0*(2.0-1.0*3.75)] = [-1.3125, -2.625]
        // dgamma_i = sum_row(dy_i*x_hat_i)
        // [1.0*0.5, 2.0*1.0] = [0.5, 2.0]

        let exp_dx: Vec<f32> = vec![-1.3125, -2.625];
        let exp_dgamma = vec![0.5, 2.0];
        assert_eq!(cpu_dx, exp_dx);
        assert_eq!(cpu_dgamma, exp_dgamma);
        assert_eq!(gpu_dx, exp_dx);
        assert_eq!(gpu_dgamma, exp_dgamma);
    }

    #[test]
    fn test_rms_norm_random() {
        let seq = 4usize;
        let d_model = 64usize;
        let len = seq * d_model;
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let scale = 0.1f32;
        let x = random_f32(len, 42, scale);
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let gpu = rms_norm(&ctx, &x, &gamma, eps, d_model as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_random() {
        let seq = 4usize;
        let d_model = 64usize;
        let len = seq * d_model;
        let scale = 0.1f32;
        let dy = random_f32(len, 41, scale);
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let x = random_f32(len, 42, scale);
        let (cpu_dx, cpu_dgamma) = rms_norm_backward_cpu(&dy, &x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let (gpu_dx, gpu_dgamma) = rms_norm_backward(&ctx, &dy, &x, &gamma, eps, d_model as u32);

        assert_close(&gpu_dx, &cpu_dx, 1e-4, 1e-5);
        assert_close(&gpu_dgamma, &cpu_dgamma, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_d512() {
        let seq = 4usize;
        let d_model = 512usize;
        let len = seq * d_model;
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let scale = 0.1f32;
        let x = random_f32(len, 42, scale);
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let gpu = rms_norm(&ctx, &x, &gamma, eps, d_model as u32);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_d128() {
        let seq = 128usize;
        let d_model = 128usize;
        let len = seq * d_model;
        let scale = 0.1f32;
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let x = random_f32(len, 42, scale);
        let cpu = rms_norm_cpu(&x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let gpu = rms_norm(&ctx, &x, &gamma, eps, d_model as u32);
        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_real_add1_out() {
        use crate::{
            kernel::rope::create_table,
            model::transformer_block::{TransformerBlock, TransformerBlockForwardCache},
            model_config::ModelConfig,
        };
        let cfg = ModelConfig::default();
        let ctx = GpuContext::new();
        let mut cache_gpu = TransformerBlockForwardCache::default();
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let seq = 128usize;
        let x = random_f32(seq * cfg.d_model, 31, 0.1);
        let (cos, sin) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);
        let tf = TransformerBlock::new(&cfg);
        let _ = tf.forward(&ctx, &cfg, &x, &cos, &sin, &mut cache_gpu);
        let _ = tf.forward_cpu(&cfg, &x, &cos, &sin, &mut cache_cpu);
        assert_close(&cache_gpu.add1_out, &cache_cpu.add1_out, 1e-4, 1e-5);
        let cache = cache_gpu;
        let dy = random_f32(seq * cfg.d_model, 41, 0.1);
        let (cpu_dx, _) =
            rms_norm_backward_cpu(&dy, &cache.add1_out, &tf.gamma_2, cfg.eps, cfg.d_model);
        let (gpu_dx, _) = rms_norm_backward(
            &ctx,
            &dy,
            &cache.add1_out,
            &tf.gamma_2,
            cfg.eps,
            cfg.d_model as u32,
        );
        assert_close(&gpu_dx, &cpu_dx, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_d128_seq4() {
        let seq = 4usize;
        let d_model = 128usize;
        let len = seq * d_model;
        let scale = 0.1f32;
        let dy = random_f32(len, 41, scale);
        let gamma: Vec<f32> = random_f32(d_model, 38, (1.0 / d_model as f32).sqrt());
        let eps = 1e-6;
        let x = random_f32(len, 42, scale);
        let (cpu_dx, _) = rms_norm_backward_cpu(&dy, &x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let (gpu_dx, _) = rms_norm_backward(&ctx, &dy, &x, &gamma, eps, d_model as u32);
        assert_close(&gpu_dx, &cpu_dx, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_d128() {
        let seq = 128usize;
        let d_model = 128usize;
        let len = seq * d_model;
        let scale = 0.1f32;
        let dy = random_f32(len, 41, scale);
        let gamma: Vec<f32> = random_f32(d_model, 38, (1.0 / d_model as f32).sqrt());
        let eps = 1e-6;
        let x = random_f32(len, 42, scale);
        let (cpu_dx, cpu_dgamma) = rms_norm_backward_cpu(&dy, &x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let (gpu_dx, gpu_dgamma) = rms_norm_backward(&ctx, &dy, &x, &gamma, eps, d_model as u32);
        assert_close(&gpu_dx, &cpu_dx, 1e-4, 1e-5);
        assert_close(&gpu_dgamma, &cpu_dgamma, 1e-4, 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_d512() {
        let seq = 4usize;
        let d_model = 512usize;
        let len = seq * d_model;
        let scale = 0.1f32;
        let dy = random_f32(len, 41, scale);
        let gamma: Vec<f32> = vec![1.0; d_model];
        let eps = 1e-6;
        let x = random_f32(len, 42, scale);
        let (cpu_dx, cpu_dgamma) = rms_norm_backward_cpu(&dy, &x, &gamma, eps, d_model);
        let ctx = GpuContext::new();
        let (gpu_dx, gpu_dgamma) = rms_norm_backward(&ctx, &dy, &x, &gamma, eps, d_model as u32);

        assert_close(&gpu_dx, &cpu_dx, 1e-4, 1e-5);
        assert_close(&gpu_dgamma, &cpu_dgamma, 1e-4, 1e-5);
    }
}
