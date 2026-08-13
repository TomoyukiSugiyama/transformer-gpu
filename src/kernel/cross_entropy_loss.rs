use crate::gpu_context::GpuContext;
use wgpu::util::DeviceExt;

pub fn cross_entropy_loss(
    ctx: &GpuContext,
    logits: &[f32],    // (seq * vocab_size)
    targets: &[usize], // (seq,)
    seq: usize,
    vocab_size: usize,
) -> (f32, Vec<f32>) {
    let grad_size = (seq * vocab_size) as u32;
    let grad_byte_size = (grad_size * 4) as u64;
    let losses_byte_size = (seq * 4) as u64;
    let buf_logits = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x1"),
            contents: bytemuck::cast_slice(&logits),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
    let buf_targets = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("targets"),
            contents: bytemuck::cast_slice(&targets_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_losses = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("losses"),
        size: losses_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_grad = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad"),
        size: grad_byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [seq as u32, vocab_size as u32, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/cross_entropy_loss.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cross_entropy_loss"),
            layout: None,
            module: &module,
            entry_point: Some("cross_entropy_loss"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_logits.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_targets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_losses.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_grad.as_entire_binding(),
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

        pass.dispatch_workgroups(seq as u32, 1, 1);
    }

    let grad_buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: grad_byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let losses_buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: losses_byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_grad, 0, &grad_buf_read, 0, grad_byte_size);
    encoder.copy_buffer_to_buffer(&buf_losses, 0, &losses_buf_read, 0, losses_byte_size);

    ctx.queue.submit([encoder.finish()]);

    let grad_slice = grad_buf_read.slice(..);
    grad_slice.map_async(wgpu::MapMode::Read, |_| {});

    let losses_slice = losses_buf_read.slice(..);
    losses_slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let grad_data = grad_slice.get_mapped_range();
    let grad = bytemuck::allocation::pod_collect_to_vec(&grad_data);

    let losses_data = losses_slice.get_mapped_range();
    let losses: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&losses_data);

    let loss = losses.iter().sum::<f32>() / seq as f32;

    (loss, grad)
}

pub fn cross_entropy_loss_cpu(
    logits: &[f32],    // (seq * vocab_size)
    targets: &[usize], // (seq,)
    seq: usize,
    vocab_size: usize,
) -> (f32, Vec<f32>) {
    assert_eq!(logits.len(), seq * vocab_size);
    assert_eq!(targets.len(), seq);

    let mut grad_data = vec![0.0f32; seq * vocab_size];

    let total_loss: f32 = grad_data
        .chunks_mut(vocab_size)
        .enumerate()
        .map(|(t, g_row)| {
            let logits_row = &logits[t * vocab_size..(t + 1) * vocab_size];
            let (loss, grad) = cross_entropy_loss_row_cpu(logits_row, targets[t]);
            for j in 0..vocab_size {
                g_row[j] = grad[j];
            }
            loss
        })
        .sum::<f32>()
        / seq as f32;

    (total_loss, grad_data)
}

pub fn cross_entropy_loss_row_cpu(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Loss
    let loss = -(probs[target] + 1e-10).ln();

    // 勾配: dL/d_logit_i = probs[i] - 1(i == target)
    let grad: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| if i == target { p - 1.0 } else { p })
        .collect();

    (loss, grad)
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::cross_entropy_loss::{cross_entropy_loss, cross_entropy_loss_cpu},
        test_utils::assert_close,
    };

    #[test]
    fn test_cross_entropy_known_value() {
        let vocab_size = 3;
        let targets = vec![1];
        let logits = vec![1.0, 2.0, 0.0];
        let seq = 1;

        let (cpu_loss, cpu_grad) = cross_entropy_loss_cpu(&logits, &targets, seq, vocab_size);
        let ctx = GpuContext::new();
        let (gpu_loss, gpu_grad) = cross_entropy_loss(&ctx, &logits, &targets, seq, vocab_size);

        // vocab_size=3, targets=[1], logits=[1.0, 2.0, 0.0]
        // max=2.0, exps=[e^-1, 1, e^-2]
        // probs[1] = 1 / (e^-1 + 1 + e^-2) = 1/(0.36787 + 1 + 0.13533) = 0.66524
        // loss = - ln(0.66524 + 1e-10) = 0.40760
        // grad[0] = probs[0] = 0.36787 / (0.36787 + 1 + 0.13533) = 0.24472
        // grad[2] = probs[2] = 0.13533 / (0.36787 + 1 + 0.13533) = 0.09002
        // grad[1] = 0.66524 - 1 = -0.33476
        let exp_loss = 0.40760;
        let exp_grad = vec![0.24472, -0.33476, 0.09002];

        assert!(
            (cpu_loss - exp_loss).abs() < 1e-4,
            "cpu_loss={:.6}, exp_loss={:.6}",
            cpu_loss,
            exp_loss
        );
        cpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "cpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert!(
            (gpu_loss - exp_loss).abs() < 1e-4,
            "gpu_loss={:.6}, exp_loss={:.6}",
            gpu_loss,
            exp_loss
        );
        gpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "gpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert_close(&gpu_grad, &cpu_grad, 1e-4, 1e-5);
        assert!((gpu_loss - cpu_loss).abs() < 1e-4);
    }

    #[test]
    fn test_target_boundary() {
        let vocab_size = 3;
        let targets = vec![2];
        let logits = vec![1.0, 0.0, 2.0];
        let seq = 1;

        let (cpu_loss, cpu_grad) = cross_entropy_loss_cpu(&logits, &targets, seq, vocab_size);
        let ctx = GpuContext::new();
        let (gpu_loss, gpu_grad) = cross_entropy_loss(&ctx, &logits, &targets, seq, vocab_size);

        let exp_loss = 0.40760;
        let exp_grad = vec![0.24472, 0.09002, -0.33476];

        assert!(
            (cpu_loss - exp_loss).abs() < 1e-4,
            "cpu_loss={:.6}, exp_loss={:.6}",
            cpu_loss,
            exp_loss
        );
        cpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "cpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert!(
            (gpu_loss - exp_loss).abs() < 1e-4,
            "gpu_loss={:.6}, exp_loss={:.6}",
            gpu_loss,
            exp_loss
        );
        gpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "gpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert_close(&gpu_grad, &cpu_grad, 1e-4, 1e-5);
        assert!((gpu_loss - cpu_loss).abs() < 1e-4);
    }

    #[test]
    fn test_loss_is_mean_over_seq() {
        let vocab_size = 3;
        let targets = vec![1, 1];
        let logits = vec![1.0, 2.0, 0.0, 1.0, 2.0, 0.0];
        let seq = 2;

        let (cpu_loss, cpu_grad) = cross_entropy_loss_cpu(&logits, &targets, seq, vocab_size);
        let ctx = GpuContext::new();
        let (gpu_loss, gpu_grad) = cross_entropy_loss(&ctx, &logits, &targets, seq, vocab_size);

        let exp_loss = 0.4076;
        let exp_grad = vec![0.24472, -0.33476, 0.09002, 0.24472, -0.33476, 0.09002];

        assert!(
            (cpu_loss - exp_loss).abs() < 1e-4,
            "cpu_loss={:.6}, exp_loss={:.6}",
            cpu_loss,
            exp_loss
        );
        cpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "cpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert!(
            (gpu_loss - exp_loss).abs() < 1e-4,
            "gpu_loss={:.6}, exp_loss={:.6}",
            gpu_loss,
            exp_loss
        );
        gpu_grad
            .iter()
            .zip(exp_grad.iter())
            .enumerate()
            .for_each(|(i, (g, e))| {
                assert!(
                    (g - e).abs() < 1e-4,
                    "gpu_grad[{i}]: grad={:.6}, exp={:.6}",
                    g,
                    e
                )
            });

        assert_close(&gpu_grad, &cpu_grad, 1e-4, 1e-5);
        assert!((gpu_loss - cpu_loss).abs() < 1e-4);
    }
}
