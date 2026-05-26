use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

pub fn create_table(d_head: usize, max_len: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = d_head / 2;
    let mut cos_table: Vec<f32> = vec![0.0; max_len * half];
    let mut sin_table: Vec<f32> = vec![0.0; max_len * half];
    for pos in 0..max_len {
        for i in 0..half {
            let theta = (base as f64).powf(-2.0 * i as f64 / d_head as f64);
            let angle = pos as f64 * theta;
            cos_table[pos * half + i] = angle.cos() as f32;
            sin_table[pos * half + i] = angle.sin() as f32;
        }
    }
    (cos_table, sin_table)
}

pub fn rope(
    ctx: &GpuContext,
    x: &[f32],
    d_head: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    assert!(d_head % 2 == 0, "d_head must be even for RoPE");
    assert_eq!(x.len() % d_head, 0, "x.len() must be divisible by d_head");
    let half = d_head / 2;
    assert_eq!(
        cos_table.len() % half,
        0,
        "cos_table.len() must be divisible by d_head/2"
    );
    assert_eq!(
        sin_table.len() % half,
        0,
        "sin_table.len() must be divisible by d_head/2"
    );
    assert_eq!(
        cos_table.len(),
        sin_table.len(),
        "cos_table and sin_table must have the same length"
    );
    let size = x.len() as u32;
    let byte_size = (size * 4) as u64;
    let buf_x = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_cos_table = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cos_table"),
            contents: bytemuck::cast_slice(&cos_table),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_sin_table = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sin_table"),
            contents: bytemuck::cast_slice(&sin_table),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [d_head as u32, 0, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/rope.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope"),
            layout: None,
            module: &module,
            entry_point: Some("rope"),
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
                resource: buf_cos_table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_sin_table.as_entire_binding(),
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

        let num_pos = (x.len() / d_head) as u32;
        pass.dispatch_workgroups(num_pos, 1, 1);
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
pub fn rope_cpu(x: &[f32], d_head: usize, cos_table: &[f32], sin_table: &[f32]) -> Vec<f32> {
    assert!(d_head % 2 == 0, "d_head must be even for RoPE");
    assert_eq!(x.len() % d_head, 0, "x.len() must be divisible by d_head");
    let half = d_head / 2;
    assert_eq!(
        cos_table.len() % half,
        0,
        "cos_table.len() must be divisible by d_head/2"
    );
    assert_eq!(
        sin_table.len() % half,
        0,
        "sin_table.len() must be divisible by d_head/2"
    );
    assert_eq!(
        cos_table.len(),
        sin_table.len(),
        "cos_table and sin_table must have the same length"
    );
    let half = d_head / 2;
    let seq = x.len() / d_head;

    let mut out = vec![0.0; seq * d_head];
    for pos in 0..seq {
        for i in 0..half {
            let c = cos_table[pos * half + i];
            let s = sin_table[pos * half + i];
            let pos_e = pos * d_head + 2 * i;
            let pos_o = pos * d_head + 2 * i + 1;

            let x0 = x[pos_e];
            let x1 = x[pos_o];
            out[pos_e] = c * x0 - s * x1;
            out[pos_o] = s * x0 + c * x1;
        }
    }

    out
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::rope::{create_table, rope, rope_cpu},
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_position_zero() {
        // 位置 0 では cos=1, sin=0 で回転なし → 入力そのまま
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let d_head = 4;
        let max_len = 8;
        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(d_head, max_len, base);
        let cpu = rope_cpu(&x, d_head, &cos_table, &sin_table);
        let ctx = GpuContext::new();
        let gpu = rope(&ctx, &x, d_head, &cos_table, &sin_table);

        cpu.iter()
            .zip(x.iter())
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
            .zip(x.iter())
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
    fn test_rope() {
        let x: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let d_head = 4;
        let max_len = 8;
        let base: f32 = 10000.0;

        // x
        // | 0.0 0.0 0.0 0.0 |
        // | 1.0 2.0 3.0 4.0 |
        // pe(pos,2i) = sin(pos / 10000^(2*i/d_head))
        // pe(pos,2i+1) = cos(pos / 10000^(2*i/d_head))
        // pos / 10000^(2*i/d_head)
        // | 1/10000^(2*0/4) 1/10000^(2*0/4) 1/10000^(2*1/4) 1/10000^(2*1/4) |
        // =
        // | 1 1 1/100 1/100 |
        // pe(pos,2) = sin(1 rad), pe(pos,3) = cos(1 rad)
        // pe(pos,4) = sin(0.01 rad), pe(pos,5) = cos(0.01 rad)
        // out(4) = cos(1 rad) * 1.0 - sin(1 rad) * 2.0 = -1.14263
        // out(5) = sin(1 rad) * 1.0 + cos(1 rad) * 2.0 = 1.92207
        // out(6) = cos(0.01 rad) * 3.0 - sin(0.01 rad) * 4.0 = 2.95985
        // out(7) = sin(0.01 rad) * 3.0 + cos(0.01 rad) * 4.0 = 4.02979

        let exp = vec![0.0, 0.0, 0.0, 0.0, -1.14263, 1.92207, 2.95985, 4.02979];
        let (cos_table, sin_table) = create_table(d_head, max_len, base);
        let cpu = rope_cpu(&x, d_head, &cos_table, &sin_table);
        let ctx = GpuContext::new();
        let gpu = rope(&ctx, &x, d_head, &cos_table, &sin_table);

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
    fn test_rope_random() {
        let seq: usize = 4;
        let d_head = 64;
        let max_len = 128;
        let base: f32 = 10000.0;
        let x: Vec<f32> = random_f32(seq * d_head, 33);
        let (cos_table, sin_table) = create_table(d_head, max_len, base);
        let cpu = rope_cpu(&x, d_head, &cos_table, &sin_table);
        let ctx = GpuContext::new();
        let gpu = rope(&ctx, &x, d_head, &cos_table, &sin_table);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
