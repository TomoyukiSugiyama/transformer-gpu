use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
}

impl DType {
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::U32 => std::mem::size_of::<u32>() as u64,
        }
    }
}

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub len: usize,
    pub dtype: DType,
    pub label: Option<String>,
}

impl GpuTensor {
    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    pub fn byte_len(&self) -> u64 {
        self.len as u64 * self.dtype.byte_size()
    }

    pub fn new_f32(
        device: &wgpu::Device,
        shape: impl Into<Vec<usize>>,
        usage: wgpu::BufferUsages,
        label: impl Into<Option<String>>,
    ) -> Self {
        let shape = shape.into();
        let len = Self::numel(&shape);
        let label = label.into();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.as_deref(),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            shape,
            len,
            dtype: DType::F32,
            label,
        }
    }

    pub fn from_f32(
        device: &wgpu::Device,
        values: &[f32],
        shape: impl Into<Vec<usize>>,
        usage: wgpu::BufferUsages,
        label: impl Into<Option<String>>,
    ) -> Self {
        let shape = shape.into();
        let len = Self::numel(&shape);

        assert_eq!(
            values.len(),
            len,
            "GpuTensor input size does not match shape"
        );

        let label = label.into();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.as_deref(),
            contents: bytemuck::cast_slice(values),
            usage,
        });

        Self {
            buffer,
            shape,
            len,
            dtype: DType::F32,
            label,
        }
    }

    pub fn write_f32(&self, queue: &wgpu::Queue, values: &[f32]) {
        assert_eq!(values.len(), self.len);
        assert_eq!(self.dtype, DType::F32);

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(values));
    }

    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: NonZeroU64::new(self.byte_len()),
        })
    }
}

// TODO: GPUTensor へ移行が完了するまでの仮実装
pub fn read_f32_tensor(ctx: &GpuContext, tensor: &GpuTensor) -> Vec<f32> {
    let byte_len = tensor.byte_len();

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_f32_tensor_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_f32_tensor_encoder"),
        });

    encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, byte_len);

    ctx.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);

    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");

    let mapped = slice.get_mapped_range();

    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();

    drop(mapped);
    staging.unmap();

    assert_eq!(values.len(), tensor.len);

    values
}
