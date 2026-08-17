pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    // pub matmul_pipeline: wgpu::ComputePipeline,
    // pub rms_norm_pipeline: wgpu::ComputePipeline,
    // pub rope_pipeline: wgpu::ComputePipeline,
    // pub attention_pipeline: wgpu::ComputePipeline,
    // pub swiglu_pipeline: wgpu::ComputePipeline,
    // pub adamw_pipeline: wgpu::ComputePipeline,

    // pub dims_buffer: wgpu::Buffer,
}

impl GpuContext {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .unwrap();
        Self { device, queue }
    }
}
