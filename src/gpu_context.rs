pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub matmul_bind_group_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,
    // pub rms_norm_pipeline: wgpu::ComputePipeline,
    // pub rope_pipeline: wgpu::ComputePipeline,
    // pub attention_pipeline: wgpu::ComputePipeline,
    // pub swiglu_pipeline: wgpu::ComputePipeline,
    // pub adamw_pipeline: wgpu::ComputePipeline,
    pub cross_entropy_bind_group_layout: wgpu::BindGroupLayout,
    pub cross_entropy_pipeline: wgpu::ComputePipeline,
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

        let matmul_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let matmul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[Some(&matmul_bind_group_layout)],
                immediate_size: 0,
            });
        let matmul_shader = device.create_shader_module(wgpu::include_wgsl!("shader/matmul.wgsl"));
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("matmul"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cross_entropy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cross_entropy_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let cross_entropy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cross_entropy_pipeline_layout"),
                bind_group_layouts: &[Some(&cross_entropy_bind_group_layout)],
                immediate_size: 0,
            });
        let cross_entropy_shader =
            device.create_shader_module(wgpu::include_wgsl!("shader/cross_entropy_loss.wgsl"));
        let cross_entropy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cross_entropy_pipeline"),
                layout: Some(&cross_entropy_pipeline_layout),
                module: &cross_entropy_shader,
                entry_point: Some("cross_entropy_loss"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            matmul_bind_group_layout,
            matmul_pipeline,
            cross_entropy_bind_group_layout,
            cross_entropy_pipeline,
        }
    }
}
