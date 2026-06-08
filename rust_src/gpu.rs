#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::mem::size_of;
#[cfg(feature = "gpu")]
use std::sync::mpsc;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
const DABC_WGSL: &str = include_str!("dabc_kernel.wgsl");

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub width: u32,
    pub height: u32,
    pub pool_len: u32,
    pub colony_size: u32,
    pub payload_len: u32,
}

#[cfg(feature = "gpu")]
pub struct GpuDabcBuffers {
    pub params: GpuParams,
    pub image_buffer: wgpu::Buffer,
    pub pool_y_buffer: wgpu::Buffer,
    pub pool_x_buffer: wgpu::Buffer,
    pub foods_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub score_table_buffer: wgpu::Buffer,
    pub fitness_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

#[cfg(feature = "gpu")]
pub struct GpuDabcContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub score_pipeline: wgpu::ComputePipeline,
    pub fitness_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuDabcContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| "No compatible GPU adapter found".to_string())?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|err| format!("Failed to request device: {err}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dabc_kernel.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DABC_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dabc_bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dabc_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dabc_score_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "compute_score_table",
            compilation_options: Default::default(),
        });

        let fitness_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dabc_fitness_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "compute_fitness",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            score_pipeline,
            fitness_pipeline,
            bind_group_layout,
        })
    }

    pub fn create_buffers(
        &self,
        image_u32: &[u32],
        pool_y: &[u32],
        pool_x: &[u32],
        foods: &[u32],
        params: GpuParams,
    ) -> GpuDabcBuffers {
        let image_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dabc_image_buffer"),
            contents: bytemuck::cast_slice(image_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pool_y_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dabc_pool_y_buffer"),
            contents: bytemuck::cast_slice(pool_y),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pool_x_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dabc_pool_x_buffer"),
            contents: bytemuck::cast_slice(pool_x),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let foods_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dabc_foods_buffer"),
            contents: bytemuck::cast_slice(foods),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dabc_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let score_table_size = (params.pool_len as usize * size_of::<f32>()) as u64;
        let fitness_size = (params.colony_size as usize * size_of::<f32>()) as u64;

        let score_table_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dabc_score_table_buffer"),
            size: score_table_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fitness_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dabc_fitness_buffer"),
            size: fitness_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dabc_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pool_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: foods_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: score_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: fitness_buffer.as_entire_binding(),
                },
            ],
        });

        GpuDabcBuffers {
            params,
            image_buffer,
            pool_y_buffer,
            pool_x_buffer,
            foods_buffer,
            params_buffer,
            score_table_buffer,
            fitness_buffer,
            bind_group,
        }
    }

    pub fn dispatch_score_and_fitness(&self, buffers: &GpuDabcBuffers) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dabc_compute_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dabc_score_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            let wg = (buffers.params.pool_len + 255) / 256;
            pass.dispatch_workgroups(wg, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dabc_fitness_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fitness_pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(buffers.params.colony_size, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn readback_score_and_fitness(
        &self,
        buffers: &GpuDabcBuffers,
    ) -> Result<(Vec<f32>, Vec<f32>), wgpu::BufferAsyncError> {
        let score_size = (buffers.params.pool_len as usize * size_of::<f32>()) as u64;
        let fitness_size = (buffers.params.colony_size as usize * size_of::<f32>()) as u64;

        let score_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dabc_score_readback"),
            size: score_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fitness_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dabc_fitness_readback"),
            size: fitness_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dabc_readback_encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &buffers.score_table_buffer,
            0,
            &score_readback,
            0,
            score_size,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.fitness_buffer,
            0,
            &fitness_readback,
            0,
            fitness_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let (score_tx, score_rx) = mpsc::channel();
        score_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = score_tx.send(res);
            });

        let (fitness_tx, fitness_rx) = mpsc::channel();
        fitness_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = fitness_tx.send(res);
            });

        self.device.poll(wgpu::Maintain::Wait);

        score_rx.recv().unwrap()?;
        fitness_rx.recv().unwrap()?;

        let score_view = score_readback.slice(..).get_mapped_range();
        let fitness_view = fitness_readback.slice(..).get_mapped_range();

        let scores: Vec<f32> = bytemuck::cast_slice(&score_view).to_vec();
        let fitness: Vec<f32> = bytemuck::cast_slice(&fitness_view).to_vec();

        drop(score_view);
        drop(fitness_view);
        score_readback.unmap();
        fitness_readback.unmap();

        Ok((scores, fitness))
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuDabcContext;
