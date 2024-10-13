//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::texture_storage_2d, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
    sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle},
};
use binding_types::uniform_buffer;
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const ADVECTION_SHADER_ASSET_PATH: &str = "shaders/advection.wgsl";
const JACOBI_SHADER_ASSET_PATH: &str = "shaders/jacobi.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (
                            (SIZE.0 * DISPLAY_FACTOR) as f32,
                            (SIZE.1 * DISPLAY_FACTOR) as f32,
                        )
                            .into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            GameOfLifeComputePlugin,
            Material2dPlugin::<CustomMaterial>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(image.clone());
    let image1 = images.add(image);

    let mut velocity_image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rg32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    velocity_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let velocity0 = images.add(velocity_image.clone());
    let velocity1 = images.add(velocity_image);

    commands.spawn(MaterialMesh2dBundle {
        // material: materials.add(Color::WHITE),
        mesh: Mesh2dHandle(meshes.add(Rectangle::new(SIZE.0 as f32, SIZE.1 as f32))),
        material: materials.add(CustomMaterial {
            color: LinearRgba::RED,
            color_texture: velocity0.clone(),
        }),
        ..Default::default()
    });

    // commands.spawn(SpriteBundle {
    //     sprite: Sprite {
    //         custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
    //         ..default()
    //     },
    //     texture: velocity0.clone(),
    //     transform: Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    //     ..default()
    // });
    commands.spawn(Camera2dBundle::default());

    commands.insert_resource(GameOfLifeImages {
        texture_a: image0,
        texture_b: image1,
    });
    commands.insert_resource(FluidSimulationParameters {
        time_step: 0.01,
        grid_step: 0.01,
        viscosity: 100.0,
    });
    commands.insert_resource(FluidSimulationImages {
        velocity_a: velocity0,
        velocity_b: velocity1,
    });
}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<FluidSimulationImages>, mut displayed: Query<&mut Handle<Image>>) {
    // let mut displayed = displayed.single_mut();
    // if *displayed == images.velocity_a {
    //     *displayed = images.velocity_b.clone_weak();
    // } else {
    //     *displayed = images.velocity_a.clone_weak();
    // }
}

struct GameOfLifeComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GameOfLifeLabel;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImages>::default());
        app.add_plugins(ExtractResourcePlugin::<FluidSimulationImages>::default());
        app.add_plugins(ExtractResourcePlugin::<FluidSimulationParameters>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );
        render_app.add_systems(Render, prepare_uniforms.in_set(RenderSet::Prepare));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GameOfLifeLabel, GameOfLifeNode::default());
        render_graph.add_node_edge(GameOfLifeLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<FluidSimulationPipeline>();
        render_app.init_resource::<FluidSimulationUniforms>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct GameOfLifeImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimulationParameters {
    time_step: f32,
    grid_step: f32,
    viscosity: f32,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct JacobiUniform {
    pub alpha: f32,
    pub r_beta: f32,
}

#[derive(Resource)]
struct FluidSimulationUniforms {
    advection: UniformBuffer<AdvectionUniform>,
    pressure: UniformBuffer<JacobiUniform>,
    diffusion: UniformBuffer<JacobiUniform>,
}

impl FromWorld for FluidSimulationUniforms {
    fn from_world(world: &mut World) -> Self {
        let mut advection = UniformBuffer::default();
        advection.set_label(Some("advection_uniforms_buffer"));

        let mut pressure = UniformBuffer::default();
        advection.set_label(Some("pressure_uniforms_buffer"));

        let mut diffusion = UniformBuffer::default();
        advection.set_label(Some("diffusion_uniforms_buffer"));

        advection.add_usages(BufferUsages::UNIFORM);
        pressure.add_usages(BufferUsages::UNIFORM);
        diffusion.add_usages(BufferUsages::UNIFORM);

        Self {
            advection,
            pressure,
            diffusion,
        }
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimulationImages {
    velocity_a: Handle<Image>,
    velocity_b: Handle<Image>,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct AdvectionUniform {
    pub time_step: f32,
}

#[derive(Resource)]
struct FluidSimulationBindGroups {
    advection_uniform: BindGroup,
    advection_image: [BindGroup; 2],

    pressure_uniform: BindGroup,
    pressure_image: [BindGroup; 2],

    diffusion_uniform: BindGroup,
    diffusion_image: [BindGroup; 2],
}

fn prepare_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut fluid_simulation_uniforms: ResMut<FluidSimulationUniforms>,
    parameters: Res<FluidSimulationParameters>,
) {
    fluid_simulation_uniforms.advection.set(AdvectionUniform {
        time_step: parameters.time_step,
    });

    let dx = parameters.grid_step;
    let dt = parameters.time_step;
    let pressure_alpha = -dx * dx;
    let diffuse_alpha = dx * dx / (dt * parameters.viscosity);
    let diffuse_beta = 4.0 + diffuse_alpha;
    fluid_simulation_uniforms.pressure.set(JacobiUniform {
        alpha: pressure_alpha,
        r_beta: 0.25,
    });
    fluid_simulation_uniforms.diffusion.set(JacobiUniform {
        alpha: diffuse_alpha,
        r_beta: 1.0 / diffuse_beta,
    });

    fluid_simulation_uniforms
        .advection
        .write_buffer(&render_device, &render_queue);
    fluid_simulation_uniforms
        .pressure
        .write_buffer(&render_device, &render_queue);
    fluid_simulation_uniforms
        .diffusion
        .write_buffer(&render_device, &render_queue);
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<FluidSimulationPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    uniforms: Res<FluidSimulationUniforms>,
    game_of_life_images: Res<GameOfLifeImages>,
    fluid_simulation_images: Res<FluidSimulationImages>,
    render_device: Res<RenderDevice>,
) {
    // Uniform Bing Groups
    let advection_uniform_bind_group = render_device.create_bind_group(
        None,
        &pipeline.advection_uniform_bind_group_layout,
        &BindGroupEntries::single(uniforms.advection.into_binding()),
    );
    let pressure_uniform_bind_group = render_device.create_bind_group(
        None,
        &pipeline.jacobi_uniform_bind_group_layout,
        &BindGroupEntries::single(uniforms.pressure.into_binding()),
    );

    let diffusion_uniform_bind_group = render_device.create_bind_group(
        None,
        &pipeline.jacobi_uniform_bind_group_layout,
        &BindGroupEntries::single(uniforms.diffusion.into_binding()),
    );

    // Advection
    let view_buffer_a = gpu_images.get(&game_of_life_images.texture_a).unwrap();
    let view_buffer_b = gpu_images.get(&game_of_life_images.texture_b).unwrap();
    let velocity_buffer_a = gpu_images.get(&fluid_simulation_images.velocity_a).unwrap();
    let velocity_buffer_b = gpu_images.get(&fluid_simulation_images.velocity_b).unwrap();
    let advection_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.advection_image_group_layout,
            &BindGroupEntries::sequential((
                &view_buffer_a.texture_view,
                &view_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.advection_image_group_layout,
            &BindGroupEntries::sequential((
                &view_buffer_b.texture_view,
                &view_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
            )),
        ),
    ];

    // Pressure
    let pressure_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_a.texture_view,
                &velocity_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_b.texture_view,
                &velocity_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
            )),
        ),
    ];

    // Diffusion
    let diffusion_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_a.texture_view,
                &velocity_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_b.texture_view,
                &velocity_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
            )),
        ),
    ];
    commands.insert_resource(FluidSimulationBindGroups {
        advection_uniform: advection_uniform_bind_group,
        advection_image: advection_image_bind_groups,
        pressure_uniform: pressure_uniform_bind_group,
        pressure_image: pressure_image_bind_groups,
        diffusion_uniform: diffusion_uniform_bind_group,
        diffusion_image: diffusion_image_bind_groups,
    })
}

#[derive(Resource)]
struct FluidSimulationPipeline {
    advection_uniform_bind_group_layout: BindGroupLayout,
    advection_image_group_layout: BindGroupLayout,
    jacobi_uniform_bind_group_layout: BindGroupLayout,
    jacobi_image_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    advection_pipeline: CachedComputePipelineId,
    pressure_pipeline: CachedComputePipelineId,
    diffusion_pipeline: CachedComputePipelineId,
}

impl FromWorld for FluidSimulationPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let advection_uniform_bind_group_layout = render_device.create_bind_group_layout(
            "AdvectionUniformLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<AdvectionUniform>(false),),
            ),
        );
        let jacobi_uniform_bind_group_layout = render_device.create_bind_group_layout(
            "JacobiUniformLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<JacobiUniform>(false),),
            ),
        );

        let advection_image_group_layout = render_device.create_bind_group_layout(
            "AdvectionImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let jacobi_image_bind_group_layout = render_device.create_bind_group_layout(
            "JacobiImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let advection_shader = world.load_asset(ADVECTION_SHADER_ASSET_PATH);
        let jacobi_shader = world.load_asset(JACOBI_SHADER_ASSET_PATH);

        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                advection_uniform_bind_group_layout.clone(),
                advection_image_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: advection_shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let advection_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                advection_uniform_bind_group_layout.clone(),
                advection_image_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: advection_shader,
            shader_defs: vec![],
            entry_point: Cow::from("advect"),
        });
        let pressure_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                jacobi_uniform_bind_group_layout.clone(),
                jacobi_image_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: jacobi_shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("jacobi"),
        });
        let diffusion_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                jacobi_uniform_bind_group_layout.clone(),
                jacobi_image_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: jacobi_shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("jacobi"),
        });
        FluidSimulationPipeline {
            advection_uniform_bind_group_layout,
            advection_image_group_layout,
            jacobi_image_bind_group_layout,
            jacobi_uniform_bind_group_layout,
            init_pipeline,
            advection_pipeline,
            diffusion_pipeline,
            pressure_pipeline,
        }
    }
}

enum GameOfLifeState {
    Loading,
    Init,
    Update(usize),
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<FluidSimulationPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = GameOfLifeState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{ADVECTION_SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.advection_pipeline)
                {
                    self.state = GameOfLifeState::Update(1);
                }
            }
            GameOfLifeState::Update(0) => {
                self.state = GameOfLifeState::Update(1);
            }
            GameOfLifeState::Update(1) => {
                self.state = GameOfLifeState::Update(0);
            }
            GameOfLifeState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<FluidSimulationBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FluidSimulationPipeline>();

        let encoder = render_context.command_encoder();
        let mut pass: ComputePass<'_> =
            encoder.begin_compute_pass(&ComputePassDescriptor::default());
        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups.advection_uniform, &[]);
                pass.set_bind_group(1, &bind_groups.advection_image[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update(index) => {
                let advection_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.advection_pipeline)
                    .unwrap();
                pass.set_pipeline(advection_pipeline);

                pass.set_bind_group(0, &bind_groups.advection_uniform, &[]);

                pass.set_bind_group(1, &bind_groups.advection_image[1], &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                pass.set_bind_group(1, &bind_groups.advection_image[0], &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                let diffusion_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.diffusion_pipeline)
                    .unwrap();
                pass.set_pipeline(diffusion_pipeline);

                pass.set_bind_group(0, &bind_groups.diffusion_uniform, &[]);

                for _ in 0..20 {
                    pass.set_bind_group(1, &bind_groups.diffusion_image[1], &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                    pass.set_bind_group(1, &bind_groups.diffusion_image[0], &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }
            }
        }

        Ok(())
    }
}

#[derive(AsBindGroup, Debug, Clone, Asset, TypePath)]
pub struct CustomMaterial {
    // Uniform bindings must implement `ShaderType`, which will be used to convert the value to
    // its shader-compatible equivalent. Most core math types already implement `ShaderType`.
    #[uniform(0)]
    color: LinearRgba,
    // Images can be bound as textures in shaders. If the Image's sampler is also needed, just
    // add the sampler attribute with a different binding index.
    #[texture(1)]
    #[sampler(2)]
    color_texture: Handle<Image>,
}

// All functions on `Material2d` have default impls. You only need to implement the
// functions that are relevant for your material.
impl Material2d for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/custom_material.wgsl".into()
    }
}
