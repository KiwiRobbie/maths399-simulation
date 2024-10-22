use bevy::{
    input::keyboard::KeyboardInput,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
    sprite::{Material2d, Material2dPlugin},
};

use image::RgbImage;
use std::ops::Rem;

// use bevy_egui::{egui, EguiContexts, EguiPlugin};
use binding_types::{texture_storage_2d, uniform_buffer};
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const ADVECTION_SHADER_ASSET_PATH: &str = "shaders/advection.wgsl";
const JACOBI_SHADER_ASSET_PATH: &str = "shaders/jacobi.wgsl";
const DIVERGENCE_SHADER_ASSET_PATH: &str = "shaders/divergence.wgsl";
const GRADIENT_SUBTRACTION_SHADER_ASSET_PATH: &str = "shaders/gradient_subtraction.wgsl";
const OUTPUT_SHADER_ASSET_PATH: &str = "shaders/output.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: (u32, u32) = (1024 / DISPLAY_FACTOR, 1024 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(
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
        )
        .add_plugins((
            FluidSimulationComputePlugin,
            // GpuReadbackPlugin::default(),
            // ExtractResourcePlugin::<ReadbackBuffer>::default(),
            // EguiPlugin,
            Material2dPlugin::<CustomMaterial>::default(),
            bevy::diagnostic::FrameTimeDiagnosticsPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, keyboard)
        // .add_systems(Update, ui_system.after(keyboard))
        .init_resource::<ResetSimulation>()
        .run();
}

#[derive(Resource, Clone, ExtractResource, Deref, DerefMut, Default)]
struct ResetSimulation(bool);

fn keyboard(mut ev_keyboard: EventReader<KeyboardInput>, mut res_reset: ResMut<ResetSimulation>) {
    *res_reset.as_deref_mut() = false;
    for ev in ev_keyboard.read() {
        if ev.key_code == KeyCode::KeyR {
            *res_reset.as_deref_mut() = true;
        }
    }
}

// fn ui_system(
//     mut contexts: EguiContexts,
//     mut res_reset: ResMut<ResetSimulation>,
//     mut parameters: ResMut<FluidSimulationParameters>,
//     diagnostics: Res<DiagnosticsStore>,
// ) {
//     egui::Window::new("Simulation Options").show(contexts.ctx_mut(), |ui| {
//         ui.heading("Performance");
//         egui::Grid::new("performance_grid").show(ui, |ui| {
//             for diagnostic in diagnostics.iter() {
//                 let (Some(value), Some(avg)) = (diagnostic.smoothed(), diagnostic.average()) else {
//                     continue;
//                 };
//                 ui.label(diagnostic.path().as_str());
//                 ui.label(format!("{:4.3} {}", value, diagnostic.suffix));
//                 ui.label(format!("{:4.3} {}", avg, diagnostic.suffix));
//                 ui.end_row();
//             }
//         });

//         ui.spacing();

//         ui.heading("Controls");
//         if ui.button("Reset Simulation").clicked() {
//             *res_reset.as_deref_mut() = true;
//         }
//         ui.spacing();

//         ui.heading("Parameters");
//         egui::Grid::new("grid").show(ui, |ui| {
//             ui.label("Viscosity");
//             ui.add(egui::Slider::new(&mut parameters.viscosity, 0.0..=10.0).logarithmic(true));
//             ui.end_row();

//             ui.label("Grid Step");
//             ui.add(egui::Slider::new(&mut parameters.grid_step, 0.0..=1.0).logarithmic(true));
//             ui.end_row();

//             ui.label("Time Step");
//             ui.add(egui::Slider::new(&mut parameters.time_step, 0.0..=1.0).logarithmic(true));
//             ui.end_row();

//             ui.label("Diffusion Iterations");
//             ui.add(egui::DragValue::new(&mut parameters.diffusion_iterations).range(0..=100));
//             ui.end_row();

//             ui.label("Pressure Iterations");
//             ui.add(egui::DragValue::new(&mut parameters.pressure_iterations).range(0..=100));
//             ui.end_row();
//         });
//     });
// }
#[derive(Component)]
struct ReadbackTextureTag(String);

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut scalar_image = Image::new_fill(
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
    scalar_image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(scalar_image.clone());
    let image1 = images.add(scalar_image.clone());

    let mut vector_image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    vector_image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;

    let output = images.add(vector_image.clone());
    let velocity_a = images.add(vector_image.clone());
    let velocity_b = images.add(vector_image.clone());
    let pressure_a: Handle<Image> = images.add(vector_image.clone());
    let pressure_b = images.add(vector_image.clone());
    let divergence = images.add(vector_image);

    commands.spawn((
        MeshMaterial2d(materials.add(ColorMaterial {
            texture: Some(output.clone()),
            color: Color::WHITE,
            alpha_mode: bevy::sprite::AlphaMode2d::Opaque,
        })),
        // MeshMaterial2d(materials.add(CustomMaterial {
        //     color_texture: image0.clone(),
        //     velocity_texture: velocity0.clone(),
        //     pressure_texture: pressure0.clone(),
        //     divergence_texture: divergence.clone(),
        // })),
        Mesh2d(meshes.add(Rectangle::new(SIZE.0 as f32, SIZE.1 as f32))),
    ));

    commands.spawn(Camera2d::default());

    commands.insert_resource(GameOfLifeImages {
        texture_a: image0,
        texture_b: image1,
    });
    commands.insert_resource(FluidSimulationParameters {
        time_step: 0.025,
        grid_step: 0.02,
        viscosity: 0.001,
        diffusion_iterations: 20,
        pressure_iterations: 50,
    });
    commands
        .spawn((
            ReadbackTextureTag("velocity".into()),
            Readback::texture(velocity_a.clone()),
        ))
        .observe(readback_observer);
    commands
        .spawn((
            ReadbackTextureTag("pressure".into()),
            Readback::texture(pressure_a.clone()),
        ))
        .observe(readback_observer);
    commands
        .spawn((
            ReadbackTextureTag("divergence".into()),
            Readback::texture(divergence.clone()),
        ))
        .observe(readback_observer);
    commands
        .spawn((
            ReadbackTextureTag("output".into()),
            Readback::texture(output.clone()),
        ))
        .observe(readback_observer);
    commands.insert_resource(FluidSimulationImages {
        output,
        velocity_a,
        velocity_b,
        pressure_a,
        pressure_b,
        divergence,
    });
}

fn readback_observer(
    trigger: Trigger<ReadbackComplete>,
    mut count: Local<u32>,
    q_tag: Query<&ReadbackTextureTag>,
) {
    let name = &q_tag.get(trigger.entity()).unwrap().0;
    let data: Vec<f32> = trigger.event().to_shader_type();
    let mut image_data = Vec::<u8>::new();
    let scale = match name.as_str() {
        "pressure" => 10.0,
        "velocity" => 1.0,
        "divergence" => 0.5,
        _ => 1.0,
    };

    let true_color = name.as_str() == "output";

    for (i, pixel) in data.chunks(4).enumerate() {
        if i.rem(1040) >= 1024 {
            continue;
        }

        if true_color {
            image_data.extend(
                pixel
                    .into_iter()
                    .map(|p| (255.0 * p.clamp(0.0, 1.0)) as u8)
                    .take(3),
            );
        } else {
            let x = scale * pixel[0];
            let y = scale * pixel[1];
            let u = (x + 255.0 / 2.0).clamp(0.0, 255.0) as u8;
            let v = (y + 255.0 / 2.0).clamp(0.0, 255.0) as u8;

            image_data.push(u);
            image_data.push(v);
            image_data.push(127);
        }
    }
    let img = RgbImage::from_vec(1024, 1024, image_data).unwrap();

    img.save(format!("out/{}-{}.png", name, *count));
    *count += 1;
}

struct FluidSimulationComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GameOfLifeLabel;

impl Plugin for FluidSimulationComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins((
            ExtractResourcePlugin::<GameOfLifeImages>::default(),
            ExtractResourcePlugin::<FluidSimulationImages>::default(),
            ExtractResourcePlugin::<FluidSimulationParameters>::default(),
            ExtractResourcePlugin::<ResetSimulation>::default(),
        ));

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
    diffusion_iterations: usize,
    pressure_iterations: usize,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct JacobiUniform {
    pub alpha: f32,
    pub r_beta: f32,
}

#[derive(Resource)]
struct FluidSimulationUniforms {
    general: UniformBuffer<GeneralUniform>,
    pressure: UniformBuffer<JacobiUniform>,
    diffusion: UniformBuffer<JacobiUniform>,
}

impl Default for FluidSimulationUniforms {
    fn default() -> Self {
        let mut advection = UniformBuffer::default();
        advection.set_label(Some("advection_uniforms_buffer"));

        let mut pressure = UniformBuffer::default();
        pressure.set_label(Some("pressure_uniforms_buffer"));

        let mut diffusion = UniformBuffer::default();
        diffusion.set_label(Some("diffusion_uniforms_buffer"));

        advection.add_usages(BufferUsages::UNIFORM);
        pressure.add_usages(BufferUsages::UNIFORM);
        diffusion.add_usages(BufferUsages::UNIFORM);

        Self {
            general: advection,
            pressure,
            diffusion,
        }
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct FluidSimulationImages {
    output: Handle<Image>,
    velocity_a: Handle<Image>,
    velocity_b: Handle<Image>,
    divergence: Handle<Image>,
    pressure_a: Handle<Image>,
    pressure_b: Handle<Image>,
}

#[derive(Component, ShaderType, Clone, Default)]
pub struct GeneralUniform {
    pub time_step: f32,
    pub half_rdx: f32,
    pub grid_step: f32,
}

#[derive(Resource)]
struct FluidSimulationBindGroups {
    general_uniform: BindGroup,
    advection_image: [BindGroup; 2],

    diffusion_uniform: BindGroup,
    diffusion_image: [BindGroup; 2],

    divergence_image: [BindGroup; 2],

    pressure_uniform: BindGroup,
    pressure_image: [BindGroup; 2],

    gradient_subtraction_image: [BindGroup; 2],

    output_image: BindGroup,
}

fn prepare_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut fluid_simulation_uniforms: ResMut<FluidSimulationUniforms>,
    parameters: Res<FluidSimulationParameters>,
) {
    fluid_simulation_uniforms.general.set(GeneralUniform {
        time_step: parameters.time_step,
        grid_step: parameters.grid_step,
        half_rdx: 0.5 / parameters.grid_step,
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
        .general
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
        &pipeline.general_uniform_bind_group_layout,
        &BindGroupEntries::single(uniforms.general.into_binding()),
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

    let color_buffer_a = gpu_images.get(&game_of_life_images.texture_a).unwrap();
    let color_buffer_b = gpu_images.get(&game_of_life_images.texture_b).unwrap();
    let velocity_buffer_a = gpu_images.get(&fluid_simulation_images.velocity_a).unwrap();
    let velocity_buffer_b = gpu_images.get(&fluid_simulation_images.velocity_b).unwrap();
    let divergence_buffer = gpu_images.get(&fluid_simulation_images.divergence).unwrap();
    let pressure_buffer_a = gpu_images.get(&fluid_simulation_images.pressure_a).unwrap();
    let pressure_buffer_b = gpu_images.get(&fluid_simulation_images.pressure_b).unwrap();
    let output_buffer = gpu_images.get(&fluid_simulation_images.output).unwrap();

    // Advection (Prime -> Off-prime)
    let advection_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.advection_image_group_layout,
            &BindGroupEntries::sequential((
                &color_buffer_a.texture_view,
                &color_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
                &divergence_buffer.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.advection_image_group_layout,
            &BindGroupEntries::sequential((
                &color_buffer_b.texture_view,
                &color_buffer_a.texture_view,
                &velocity_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
                &divergence_buffer.texture_view,
            )),
        ),
    ];

    // Diffusion (Off-prime first)
    let diffusion_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_b.texture_view,
                &divergence_buffer.texture_view,
                &velocity_buffer_a.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_a.texture_view,
                &divergence_buffer.texture_view,
                &velocity_buffer_b.texture_view,
            )),
        ),
    ];
    // Divergence (Off-prime first)
    let divergence_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.divergence_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_b.texture_view,
                &divergence_buffer.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.divergence_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_a.texture_view,
                &divergence_buffer.texture_view,
            )),
        ),
    ];

    // Pressure (Own buffers)
    let pressure_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &pressure_buffer_a.texture_view,
                &divergence_buffer.texture_view,
                &pressure_buffer_b.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.jacobi_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &pressure_buffer_b.texture_view,
                &divergence_buffer.texture_view,
                &pressure_buffer_a.texture_view,
            )),
        ),
    ];

    // Gradient Subtraction (Velocity: Off-prime -> Prime, Pressure: Off-prime)
    let gradient_subtraction_image_bind_groups = [
        render_device.create_bind_group(
            None,
            &pipeline.gradient_subtraction_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_b.texture_view,
                &color_buffer_b.texture_view,
                &pressure_buffer_b.texture_view,
                &velocity_buffer_a.texture_view,
                &color_buffer_a.texture_view,
            )),
        ),
        render_device.create_bind_group(
            None,
            &pipeline.gradient_subtraction_image_bind_group_layout,
            &BindGroupEntries::sequential((
                &velocity_buffer_a.texture_view,
                &color_buffer_a.texture_view,
                &pressure_buffer_b.texture_view,
                &velocity_buffer_b.texture_view,
                &color_buffer_b.texture_view,
            )),
        ),
    ];

    let output_image_bind_groups = render_device.create_bind_group(
        None,
        &pipeline.output_image_bind_group_layout,
        &BindGroupEntries::sequential((
            &color_buffer_a.texture_view,
            &velocity_buffer_a.texture_view,
            &pressure_buffer_b.texture_view,
            &output_buffer.texture_view,
        )),
    );

    commands.insert_resource(FluidSimulationBindGroups {
        general_uniform: advection_uniform_bind_group,
        advection_image: advection_image_bind_groups,
        diffusion_uniform: diffusion_uniform_bind_group,
        diffusion_image: diffusion_image_bind_groups,
        divergence_image: divergence_image_bind_groups,
        pressure_uniform: pressure_uniform_bind_group,
        pressure_image: pressure_image_bind_groups,
        gradient_subtraction_image: gradient_subtraction_image_bind_groups,
        output_image: output_image_bind_groups,
    })
}

#[derive(Resource)]
struct FluidSimulationPipeline {
    general_uniform_bind_group_layout: BindGroupLayout,
    advection_image_group_layout: BindGroupLayout,
    jacobi_uniform_bind_group_layout: BindGroupLayout,
    jacobi_image_bind_group_layout: BindGroupLayout,
    divergence_image_bind_group_layout: BindGroupLayout,
    gradient_subtraction_image_bind_group_layout: BindGroupLayout,
    output_image_bind_group_layout: BindGroupLayout,

    init_pipeline: CachedComputePipelineId,
    advection_pipeline: CachedComputePipelineId,
    diffusion_pipeline: CachedComputePipelineId,
    divergence_pipeline: CachedComputePipelineId,
    pressure_pipeline: CachedComputePipelineId,
    gradient_subtraction_pipeline: CachedComputePipelineId,
    output_pipeline: CachedComputePipelineId,
}

impl FromWorld for FluidSimulationPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let general_uniform_bind_group_layout = render_device.create_bind_group_layout(
            "AdvectionUniformLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<GeneralUniform>(false),),
            ),
        );
        let jacobi_uniform_bind_group_layout = render_device.create_bind_group_layout(
            "JacobiUniformLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<JacobiUniform>(false),),
            ),
        );

        let r_read = texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly);
        let r_write = texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly);

        let rgba_read =
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly);
        let rgba_write =
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly);

        let advection_image_group_layout = render_device.create_bind_group_layout(
            "AdvectionImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (r_read, r_write, rgba_read, rgba_write, rgba_write),
            ),
        );
        let jacobi_image_bind_group_layout = render_device.create_bind_group_layout(
            "JacobiImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (rgba_read, rgba_read, rgba_write),
            ),
        );
        let divergence_image_bind_group_layout = render_device.create_bind_group_layout(
            "DiffusionImageLayout",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (rgba_read, rgba_write)),
        );
        let gradient_subtraction_image_bind_group_layout = render_device.create_bind_group_layout(
            "GradientSubtractionImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (rgba_read, r_read, rgba_read, rgba_write, r_write),
            ),
        );
        let output_image_bind_group_layout = render_device.create_bind_group_layout(
            "OutputImageLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (r_read, rgba_read, rgba_read, rgba_write),
            ),
        );
        let advection_shader = world.load_asset(ADVECTION_SHADER_ASSET_PATH);
        let jacobi_shader = world.load_asset(JACOBI_SHADER_ASSET_PATH);
        let divergence_shader = world.load_asset(DIVERGENCE_SHADER_ASSET_PATH);
        let gradient_subtraction_shader = world.load_asset(GRADIENT_SUBTRACTION_SHADER_ASSET_PATH);
        let output_shader = world.load_asset(OUTPUT_SHADER_ASSET_PATH);

        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                general_uniform_bind_group_layout.clone(),
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
                general_uniform_bind_group_layout.clone(),
                advection_image_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: advection_shader,
            shader_defs: vec![],
            entry_point: Cow::from("advect"),
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
        let divergence_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: vec![
                    general_uniform_bind_group_layout.clone(),
                    divergence_image_bind_group_layout.clone(),
                ],
                push_constant_ranges: Vec::new(),
                shader: divergence_shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from("divergence"),
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
        let gradient_subtraction_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: vec![
                    general_uniform_bind_group_layout.clone(),
                    gradient_subtraction_image_bind_group_layout.clone(),
                ],
                push_constant_ranges: Vec::new(),
                shader: gradient_subtraction_shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from("gradient_subtraction"),
            });
        let output_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![
                general_uniform_bind_group_layout.clone(),
                output_image_bind_group_layout.clone(),
            ],
            push_constant_ranges: Vec::new(),
            shader: output_shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("output"),
        });
        FluidSimulationPipeline {
            general_uniform_bind_group_layout,
            advection_image_group_layout,
            jacobi_image_bind_group_layout,
            jacobi_uniform_bind_group_layout,
            divergence_image_bind_group_layout,
            gradient_subtraction_image_bind_group_layout,
            output_image_bind_group_layout,
            init_pipeline,
            advection_pipeline,
            diffusion_pipeline,
            divergence_pipeline,
            pressure_pipeline,
            gradient_subtraction_pipeline,
            output_pipeline,
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
        let ResetSimulation(reset) = world.resource::<ResetSimulation>();
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
            GameOfLifeState::Update(index) => {
                self.state = GameOfLifeState::Update(1 - index);
                if *reset {
                    self.state = GameOfLifeState::Init;
                }
            }
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let parameters = world.resource::<FluidSimulationParameters>();
        let bind_groups = world.resource::<FluidSimulationBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FluidSimulationPipeline>();

        let encoder = render_context.command_encoder();

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let Some(init_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.init_pipeline)
                else {
                    return Ok(());
                };
                let gpu_images = world.resource::<RenderAssets<GpuImage>>();
                let fluid_simulation_images = world.resource::<FluidSimulationImages>();
                let game_of_life_images = world.resource::<GameOfLifeImages>();

                let textures = [
                    gpu_images.get(&game_of_life_images.texture_a).unwrap(),
                    gpu_images.get(&game_of_life_images.texture_b).unwrap(),
                    gpu_images.get(&fluid_simulation_images.velocity_a).unwrap(),
                    gpu_images.get(&fluid_simulation_images.velocity_b).unwrap(),
                    gpu_images.get(&fluid_simulation_images.divergence).unwrap(),
                    gpu_images.get(&fluid_simulation_images.pressure_a).unwrap(),
                    gpu_images.get(&fluid_simulation_images.pressure_b).unwrap(),
                ];

                for image in textures.into_iter() {
                    encoder.clear_texture(&image.texture, &Default::default());
                }

                let mut pass: ComputePass<'_> =
                    encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_bind_group(0, &bind_groups.general_uniform, &[]);
                pass.set_bind_group(1, &bind_groups.advection_image[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update(_) => {
                let index = 1;
                // Advection
                let Some(advection_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.advection_pipeline)
                else {
                    return Ok(());
                };
                // let gpu_images = world.resource::<RenderAssets<GpuImage>>();
                // let fluid_simulation_images = world.resource::<FluidSimulationImages>();

                // for image in [
                //     gpu_images.get(&fluid_simulation_images.pressure_a).unwrap(),
                //     gpu_images.get(&fluid_simulation_images.pressure_b).unwrap(),
                // ] {
                //     encoder.clear_texture(&image.texture, &Default::default());
                // }

                let mut pass: ComputePass<'_> =
                    encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(advection_pipeline);
                pass.set_bind_group(0, &bind_groups.general_uniform, &[]);

                // prime -> off-prime
                pass.set_bind_group(1, &bind_groups.advection_image[index], &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                let Some(diffusion_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.diffusion_pipeline)
                else {
                    return Ok(());
                };

                pass.set_pipeline(diffusion_pipeline);
                pass.set_bind_group(0, &bind_groups.diffusion_uniform, &[]);

                // Iterate diffusion, result in off-prime buffer
                for _ in 0..parameters.diffusion_iterations {
                    // off-prime -> prime
                    pass.set_bind_group(1, &bind_groups.diffusion_image[index], &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                    // prime -> off-prime
                    pass.set_bind_group(1, &bind_groups.diffusion_image[1 - index], &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }

                // Divergence
                let Some(divergence_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.divergence_pipeline)
                else {
                    return Ok(());
                };

                pass.set_pipeline(divergence_pipeline);
                pass.set_bind_group(0, &bind_groups.general_uniform, &[]);
                pass.set_bind_group(1, &bind_groups.divergence_image[index], &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                // Pressure
                let Some(pressure_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.pressure_pipeline)
                else {
                    return Ok(());
                };

                pass.set_pipeline(pressure_pipeline);
                pass.set_bind_group(0, &bind_groups.pressure_uniform, &[]);
                for j in 0usize..parameters.pressure_iterations {
                    pass.set_bind_group(1, &bind_groups.pressure_image[j.rem_euclid(2)], &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }

                // Gradient Subtraction
                let Some(gradient_subtraction_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.gradient_subtraction_pipeline)
                else {
                    return Ok(());
                };

                pass.set_pipeline(gradient_subtraction_pipeline);
                pass.set_bind_group(0, &bind_groups.general_uniform, &[]);
                pass.set_bind_group(1, &bind_groups.gradient_subtraction_image[index], &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);

                // Output
                let Some(output_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipeline.output_pipeline)
                else {
                    return Ok(());
                };

                pass.set_pipeline(output_pipeline);
                pass.set_bind_group(0, &bind_groups.general_uniform, &[]);
                pass.set_bind_group(1, &bind_groups.output_image, &[]);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}

#[derive(AsBindGroup, Debug, Clone, Asset, TypePath)]
pub struct CustomMaterial {
    #[texture(0)]
    #[sampler(1)]
    velocity_texture: Handle<Image>,

    #[texture(2)]
    #[sampler(3)]
    color_texture: Handle<Image>,

    #[texture(4)]
    #[sampler(5)]
    pressure_texture: Handle<Image>,

    #[texture(6)]
    #[sampler(7)]
    divergence_texture: Handle<Image>,
}

// All functions on `Material2d` have default impls. You only need to implement the
// functions that are relevant for your material.
impl Material2d for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/custom_material.wgsl".into()
    }
}
