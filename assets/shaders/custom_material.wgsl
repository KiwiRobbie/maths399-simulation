#import bevy_pbr::forward_io::VertexOutput


struct CustomMaterial {
    color: vec4<f32>,
}

@group(2) @binding(0) var velocity_texture: texture_2d<f32>;
@group(2) @binding(1) var velocity_sampler: sampler;

@group(2) @binding(2) var color_texture: texture_2d<f32>;
@group(2) @binding(3) var color_sampler: sampler;

@group(2) @binding(4) var pressure_texture: texture_2d<f32>;
@group(2) @binding(5) var pressure_sampler: sampler;

@group(2) @binding(6) var divergence_texture: texture_2d<f32>;
@group(2) @binding(7) var divergence_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(
        in.position.x / f32(textureDimensions(velocity_texture).x),
        in.position.y / f32(textureDimensions(velocity_texture).y),
    );
    let velocity = textureSample(velocity_texture, velocity_sampler, uv).xy ;
    let color = textureSample(color_texture, color_sampler, uv).r;
    let pressure = textureSample(pressure_texture, pressure_sampler, uv).r;
    let divergence = textureSample(divergence_texture, divergence_sampler, uv).r;

    // return  vec4<f32>(velocity.x, velocity.y, 0.0, 1.0) + vec4<f32>(0.0, 0.0, color, 0.0);
    return vec4<f32>(color, color, color, 1.0) * vec4<f32>(max(0.0, pressure), 0.25, max(0.0, -pressure), 1.0);
    // return vec4<f32>(textureLoad(color_texture, vec2<i32>(100), 0).x, 0.0, 0.0, 1.0);
}