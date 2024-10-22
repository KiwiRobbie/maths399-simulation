struct GeneralUniform {
    time_step: f32,
    half_rdx: f32,
    grid_step: f32,
}

@group(0) @binding(0) var<uniform> uniforms: GeneralUniform;

@group(1) @binding(0) var color_input: texture_storage_2d<r32float, read>;
@group(1) @binding(1) var velocity_input: texture_storage_2d<rgba32float, read>;
@group(1) @binding(2) var pressure_input: texture_storage_2d<rgba32float, read>;
@group(1) @binding(3) var texture_output: texture_storage_2d<rgba32float, write>;


@compute @workgroup_size(8, 8, 1)
fn output(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let color = textureLoad(color_input, location).x;
    let velocity = textureLoad(velocity_input, location).xy;
    let pressure = textureLoad(pressure_input, location).x;

    let out = vec4<f32>(color, color, color, 1.0) * vec4<f32>(max(0.0, pressure), 0.25, max(0.0, -pressure), 1.0);
    textureStore(texture_output, location, out);
}
