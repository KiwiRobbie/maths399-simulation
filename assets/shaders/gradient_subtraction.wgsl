struct GeneralUniform {
    time_step: f32,
    half_rdx: f32,
    grid_step: f32,
}

@group(0) @binding(0) var<uniform> uniforms: GeneralUniform;
@group(1) @binding(0) var w_input: texture_storage_2d<rg32float, read>;
@group(1) @binding(1) var c_input: texture_storage_2d<r32float, read>;
@group(1) @binding(2) var p_input: texture_storage_2d<rg32float, read>;
@group(1) @binding(3) var u_output: texture_storage_2d<rg32float, write>;
@group(1) @binding(4) var c_output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn gradient_subtraction(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    // left, right, bottom, and top pressure samples
    let  p_left = textureLoad(p_input, location - vec2<i32>(1, 0)).x;
    let  p_right = textureLoad(p_input, location + vec2<i32>(1, 0)).x;
    let  p_down = textureLoad(p_input, location - vec2<i32>(0, 1)).x;
    let  p_up = textureLoad(p_input, location + vec2<i32>(0, 1)).x;

    let velocity = textureLoad(w_input, location).xy - uniforms.half_rdx * vec2<f32>(p_right - p_left, p_up - p_down);

    let color = max(textureLoad(c_input, location) * exp(-uniforms.time_step), vec4<f32>(f32(location.x % 32 == 0 && location.y % 32 == 0)));
    let velocity_allow_x = f32(location.x != 0) * f32(location.x < 1023);
    let velocity_allow_y = f32(location.y != 0) * f32(location.y < 1023);
    let velocity_allow = velocity_allow_x * velocity_allow_y;
    let boundary_velocity = vec2<f32>(0.0, -10.0) * f32(location.y > 1000) * f32(location.x > 512);

    textureStore(u_output, location, vec4<f32>(
        velocity.x * velocity_allow,
        velocity.y * velocity_allow,
        0.0, 0.0
    ));
    textureStore(c_output, location, color);
}

