struct GeneralUniform {
    time_step: f32,
    half_rdx: f32,
    grid_step: f32,
}


@group(0) @binding(0) var<uniform> uniforms: GeneralUniform;
@group(1) @binding(0) var w_input: texture_storage_2d<rgba32float, read>;
@group(1) @binding(1) var div_output: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn divergence(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    // left, right, bottom, and top x samples
    let  w_left = textureLoad(w_input, location - vec2<i32>(1, 0));
    let  w_right = textureLoad(w_input, location + vec2<i32>(1, 0));
    let  w_down = textureLoad(w_input, location - vec2<i32>(0, 1));
    let  w_up = textureLoad(w_input, location + vec2<i32>(0, 1));

    // evaluate Jacobi iteration
    let div = uniforms.half_rdx * ((w_right.x - w_left.x) + (w_up.y - w_down.y));
    textureStore(div_output, location, vec4<f32>(div, 0.0, 0.0, 0.0));
}

