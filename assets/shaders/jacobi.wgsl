// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for the game of life as each pixel of step N depends on the state of its
// neighbors at step N-1.

@group(0) @binding(0) var<uniform> alpha: f32;
@group(0) @binding(1) var<uniform> r_beta: f32;

@group(1) @binding(0) var input_x: texture_storage_2d<r32float, read>;
@group(1) @binding(1) var input_b: texture_storage_2d<r32float, read>;
@group(1) @binding(2) var output_x: texture_storage_2d<r32float, write>;


@compute @workgroup_size(8, 8, 1)
fn jacobi(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

  // left, right, bottom, and top x samples
    half4 xL = textureLoad(input_x, location - vec2<i32>(1, 0));
    half4 xR = textureLoad(input_x, location + vec2<i32>(1, 0));
    half4 xB = textureLoad(input_x, location - vec2<i32>(0, 1));
    half4 xT = textureLoad(input_x, location + vec2<i32>(0, 1));
    
    // b sample, from center
    half4 bC = textureLoad(input_b, location);

    // evaluate Jacobi iteration
    let x_new = (xL + xR + xB + xT + alpha * bC) * rBeta;
    textureStore(output_x, location, x_new)
}