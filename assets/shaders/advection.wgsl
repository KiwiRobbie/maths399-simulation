struct GeneralUniform {
    time_step: f32,
    half_rdx: f32,
    grid_step: f32,
}

@group(0) @binding(0) var<uniform> uniforms: GeneralUniform;

@group(1) @binding(0) var input: texture_storage_2d<r32float, read>;
@group(1) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var velocity_input: texture_storage_2d<rgba32float, read>;
@group(1) @binding(3) var velocity_output: texture_storage_2d<rgba32float, write>;
@group(1) @binding(4) var velocity_copy: texture_storage_2d<rgba32float, write>;


fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let randomNumber = randomFloat(invocation_id.y << 16u | invocation_id.x);
    let alive = randomNumber > 0.925;
    let color = 0.0 * vec4<f32>(f32(alive));

    let uv = 2.0 * vec2<f32>(
        f32(location.x) / f32(textureDimensions(velocity_output).x),
        f32(location.y) / f32(textureDimensions(velocity_output).y),
    ) - 1.0;
    let x = randomFloat(invocation_id.y << 16u | invocation_id.x + 282873452);
    let y = randomFloat(invocation_id.y << 16u | invocation_id.x + 178712234);
    let rand = vec2<f32>(x - 0.5, y - 0.5) * 2.0;


    
    // let speed = 1000.0 * (uv - vec2<f32>(0.5));
    // let velocity = vec2<f32>(speed.y, -speed.x);

    // let velocity = 100.0 * vec2<f32>(2.0 * x - 1.0, 2.0 * round(uv.x) - 1.0);
    // let velocity = vec2<f32>(1000.0);

    let coord = 2.0 * fract(uv) - 1.0;
    let velocity = 500.0 * vec2<f32>(0.0, sign(uv.x)) * f32(dot(coord, coord) < 0.1) + 200.0 * rand;




    // let velocity = vec2<f32>(0.0) + vec2<f32>(200.0, 0.0) * f32(dot(2.0 * uv - 1.0, 2.0 * uv - 1.0) < 0.1 * 0.1);
    textureStore(output, location, color);
    textureStore(velocity_output, location, vec4<f32>(velocity.x, velocity.y, 0.0, 0.0));
}



fn bilinear_sample_x(texture: texture_storage_2d<r32float, read>, pixelPos: vec2<f32>) -> f32 {
    // Get the texture size using textureDimensions
    let textureSize: vec2<i32> = vec2<i32>(
        i32(textureDimensions(texture).x),
        i32(textureDimensions(texture).y),
    );

    // Calculate the texture size as floating point
    let size = vec2<f32>(textureSize);

    // Get the pixel coordinates in the texture
    // let pixelPos = uv * size - 0.5;

    // Get the integer coordinates of the top-left texel
    let p0 = vec2<i32>(floor(pixelPos));

    // Calculate fractional offset within the pixel
    let frac = fract(pixelPos);

    // Calculate texel coordinates of the four surrounding texels
    let p00 = max(vec2<i32>(0), min(p0, textureSize - 1));
    let p10 = max(vec2<i32>(0), min(p0 + vec2<i32>(1, 0), textureSize - 1));
    let p01 = max(vec2<i32>(0), min(p0 + vec2<i32>(0, 1), textureSize - 1));
    let p11 = max(vec2<i32>(0), min(p0 + vec2<i32>(1, 1), textureSize - 1));

    // Load the four nearest texels
    let tex00 = textureLoad(texture, p00);
    let tex10 = textureLoad(texture, p10);
    let tex01 = textureLoad(texture, p01);
    let tex11 = textureLoad(texture, p11);

    // Perform bilinear interpolation
    let top = mix(tex00, tex10, frac.x);
    let bottom = mix(tex01, tex11, frac.x);
    return mix(top, bottom, frac.y).x;
}

fn bilinear_sample_xy(texture: texture_storage_2d<rgba32float, read>, pixelPos: vec2<f32>) -> vec2<f32> {
    // Get the texture size using textureDimensions
    let textureSize: vec2<i32> = vec2<i32>(
        i32(textureDimensions(texture).x),
        i32(textureDimensions(texture).y),
    );

    // Calculate the texture size as floating point
    let size = vec2<f32>(textureSize);

    // Get the pixel coordinates in the texture
    // let pixelPos = uv * size - 0.5;

    // Get the integer coordinates of the top-left texel
    let p0 = vec2<i32>(floor(pixelPos));

    // Calculate fractional offset within the pixel
    let frac = fract(pixelPos);

    // Calculate texel coordinates of the four surrounding texels
    let p00 = max(vec2<i32>(0), min(p0, textureSize - 2));
    let p10 = max(vec2<i32>(0), min(p0 + vec2<i32>(1, 0), textureSize - 1));
    let p01 = max(vec2<i32>(0), min(p0 + vec2<i32>(0, 1), textureSize - 1));
    let p11 = max(vec2<i32>(0), min(p0 + vec2<i32>(1, 1), textureSize - 1));

    // Load the four nearest texels
    let tex00 = textureLoad(texture, p00);
    let tex10 = textureLoad(texture, p10);
    let tex01 = textureLoad(texture, p01);
    let tex11 = textureLoad(texture, p11);

    // Perform bilinear interpolation
    let top = mix(tex00, tex10, frac.x);
    let bottom = mix(tex01, tex11, frac.x);
    return mix(top, bottom, frac.y).xy;
}

@compute @workgroup_size(8, 8, 1)
fn advect(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    // Advection 
    let velocity = textureLoad(velocity_input, location).xy;
    let advection_source = vec2<f32>(f32(location.x), f32(location.y)) - uniforms.time_step * velocity.xy;


    let advected_color = bilinear_sample_x(input, advection_source);
    let advected_velocity = bilinear_sample_xy(velocity_input, advection_source).xy;
    textureStore(output, location, vec4<f32>(advected_color));
    textureStore(velocity_output, location, vec4<f32>(advected_velocity.x, advected_velocity.y, 0.0, 0.0));
    textureStore(velocity_copy, location, vec4<f32>(advected_velocity.x, advected_velocity.y, 0.0, 0.0));
}

