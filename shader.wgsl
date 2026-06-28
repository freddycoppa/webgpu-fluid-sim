override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;

struct Params {
    dt: f32,
    splatFlag: u32,
    splatRadius: f32,
    vorticity: f32,
    splatCenter: vec2f,
    splatVelocity: vec2f,
    splatDensity: vec4f,
    densityDiffusion: f32,
    velocityDiffusion: f32,
    contour: u32,
    sharpen: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var linearClampedSampler: sampler;
@group(0) @binding(2) var nearestClampedSampler: sampler;

@group(1) @binding(0) var density_in: texture_2d<f32>;
@group(1) @binding(1) var density_out: texture_storage_2d<rgba16float, write>;

/*
@group(2) @binding(0) var pressure_in: texture_2d<f32>;
@group(2) @binding(1) var pressure_out: texture_storage_2d<r32float, write>;
*/

@group(2) @binding(0) var curl_in: texture_2d<f32>;
@group(2) @binding(1) var curl_out: texture_storage_2d<r32float, write>;

@group(3) @binding(0) var u_in: texture_2d<f32>;
@group(3) @binding(1) var u_out: texture_storage_2d<r32float, write>;
@group(3) @binding(2) var v_in: texture_2d<f32>;
@group(3) @binding(3) var v_out: texture_storage_2d<r32float, write>;

fn d_dims() -> vec2f {
    return vec2f(textureDimensions(density_in));
}

fn v_dims() -> vec2f {
    return vec2f(textureDimensions(u_in) - vec2u(1, 0));
}

fn SIM_WIDTH() -> u32 {
    return textureDimensions(v_in).x;
}

fn SIM_HEIGHT() -> u32 {
    return textureDimensions(u_in).y;
}

fn outOfBounds(id: vec3u, tex: texture_2d<f32>) -> bool {
    let size = textureDimensions(tex);
    return id.x >= size.x || id.y >= size.y;
}

fn texindex(tex: texture_2d<f32>, ij: vec2u) -> f32 {
    return textureLoad(tex, ij, 0).x;
}

fn texwrite(
    tex: texture_storage_2d<r32float, write>,
    ij: vec2u,
    value: f32
) {
    textureStore(tex, ij, vec4f(value, 0.0, 0.0, 1.0));
}

fn sample(tex: texture_2d<f32>, xy: vec2f) -> f32 {
    let size = vec2i(textureDimensions(tex));

    // Convert from cell-center coordinates to index coordinates.
    // So xy = vec2f(i + 0.5, j + 0.5) maps exactly to cell (i, j).
    let p = xy - vec2f(0.5);

    let i0 = vec2i(floor(p));
    let f = fract(p);

    // Clamp to edge. Replace this with wrapping if your sim is toroidal.
    let a = clamp(i0, vec2i(0), size - vec2i(1));
    let b = clamp(i0 + vec2i(1, 0), vec2i(0), size - vec2i(1));
    let c = clamp(i0 + vec2i(0, 1), vec2i(0), size - vec2i(1));
    let d = clamp(i0 + vec2i(1, 1), vec2i(0), size - vec2i(1));

    let va = textureLoad(tex, a, 0).x;
    let vb = textureLoad(tex, b, 0).x;
    let vc = textureLoad(tex, c, 0).x;
    let vd = textureLoad(tex, d, 0).x;

    let x0 = mix(va, vb, f.x);
    let x1 = mix(vc, vd, f.x);

    return mix(x0, x1, f.y);
}

fn texSampleUV(tex: texture_2d<f32>, uv: vec2f) -> f32 {
    return sample(tex, uv * vec2f(textureDimensions(tex)));
}

fn sampleU(xy: vec2f) -> f32 {
    return sample(u_in, xy + vec2f(0.5, 0));
}

fn sampleV(xy: vec2f) -> f32 {
    return sample(v_in, xy + vec2f(0, 0.5));
}

fn sampleVelocity(xy: vec2f) -> vec2f {
    return vec2f(sampleU(xy), sampleV(xy));
}

fn gaussian2d(p: vec2f, r: f32) -> f32 {
    let r_safe = max(r, 1e-6);
    let r2 = r_safe * r_safe;
    return exp(-dot(p, p) / r2);
}

fn gaussian2dVec(p: vec2f, v: vec2f) -> f32 {
    let r = min(v.x, v.y);
    return exp( - (dot(p, p) / (r * r)) );
}

fn rk1_backtrace(xy: vec2f, dt: f32) -> vec2f {
    let k1 = sampleVelocity(xy);
    return xy - dt * k1;
}

fn rk4_backtrace(xy: vec2f, dt: f32) -> vec2f {
    let k1 = sampleVelocity(xy);
    let k2 = sampleVelocity(xy - dt/2 * k1);
    let k3 = sampleVelocity(xy - dt/2 * k2);
    let k4 = sampleVelocity(xy -  dt  * k3);
    return xy - dt/6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

fn backtrace(xy: vec2f) -> vec2f {
    return rk4_backtrace(xy, params.dt);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn splatDensity(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, density_in)) { return; }

    let cell = id.xy;
    let x = vec2f(cell) + vec2f(0.5, 0.5);
    let base = textureLoad(density_in, cell, 0);
    let splat = gaussian2dVec(
        x - params.splatCenter * d_dims(),
        params.splatRadius * d_dims()
    ) * params.splatDensity;
    textureStore(density_out, cell, base + splat);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn splatU(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, u_in)) { return; }

    let cell = id.xy;
    if (cell.x == 0 || cell.x == SIM_WIDTH()) {
        texwrite(u_out, cell, 0);
        return;
    }

    let x = vec2f(cell) + vec2f(0, 0.5);
    let base = texindex(u_in, cell);
    let splat = gaussian2dVec(
        x - params.splatCenter * v_dims(),
        params.splatRadius * v_dims()
    ) * params.splatVelocity.x * v_dims().x;
    texwrite(u_out, cell, base + splat);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn splatV(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, v_in)) { return; }

    let cell = id.xy;
    if (cell.y == 0 || cell.y == SIM_HEIGHT()) {
        texwrite(v_out, cell, 0);
        return;
    }

    let x = vec2f(cell) + vec2f(0.5, 0);
    let base = texindex(v_in, cell);
    let splat = gaussian2dVec(
        x - params.splatCenter * v_dims(),
        params.splatRadius * v_dims()
    ) * params.splatVelocity.y * v_dims().y;
    texwrite(v_out, cell, base + splat);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn advectDensity(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, density_in)) { return; }

    let cell = id.xy;
    let x = vec2f(cell) + vec2f(0.5, 0.5);
    let x_prev_rel = backtrace(v_dims() * x / d_dims()) / v_dims();
    let rho = textureSampleLevel(density_in, linearClampedSampler, x_prev_rel, 0);
    let decay = 1 + params.densityDiffusion * params.dt;
    textureStore(density_out, cell, rho / decay);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn advectU(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, u_in)) { return; }

    let cell = id.xy;
    if (cell.x == 0 || cell.x == SIM_WIDTH()) {
        texwrite(u_out, cell, 0);
        return;
    }
    let x = vec2f(cell) + vec2f(0, 0.5);
    let x_prev = backtrace(x);
    let decay = 1 + params.velocityDiffusion * params.dt;  
    texwrite(u_out, cell, sampleU(x_prev) / decay);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn advectV(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, v_in)) { return; }

    let cell = id.xy;
    if (cell.y == 0 || cell.y == SIM_HEIGHT()) {
        texwrite(v_out, cell, 0);
        return;
    }
    let x = vec2f(cell) + vec2f(0.5, 0);
    let x_prev = backtrace(x);   
    let decay = 1 + params.velocityDiffusion * params.dt; 
    texwrite(v_out, cell, sampleV(x_prev) / decay);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn computeCurl(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, curl_in)) { return; }

    let cell = id.xy;

    /*if (cell.x == 0) {
        texwrite(curl_out, cell, texindex(v_in, cell));
        return;
    }

    if (cell.x == SIM_WIDTH()) {
        texwrite(curl_out, cell, -texindex(v_in, cell - vec2u(1, 0)));
        return;
    }

    if (cell.y == 0) {
        texwrite(curl_out, cell, -texindex(u_in, cell));
        return;
    }

    if (cell.y == SIM_HEIGHT()) {
        texwrite(curl_out, cell, texindex(u_in, cell - vec2u(0, 1)));
        return;
    }*/

    if (
           cell.x == 0
        || cell.x == SIM_WIDTH()
        || cell.y == 0
        || cell.y == SIM_HEIGHT()
    ) {
        texwrite(curl_out, cell, 0);
        return;
    }

    let dvdx = texindex(v_in, cell) - texindex(v_in, cell - vec2u(1, 0));
    let dudy = texindex(u_in, cell) - texindex(u_in, cell - vec2u(0, 1));

    texwrite(curl_out, cell, dvdx - dudy);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn applyVorticityU(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, u_in)) { return; }

    let cell = id.xy;

    if (cell.x == 0 || cell.x == SIM_WIDTH()) {
        texwrite(u_out, cell, 0);
        return;
    }

    let u = texindex(u_in, cell);

    if (cell.y == 0 || cell.y == SIM_HEIGHT() - 1) {
        texwrite(u_out, cell, u);
        return;
    }

    let curl_t = texindex(curl_in, cell + vec2u(0, 1));
    let curl_b = texindex(curl_in, cell + vec2u(0, 0));
    let mag_l = 0.5 * (
        abs(texindex(curl_in, cell - vec2u(1, 0))) +
        abs(texindex(curl_in, cell - vec2u(1, 0) + vec2u(0, 1)))
    );
    let mag_r = 0.5 * (
        abs(texindex(curl_in, cell + vec2u(1, 0))) +
        abs(texindex(curl_in, cell + vec2u(1, 0) + vec2u(0, 1)))
    );
    
    let omega = 0.5 * (curl_b + curl_t);
    let gy = abs(curl_t) - abs(curl_b);
    let gx = mag_r - mag_l;

    let grad = vec2f(gx, gy);
    let n = grad / max(length(grad), 1e-5);

    let force_x = params.vorticity * omega * n.y;

    texwrite(u_out, cell, u + params.dt * force_x);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn applyVorticityV(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, v_in)) { return; }

    let cell = id.xy;
    if (cell.y == 0 || cell.y == SIM_HEIGHT()) {
        texwrite(v_out, cell, 0);
        return;
    }

    let v = texindex(v_in, cell);

    if (cell.x == 0 || cell.x == SIM_WIDTH() - 1) {
        texwrite(v_out, cell, v);
        return;
    }

    let curl_r = texindex(curl_in, cell + vec2u(1, 0));
    let curl_l = texindex(curl_in, cell + vec2u(0, 0));
    let mag_b = 0.5 * (
        abs(texindex(curl_in, cell - vec2u(0, 1))) +
        abs(texindex(curl_in, cell - vec2u(0, 1) + vec2u(1, 0)))
    );

    let mag_t = 0.5 * (
        abs(texindex(curl_in, cell + vec2u(0, 1))) +
        abs(texindex(curl_in, cell + vec2u(0, 1) + vec2u(1, 0)))
    );

    let omega = 0.5 * (curl_l + curl_r);
    let gx = abs(curl_r) - abs(curl_l);
    let gy = mag_t - mag_b;

    let grad = vec2f(gx, gy);
    let n = grad / max(length(grad), 1e-5);
    
    let force_y = params.vorticity * omega * -n.x;

    texwrite(v_out, cell, v + params.dt * force_y);
}

/*
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn pressureStep(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, pressure_in)) { return; }

    let cell = id.xy;

    let d = (
          texindex(u_in, cell + vec2u(1, 0))
        - texindex(u_in, cell + vec2u(0, 0))
        + texindex(v_in, cell + vec2u(0, 1))
        - texindex(v_in, cell + vec2u(0, 0))
    );

    var p_new = -d;
    var neighbors: f32 = 0.0;

    if (cell.x > 0) {
        p_new += texindex(pressure_in, cell - vec2u(1, 0));
        neighbors += 1.0;
    }

    if (cell.x < SIM_WIDTH() - 1) {
        p_new += texindex(pressure_in, cell + vec2u(1, 0));
        neighbors += 1.0;
    }

    if (cell.y > 0) {
        p_new += texindex(pressure_in, cell - vec2u(0, 1));
        neighbors += 1.0;
    }

    if (cell.y < SIM_HEIGHT() - 1) {
        p_new += texindex(pressure_in, cell + vec2u(0, 1));
        neighbors += 1.0;
    }

    if (neighbors == 0.0) {
        texwrite(pressure_out, cell, texindex(pressure_in, cell));
        return;
    }

    p_new /= neighbors;

    texwrite(pressure_out, cell, p_new);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn subtractGradientU(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, u_in)) { return; }

    let cell = id.xy;
    if (cell.x == 0 || cell.x == SIM_WIDTH()) {
        texwrite(u_out, cell, 0.0);
        return;
    }

    let u = texindex(u_in, cell);
    let grad = (
          texindex(pressure_in, cell)
        - texindex(pressure_in, cell - vec2u(1, 0))
    );

    texwrite(u_out, cell, u - grad);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn subtractGradientV(@builtin(global_invocation_id) id: vec3u) {
    if (outOfBounds(id, v_in)) { return; }

    let cell = id.xy;
    if (cell.y == 0 || cell.y == SIM_HEIGHT()) {
        texwrite(v_out, cell, 0.0);
        return;
    }

    let v = texindex(v_in, cell);
    let grad = (
          texindex(pressure_in, cell)
        - texindex(pressure_in, cell - vec2u(0, 1))
    );

    texwrite(v_out, cell, v - grad);
}
*/

fn sValue(cell: vec2i) -> f32 {
    let i = cell.x;
    let j = cell.y;
    if (i <                   0) { return 0.0; }
    if (i > i32( SIM_WIDTH()) - 1) { return 0.0; }
    if (j <                   0) { return 0.0; }
    if (j > i32(SIM_HEIGHT()) - 1) { return 0.0; }
    return 1.0;
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn projectRed(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= SIM_WIDTH() || id.y >= SIM_HEIGHT()) { return; }
    projectRedBlack(0, id.xy);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn projectBlack(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= SIM_WIDTH() || id.y >= SIM_HEIGHT()) { return; }
    projectRedBlack(1, id.xy);
}

fn projectRedBlack(red_black_flag: u32, cell: vec2u) {
    if ((cell.x + cell.y) % 2 != red_black_flag) {
        if (cell.x ==              0) { texwrite(u_out, cell + vec2u(0, 0), 0.0); }
        if (cell.x == SIM_WIDTH()  - 1) { texwrite(u_out, cell + vec2u(1, 0), 0.0); }
        if (cell.y ==              0) { texwrite(v_out, cell + vec2u(0, 0), 0.0); }
        if (cell.y == SIM_HEIGHT() - 1) { texwrite(v_out, cell + vec2u(0, 1), 0.0); }
        return; 
    }
    let u0 = texindex(u_in, cell + vec2u(0, 0));
    let u1 = texindex(u_in, cell + vec2u(1, 0));
    let v0 = texindex(v_in, cell + vec2u(0, 0));
    let v1 = texindex(v_in, cell + vec2u(0, 1));
    let d  = (u1 - u0 + v1 - v0) * 1.9;
    let sn = sValue(vec2i(cell) + vec2i(0, 1));
    let se = sValue(vec2i(cell) + vec2i(1, 0));
    let ss = sValue(vec2i(cell) - vec2i(0, 1));
    let sw = sValue(vec2i(cell) - vec2i(1, 0));
    let s  = sn + se + ss + sw;
    if (s == 0.0) { return; }
    texwrite(u_out, cell + vec2u(0, 0), u0 + d * sw / s);
    texwrite(u_out, cell + vec2u(1, 0), u1 - d * se / s);
    texwrite(v_out, cell + vec2u(0, 0), v0 + d * ss / s);
    texwrite(v_out, cell + vec2u(0, 1), v1 - d * sn / s);
}

struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );

    let uvs = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );

    var out: VSOut;
    out.position = vec4f(positions[i], 0.0, 1.0);
    out.uv = uvs[i];

    return out;
}

fn densityAmount(rgb: vec3f) -> f32 {
    return max(max(rgb.r, rgb.g), rgb.b);
    //return 0.3 * rgb.r + 0.59 * rgb.g + 0.11 * rgb.b;
}

fn sharpenRgbDensity(rgb: vec3f) -> vec3f {
    // Pick a scalar "amount of smoke/color"
    let intensity = densityAmount(rgb);

    // Remove faint haze and increase contrast
    let mask = smoothstep(0.04, 0.35, intensity);

    // Preserve hue
    let hue = rgb / max(intensity, 1e-5);

    return hue * mask;
}

fn contourDensity(rgb: vec3f) -> vec3f {
    let d = densityAmount(rgb);

    // Number of visible density layers
    let bands = 12.0;

    // Quantized layer value
    let q = floor(d * bands) / bands;

    // Preserve hue, but quantize brightness
    let hue = rgb / max(d, 1e-5);

    return hue * q;
}

@fragment fn fs(vsOutput: VSOut) -> @location(0) vec4f {
    let rho = textureSample(density_in, nearestClampedSampler, vsOutput.uv);
    var rgb = rho.rgb;
    if (params.contour == 1) { rgb = contourDensity(rgb); }
    if (params.sharpen == 1) { rgb = sharpenRgbDensity(rgb); }
    return vec4f(rgb, 1.0);
}
