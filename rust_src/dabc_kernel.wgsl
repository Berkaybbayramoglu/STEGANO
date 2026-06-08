// WGSL: D-ABC GPU Kernels (WGPU)

struct Params {
    width: u32,
    height: u32,
    pool_len: u32,
    colony_size: u32,
    payload_len: u32,
};

@group(0) @binding(0) var<storage, read> image_u8 : array<u32>;
@group(0) @binding(1) var<storage, read> pool_y   : array<u32>;
@group(0) @binding(2) var<storage, read> pool_x   : array<u32>;
@group(0) @binding(3) var<storage, read> foods    : array<u32>;
@group(0) @binding(4) var<uniform> params         : Params;

@group(0) @binding(5) var<storage, read_write> score_table : array<f32>;
@group(0) @binding(6) var<storage, read_write> fitness     : array<f32>;

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    return max(lo, min(hi, v));
}

fn img_at(x: i32, y: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let xi = clamp_i32(x, 0, w - 1);
    let yi = clamp_i32(y, 0, h - 1);
    let idx = u32(yi * w + xi);
    return f32(image_u8[idx] & 0xFFu);
}

@compute @workgroup_size(256)
fn compute_score_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.pool_len) {
        return;
    }

    let y = i32(pool_y[i]);
    let x = i32(pool_x[i]);

    var sum: f32 = 0.0;
    var sum2: f32 = 0.0;
    var count: f32 = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let v = img_at(x + dx, y + dy);
            sum = sum + v;
            sum2 = sum2 + v * v;
            count = count + 1.0;
        }
    }

    let mean = sum / count;
    let variance = (sum2 / count) - (mean * mean);
    score_table[i] = variance;
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn compute_fitness(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let bee_id = wid.x;
    if (bee_id >= params.colony_size) {
        return;
    }

    let L = params.payload_len;
    let base = bee_id * L;

    var local_sum: f32 = 0.0;
    var idx = lid.x;

    while (idx < L) {
        let pool_idx = foods[base + idx];
        local_sum = local_sum + score_table[pool_idx];
        idx = idx + 256u;
    }

    partial[lid.x] = local_sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lid.x < stride) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        fitness[bee_id] = partial[0];
    }
}
