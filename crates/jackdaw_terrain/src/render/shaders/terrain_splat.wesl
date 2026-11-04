// Terrain splat shading: two texture ids per control texel, blended by comparing
// their height maps.
//
// The control map is `texture_2d<u32>` read with `textureLoad` only: a packed
// control word must never be filtered into a different packed word. Four texels
// are read per fragment and their decoded contributions are mixed with bilinear
// weights.
//
// A terrain with autoterrain on replaces the word of every cell no hand has
// claimed with one its own slope stands for. That slope comes from `slope_map`,
// which holds the ground's own slope per grid point, and never from the surface
// being drawn: the surface is a clipmap, so a slope taken from the mesh normal
// reads gentler with distance.
//
// Every texture read is `textureSampleGrad` with derivatives taken once in
// uniform control flow at the top of `fragment`: the per-layer branches below
// make implicit-derivative sampling illegal.

import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    mesh_bindings::mesh,
    mesh_view_bindings::view,
    pbr_functions,
    pbr_types,
};

const BASE_SHIFT: u32 = 0u;
const BASE_MASK: u32 = 0x1Fu;
const OVERLAY_SHIFT: u32 = 5u;
const OVERLAY_MASK: u32 = 0x1Fu;
const BLEND_SHIFT: u32 = 10u;
const BLEND_MASK: u32 = 0xFFu;
// A hand painted this cell, so its own word is what it draws.
const MANUAL_BIT: u32 = 0x40000u;

// Entries in `uv_scales`, which is four `vec4`s wide.
const MAX_LAYERS: u32 = 16u;

// What a slot that sets no reflectance shades at: a StandardMaterial's
// default, 4% at normal incidence.
const DEFAULT_REFLECTANCE: f32 = 0.5;

// height a slot with no height map blends at
const FLAT_HEIGHT: f32 = 128.0 / 255.0;

struct SplatUniform {
    // 16 per-id UV scales, packed four to a vec4 for uniform alignment.
    uv_scales: array<vec4<f32>, 4>,
    // 16 per-id detiling strengths, packed the same way. 0 is off.
    detile_strengths: array<vec4<f32>, 4>,
    // 16 per-id perceptual roughness values a slot sets of its own, packed
    // the same way: a scale on the slot's roughness map, or the roughness
    // where it has none. Negative where it sets none, which leaves the map
    // as it is, or the terrain's own roughness.
    slot_roughness: array<vec4<f32>, 4>,
    // 16 per-id reflectance values, packed and unset the same way.
    slot_reflectance: array<vec4<f32>, 4>,
    // Terrain XZ extent in world units, for turning UV0 back into a
    // terrain-local position. Local, not world, so a terrain that moves
    // carries its texturing with it.
    terrain_size: vec2<f32>,
    blend_sharpness: f32,
    perceptual_roughness: f32,
    control_resolution: u32,
    layer_count: u32,
    // Nonzero where this terrain textures the cells no hand has claimed
    // from their slope. 0 leaves every cell to its own control word.
    autoterrain_enabled: u32,
    autoterrain_base_slot: u32,
    autoterrain_slope_slot: u32,
    // The slope range in radians, converted from the authored degrees on
    // the way in so this side carries no conversion of its own.
    autoterrain_slope_start: f32,
    autoterrain_slope_end: f32,
    // How much of the tint texture reaches the finished albedo, 0..1.
    // The layer's own white is the identity, so a terrain that has never
    // been tinted draws the same at every strength.
    tint_strength: f32,
    // Bit i is set where texture id i has an occlusion map. An id without
    // one shades unoccluded instead of reading its filled array layer.
    occlusion_slots: u32,
    // Bit i is set where texture id i has a roughness map. An id without
    // one takes `perceptual_roughness`.
    roughness_slots: u32,
    normal_slots: u32,
    height_slots: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> splat: SplatUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var albedo_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var layer_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var normal_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var height_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var control_map: texture_2d<u32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var slope_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var tint_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(8) var tint_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(9) var occlusion_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(10) var roughness_array: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(11) var point_sampler: sampler;

// The last id this material can address: one short of the bound set's
// layer count, and never past the UV-scale array. Every id is clamped
// through here exactly once, so the layer sampled and the UV scale
// applied can never disagree.
fn last_layer() -> u32 {
    return min(max(splat.layer_count, 1u), MAX_LAYERS) - 1u;
}

// `id` must already be clamped by `last_layer`.
fn uv_scale_for(id: u32) -> f32 {
    return splat.uv_scales[id / 4u][id % 4u];
}

// `id` must already be clamped by `last_layer`.
// A slot's own roughness or reflectance, negative where it sets none.
fn slot_value(values: array<vec4<f32>, 4>, id: u32) -> f32 {
    return values[id / 4u][id % 4u];
}

fn detile_for(id: u32) -> f32 {
    return splat.detile_strengths[id / 4u][id % 4u];
}

// Two uncorrelated values in 0..1 from a tile coordinate. Integer-valued
// input, so the hash only has to separate whole numbers.
fn tile_hash(tile: vec2<f32>) -> vec2<f32> {
    var h = fract(vec3<f32>(tile.x, tile.y, tile.x + tile.y) * vec3<f32>(0.1031, 0.1030, 0.0973));
    h += dot(h, h.yzx + 33.33);
    return fract((h.xx + h.yz) * h.zy);
}

// One tile's own rigid sample frame: the tile's rotation and shift are applied
// whole, and the four tiles around a fragment are blended after each has been
// sampled in its own frame.
struct TileTap {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
    // The rotation that produced `uv`, as a unit `(cos, sin)`. Anything
    // sampled through it that is itself a direction - a tangent-space
    // normal - has to turn back by the same angle.
    turn: vec2<f32>,
    weight: f32,
}

// The tiles one fragment reads, with the share each of them takes.
struct Sampling {
    taps: array<TileTap, 4>,
    count: u32,
}

// `v` turned by the unit `(cos, sin)` in `turn`.
fn turned_by(v: vec2<f32>, turn: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(turn.x * v.x - turn.y * v.y, turn.y * v.x + turn.x * v.y);
}

fn texel_sharp(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let texel = uv * size;
    let footprint = clamp((abs(ddx) + abs(ddy)) * size, vec2<f32>(1e-5), vec2<f32>(1.0));
    let edge = floor(texel + 0.5);
    let across = smoothstep(-0.5 * footprint, 0.5 * footprint, texel - edge) - 0.5;
    return (edge + across) / size;
}

// Break a tiled texture out of its grid.
//
// Every tile of the repeat is turned and shifted by its own amount, drawn from a
// hash of its coordinate. The four nearest tiles are each sampled in their own
// frame and the sampled values blended with the bilinear weights of the position
// between their centres, so a tile's weight reaches zero where it is left and the
// blend is continuous across the plane.
//
// Sampling per tile, rather than blending the four tiles' coordinates and
// sampling once, is the whole point: interpolating between rigid transforms is
// not a rigid transform, and the blended coordinate field stretches and shears
// between tile centres. This is the standard technique (Inigo Quilez's texture
// repetition; Unity HDRP and Unreal call it procedural or stochastic tiling).
//
// Each tap carries its own derivatives, and a rigid frame makes them exact. The
// rotation pivots on the tile centres rather than the texture origin, so the
// displacement does not grow with how far the terrain runs.
//
// `strength` 0 returns a single tap holding exactly what came in.
fn detile(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, strength: f32) -> Sampling {
    var out: Sampling;
    out.count = 1u;
    out.taps[0].uv = uv;
    out.taps[0].ddx = ddx;
    out.taps[0].ddy = ddy;
    out.taps[0].turn = vec2<f32>(1.0, 0.0);
    out.taps[0].weight = 1.0;
    if strength <= 0.0 {
        return out;
    }

    // Half a tile back, so `base` names the four tile centres around the
    // sample rather than the corners of the tile it is inside.
    let p = uv - vec2<f32>(0.5);
    let base = floor(p);
    let f = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), fract(p));

    out.count = 4u;
    for (var j = 0u; j < 2u; j++) {
        for (var i = 0u; i < 2u; i++) {
            let corner = base + vec2<f32>(f32(i), f32(j));
            let weight = mix(1.0 - f.x, f.x, f32(i)) * mix(1.0 - f.y, f.y, f32(j));
            let hash = tile_hash(corner);
            let angle = (hash.x * 2.0 - 1.0) * strength * 3.14159265;
            let shift = (hash - vec2<f32>(0.5)) * strength;
            let pivot = corner + vec2<f32>(0.5);
            let turn = vec2<f32>(cos(angle), sin(angle));
            let index = j * 2u + i;
            out.taps[index].uv = pivot + turned_by(uv - pivot, turn) + shift;
            out.taps[index].ddx = turned_by(ddx, turn);
            out.taps[index].ddy = turned_by(ddy, turn);
            out.taps[index].turn = turn;
            out.taps[index].weight = weight;
        }
    }
    return out;
}

// The texel a grid coordinate's lower corner sits in. `control_resolution`
// texels cover grid points 0..resolution-1, and the upper corner of the
// bilinear tap is this + 1, so the base clamps one short of the end.
fn control_corner(g: vec2<f32>) -> vec2<i32> {
    let last = i32(max(splat.control_resolution, 2u) - 2u);
    return clamp(vec2<i32>(floor(g)), vec2<i32>(0), vec2<i32>(last));
}

fn gather_uv(corner: vec2<i32>, size: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(corner) + 1.0) / vec2<f32>(size);
}

// The stored slope under a fragment, bilinear across the four grid points around
// it.
//
// Filtered rather than nearest so the band between the two textures moves
// smoothly across a cell, and taken on the slope rather than on the two control
// words it produces so the four corners agree wherever the ground is uniformly
// flat or steep, which keeps the single-word fast path below covering most of a
// terrain.
fn slope_at(corner: vec2<i32>, f: vec2<f32>) -> f32 {
    // x = (0, 1), y = (1, 1), z = (1, 0), w = (0, 0)
    let s = textureGather(0, slope_map, point_sampler, gather_uv(corner, textureDimensions(slope_map)));
    return mix(mix(s.w, s.z, f.x), mix(s.x, s.y, f.x), f.y);
}

struct LayerSlots {
    layer: i32,
    has_height: bool,
    has_normal: bool,
    has_occlusion: bool,
    has_roughness: bool,
}

struct LayerSample {
    albedo: vec3<f32>,
    height: f32,
    normal: vec3<f32>,
    occlusion: f32,
    roughness: f32,
}

fn sample_tap(tap: TileTap, slots: LayerSlots) -> LayerSample {
    let layer_size = vec2<f32>(textureDimensions(albedo_array));
    let uv = texel_sharp(tap.uv, tap.ddx, tap.ddy, layer_size);
    var out: LayerSample;
    out.albedo = textureSampleGrad(albedo_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).rgb;
    out.height = FLAT_HEIGHT;
    if slots.has_height {
        out.height = textureSampleGrad(height_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).r;
    }
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    if slots.has_normal {
        let packed = textureSampleGrad(normal_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).rgb;
        let unpacked = packed * 2.0 - 1.0;
        let back = turned_by(unpacked.xy, vec2<f32>(tap.turn.x, -tap.turn.y));
        out.normal = vec3<f32>(back, unpacked.z);
    }
    out.occlusion = 1.0;
    if slots.has_occlusion {
        out.occlusion = textureSampleGrad(
            occlusion_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).r;
    }
    out.roughness = 1.0;
    if slots.has_roughness {
        out.roughness = textureSampleGrad(
            roughness_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).g;
    }
    return out;
}

fn sample_detiled(tiled: Sampling, slots: LayerSlots) -> LayerSample {
    var sum: LayerSample;
    sum.albedo = vec3<f32>(0.0);
    sum.height = 0.0;
    sum.normal = vec3<f32>(0.0);
    sum.occlusion = 0.0;
    sum.roughness = 0.0;
    var taken = 0.0;
    for (var t = 0u; t < tiled.count; t++) {
        let tap = tiled.taps[t];
        if tap.weight <= 1e-4 {
            continue;
        }
        let one = sample_tap(tap, slots);
        sum.albedo += tap.weight * one.albedo;
        sum.height += tap.weight * one.height;
        sum.normal += tap.weight * one.normal;
        sum.occlusion += tap.weight * one.occlusion;
        sum.roughness += tap.weight * one.roughness;
        taken += tap.weight;
    }
    let inv = 1.0 / max(taken, 1e-8);
    sum.albedo *= inv;
    sum.height *= inv;
    sum.normal *= inv;
    sum.occlusion *= inv;
    sum.roughness *= inv;
    return sum;
}

const SIDE_START_COS: f32 = 0.766; // 40 degrees
const SIDE_END_COS: f32 = 0.574; // 55 degrees

struct Surface {
    p: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
    // 0 on ground flat enough to project from above, 1 on cliffs.
    steep: f32,
    faces_x: bool,
}

// Hash Functions for GPU Rendering https://jcgt.org/published/0009/03/02/
fn pcg3d(input: vec3<u32>) -> vec3<u32> {
    var v = input * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn cell_hash(cell: vec3<i32>) -> f32 {
    return f32(pcg3d(vec3<u32>(cell)).x >> 8u) / 16777216.0;
}

fn project(surface: Surface, scale: f32, texels: f32) -> TileTap {
    let p = surface.p * scale;
    let dx = surface.ddx * scale;
    let dy = surface.ddy * scale;
    let side = cell_hash(vec3<i32>(floor(p * texels))) < surface.steep;
    if !side {
        return TileTap(p.xz, dx.xz, dy.xz, vec2<f32>(1.0, 0.0), 1.0);
    }
    // texture v runs down, height runs up
    if surface.faces_x {
        return TileTap(
            vec2<f32>(p.z, -p.y), vec2<f32>(dx.z, -dx.y), vec2<f32>(dy.z, -dy.y), vec2<f32>(1.0, 0.0), 1.0);
    }
    return TileTap(
        vec2<f32>(p.x, -p.y), vec2<f32>(dx.x, -dx.y), vec2<f32>(dy.x, -dy.y), vec2<f32>(1.0, 0.0), 1.0);
}

fn sample_layer(layer_id: u32, surface: Surface) -> LayerSample {
    let slots = LayerSlots(
        i32(layer_id),
        ((splat.height_slots >> layer_id) & 1u) != 0u,
        ((splat.normal_slots >> layer_id) & 1u) != 0u,
        ((splat.occlusion_slots >> layer_id) & 1u) != 0u,
        ((splat.roughness_slots >> layer_id) & 1u) != 0u,
    );
    let projected = project(surface, uv_scale_for(layer_id), f32(textureDimensions(albedo_array).x));
    let strength = detile_for(layer_id);
    if strength <= 0.0 {
        return sample_tap(projected, slots);
    }
    return sample_detiled(detile(projected.uv, projected.ddx, projected.ddy, strength), slots);
}

struct Accum {
    base_ids: vec4<u32>,
    base_weights: vec4<f32>,
    overlay_ids: vec4<u32>,
    overlay_weights: vec4<f32>,
}

// you have a 4 texture mask like (0, 0, 1, 1)
// with a weighting like (0.4, 0.2, 0.1, 0.3)
// the share_below(1) is the mask (1, 1, 0, 0) i.e. (true true false false)
// sum the matching weights 0.4 + 0.2 = 0.6
fn share_below(accum: Accum, id: u32) -> f32 {
    let one = vec4<f32>(1.0);
    let zero = vec4<f32>(0.0);
    return dot(accum.base_weights, select(zero, one, accum.base_ids < vec4<u32>(id)))
        + dot(accum.overlay_weights, select(zero, one, accum.overlay_ids < vec4<u32>(id)));
}

// same as share below but with the inverse masking rule
fn share_of(accum: Accum, id: u32) -> f32 {
    let one = vec4<f32>(1.0);
    let zero = vec4<f32>(0.0);
    return dot(accum.base_weights, select(zero, one, accum.base_ids == vec4<u32>(id)))
        + dot(accum.overlay_weights, select(zero, one, accum.overlay_ids == vec4<u32>(id)));
}

fn dithered_layer(accum: Accum, threshold: f32) -> u32 {
    let level = threshold * (dot(accum.base_weights, vec4<f32>(1.0)) + dot(accum.overlay_weights, vec4<f32>(1.0)));
    var chosen = accum.base_ids.x;
    for (var k = 0u; k < 4u; k++) {
        let base = share_below(accum, accum.base_ids[k]);
        if base <= level && level < base + share_of(accum, accum.base_ids[k]) {
            chosen = accum.base_ids[k];
        }
        let overlay = share_below(accum, accum.overlay_ids[k]);
        if overlay <= level && level < overlay + share_of(accum, accum.overlay_ids[k]) {
            chosen = accum.overlay_ids[k];
        }
    }
    return chosen;
}


fn bayer(cell: vec2<u32>) -> f32 {
    let lo = cell & vec2<u32>(1u);
    let hi = (cell >> vec2<u32>(1u)) & vec2<u32>(1u);
    let rank = 4u * (((lo.x ^ lo.y) << 1u) | lo.y) + (((hi.x ^ hi.y) << 1u) | hi.y);
    return (f32(rank) + 0.5) / 16.0;
}

// The control word a cell's own slope stands for.
//
// A word, not a colour: the two slots and the blend between them are what a hand
// would have painted for this slope, so the fragment carries on into
// `accumulate_control` and picks up the height blend, the detiling and the mip
// derivatives that painted ground gets.
//
// `smoothstep` across the authored range means flat ground is purely the base
// slot and anything past the far end purely the slope slot.
fn autoterrain_control(slope: f32) -> u32 {
    let t = smoothstep(splat.autoterrain_slope_start, splat.autoterrain_slope_end, slope);
    let blend = u32(clamp(t, 0.0, 1.0) * 255.0 + 0.5);
    return ((splat.autoterrain_base_slot & BASE_MASK) << BASE_SHIFT)
        | ((splat.autoterrain_slope_slot & OVERLAY_MASK) << OVERLAY_SHIFT)
        | ((blend & BLEND_MASK) << BLEND_SHIFT);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    // UV0 is 0..1 across the whole terrain, shared bit for bit by chunks
    // that meet on a grid line, which is what keeps the control lookup
    // seamless across a chunk join.
    let local_xz = (in.uv - vec2<f32>(0.5)) * splat.terrain_size;
    let p = vec3<f32>(local_xz.x, in.world_position.y, local_xz.y);
    let world_normal = pbr_functions::prepare_world_normal(in.world_normal, false, is_front);
    let n = normalize(world_normal);
    var surface: Surface;
    surface.p = p;
    surface.ddx = dpdx(p);
    surface.ddy = dpdy(p);
    surface.steep = saturate((SIDE_START_COS - n.y) / (SIDE_START_COS - SIDE_END_COS));
    surface.faces_x = abs(n.x) > abs(n.z);

    // The tint is sampled, not loaded: it is a colour, so two of them interpolate
    // into a colour. Taken here, with the derivatives, because the per-layer
    // branches below are not uniform control flow.
    //
    // Grid point i sits at i/(res-1) in UV0, while texel i is sampled at its
    // centre, (i+0.5)/res. Without the remap the layer would be half a cell out of
    // register with the control map, in opposite directions at the two edges.
    // `render::tint_uv` spells the same mapping in Rust.
    let tint_res = f32(max(splat.control_resolution, 2u));
    let tint_uv = (in.uv * (tint_res - 1.0) + 0.5) / tint_res;
    var tint = vec3<f32>(1.0);
    if splat.tint_strength > 0.0 {
        tint = textureSample(tint_map, tint_sampler, tint_uv).rgb;
    }

    // manipulate uvs to be whole texels
    var finest = 0.0;
    for (var i = 0u; i <= last_layer(); i++) {
        finest = max(finest, uv_scale_for(i));
    }
    let texel = 1.0 / (finest * f32(textureDimensions(albedo_array).x));
    let texel_cell = vec2<i32>(floor(local_xz / texel));
    let texel_uv = (vec2<f32>(texel_cell) + 0.5) * texel / splat.terrain_size + vec2<f32>(0.5);
    let g = texel_uv * f32(max(splat.control_resolution, 2u) - 1u);
    let corner = control_corner(g);
    let f = clamp(g - vec2<f32>(corner), vec2<f32>(0.0), vec2<f32>(1.0));

    // gather order: x = (0, 1), y = (1, 1), z = (1, 0), w = (0, 0)
    let words = textureGather(0, control_map, point_sampler, gather_uv(corner, textureDimensions(control_map)));
    var c00 = words.w;
    var c10 = words.z;
    var c01 = words.x;
    var c11 = words.y;

    // A terrain with autoterrain off never reaches this, and a cell a
    // hand claimed keeps the word it was painted with. Everything
    // downstream reads these four words and cannot tell which of them
    // came from the control map.
    if splat.autoterrain_enabled != 0u {
        let unclaimed = autoterrain_control(slope_at(corner, f));
        c00 = select(unclaimed, c00, (c00 & MANUAL_BIT) != 0u);
        c10 = select(unclaimed, c10, (c10 & MANUAL_BIT) != 0u);
        c01 = select(unclaimed, c01, (c01 & MANUAL_BIT) != 0u);
        c11 = select(unclaimed, c11, (c11 & MANUAL_BIT) != 0u);
    }

    let inv = vec2<f32>(1.0) - f;
    let w = vec4<f32>(inv.x * inv.y, f.x * inv.y, inv.x * f.y, f.x * f.y);

    // each texel has a certain share of each texture
    // the dither is calculated based on that share
    let corner_words = vec4<u32>(c00, c10, c01, c11);
    var accum: Accum;
    for (var k = 0u; k < 4u; k++) {
        let control = corner_words[k];
        let base_id = min((control >> BASE_SHIFT) & BASE_MASK, last_layer());
        let overlay_id = min((control >> OVERLAY_SHIFT) & OVERLAY_MASK, last_layer());
        let blend = f32((control >> BLEND_SHIFT) & BLEND_MASK) / 255.0;
        accum.base_ids[k] = base_id;
        accum.overlay_ids[k] = overlay_id;
        accum.base_weights[k] = 0.0;
        accum.overlay_weights[k] = 0.0;
        if blend < 1.0 {
            accum.base_weights[k] = w[k] * (1.0 - blend);
        }
        if blend > 0.0 {
            accum.overlay_weights[k] = w[k] * blend;
        }
    }

    // "pick" what texture this texel will be based on the shares above
    // the second argument is the threshold to use and bayer is what gives us that dither pattern
    let layer_id = dithered_layer(accum, bayer(vec2<u32>(texel_cell & vec2<i32>(3))));
    let sampled = sample_layer(layer_id, surface);
    let own_roughness = slot_value(splat.slot_roughness, layer_id);
    var roughness_scale = 1.0;
    if own_roughness >= 0.0 {
        roughness_scale = own_roughness;
    } else if ((splat.roughness_slots >> layer_id) & 1u) == 0u {
        roughness_scale = splat.perceptual_roughness;
    }
    let own_reflectance = slot_value(splat.slot_reflectance, layer_id);
    let albedo = sampled.albedo * mix(vec3<f32>(1.0), tint, splat.tint_strength);
    let tangent_normal = sampled.normal;
    let occlusion = sampled.occlusion;
    let roughness = min(sampled.roughness * roughness_scale, 1.0);
    let reflectance = select(DEFAULT_REFLECTANCE, own_reflectance, own_reflectance >= 0.0);

    // The mesher emits no tangents: UV0 runs along world X and Z, so the tangent
    // is world +X projected onto the surface normal. That projection collapses
    // where the surface stands vertical and faces X, so +Z stands in there.
    // Picking the axis the normal leans on least would swap the frame by 90
    // degrees wherever |n.x| and |n.z| cross, seaming flat ground.
    var reference = vec3<f32>(1.0, 0.0, 0.0);
    if abs(n.x) > 0.999 {
        reference = vec3<f32>(0.0, 0.0, 1.0);
    }
    let t = normalize(reference - n * dot(reference, n));
    let tbn = mat3x3<f32>(t, cross(t, n), n);
    let mapped_normal = normalize(tbn * normalize(tangent_normal));

    var pbr_input = pbr_types::pbr_input_new();
    // Mesh flags carry the shadow-receiver and transmitted-shadow-receiver
    // bits. Leaving them zero renders terrain that no shadow ever falls on.
    pbr_input.flags = mesh[in.instance_index].flags;
    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.material.perceptual_roughness = roughness;
    pbr_input.material.reflectance = vec3<f32>(reflectance);
    pbr_input.material.metallic = 0.0;
    pbr_input.diffuse_occlusion = vec3<f32>(occlusion);
    // `standard_material_new` leaves fog off; `StandardMaterial`'s Rust
    // default has it on, and terrain that ignores the scene's fog while
    // everything beside it obeys reads as a hole in the distance.
    pbr_input.material.flags |= pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    pbr_input.world_normal = n;
    pbr_input.N = mapped_normal;
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = pbr_functions::calculate_view(in.world_position, pbr_input.is_orthographic);

    var out: FragmentOutput;
    out.color = pbr_functions::apply_pbr_lighting(pbr_input);
    out.color = pbr_functions::main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
