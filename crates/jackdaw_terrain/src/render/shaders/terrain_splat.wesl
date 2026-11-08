// Terrain splat shading: two texture ids per control texel, blended by comparing
// their height maps.
//
// The control map is `texture_2d<u32>` read with `textureLoad` only: a packed
// control word must never be filtered into a different packed word. Four texels
// are read per fragment and their decoded contributions are mixed with bilinear
// weights.
//
//
// Every texture read is `textureSampleGrad` with derivatives taken once in
// uniform control flow at the top of `fragment`: the per-layer branches below
// make implicit-derivative sampling illegal.

import bevy_pbr::render::{
    clustered_forward as clustering,
    forward_io::{VertexOutput, FragmentOutput},
    mesh_bindings::mesh,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    mesh_view_bindings::{clustered_lights, lights, view},
    mesh_view_types::{
        DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
        POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
    },
    pbr_functions,
    pbr_lighting::getDistanceAttenuation,
    pbr_types,
    shadows,
};
import bevy_render::maths::{PI, quat_rotate};
@if(ENVIRONMENT_MAP) import bevy_pbr::render::mesh_view_bindings::{environment_map_sampler, light_probes};
@if(ENVIRONMENT_MAP && MULTIPLE_LIGHT_PROBES_IN_ARRAY)
import bevy_pbr::render::mesh_view_bindings::diffuse_environment_maps;
@if(ENVIRONMENT_MAP && !MULTIPLE_LIGHT_PROBES_IN_ARRAY)
import bevy_pbr::render::mesh_view_bindings::diffuse_environment_map;

const BASE_SHIFT: u32 = 0u;
const BASE_MASK: u32 = 0x1Fu;
const OVERLAY_SHIFT: u32 = 5u;
const OVERLAY_MASK: u32 = 0x1Fu;
const BLEND_SHIFT: u32 = 10u;
const BLEND_MASK: u32 = 0xFFu;

// Entries in `uv_scales`, which is four `vec4`s wide.
const MAX_LAYERS: u32 = 16u;


struct SplatUniform {
    // 16 per-id UV scales, packed four to a vec4 for uniform alignment.
    uv_scales: array<vec4<f32>, 4>,
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
    variant_ranges: array<vec4<u32>, 4>,
    variant_weights: array<vec4<f32>, 16>,
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


// Where one layer is read: its texture coordinate and how that coordinate
// changes across the screen, for sampling with explicit derivatives.
struct TileTap {
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
}

fn texel_sharp(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let texel = uv * size;
    let footprint = clamp((abs(ddx) + abs(ddy)) * size, vec2<f32>(1e-5), vec2<f32>(1.0));
    let edge = floor(texel + 0.5);
    let across = smoothstep(-0.5 * footprint, 0.5 * footprint, texel - edge) - 0.5;
    return (edge + across) / size;
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

fn ground_normal(corner: vec2<i32>, f: vec2<f32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(slope_map)) - 1;
    let g00 = textureLoad(slope_map, clamp(corner, vec2<i32>(0), last), 0).yz;
    let g10 = textureLoad(slope_map, clamp(corner + vec2<i32>(1, 0), vec2<i32>(0), last), 0).yz;
    let g01 = textureLoad(slope_map, clamp(corner + vec2<i32>(0, 1), vec2<i32>(0), last), 0).yz;
    let g11 = textureLoad(slope_map, clamp(corner + vec2<i32>(1, 1), vec2<i32>(0), last), 0).yz;
    let g = mix(mix(g00, g10, f.x), mix(g01, g11, f.x), f.y);
    return normalize(vec3<f32>(g.x, sqrt(max(1.0 - dot(g, g), 0.0)), g.y));
}

// what the matte lighting reads from a layer; its height and roughness maps
// change nothing it draws, so they aren't sampled
struct LayerSlots {
    layer: i32,
    has_normal: bool,
    has_occlusion: bool,
}

struct LayerSample {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    occlusion: f32,
}

// `varies` where neighbouring tiles may draw different variants or turns: the
// texel past a tile's edge then isn't the one its wrap would read, so the
// smoothing between texels stops at the tile's edge rather than reading it
fn sample_tap(tap: TileTap, slots: LayerSlots, albedo_layer: i32, turn: u32, varies: bool) -> LayerSample {
    let layer_size = vec2<f32>(textureDimensions(albedo_array));
    var uv = texel_sharp(tap.uv, tap.ddx, tap.ddy, layer_size);
    if varies {
        let tile = floor(tap.uv);
        let inset = 0.5 / layer_size;
        uv = clamp(uv, tile + inset, tile + 1.0 - inset);
    }
    var out: LayerSample;
    out.albedo = textureSampleGrad(albedo_array, layer_sampler, uv, albedo_layer, tap.ddx, tap.ddy).rgb;
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    if slots.has_normal {
        let packed = textureSampleGrad(normal_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).rgb;
        out.normal = packed * 2.0 - 1.0;
        out.normal = vec3<f32>(unturned(out.normal.xy, turn), out.normal.z);
    }
    out.occlusion = 1.0;
    if slots.has_occlusion {
        out.occlusion = textureSampleGrad(
            occlusion_array, layer_sampler, uv, slots.layer, tap.ddx, tap.ddy).r;
    }
    return out;
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
    // ground that isn't steep at all never takes a side projection, so it skips the hash
    let side = surface.steep > 0.0 && cell_hash(vec3<i32>(floor(p * texels))) < surface.steep;
    if !side {
        return TileTap(p.xz, dx.xz, dy.xz);
    }
    // texture v runs down, height runs up
    if surface.faces_x {
        return TileTap(
            vec2<f32>(p.z, -p.y), vec2<f32>(dx.z, -dx.y), vec2<f32>(dy.z, -dy.y));
    }
    return TileTap(
        vec2<f32>(p.x, -p.y), vec2<f32>(dx.x, -dx.y), vec2<f32>(dy.x, -dy.y));
}

fn turned(v: vec2<f32>, turn: u32) -> vec2<f32> {
    var out = select(v, v.yx, (turn & 1u) != 0u);
    out.x = select(out.x, -out.x, (turn & 2u) != 0u);
    out.y = select(out.y, -out.y, (turn & 4u) != 0u);
    return out;
}

fn unturned(v: vec2<f32>, turn: u32) -> vec2<f32> {
    var out = v;
    out.x = select(out.x, -out.x, (turn & 2u) != 0u);
    out.y = select(out.y, -out.y, (turn & 4u) != 0u);
    return select(out, out.yx, (turn & 1u) != 0u);
}

fn sample_layer(layer_id: u32, surface: Surface) -> LayerSample {
    let slots = LayerSlots(
        i32(layer_id),
        ((splat.normal_slots >> layer_id) & 1u) != 0u,
        ((splat.occlusion_slots >> layer_id) & 1u) != 0u,
    );
    var tap = project(surface, uv_scale_for(layer_id), f32(textureDimensions(albedo_array).x));
    let tile = floor(tap.uv);
    let hash = pcg3d(vec3<u32>(vec3<i32>(vec2<i32>(tile), i32(layer_id))));
    let range = splat.variant_ranges[layer_id / 4u][layer_id % 4u];

    var turn = 0u;
    if (range >> 31u) != 0u {
        turn = hash.y & 7u;
        tap = TileTap(
            tile + 0.5 + turned(tap.uv - tile - 0.5, turn),
            turned(tap.ddx, turn),
            turned(tap.ddy, turn),
        );
    }

    var albedo_layer = i32(layer_id);
    let first = range & 0xFFFFu;
    let count = (range >> 16u) & 0xFFu;
    if count > 0u {
        // the variant whose share of the weight the roll lands in
        let roll = f32(hash.x >> 8u) / 16777216.0;
        var chosen = 0u;
        for (var i = 0u; i + 1u < count; i++) {
            let at = first + i;
            chosen += u32(roll >= splat.variant_weights[at / 4u][at % 4u]);
        }
        albedo_layer = i32(splat.layer_count + first + chosen);
    }
    return sample_tap(tap, slots, albedo_layer, turn, (range >> 16u) != 0u);
}

struct Accum {
    base_ids: vec4<u32>,
    base_weights: vec4<f32>,
    overlay_ids: vec4<u32>,
    overlay_weights: vec4<f32>,
}

// The share of the corners' weight that goes to layers numbered `id` or lower:
// with ids (0, 0, 1, 1) and weights (0.4, 0.2, 0.1, 0.3), layer 0 has 0.6.
fn share_upto(accum: Accum, id: u32) -> f32 {
    let one = vec4<f32>(1.0);
    let zero = vec4<f32>(0.0);
    return dot(accum.base_weights, select(zero, one, accum.base_ids <= vec4<u32>(id)))
        + dot(accum.overlay_weights, select(zero, one, accum.overlay_ids <= vec4<u32>(id)));
}
fn dithered_layer(accum: Accum, threshold: f32) -> u32 {
    let only = accum.base_ids.x;
    let overlay_same = (accum.overlay_ids == vec4<u32>(only)) | (accum.overlay_weights == vec4<f32>(0.0));
    if all(accum.base_ids == vec4<u32>(only)) && all(overlay_same) {
        return only;
    }
    let level = threshold * (dot(accum.base_weights, vec4<f32>(1.0)) + dot(accum.overlay_weights, vec4<f32>(1.0)));
    var chosen = 0xFFFFFFFFu;
    for (var k = 0u; k < 4u; k++) {
        let base = accum.base_ids[k];
        if base < chosen && share_upto(accum, base) > level {
            chosen = base;
        }
        let overlay = accum.overlay_ids[k];
        if overlay < chosen && share_upto(accum, overlay) > level {
            chosen = overlay;
        }
    }
    return select(chosen, accum.base_ids.x, chosen == 0xFFFFFFFFu);
}

fn texel_position(
    world: vec4<f32>,
    surface: Surface,
    at: vec2<f32>,
    cell: vec2<i32>,
    texel: f32,
) -> vec4<f32> {
    let to_centre = (vec2<f32>(cell) + 0.5) * texel - at;
    let a = surface.ddx;
    let b = surface.ddy;
    // the screen steps in the grid's own two directions
    var a_grid = a.xz;
    var b_grid = b.xz;
    if surface.steep >= 0.5 {
        let across = select(vec2<f32>(a.x, b.x), vec2<f32>(a.z, b.z), surface.faces_x);
        a_grid = vec2<f32>(across.x, -a.y);
        b_grid = vec2<f32>(across.y, -b.y);
    }
    let det = a_grid.x * b_grid.y - a_grid.y * b_grid.x;
    if abs(det) < 1e-12 {
        return world;
    }
    let along_a = (to_centre.x * b_grid.y - to_centre.y * b_grid.x) / det;
    let along_b = (a_grid.x * to_centre.y - a_grid.y * to_centre.x) / det;
    return world + vec4<f32>(a * along_a + b * along_b, 0.0);
}

@if(ENVIRONMENT_MAP)
fn sky_irradiance(direction: vec3<f32>) -> vec3<f32> {
    var toward = quat_rotate(light_probes.view_rotation, direction);
    toward.z = -toward.z;
    @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY) {
        return textureSampleLevel(
            diffuse_environment_maps[light_probes.view_cubemap_index],
            environment_map_sampler,
            toward,
            0.0,
        ).rgb;
    } @else {
        return textureSampleLevel(diffuse_environment_map, environment_map_sampler, toward, 0.0).rgb;
    }
}

fn point_arriving(id: u32, at: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let light = &clustered_lights.data[id];
    let to_light = (*light).position_radius.xyz - at;
    let attenuation = getDistanceAttenuation(dot(to_light, to_light), (*light).color_inverse_square_range.w);
    return (*light).color_inverse_square_range.rgb * attenuation * saturate(dot(n, normalize(to_light)));
}

fn spot_cone(id: u32, at: vec3<f32>) -> f32 {
    let light = &clustered_lights.data[id];
    let packed = (*light).light_custom_data;
    var axis = vec3<f32>(packed.x, 0.0, packed.y);
    axis.y = sqrt(max(0.0, 1.0 - axis.x * axis.x - axis.z * axis.z));
    if ((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
        axis.y = -axis.y;
    }
    let toward = normalize((*light).position_radius.xyz - at);
    let cone = saturate(dot(-axis, toward) * packed.z + packed.w);
    return cone * cone;
}

fn terrain_lighting(in: pbr_types::PbrInput) -> vec3<f32> {
    let diffuse = in.material.base_color.rgb;
    let n = in.N;
    let at = in.world_position;
    let receives = (in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u;
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z,
    ), at);
    var arriving = vec3<f32>(0.0);

    for (var i = 0u; i < lights.n_directional_lights; i += 1u) {
        let light = &lights.directional_lights[i];
        let facing = saturate(dot(n, (*light).direction_to_light));
        if facing <= 0.0 {
            continue;
        }
        var shadow = 1.0;
        if receives && ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_directional_shadow(i, at, in.world_normal, view_z, in.frag_coord.xy);
        }
        arriving += (*light).color.rgb * facing * shadow;
    }

    let ranges = clustering::unpack_clusterable_object_index_ranges(
        clustering::view_fragment_cluster_index(in.frag_coord.xy, view_z, in.is_orthographic));
    for (var i = ranges.first_point_light_index_offset; i < ranges.first_spot_light_index_offset; i += 1u) {
        let id = clustering::get_clusterable_object_id(i);
        var shadow = 1.0;
        if receives && (clustered_lights.data[id].flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_point_shadow(id, at, in.world_normal, in.frag_coord.xy);
        }
        arriving += point_arriving(id, at.xyz, n) * shadow;
    }
    for (var i = ranges.first_spot_light_index_offset; i < ranges.first_rect_light_index_offset; i += 1u) {
        let id = clustering::get_clusterable_object_id(i);
        let light = &clustered_lights.data[id];
        var shadow = 1.0;
        if receives && ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_spot_shadow(id, at, in.world_normal, (*light).shadow_map_near_z, in.frag_coord.xy);
        }
        arriving += point_arriving(id, at.xyz, n) * spot_cone(id, at.xyz) * shadow;
    }

    var ambient = diffuse * lights.ambient_color.rgb * in.diffuse_occlusion;
    @if(ENVIRONMENT_MAP) {
        if light_probes.view_cubemap_index >= 0 {
            ambient += diffuse * sky_irradiance(n) * light_probes.intensity_for_view * in.diffuse_occlusion;
        }
    }
    return (diffuse * arriving / PI + ambient) * view.exposure;
}

fn bayer(cell: vec2<u32>) -> f32 {
    let lo = cell & vec2<u32>(1u);
    let hi = (cell >> vec2<u32>(1u)) & vec2<u32>(1u);
    let rank = 4u * (((lo.x ^ lo.y) << 1u) | lo.y) + (((hi.x ^ hi.y) << 1u) | hi.y);
    return (f32(rank) + 0.5) / 16.0;
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
    let c00 = words.w;
    let c10 = words.z;
    let c01 = words.x;
    let c11 = words.y;

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
    // the second argument is the threshold to use and bayer is what gives us that dither pattern.
    let along_side = select(vec2<f32>(p.x, -p.y), vec2<f32>(p.z, -p.y), surface.faces_x);
    let dither_at = select(local_xz, along_side, surface.steep >= 0.5);
    let dither_cell = vec2<i32>(floor(dither_at / texel));
    let layer_id = dithered_layer(accum, bayer(vec2<u32>(dither_cell & vec2<i32>(3))));
    let sampled = sample_layer(layer_id, surface);
    let albedo = sampled.albedo;
    let tangent_normal = sampled.normal;
    let occlusion = sampled.occlusion;

    // The mesher emits no tangents: UV0 runs along world X and Z, so the tangent
    // is world +X projected onto the surface normal. That projection collapses
    // where the surface stands vertical and faces X, so +Z stands in there.
    // Picking the axis the normal leans on least would swap the frame by 90
    // degrees wherever |n.x| and |n.z| cross, seaming flat ground.
    let lit_at = texel_position(in.world_position, surface, dither_at, dither_cell, texel);
    let lit_n = ground_normal(corner, f);
    var reference = vec3<f32>(1.0, 0.0, 0.0);
    if abs(lit_n.x) > 0.999 {
        reference = vec3<f32>(0.0, 0.0, 1.0);
    }
    let t = normalize(reference - lit_n * dot(reference, lit_n));
    let tbn = mat3x3<f32>(t, cross(t, lit_n), lit_n);
    let mapped_normal = normalize(tbn * normalize(tangent_normal));

    var pbr_input = pbr_types::pbr_input_new();
    // Mesh flags carry the shadow-receiver and transmitted-shadow-receiver
    // bits. Leaving them zero renders terrain that no shadow ever falls on.
    pbr_input.flags = mesh[in.instance_index].flags;
    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    pbr_input.diffuse_occlusion = vec3<f32>(occlusion);
    // `standard_material_new` leaves fog off; `StandardMaterial`'s Rust
    // default has it on, and terrain that ignores the scene's fog while
    // everything beside it obeys reads as a hole in the distance.
    pbr_input.material.flags |= pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;
    pbr_input.frag_coord = in.position;
    pbr_input.world_position = lit_at;
    pbr_input.world_normal = lit_n;
    pbr_input.N = mapped_normal;
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.V = pbr_functions::calculate_view(pbr_input.world_position, pbr_input.is_orthographic);

    var out: FragmentOutput;
    out.color = vec4<f32>(terrain_lighting(pbr_input), 1.0);
    out.color = pbr_functions::main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
