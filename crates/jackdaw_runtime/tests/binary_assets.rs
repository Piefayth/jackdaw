//! A document exported to the binary form still answers to the text path a
//! reference spells, through the twin-resolving asset source, so references
//! written before an export still resolve.

mod util;

use std::path::PathBuf;

use bevy::asset::{Asset, AssetApp, AssetPlugin};
use bevy::image::ImagePlugin;
use bevy::transform::TransformPlugin;
use bevy::world_serialization::WorldSerializationPlugin;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use jackdaw_runtime::{JackdawAssetSourcePlugin, JackdawPlugin, ReflectBsnAsset};

/// A reflectable stand-in for a material, so the walk has a registered asset
/// type to load without the render stack.
#[derive(Asset, Reflect, Default)]
#[reflect(Default)]
struct BinaryMaterial {
    tint: f32,
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "jackdaw-runtime-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn the_runtime_loads_a_material_held_in_the_binary_form() {
    let type_path = <BinaryMaterial as TypePath>::type_path();
    let dir = unique_temp_dir("binary-material");
    std::fs::create_dir_all(dir.join("materials")).unwrap();
    let text = jackdaw_bsn::with_asset_header(
        type_path,
        &format!("#grass\n{type_path} {{ tint: 0.5 }}\n"),
    );
    jackdaw_bsn::write_document_text(&dir.join("materials/grass.bsb"), &text).unwrap();

    let file_path = dir.to_string_lossy().into_owned();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JackdawAssetSourcePlugin {
        file_path: file_path.clone(),
        ..Default::default()
    });
    app.add_plugins(AssetPlugin {
        file_path,
        ..Default::default()
    });
    app.add_plugins(TransformPlugin);
    app.add_plugins(WorldSerializationPlugin);
    app.add_plugins(ImagePlugin::default());
    app.init_asset::<BinaryMaterial>()
        .register_asset_reflect::<BinaryMaterial>()
        .register_type_data::<BinaryMaterial, ReflectBsnAsset>();
    app.add_plugins(JackdawPlugin);

    let material: Handle<BinaryMaterial> = app
        .world()
        .resource::<AssetServer>()
        .load("materials/grass.bsn#Asset");
    let tint = util::wait_for(&mut app, "the binary material", |app| {
        app.world()
            .resource::<Assets<BinaryMaterial>>()
            .get(&material)
            .map(|material| material.tint)
    });
    assert_eq!(
        tint, 0.5,
        "the binary pair is unified"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
