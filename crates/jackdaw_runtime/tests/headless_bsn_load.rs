//! Headless load of a `.bsn` scene through the runtime asset loader: the
//! text goes through the worldless bridge and the normal spawn loop.

mod util;

use bevy::prelude::*;
use jackdaw_runtime::{JackdawPlugin, JackdawScene, JackdawSceneRoot};

#[test]
fn headless_bsn_scene_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let scene_text = r##"
bevy_ecs::hierarchy::Children [
    #Anchor
    jackdaw_scene_types::node_id::SceneNodeId(77)
    bevy_transform::components::transform::Transform {
        translation: glam::Vec3 { x: 4.0, y: 5.0, z: 6.0 },
    }
    --
    #Follower
    bevy_transform::components::transform::Transform {
        translation: glam::Vec3 { x: 9.0, y: 0.0, z: 0.0 },
    }
]
"##;
    std::fs::write(dir.path().join("scene.bsn"), scene_text).unwrap();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::transform::TransformPlugin);
    app.add_plugins(bevy::asset::AssetPlugin {
        file_path: dir.path().to_string_lossy().into_owned(),
        ..Default::default()
    });
    app.add_plugins(bevy::world_serialization::WorldSerializationPlugin);
    app.add_plugins(JackdawPlugin);

    let handle: Handle<JackdawScene> = app.world().resource::<AssetServer>().load("scene.bsn");
    app.world_mut().spawn(JackdawSceneRoot(handle));

    let translation = util::wait_for(&mut app, "the Anchor entity to load", |app| {
        let mut query = app.world_mut().query::<(&Name, &Transform)>();
        query
            .iter(app.world())
            .find(|(n, _)| n.as_str() == "Anchor")
            .map(|(_, t)| t.translation)
    });
    assert!((translation - Vec3::new(4.0, 5.0, 6.0)).length() < 1e-5);

    // The stable node id component survives (the PIE mapping depends on it).
    let mut ids = app.world_mut().query::<&jackdaw_scene_types::SceneNodeId>();
    assert!(
        ids.iter(app.world()).any(|id| id.0 == 77),
        "SceneNodeId survives the runtime .bsn load"
    );

    let mut names = app.world_mut().query::<&Name>();
    assert!(
        names.iter(app.world()).any(|n| n.as_str() == "Follower"),
        "second root loaded"
    );
}

/// A saved scene leaves out `Visibility::Inherited`, so the game gives it to a
/// placed entity that has none, and an authored visibility still wins.
#[test]
fn a_placed_entity_without_visibility_loads_inheriting_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let scene_text = r##"
bevy_ecs::hierarchy::Children [
    #Parent
    bevy_transform::components::transform::Transform
    bevy_ecs::hierarchy::Children [
        #Kid
        bevy_transform::components::transform::Transform
        bevy_camera::visibility::Visibility::Hidden
    ]
]
"##;
    std::fs::write(dir.path().join("scene.bsn"), scene_text).unwrap();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::transform::TransformPlugin);
    app.add_plugins(bevy::asset::AssetPlugin {
        file_path: dir.path().to_string_lossy().into_owned(),
        ..Default::default()
    });
    app.add_plugins(bevy::world_serialization::WorldSerializationPlugin);
    app.add_plugins(JackdawPlugin);

    let handle: Handle<JackdawScene> = app.world().resource::<AssetServer>().load("scene.bsn");
    app.world_mut().spawn(JackdawSceneRoot(handle));

    let visibility = |app: &mut App, wanted: &str| {
        let mut query = app.world_mut().query::<(&Name, Option<&Visibility>)>();
        query
            .iter(app.world())
            .find(|(name, _)| name.as_str() == wanted)
            .map(|(_, visibility)| visibility.copied())
    };
    util::wait_until(&mut app, "the Kid entity to load", |app| {
        visibility(app, "Kid").is_some()
    });
    assert_eq!(
        visibility(&mut app, "Parent"),
        Some(Some(Visibility::Inherited))
    );
    assert_eq!(visibility(&mut app, "Kid"), Some(Some(Visibility::Hidden)));
}
