use bevy::asset::LoadState;
use bevy::prelude::*;
use jackdaw_runtime::{JackdawScene, JackdawSceneRoot};
#[cfg(feature = "render")]
use jackdaw_runtime::AwaitingModelSettings;
#[cfg(feature = "render")]
use jackdaw_scene_types::model_import::ModelLodIndex;

/// spin to wait for asset loads in tests
#[expect(clippy::allow_attributes, reason = "shared in tests")]
#[allow(dead_code, reason = "shared in tests")]
pub fn wait_for<T>(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> Option<T>) -> T {
    for _ in 0..2000 {
        app.update();
        if let Some(found) = done(app) {
            return found;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}

#[expect(clippy::allow_attributes, reason = "shared in tests")]
#[allow(dead_code, reason = "shared in tests")]
pub fn wait_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    wait_for(app, what, |app| done(app).then_some(()));
}

/// update one frame after scenes are loaded so they spawn too
#[expect(clippy::allow_attributes, reason = "shared in tests")]
#[allow(dead_code, reason = "shared in tests")]
pub fn settle_scenes(app: &mut App) {
    wait_until(app, "the scenes to load", |app| {
        let handles: Vec<Handle<JackdawScene>> = app
            .world_mut()
            .query::<&JackdawSceneRoot>()
            .iter(app.world())
            .map(|root| root.0.clone())
            .collect();
        let server = app.world().resource::<AssetServer>();
        handles.iter().all(|handle| {
            server.is_loaded_with_dependencies(handle)
                || matches!(server.load_state(handle), LoadState::Failed(_))
        })
    });
    app.update();
    app.update();
}

#[cfg(feature = "render")]
#[expect(clippy::allow_attributes, reason = "shared in tests")]
#[allow(dead_code, reason = "shared in tests")]
pub fn settle_models(app: &mut App) {
    wait_until(app, "the models to settle", |app| {
        let reading = app
            .world()
            .resource::<ModelLodIndex>()
            .is_reading();
        let waiting = app
            .world_mut()
            .query_filtered::<(), With<AwaitingModelSettings>>()
            .iter(app.world())
            .next()
            .is_some();
        !reading && !waiting
    });
    app.update();
}
