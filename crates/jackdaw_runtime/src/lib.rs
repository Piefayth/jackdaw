//! Load a scene authored in the editor, in a game that does not have the
//! editor.
//!
//! A game adds [`JackdawPlugin`], points a [`JackdawSceneRoot`] at a `.bsn`
//! file, and the scene's entities spawn under that root: transforms, brushes,
//! lights, cameras, UI canvases, glTF instances, and, under the features
//! below, terrain, colliders and a baked navmesh.
//!
//! ```ignore
//! App::new()
//!     .add_plugins((DefaultPlugins, jackdaw_runtime::JackdawPlugin))
//!     .add_systems(Startup, |mut commands: Commands, assets: Res<AssetServer>| {
//!         commands.spawn(JackdawSceneRoot(assets.load("scene.bsn")));
//!     })
//!     .run();
//! ```
//!
//! # What a game provides
//!
//! - **An asset folder**, as `AssetPlugin` was configured with: the `assets/`
//!   beside the executable unless the game said otherwise. Terrain sidecars
//!   and the catalog are read from it directly rather than through the asset
//!   server. Add [`JackdawPlugin`] *after* `DefaultPlugins` so it can see how
//!   `AssetPlugin` was configured. [`JackdawCatalogPath`] overrides the
//!   location for a game that keeps its project elsewhere.
//! - **[`JackdawAssetSourcePlugin`], for a project exported to the binary
//!   form**, added *before* `DefaultPlugins`: it registers the default asset
//!   source so a `.bsn` path handed to the asset server reads the `.bsb` twin
//!   when that is the form on disk. An asset source can only be registered
//!   before `AssetPlugin`, which is why this half is its own plugin.
//! - **A camera.** Scenes usually carry one. A game that spawns its own can
//!   mark it `TerrainViewer` so terrain lays its detail around the camera
//!   the player looks through rather than a UI overlay.
//!
//! No registration call, manifest or export step: the editor and the game
//! read the same files with the same code.
//!
//! # What it reads
//!
//! - `<scene>.bsn`: the scene, its entities and their components, plus inline
//!   `#Name` assets.
//! - Every `.bsn` under the asset folder that holds a registered asset type,
//!   loaded at startup into [`JackdawCatalog`] under the path it sits at, and
//!   under the `@Name` a reference written before paths spells. `catalog.bsn`
//!   is read after them for the entries that have no file of their own.
//! - `<scene>.terrain-<n>.jdterrain`: one terrain's heights, control map,
//!   colors and material slots, named by the `Terrain` component's `data_path`
//!   relative to the scene. Read under the `terrain` feature.
//! - `<scene>.jdnav`: the navmesh baked for that scene. Read under the
//!   `navmesh` feature onto the scene root as a `JackdawNavmesh`.
//!
//! # Features
//!
//! - `render` (default): draw what the scene authored. A build without it does
//!   not compile, because `jackdaw_scene_types` registers its types through
//!   its own `render` feature.
//! - `terrain`: draw the authored terrain. Implies `render` and `navmesh`.
//! - `navmesh`: load the baked navmesh without drawing anything. A server that
//!   validates moves is `render` plus `navmesh` and no `terrain`, which leaves
//!   out the mesher, the splat material and the texture loads.
//! - `animation`: play the animation sets a scene authored, binding each to
//!   the skeleton spawned under it. Off by default: a game with no rigs links
//!   neither Bevy's animation support nor its glTF loader.
//! - `pie`: play-in-editor, streaming this world to a running editor.

use std::any::TypeId;
use std::collections::HashMap;
// Only the texture-format promotion pass gathers bound image ids, and that is
// a rendering build's concern.
#[cfg(feature = "render")]
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use bevy::asset::{AssetLoader, LoadContext, LoadState, ReflectAsset, UntypedHandle, io::Reader};
use bevy::ecs::reflect::AppTypeRegistry;
use bevy::prelude::*;
use bevy::reflect::{CreateTypeData, PartialReflect, TypeRegistry};
#[cfg(feature = "render")]
use bevy::world_serialization::{WorldAsset, WorldAssetRoot};
use jackdaw_bsn::{
    BsnApplyAssets, BsnPatch, BsnSceneAssets, BsnValue, DOCUMENT_ASSET_LABEL, PREFAB_TYPE,
    SceneBsnAst, apply_component_patch, asset_value_from_root, bsn_value_to_reflect,
    document_text_from_bytes, emit_scene, is_asset_root, is_document_path, parse_bsn_text,
    read_asset_header, reject_retired_ui_components, root_type_path, server_loads,
};
use jackdaw_prefab::{
    ISA_TYPE, isa_value, normalize_as_prefab_source, read_isa_deleted, read_isa_source,
    resolve_scene, set_whole_component,
};
#[cfg(feature = "render")]
use jackdaw_scene_types::{render_assets::drawn_texture_settings, to_asset_path};
#[cfg(feature = "render")]
use jackdaw_surface::{FoliageMaterial, LayeredSurfaceMaterial, WaterMaterial};
#[cfg(feature = "navmesh")]
use jackdaw_terrain::navmesh::NavmeshArtifact;

pub use jackdaw_scene_types::{
    Brush, BrushFaceData, CustomProperties, DetailPresser, EditorCategory, EditorDescription,
    EditorHidden, EditorPreview, GltfSource, HIDDEN_IN_GAME_TYPE_PATH, HiddenInGame,
    NAVMESH_EXCLUDE_TYPE_PATH, NavmeshExclude, PropertyValue, ScatterGroup, ScatterInstance,
    SkipSerialization,
};

#[cfg(feature = "pie")]
mod pie;
#[cfg(feature = "pie")]
mod pie_frames;
#[cfg(feature = "pie")]
mod pie_windowless;
#[cfg(feature = "pie")]
pub use pie_windowless::{maybe_windowless, windowless_requested};

pub mod frame_work;
#[cfg(feature = "render")]
mod live_levels;
#[cfg(feature = "render")]
mod lod;
#[cfg(feature = "render")]
pub use live_levels::{
    AwaitingModelSettings, LiveLevelProgress, LiveLevelSettings, LiveLevels, LiveLevelsPlugin,
    LodPart, LodPartLevel, ModelPlacement, drawn_ranges, is_level_of, is_lod_level,
    model_placement,
};
#[cfg(feature = "render")]
pub use lod::{
    ForcedLod, LodPlugin, LodQuality, LodSwitches, LodView, level_shows, lod_distance, lod_ranges,
};
#[cfg(feature = "render")]
mod material_overrides;
#[cfg(feature = "render")]
pub use material_overrides::{
    MaterialOverridesPlugin, ModelMaterial, dress_model, dress_part, material_of_reference,
    overrides_reaching,
};

#[cfg(feature = "terrain")]
mod terrain;
#[cfg(feature = "terrain")]
pub use jackdaw_terrain::render::{DetailPressers, DetailSettings, DetailViewer};
#[cfg(feature = "terrain")]
pub use terrain::{TerrainHeights, TerrainViewer};

#[cfg(feature = "navmesh")]
mod navmesh;
#[cfg(feature = "navmesh")]
pub use navmesh::JackdawNavmesh;

mod hidden_in_game;
#[cfg(feature = "import")]
pub mod texture_import;
mod twin;
pub use twin::{DocumentTwinReader, JackdawAssetSourcePlugin, with_document_twins};

mod schema_cli;
pub use schema_cli::{
    SCHEMA_FLAG, extract_schema_and_exit_if_requested, extract_schema_from_world,
    extract_schema_json, schema_extraction_requested,
};

pub mod prelude {
    #[cfg(feature = "navmesh")]
    pub use crate::JackdawNavmesh;
    pub use crate::{
        DetailPresser, EditorCategory, EditorDescription, EditorHidden, EditorPreview,
        DocumentAsset, JackdawAssetSourcePlugin, JackdawDocuments, JackdawPlugin, JackdawSceneMember,
        JackdawSceneRoot, ReflectBsnAsset, SceneRefused, SkipSerialization,
    };
    #[cfg(feature = "terrain")]
    pub use crate::{DetailPressers, DetailSettings, DetailViewer, TerrainViewer};
}

/// Marks the transform trees of entities spawned into a freed entity index as changed, which
/// Bevy's static transform optimization can miss, leaving the entity with a stale `GlobalTransform`.
pub fn propagate_into_reused_indices(app: &mut App) {
    if app.world().contains_resource::<ReusedIndexTracking>() {
        return;
    }
    app.init_resource::<ReusedIndexTracking>().add_systems(
        PostUpdate,
        mark_reused_index_trees
            .in_set(bevy::transform::TransformSystems::Propagate)
            .before(bevy::transform::systems::mark_dirty_trees),
    );
}

/// Entities whose parent link was removed in the previous frame, checked again for reuse in case
/// their index was taken after that frame's check.
#[derive(Resource, Default)]
struct ReusedIndexTracking {
    previous: Vec<Entity>,
}

fn mark_reused_index_trees(
    mut removed: RemovedComponents<ChildOf>,
    mut tracking: ResMut<ReusedIndexTracking>,
    mut marked: Local<bevy::ecs::entity::EntityHashSet>,
    entities: &bevy::ecs::entity::Entities,
    parents: Query<&ChildOf>,
    mut trees: Query<&mut bevy::transform::components::TransformTreeChanged>,
) {
    let current: Vec<Entity> = removed.read().collect();
    marked.clear();
    for freed in tracking.previous.iter().chain(&current) {
        let mut next = entities.resolve_from_index(freed.index());
        if next == *freed {
            continue;
        }
        while marked.insert(next) {
            let Ok(mut tree) = trees.get_mut(next) else {
                break;
            };
            tree.set_changed();
            let Ok(child_of) = parents.get(next) else {
                break;
            };
            next = child_of.parent();
        }
    }
    tracking.previous = current;
}

pub struct JackdawPlugin;

impl Plugin for JackdawPlugin {
    fn build(&self, app: &mut App) {
        // The editor asks for this binary's reflected types by launching
        // it with `--jackdaw-extract-schema`. The dump waits for the rest of
        // the app to finish building, where the game's events and functions get
        // registered, and exits before `App::run` opens a window.
        schema_cli::extract_schema_and_exit_if_requested(app);

        propagate_into_reused_indices(app);

        #[cfg(feature = "import")]
        texture_import::register_texture_import_processor(app);

        // Registers every scene type for reflection and installs
        // `MeshRebuildPlugin` (which embeds the bundled grid texture
        // used as the brush fallback material).
        app.add_plugins(jackdaw_scene_types::SceneTypesPlugin {
            runtime_mesh_rebuild: true,
        });

        app.add_plugins(jackdaw_prefab::PrefabTypesPlugin);

        // The whole plugin, unlike the editor's slim registration: a game has
        // no preview mode to gate evaluation behind.
        app.add_plugins(jackdaw_bind::JackdawBindPlugin);

        // `bevy_ui_widgets` supplies neither the reflected defaults a load
        // rebuilds markers from nor the observers a click goes through.
        app.add_plugins(jackdaw_widgets_runtime::AuthoredWidgetPlugin);

        // Neither crate can name the other's half of a text binding, so this
        // is where the two meet: where a string binding writes, and which half
        // runs first.
        app.insert_resource(jackdaw_bind::ValueTextTarget(jackdaw_bind::BindPath::new(
            jackdaw_widgets_runtime::text_value_write_path(),
        )));
        app.configure_sets(
            PostUpdate,
            jackdaw_widgets_runtime::AuthoredTextSystems.after(jackdaw_bind::BindEvaluationSystems),
        );
        // The same edge for the derived `Node` values, or a bound progress bar
        // is always a frame behind the number beside it.
        app.configure_sets(
            PostUpdate,
            jackdaw_widgets_runtime::AuthoredNodeSystems.after(jackdaw_bind::BindEvaluationSystems),
        );
        // And for the chrome a bound list or choice is drawn from.
        app.configure_sets(
            PostUpdate,
            jackdaw_widgets_runtime::AuthoredChromeSystems
                .after(jackdaw_bind::BindEvaluationSystems),
        );

        app.init_asset::<JackdawScene>()
            .init_asset_loader::<JackdawSceneLoader>()
            .init_resource::<JackdawDocuments>();

        app.add_systems(
            Update,
            (
                clear_modified_scene_roots,
                spawn_loaded_scenes,
                cleanup_orphaned_scene_members,
            )
                .chain(),
        );
        app.add_observer(hidden_in_game::hide_on_add);

        #[cfg(feature = "render")]
        app.add_plugins((
            MaterialTextureFormatPlugin,
            jackdaw_surface::LayeredSurfacePlugin,
            jackdaw_surface::FoliagePlugin,
            jackdaw_surface::WaterPlugin,
            jackdaw_surface::EnvironmentPlugin,
            jackdaw_surface::ReflectionProbePlugin,
            jackdaw_surface::ShadowViewCleanupPlugin,
            MaterialOverridesPlugin,
            LodPlugin,
        ));
        #[cfg(feature = "render")]
        app.register_type::<StandardMaterial>()
            .register_type_data::<StandardMaterial, ReflectBsnAsset>()
            .register_type::<LayeredSurfaceMaterial>()
            .register_type_data::<LayeredSurfaceMaterial, ReflectBsnAsset>()
            .register_type::<FoliageMaterial>()
            .register_type_data::<FoliageMaterial, ReflectBsnAsset>()
            .register_type::<WaterMaterial>()
            .register_type_data::<WaterMaterial, ReflectBsnAsset>();

        // Game code may add a model to an entity long after the scene it lives
        // in was loaded, and such a source resolves exactly as an authored one.
        #[cfg(feature = "render")]
        app.add_systems(
            Update,
            (
                attach_inserted_gltf_sources,
                attach_models_left_by_lod_groups,
                attach_awaited_models,
            )
                .chain()
                .after(spawn_loaded_scenes),
        );

        #[cfg(feature = "terrain")]
        app.add_plugins(terrain::plugin);

        // Registers the authored animation types as well as playing them, so
        // a set survives a load in the same build that runs it.
        #[cfg(feature = "animation")]
        app.add_plugins(jackdaw_animation_runtime::AnimationRuntimePlugin);

        // When `JACKDAW_PIE` is set, open the ipc-channel link to the editor
        // and attach the PIE stream / control systems. A connect failure logs
        // and leaves the runtime untouched.
        #[cfg(feature = "pie")]
        if let Some(cfg) = pie::pie_config() {
            match jackdaw_pie_protocol::connect(&cfg.server) {
                Ok(transport) => pie::attach_pie(app, transport),
                Err(err) => bevy::log::error!("PIE connect failed: {err}"),
            }
        }
    }
}

/// fork - JackdawDocuments is effectively cache of bsn documents keyed by reference
#[derive(Resource, Default)]
pub struct JackdawDocuments(HashMap<String, Handle<JackdawScene>>);

impl JackdawDocuments {

    pub fn asset(
        &mut self,
        server: &AssetServer,
        documents: &Assets<JackdawScene>,
        reference: &str,
    ) -> DocumentAsset {
        if !is_document_path(Path::new(reference)) {
            return DocumentAsset::Missing;
        }
        let handle = self
            .0
            .entry(reference.to_owned())
            .or_insert_with(|| server.load(reference.to_owned()));
        if let Some(document) = documents.get(&*handle) {
            return document
                .asset
                .clone()
                .map_or(DocumentAsset::Missing, DocumentAsset::Held);
        }
        match server.load_state(&*handle) {
            LoadState::Failed(_) => DocumentAsset::Missing,
            _ => DocumentAsset::Loading,
        }
    }
}

/// fork - state machine for document loading state
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DocumentAsset {
    /// The document has not finished loading.
    Loading,
    /// The asset the document holds.
    Held(UntypedHandle),
    /// The reference names no document, one that failed to load, or one that
    /// holds no asset.
    Missing,
}

/// fork - for referencing our bsn #Asset labels we need to reflect out the "real"
/// underlying types for them to resolve to
#[derive(Clone)]
pub struct ReflectBsnAsset {
    add: fn(
        &mut LoadContext,
        &mut dyn FnMut(&mut LoadContext) -> Option<Box<dyn PartialReflect>>,
    ) -> Option<UntypedHandle>,
}

impl<A: Asset + FromReflect> CreateTypeData<A> for ReflectBsnAsset {
    fn create_type_data(_input: ()) -> Self {
        Self {
            add: add_document_asset::<A>,
        }
    }
}

fn add_document_asset<A: Asset + FromReflect>(
    load_context: &mut LoadContext,
    value: &mut dyn FnMut(&mut LoadContext) -> Option<Box<dyn PartialReflect>>,
) -> Option<UntypedHandle> {
    load_context
        .labeled_asset_scope(DOCUMENT_ASSET_LABEL, |load_context| {
            let value = value(load_context).ok_or(())?;
            A::from_reflect(value.as_ref()).ok_or(())
        })
        .ok()
        .map(Handle::untyped)
}

/// fork - this is still a "loaded bsn scene" but now that loading is async 
/// the data we are loading needs somewhere to be for now; might as well stick it here
#[derive(Asset, TypePath, Clone)]
pub struct JackdawScene {
    bsn: String,
    parent_path: PathBuf,
    /// Source file stem (`starter` from `zones/starter.bsn`), captured at
    /// load time. Used to give the spawned scene root a readable `Name` so
    /// the editor's Live tree shows the scene name instead of an entity id.
    /// `None` when the asset was built without a source path.
    stem: Option<String>,
    asset: Option<UntypedHandle>,
    #[cfg(feature = "terrain")]
    terrain: terrain::TerrainFiles,
    #[cfg(feature = "navmesh")]
    navmesh: Option<Result<NavmeshArtifact, String>>,
}

impl JackdawScene {
    /// Build a scene asset directly from in-memory `.bsn` text.
    pub fn new(bsn: String, parent_path: PathBuf) -> Self {
        Self {
            bsn,
            parent_path,
            stem: None,
            asset: None,
            #[cfg(feature = "terrain")]
            terrain: default(),
            #[cfg(feature = "navmesh")]
            navmesh: None,
        }
    }

    fn holding(self, asset: Option<UntypedHandle>) -> Self {
        Self { asset, ..self }
    }

    /// Like [`JackdawScene::new`] but with an explicit source file stem, so
    /// tests can exercise the root-naming path without a real `.bsn` file.
    pub fn with_stem(bsn: String, parent_path: PathBuf, stem: Option<String>) -> Self {
        Self {
            bsn,
            parent_path,
            stem,
            asset: None,
            #[cfg(feature = "terrain")]
            terrain: default(),
            #[cfg(feature = "navmesh")]
            navmesh: None,
        }
    }
}

/// Scene entities spawn as children of this root.
///
/// Requires `Transform` and `Visibility` so the hierarchy has a
/// propagation backbone (otherwise every child would have
/// `GlobalTransform`/`InheritedVisibility` but no upstream
/// chain, triggering Bevy B0004 warnings and silently breaking
/// rendering). Callers can spawn `JackdawSceneRoot(handle)` by
/// itself; Bevy fills in the requires.
#[derive(Component, Deref)]
#[require(Transform, Visibility, SceneInstanceMembers)]
pub struct JackdawSceneRoot(pub Handle<JackdawScene>);

/// Associates a top-level spawned entity with its scene instance independently
/// from the ECS hierarchy.
///
/// UI canvases must remain ECS roots for Bevy layout, so [`ChildOf`] cannot be
/// used as their ownership relation.
///
/// A loaded scene's roots are not children of the entity carrying
/// [`JackdawSceneRoot`], so a `jackdaw_bind::BindContext` put on the handle
/// entity is inherited by nothing. Mark the document's root with a component of
/// the game's own and insert the context on that entity once the scene has
/// spawned.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct JackdawSceneMember {
    /// The [`JackdawSceneRoot`] that owns this entity.
    pub root: Entity,
}

#[derive(Component, Default)]
struct SceneInstanceMembers(Vec<Entity>);

#[derive(Component)]
struct SceneSpawned;

/// Marks a scene root the loader would not spawn.
///
/// A refused scene and a scene that held nothing both leave the root with no
/// children, and this tells the two apart; the reason is in the log. It
/// describes one load attempt, and the next load of a corrected file removes
/// it.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneRefused;

#[derive(TypePath)]
struct JackdawSceneLoader {
    registry: AppTypeRegistry,
}

impl FromWorld for JackdawSceneLoader {
    fn from_world(world: &mut World) -> Self {
        Self {
            registry: world.resource::<AppTypeRegistry>().clone(),
        }
    }
}

impl AssetLoader for JackdawSceneLoader {
    type Asset = JackdawScene;
    type Settings = ();
    type Error = JackdawLoadError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|e| JackdawLoadError::Io(e.to_string()))?;

        let text = document_text_from_bytes(&bytes, load_context.path().path())
            .map_err(|e| JackdawLoadError::Parse(e.to_string()))?;

        let source_path = load_context.path().path();
        let parent_path = source_path.parent().unwrap_or(Path::new("")).to_owned();
        let stem = source_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned());
        let scene_name = load_context.path().to_string();


        let (asset, sources) = {
            let mut ast = parse_scene(&text)?;
            let asset = self.document_asset(&ast, &text, load_context);
            // a file holding an asset isn't a scene
            if asset.is_some() {
                return Ok(JackdawScene::with_stem(text, parent_path, stem).holding(asset));
            }
            let sources = match normalize_isa_sources(&mut ast) {
                Ok(()) => isa_sources(&ast),
                Err(outside) => {
                    warn!(
                        "Scene '{scene_name}' references the prefab {} from outside the asset \
                         root; refusing to read it, and spawning the scene as authored",
                        outside.display()
                    );
                    Vec::new()
                }
            };
            (asset, sources)
        };
        let prefabs = read_prefabs(sources, &scene_name, load_context).await;

        #[cfg(feature = "terrain")]
        let terrain_paths;
        let bsn = {
            let ast = resolve_prefab_references(parse_scene(&text)?, &scene_name, &prefabs);
            reject_retired_ui_components(&ast)
                .map_err(|e| JackdawLoadError::RetiredComponents(e.to_string()))?;
            #[cfg(feature = "terrain")]
            {
                terrain_paths = terrain::terrain_data_paths(&ast);
            }
            if prefabs.is_empty() {
                text
            } else {
                emit_scene(&ast)
            }
        };

        Ok(JackdawScene {
            #[cfg(feature = "terrain")]
            terrain: terrain::read_terrain_files(terrain_paths, &parent_path, load_context).await,
            #[cfg(feature = "navmesh")]
            navmesh: navmesh::read_navmesh(&parent_path, stem.as_deref(), load_context).await,
            bsn,
            parent_path,
            stem,
            asset,
        })
    }

    fn extensions(&self) -> &[&str] {
        &["bsn", "bsb"]
    }
}

impl JackdawSceneLoader {
    /// fork - actually build and load the underlying asset
    fn document_asset(
        &self,
        ast: &SceneBsnAst,
        text: &str,
        load_context: &mut LoadContext,
    ) -> Option<UntypedHandle> {
        let root = *ast.roots.first()?;
        if !ast.get_children_ast(root).is_empty()
            || ast
                .find_patch_by_type_path(root, PREFAB_TYPE)
                .is_some()
        {
            return None;
        }
        let type_path = root_type_path(ast, root)
            .or_else(|| read_asset_header(text))?;
        let reg = self.registry.read();
        let registration = reg.get_with_type_path(&type_path)?;
        let Some(bsn_asset) = registration.data::<ReflectBsnAsset>().cloned() else {
            if registration.data::<ReflectAsset>().is_some() {
                warn!(
                    "{} holds a {type_path}, which this app cannot build from a file; \
                     register it with `register_type_data::<_, ReflectBsnAsset>()`",
                    load_context.path()
                );
            }
            return None;
        };
        let type_id = registration.type_id();
        let (_, value) = asset_value_from_root(ast, root)?;
        if ast.roots.len() > 1 {
            warn!(
                "{} holds {} values; only the first answers to its path",
                load_context.path(),
                ast.roots.len()
            );
        }
        (bsn_asset.add)(load_context, &mut |load_context| {
            let load_context = std::cell::RefCell::new(load_context);
            #[cfg(feature = "render")]
            let _preloaded_textures = preload_textures(ast, |path, linear| {
                load_context
                    .borrow_mut()
                    .load_builder()
                    .with_settings(drawn_texture_settings(
                        linear,
                    ))
                    .load::<Image>(path.to_owned())
                    .untyped()
            });
            let load = |type_id, path: &str| {
                load_context
                    .borrow_mut()
                    .load_builder()
                    .load_erased(type_id, path.to_owned())
            };
            let assets = BsnApplyAssets {
                load: &load,
                local: None,
            };
            bsn_value_to_reflect(&value, type_id, &reg, Some(&assets))
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JackdawLoadError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
    /// The document names components the engine does not have, so loading it
    /// would hand the game a scene missing part of itself.
    #[error("{0}")]
    RetiredComponents(String),
}

/// On `JackdawScene` change, despawn the previously-spawned
/// children and clear `SceneSpawned` so the next
/// `spawn_loaded_scenes` tick re-instantiates from the new
/// asset content. Pair with Bevy's `file_watcher` feature to get
/// hot reload of `assets/scene.bsn` in the standalone game binary.
fn clear_modified_scene_roots(
    mut events: bevy::ecs::message::MessageReader<bevy::asset::AssetEvent<JackdawScene>>,
    roots: Query<(Entity, &JackdawSceneRoot, &SceneInstanceMembers), With<SceneSpawned>>,
    mut commands: Commands,
) {
    use bevy::asset::AssetEvent;

    let modified: Vec<bevy::asset::AssetId<JackdawScene>> = events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } | AssetEvent::LoadedWithDependencies { id } => Some(*id),
            _ => None,
        })
        .collect();
    if modified.is_empty() {
        return;
    }

    for (root_entity, root, members) in &roots {
        if !modified.contains(&root.0.id()) {
            continue;
        }
        for &member in &members.0 {
            commands.entity(member).despawn();
        }
        commands
            .entity(root_entity)
            .remove::<SceneSpawned>()
            .insert(SceneInstanceMembers::default());
    }
}

/// Why the asset server gave up on a scene, when it has. A load still in
/// progress answers `None`.
fn scene_load_failure(world: &World, handle: &Handle<JackdawScene>) -> Option<String> {
    match world.get_resource::<AssetServer>()?.get_load_state(handle) {
        Some(LoadState::Failed(err)) => Some(err.to_string()),
        _ => None,
    }
}

fn cleanup_orphaned_scene_members(
    members: Query<(Entity, &JackdawSceneMember)>,
    roots: Query<(), With<JackdawSceneRoot>>,
    mut commands: Commands,
) {
    for (entity, member) in &members {
        if roots.get(member.root).is_err() {
            commands.entity(entity).despawn();
        }
    }
}

pub(crate) fn spawn_loaded_scenes(
    world: &mut World,
    scene_roots: &mut QueryState<(Entity, &JackdawSceneRoot), Without<SceneSpawned>>,
) {
    let to_spawn: Vec<(Entity, Handle<JackdawScene>)> = scene_roots
        .iter(world)
        .map(|(e, root)| (e, root.0.clone()))
        .collect();

    for (root_entity, handle) in to_spawn {
        // The reload path removes `SceneSpawned` when the document changes, so
        // the previous attempt's verdict does not carry over.
        if world.get::<SceneRefused>(root_entity).is_some() {
            world.entity_mut(root_entity).remove::<SceneRefused>();
        }

        let loaded = world
            .resource::<Assets<JackdawScene>>()
            .get(&handle)
            .cloned();
        let Some(scene) = loaded else {
            // A document the asset loader refuses never reaches the gates
            // below, so the root would otherwise sit empty with nothing to say
            // why.
            if let Some(err) = scene_load_failure(world, &handle) {
                warn!("Cannot spawn scene: {err}");
                world
                    .entity_mut(root_entity)
                    .insert((SceneSpawned, SceneRefused));
            }
            continue;
        };

        let JackdawScene {
            bsn,
            parent_path,
            stem,
            ..
        } = &scene;
        let ast = match parse_bsn_text(bsn) {
            Ok(ast) => ast,
            Err(err) => {
                warn!("Failed to parse scene .bsn: {err}");
                world
                    .entity_mut(root_entity)
                    .insert((SceneSpawned, SceneRefused));
                continue;
            }
        };

        let scene_name = match stem.as_deref().filter(|s| !s.is_empty()) {
            Some(stem) => parent_path
                .join(format!("{stem}.bsn"))
                .display()
                .to_string(),
            None => "a scene built in memory".to_string(),
        };
        // Spawning is the one place every route meets, including in-memory
        // text through `JackdawScene::new`, and it reads the resolved document
        // because a prefab base can hand retired vocabulary to an instance
        // whose own file never names it.
        if let Err(err) = reject_retired_ui_components(&ast) {
            warn!("Cannot spawn scene '{scene_name}': {err}");
            world
                .entity_mut(root_entity)
                .insert((SceneSpawned, SceneRefused));
            continue;
        }

        let members = spawn_scene_entities(world, root_entity, &ast, &scene);
        world
            .entity_mut(root_entity)
            .insert(SceneInstanceMembers(members));

        // A bake is saved under the scene's own name, which is known here.
        #[cfg(feature = "navmesh")]
        navmesh::attach_navmesh(world, root_entity, &scene.navmesh);

        // Give the container root a readable name from the scene's file
        // stem so the editor's Live tree shows the scene name instead of an
        // entity id. Never overwrite an author-supplied name, and never
        // insert an empty one.
        if world.get::<Name>(root_entity).is_none()
            && let Some(stem) = stem.clone().filter(|s| !s.is_empty())
        {
            world.entity_mut(root_entity).insert(Name::new(stem));
        }

        // Tag the container so the editor's outliner classifies it as a scene
        // root and shows the scene icon. The tag streams in the snapshot.
        if world
            .get::<jackdaw_scene_types::SceneRootTag>(root_entity)
            .is_none()
        {
            world
                .entity_mut(root_entity)
                .insert(jackdaw_scene_types::SceneRootTag);
        }

        world.entity_mut(root_entity).insert(SceneSpawned);
    }
}

fn parse_scene(text: &str) -> Result<SceneBsnAst, JackdawLoadError> {
    parse_bsn_text(text).map_err(|e| JackdawLoadError::Parse(e.to_string()))
}

pub(crate) fn asset_root_source(source: &Path) -> Option<PathBuf> {
    use std::path::Component;
    let mut folded = PathBuf::new();
    for component in source.components() {
        match component {
            Component::Normal(part) => folded.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                if !folded.pop() {
                    return None;
                }
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(folded)
}

/// fork - prefab paths can be written different ways so we make a canonical one
fn normalize_isa_sources(ast: &mut SceneBsnAst) -> Result<(), PathBuf> {
    for node in ast.entities_with_component(ISA_TYPE) {
        let Some(source) = read_isa_source(ast, node) else {
            continue;
        };
        let folded = asset_root_source(&source).ok_or_else(|| source.clone())?;
        if folded != source {
            let deleted = read_isa_deleted(ast, node);
            set_whole_component(
                ast,
                node,
                ISA_TYPE,
                isa_value(&folded.to_string_lossy(), &deleted),
            );
        }
    }
    Ok(())
}

enum PrefabRead {
    Unreadable(String),
    Outside(PathBuf),
}

impl std::fmt::Display for PrefabRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreadable(err) => f.write_str(err),
            Self::Outside(source) => write!(
                f,
                "it references {} from outside the asset root",
                source.display()
            ),
        }
    }
}

pub(crate) fn parse_prefab(path: &Path, text: &str) -> Result<SceneBsnAst, PrefabRead> {
    let mut ast = parse_bsn_text(text).map_err(|err| PrefabRead::Unreadable(err.to_string()))?;
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
    normalize_as_prefab_source(&mut ast, name);
    normalize_isa_sources(&mut ast).map_err(PrefabRead::Outside)?;
    Ok(ast)
}

async fn read_prefabs(
    mut pending: Vec<PathBuf>,
    scene_name: &str,
    load_context: &mut LoadContext<'_>,
) -> HashMap<PathBuf, String> {
    let mut texts = HashMap::new();
    while let Some(path) = pending.pop() {
        if texts.contains_key(&path) {
            continue;
        }
        let text = match load_context.read_asset_bytes(path.clone()).await {
            Ok(bytes) => document_text_from_bytes(&bytes, &path)
                .map_err(|err| PrefabRead::Unreadable(err.to_string())),
            Err(err) => Err(PrefabRead::Unreadable(err.to_string())),
        };
        let read = text.and_then(|text| {
            let sources = isa_sources(&parse_prefab(&path, &text)?);
            Ok((text, sources))
        });
        match read {
            Ok((text, sources)) => {
                pending.extend(sources);
                texts.insert(path, text);
            }
            Err(PrefabRead::Unreadable(err)) => warn!(
                "Scene '{scene_name}': failed to read prefab {}: {err}",
                path.display()
            ),
            Err(PrefabRead::Outside(outside)) => {
                warn!(
                    "Scene '{scene_name}' references the prefab {} from outside the asset \
                     root",
                    outside.display()
                );
                return HashMap::new();
            }
        }
    }
    texts
}

/// Materialize the prefab instances a scene document names.
///
/// A scene stores an instance as an `IsA` pointing at another document plus the
/// fields that differ from what it inherits; this turns that reference into the
/// entities it stands for before anything spawns.
///
/// Fail-soft: a scene whose references cannot be followed spawns as authored,
/// with the reason and the scene named. A game follows references only inside
/// its own asset root, since a reference is an instruction to open whatever it
/// names and a scene file is content a player can replace.
fn resolve_prefab_references(
    mut ast: SceneBsnAst,
    scene_name: &str,
    prefabs: &HashMap<PathBuf, String>,
) -> SceneBsnAst {
    if ast
        .entities_with_component(ISA_TYPE)
        .is_empty()
    {
        return ast;
    }
    if normalize_isa_sources(&mut ast).is_err() {
        return ast;
    }
    let documents: HashMap<PathBuf, SceneBsnAst> = prefabs
        .iter()
        .filter_map(|(path, text)| Some((path.clone(), parse_prefab(path, text).ok()?)))
        .collect();
    let get_prefab = |path: &Path| documents.get(path);
    match resolve_scene(&ast, &get_prefab) {
        Ok(resolved) => resolved,
        Err(err) => {
            warn!(
                "Prefab resolution failed for scene '{scene_name}': {err}; spawning it as authored"
            );
            ast
        }
    }
}

/// The sources every `IsA` in `ast` names, meaningful only once the document
/// has been through [`normalize_isa_sources`].
fn isa_sources(ast: &SceneBsnAst) -> Vec<PathBuf> {
    ast.entities_with_component(ISA_TYPE)
        .into_iter()
        .filter_map(|node| read_isa_source(ast, node))
        .collect()
}

/// Spawn a scene document's entities under `root_entity`.
///
/// Embedded named-asset roots load into their `Assets<T>` stores first (so
/// `#Name` references resolve), then entity roots spawn parent-first
/// by walking `roots` and each node's `Children` relation.
fn spawn_scene_entities(
    world: &mut World,
    root_entity: Entity,
    ast: &SceneBsnAst,
    scene: &JackdawScene,
) -> Vec<Entity> {
    let registry = world.resource::<AppTypeRegistry>().clone();

    #[cfg(feature = "render")]
    let _preloaded_textures = {
        let server = world.resource::<AssetServer>();
        preload_textures(ast, |path, linear| {
            server
                .load_builder()
                .with_settings(drawn_texture_settings(
                    linear,
                ))
                .load::<Image>(path.to_owned())
                .untyped()
        })
    };

    // Embedded assets keyed as `#Name`, kept in `BsnSceneAssets` so
    // `apply_component_patch` resolves reference strings. A reference to a
    // file loads through the asset server instead.
    let local_assets = load_embedded_assets(world, ast, &registry);
    let mut scene_assets = bevy::platform::collections::HashMap::default();
    for (name, handle) in &local_assets {
        scene_assets.insert(name.clone(), handle.clone());
    }
    world.insert_resource(BsnSceneAssets(scene_assets));

    let mut spawned: Vec<Entity> = Vec::new();
    let mut members: Vec<Entity> = Vec::new();
    for root in ast.roots.clone() {
        let is_asset = {
            let reg = registry.read();
            is_asset_root(ast, root, &reg)
        };
        if is_asset {
            continue;
        }
        if let Some(member) = spawn_node(
            world,
            ast,
            root,
            root_entity,
            root_entity,
            true,
            &registry,
            &mut spawned,
        ) {
            members.push(member);
        }
    }

    #[cfg(feature = "render")]
    {
        let asset_server = world.resource::<AssetServer>().clone();
        let gltf_entities: Vec<(Entity, jackdaw_scene_types::GltfSource)> = spawned
            .iter()
            .filter_map(|&e| {
                world
                    .get::<jackdaw_scene_types::GltfSource>(e)
                    .map(|source| (e, source.clone()))
            })
            .collect();
        for (entity, source) in gltf_entities {
            match live_levels::model_placement(world, entity) {
                live_levels::ModelPlacement::Levels => {}
                live_levels::ModelPlacement::Waiting => {
                    world
                        .entity_mut(entity)
                        .insert(live_levels::AwaitingModelSettings);
                }
                live_levels::ModelPlacement::Instance => {
                    let root = world_asset_root(
                        &asset_server,
                        world.get_resource(),
                        &source,
                    );
                    world.entity_mut(entity).insert(root);
                }
            }
        }
    }
    #[cfg(feature = "terrain")]
    terrain::attach_sidecars(world, &spawned, &scene.terrain);

    #[cfg(not(feature = "terrain"))]
    let _ = scene;

    members
}

/// The glTF scene an authored [`GltfSource`] names, as the handle the
/// world-asset spawner instantiates.
///
/// The path is read the way the editor reads it, from the assets root rather
/// than from beside the scene file, so a prefab stored in a subdirectory shows
/// the same model in a built game that it showed while it was authored.
///
/// [`GltfSource`]: jackdaw_scene_types::GltfSource
#[cfg(feature = "render")]
fn world_asset_root(
    asset_server: &AssetServer,
    imports: Option<&jackdaw_scene_types::model_import::ModelLodIndex>,
    source: &jackdaw_scene_types::GltfSource,
) -> WorldAssetRoot {
    let path = to_asset_path(&source.path, None);
    let scene: Handle<WorldAsset> = jackdaw_scene_types::render_assets::load_model(
        asset_server,
        imports,
        format!("{path}#Scene{}", source.scene_index),
    );
    WorldAssetRoot(scene)
}

/// Give a [`GltfSource`] inserted by game code the same `WorldAssetRoot` a
/// scene-authored one gets at spawn.
///
/// Scene loading attaches the handle as it spawns, so this finds only sources
/// that arrived some other way; re-inserting an equal handle would make the
/// world-asset spawner despawn and rebuild the instance, so an entity that
/// already points at the same glTF scene is left alone.
///
/// [`GltfSource`]: jackdaw_scene_types::GltfSource
#[cfg(feature = "render")]
fn attach_inserted_gltf_sources(
    added: Query<Entity, Added<jackdaw_scene_types::GltfSource>>,
    mut commands: Commands,
) {
    for entity in &added {
        commands.queue(move |world: &mut World| attach_model(world, entity));
    }
}

/// Bring on the models of children a LOD group no longer keeps as levels:
/// those past its last level after a change, and all of them once the group is
/// gone.
#[cfg(feature = "render")]
fn attach_models_left_by_lod_groups(
    groups: Query<
        (&jackdaw_scene_types::LodGroup, &Children),
        Changed<jackdaw_scene_types::LodGroup>,
    >,
    mut removed: RemovedComponents<jackdaw_scene_types::LodGroup>,
    mut unleveled: RemovedComponents<jackdaw_scene_types::model_import::ModelLevels>,
    children_of: Query<&Children>,
    unplaced: Query<
        (),
        (
            With<jackdaw_scene_types::GltfSource>,
            Without<WorldAssetRoot>,
        ),
    >,
    mut commands: Commands,
) {
    let past_last_level = groups
        .iter()
        .flat_map(|(group, children)| children.iter().skip(group.levels.len()));
    let ungrouped: Vec<Entity> = removed
        .read()
        .flat_map(|group| {
            let levels = children_of
                .get(group)
                .into_iter()
                .flat_map(RelationshipTarget::iter);
            std::iter::once(group).chain(levels)
        })
        .chain(unleveled.read())
        .collect();
    for model in past_last_level.chain(ungrouped) {
        if unplaced.contains(model) {
            commands.queue(move |world: &mut World| attach_model(world, model));
        }
    }
}

/// Give the models that waited for their import settings an instance, now
/// the settings are read, unless they draw live levels instead.
#[cfg(feature = "render")]
fn attach_awaited_models(
    awaiting: Query<
        (Entity, &jackdaw_scene_types::GltfSource),
        With<live_levels::AwaitingModelSettings>,
    >,
    index: Res<jackdaw_scene_types::model_import::ModelLodIndex>,
    mut commands: Commands,
) {
    for (entity, source) in &awaiting {
        if !index.is_known(&jackdaw_scene_types::model_parts::source_path(source)) {
            continue;
        }
        commands
            .entity(entity)
            .remove::<live_levels::AwaitingModelSettings>();
        commands.queue(move |world: &mut World| attach_model(world, entity));
    }
}

/// Give `entity` its model's instance unless it draws live levels, or mark it
/// to wait while its model's import settings are read.
///
/// An entity that already points at the same glTF scene is left alone:
/// re-inserting an equal handle would make the world-asset spawner despawn and
/// rebuild the instance.
#[cfg(feature = "render")]
fn attach_model(world: &mut World, entity: Entity) {
    let Some(source) = world
        .get::<jackdaw_scene_types::GltfSource>(entity)
        .cloned()
    else {
        return;
    };
    match live_levels::model_placement(world, entity) {
        live_levels::ModelPlacement::Levels => {}
        live_levels::ModelPlacement::Waiting => {
            world
                .entity_mut(entity)
                .insert(live_levels::AwaitingModelSettings);
        }
        live_levels::ModelPlacement::Instance => {
            let root = world_asset_root(
                world.resource::<AssetServer>(),
                world.get_resource(),
                &source,
            );
            let mut entity = world.entity_mut(entity);
            if entity
                .get::<WorldAssetRoot>()
                .is_none_or(|current| current.0 != root.0)
            {
                entity.insert(root);
            }
        }
    }
}

/// Spawn one document node and its subtree.
///
/// `Transform` and `Visibility` are pulled into a single `world.spawn` along
/// with a `GlobalTransform`/`InheritedVisibility` computed from the parent's
/// already-final values, so the entity reaches its structural state in one
/// archetype move. `On<Insert<T>>` observers for the remaining components then
/// see correct globals. Children are spawned parent-first via recursion.
///
/// Limitation: component fields of type `Entity` that reference another node
/// in the same scene are not remapped to the spawned entity. No built-in
/// component uses such a field today; a cross-entity reference feature must add
/// a post-spawn pass mapping document node order to spawned entities.
fn spawn_node(
    world: &mut World,
    ast: &SceneBsnAst,
    node: Entity,
    parent_entity: Entity,
    scene_root: Entity,
    is_document_root: bool,
    registry: &AppTypeRegistry,
    spawned: &mut Vec<Entity>,
) -> Option<Entity> {
    let patches = ast.get_patches(node).map(|p| p.0.clone())?;
    if patches.is_empty() {
        // A document with no content parses to a single empty root; skip it
        // rather than spawn a phantom entity.
        return None;
    }

    let mut name: Option<String> = None;
    let mut children: Vec<Entity> = Vec::new();
    let mut transform = Transform::default();
    let mut visibility = Visibility::default();
    let mut deferred: Vec<BsnPatch> = Vec::new();
    let mut is_ui_root = false;

    {
        let reg = registry.read();
        for &pe in &patches {
            let Some(patch) = ast.get_patch(pe) else {
                continue;
            };
            match patch {
                BsnPatch::Name(n) => name = Some(n.clone()),
                BsnPatch::Children(kids) => children = kids.clone(),
                BsnPatch::Base(_) | BsnPatch::Template(_, _) => {}
                BsnPatch::Type(_) | BsnPatch::Struct(_) | BsnPatch::TupleStruct(_) => {
                    let Some(type_path) = patch_type_path(patch) else {
                        continue;
                    };
                    match resolve_component_type_id(&reg, type_path) {
                        Some(id) if id == TypeId::of::<Transform>() => {
                            if let Some(t) = convert_component::<Transform>(patch, &reg) {
                                transform = t;
                            }
                        }
                        Some(id) if id == TypeId::of::<Visibility>() => {
                            if let Some(v) = convert_component::<Visibility>(patch, &reg) {
                                visibility = v;
                            }
                        }
                        Some(id) if id == TypeId::of::<jackdaw_scene_types::UiSceneRoot>() => {
                            is_ui_root = true;
                            deferred.push(patch.clone());
                        }
                        _ => deferred.push(patch.clone()),
                    }
                }
            }
        }
    }

    // GT / IV from the parent's already-final values + local overrides.
    // A UI scene root must be a real ECS root: Bevy's layout only lays out
    // `Node` trees that start at an unparented entity.
    let unparented_ui_root = is_document_root && is_ui_root;
    let parent_gt = if unparented_ui_root {
        GlobalTransform::IDENTITY
    } else {
        world
            .get::<GlobalTransform>(parent_entity)
            .copied()
            .unwrap_or(GlobalTransform::IDENTITY)
    };
    let computed_gt = parent_gt.mul_transform(transform);

    let parent_iv = if unparented_ui_root {
        InheritedVisibility::VISIBLE
    } else {
        world
            .get::<InheritedVisibility>(parent_entity)
            .copied()
            .unwrap_or(InheritedVisibility::VISIBLE)
    };
    let computed_iv = match visibility {
        Visibility::Hidden => InheritedVisibility::HIDDEN,
        Visibility::Visible => InheritedVisibility::VISIBLE,
        Visibility::Inherited => parent_iv,
    };

    // One archetype move for all structural state. `AuthoredWidget` rides
    // along: everything a scene document spawns is authored content.
    let entity = world
        .spawn((
            transform,
            visibility,
            computed_gt,
            computed_iv,
            jackdaw_widgets_runtime::AuthoredWidget,
        ))
        .id();
    if !unparented_ui_root {
        world.entity_mut(entity).insert(ChildOf(parent_entity));
    }
    if is_document_root {
        world
            .entity_mut(entity)
            .insert(JackdawSceneMember { root: scene_root });
    }
    spawned.push(entity);

    if let Some(name) = name {
        world.entity_mut(entity).insert(Name::new(name));
    }

    // User components on top. `On<Insert<T>>` fires here with
    // GlobalTransform / InheritedVisibility already correct. `SceneNodeId`
    // rides through this path as a normal registered tuple-struct component.
    for patch in &deferred {
        apply_component_patch(world, entity, patch);
    }

    for child in children {
        spawn_node(
            world, ast, child, entity, scene_root, false, registry, spawned,
        );
    }

    Some(entity)
}

/// The `BsnValue` form of a component patch (`Type`/`Struct`/`TupleStruct`),
/// or `None` for the relational/name patches that carry no component value.
fn patch_to_bsn_value(patch: &BsnPatch) -> Option<BsnValue> {
    match patch {
        BsnPatch::Type(tp) => Some(BsnValue::Type(tp.clone())),
        BsnPatch::Struct(data) => Some(BsnValue::Struct(data.clone())),
        BsnPatch::TupleStruct(data) => Some(BsnValue::TupleStruct(data.clone())),
        _ => None,
    }
}

/// The authored type path of a component patch, which may be enum-variant
/// qualified (`Enum::Variant`).
fn patch_type_path(patch: &BsnPatch) -> Option<&str> {
    match patch {
        BsnPatch::Type(tp) => Some(tp),
        BsnPatch::Struct(data) => Some(&data.type_path),
        BsnPatch::TupleStruct(data) => Some(&data.type_path),
        _ => None,
    }
}

/// The registered component's `TypeId` for a patch type path, resolving an
/// enum-variant path (`Enum::Variant`) back to its base enum registration.
fn resolve_component_type_id(reg: &TypeRegistry, type_path: &str) -> Option<TypeId> {
    let registration = reg.get_with_type_path(type_path).or_else(|| {
        type_path
            .rfind("::")
            .and_then(|sep| reg.get_with_type_path(&type_path[..sep]))
    })?;
    Some(registration.type_id())
}

/// Convert a component patch to a concrete `T` via the BSN document layer.
/// `assets` is `None`: the structural components this is used for (`Transform`,
/// `Visibility`) carry no asset references.
fn convert_component<T: bevy::reflect::FromReflect>(
    patch: &BsnPatch,
    reg: &TypeRegistry,
) -> Option<T> {
    let value = patch_to_bsn_value(patch)?;
    let reflected = bsn_value_to_reflect(&value, TypeId::of::<T>(), reg, None)?;
    <T as bevy::reflect::FromReflect>::from_reflect(reflected.as_ref())
}

/// Load a scene's embedded named assets into their `Assets<T>` stores.
/// Returns a map of `#Name` and `@Name` reference strings to handles.
fn load_embedded_assets(
    world: &mut World,
    ast: &SceneBsnAst,
    registry: &AppTypeRegistry,
) -> HashMap<String, UntypedHandle> {
    let mut map: HashMap<String, UntypedHandle> = HashMap::new();
    let server = world.resource::<AssetServer>().clone();

    for root in ast.roots.clone() {
        let reg = registry.read();
        let Some((type_path, asset_value)) = asset_value_from_root(ast, root) else {
            continue;
        };
        let Some(registration) = reg.get_with_type_path(&type_path) else {
            warn!(
                "embedded asset type '{type_path}' is not registered in this app; \
                 entities referencing it will fall back to a default handle"
            );
            continue;
        };
        let Some(reflect_asset) = registration.data::<ReflectAsset>() else {
            continue;
        };
        let type_id = registration.type_id();
        let Some(name) = ast.get_name(root).map(str::to_owned) else {
            continue;
        };
        let load = server_loads(&server);
        let assets_ctx = BsnApplyAssets {
            load: &load,
            local: None,
        };
        let Some(value) = bsn_value_to_reflect(&asset_value, type_id, &reg, Some(&assets_ctx))
        else {
            continue;
        };
        let handle = reflect_asset.add(world, &*value);
        map.insert(format!("#{name}"), handle);
    }

    map
}

/// Every texture path the document binds to a material slot, and whether that
/// slot holds linear data.
#[cfg(feature = "render")]
fn collect_texture_slots(ast: &SceneBsnAst) -> Vec<(String, bool)> {
    use jackdaw_scene_types::render_assets::{is_file_reference, texture_slot};

    fn collect(data: &jackdaw_bsn::BsnStructData, slots: &mut Vec<(String, bool)>) {
        for field in &data.fields.0 {
            match &field.value {
                BsnValue::String(path) if is_file_reference(path) => {
                    if let Some(linear) = texture_slot(&field.name) {
                        slots.push((path.clone(), linear));
                    }
                }
                BsnValue::Struct(nested) => collect(nested, slots),
                _ => {}
            }
        }
    }

    let mut slots = Vec::new();
    let mut stack: Vec<Entity> = ast.roots.clone();
    while let Some(node) = stack.pop() {
        let Some(patches) = ast.get_patches(node) else {
            continue;
        };
        for &pe in &patches.0 {
            match ast.get_patch(pe) {
                Some(BsnPatch::Struct(data)) => collect(data, &mut slots),
                Some(BsnPatch::Children(kids)) => stack.extend(kids.iter().copied()),
                _ => {}
            }
        }
    }
    slots
}

/// Load the document's material textures with each slot's colour space before
/// the generic resolve reaches the same paths. The returned handles keep the
/// images alive until the materials hold their own.
#[cfg(feature = "render")]
fn preload_textures(
    ast: &SceneBsnAst,
    mut load: impl FnMut(&str, bool) -> UntypedHandle,
) -> Vec<UntypedHandle> {
    collect_texture_slots(ast)
        .into_iter()
        .map(|(path, linear)| load(&path, linear))
        .collect()
}

/// The `Unorm` twin of a 16-bit `Uint` texture format: same bytes per texel,
/// same channel order, but float-filterable where the `Uint` side is not.
///
/// Bevy decodes 16-bit grayscale PNGs as `R16Uint` and grayscale+alpha as
/// `Rg16Uint`. A `StandardMaterial` slot demands a filterable float sampler,
/// so binding either one fails the whole bind group.
#[cfg(feature = "render")]
fn filterable_twin(
    format: bevy::render::render_resource::TextureFormat,
) -> Option<bevy::render::render_resource::TextureFormat> {
    use bevy::render::render_resource::TextureFormat;
    match format {
        TextureFormat::R16Uint => Some(TextureFormat::R16Unorm),
        TextureFormat::Rg16Uint => Some(TextureFormat::Rg16Unorm),
        TextureFormat::Rgba16Uint => Some(TextureFormat::Rgba16Unorm),
        _ => None,
    }
}

/// The images a `StandardMaterial` binds, over the slots every build has.
#[cfg(feature = "render")]
fn material_texture_ids(
    material: &StandardMaterial,
) -> impl Iterator<Item = bevy::asset::AssetId<Image>> {
    [
        material.base_color_texture.as_ref(),
        material.emissive_texture.as_ref(),
        material.metallic_roughness_texture.as_ref(),
        material.normal_map_texture.as_ref(),
        material.occlusion_texture.as_ref(),
        material.depth_map.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(Handle::id)
}

/// Retags 16-bit `Uint` material textures as their filterable `Unorm`
/// twins.
///
/// The editor adds this plugin rather than keeping a copy of the retag, so a
/// pack of 16-bit maps renders the same in the editor and in a built game.
#[cfg(feature = "render")]
pub struct MaterialTextureFormatPlugin;

#[cfg(feature = "render")]
impl Plugin for MaterialTextureFormatPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            promote_material_texture_formats.after(bevy::asset::AssetEventSystems),
        );
    }
}

/// Retag every 16-bit `Uint` image a material binds as its `Unorm` twin.
///
/// Descriptor only: the texels already have the layout the twin declares. Runs
/// after the asset events are published and before the render world extracts.
/// Both event streams matter, since an image may decode long after the material
/// naming it, or the other way round.
#[cfg(feature = "render")]
fn promote_material_texture_formats(
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut material_events: MessageReader<AssetEvent<StandardMaterial>>,
    materials: Res<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    use bevy::asset::AssetId;

    let touched: Vec<AssetId<Image>> = image_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => Some(*id),
            _ => None,
        })
        .collect();
    let materials_changed = material_events.read().any(|event| {
        matches!(
            event,
            AssetEvent::Added { .. } | AssetEvent::Modified { .. }
        )
    });
    if touched.is_empty() && !materials_changed {
        return;
    }

    let bound: HashSet<AssetId<Image>> = materials
        .iter()
        .flat_map(|(_, material)| material_texture_ids(material))
        .collect();
    let candidates: Vec<AssetId<Image>> = if materials_changed {
        bound.into_iter().collect()
    } else {
        touched
            .into_iter()
            .filter(|id| bound.contains(id))
            .collect()
    };

    for id in candidates {
        let Some(image) = images.get(id) else {
            continue;
        };
        let Some(twin) = filterable_twin(image.texture_descriptor.format) else {
            continue;
        };
        if image.data.is_none() {
            if let Some(path) = asset_server.get_path(id) {
                asset_server.reload(path.into_owned());
            }
            continue;
        }
        if let Some(mut image) = images.get_mut(id) {
            image.texture_descriptor.format = twin;
        }
    }
}

/// The one definition of the `Uint` retag, exercised through the plugin both
/// the editor and a built game add.
#[cfg(all(test, feature = "render"))]
mod material_texture_format_tests {
    use super::*;
    use bevy::app::App;
    use bevy::asset::{AssetApp, AssetPlugin, RenderAssetUsages};
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    fn promotion_app() -> App {
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AssetPlugin::default()));
        app.init_asset::<Image>();
        app.init_asset::<StandardMaterial>();
        app.add_plugins(MaterialTextureFormatPlugin);
        app
    }

    /// A one-texel image in `format`, sized from the texel it is given.
    fn raw_image(app: &mut App, texel: &[u8], format: TextureFormat) -> Handle<Image> {
        let image = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            texel.to_vec(),
            format,
            RenderAssetUsages::default(),
        );
        app.world_mut().resource_mut::<Assets<Image>>().add(image)
    }

    fn format_of(app: &App, handle: &Handle<Image>) -> TextureFormat {
        app.world()
            .resource::<Assets<Image>>()
            .get(handle)
            .expect("image")
            .texture_descriptor
            .format
    }

    /// A 16-bit grayscale PNG decodes as `R16Uint`, which has no filterable
    /// sampler; bound to a material slot it fails the whole bind group.
    #[test]
    fn a_sixteen_bit_uint_image_a_material_binds_is_retagged_as_its_unorm_twin() {
        let mut app = promotion_app();
        let occlusion = raw_image(&mut app, &[0x00, 0x80], TextureFormat::R16Uint);
        let gray_alpha = raw_image(&mut app, &[0x00, 0x80, 0x00, 0xff], TextureFormat::Rg16Uint);
        let _material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                occlusion_texture: Some(occlusion.clone()),
                depth_map: Some(gray_alpha.clone()),
                ..default()
            });

        app.update();

        assert_eq!(format_of(&app, &occlusion), TextureFormat::R16Unorm);
        assert_eq!(format_of(&app, &gray_alpha), TextureFormat::Rg16Unorm);
    }

    /// The texels are what they were: only the descriptor is rewritten.
    #[test]
    fn promotion_leaves_the_texels_alone() {
        let mut app = promotion_app();
        let image = raw_image(&mut app, &[0x34, 0x12], TextureFormat::R16Uint);
        let _material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                depth_map: Some(image.clone()),
                ..default()
            });

        app.update();

        let images = app.world().resource::<Assets<Image>>();
        assert_eq!(
            images.get(&image).unwrap().data.as_deref(),
            Some(&[0x34u8, 0x12][..])
        );
    }

    /// An image no material binds keeps its format: an integer texture read
    /// with `textureLoad` is meant to stay integer.
    #[test]
    fn an_image_no_material_binds_keeps_its_uint_format() {
        let mut app = promotion_app();
        let unbound = raw_image(&mut app, &[0x00, 0x80], TextureFormat::R16Uint);

        app.update();

        assert_eq!(format_of(&app, &unbound), TextureFormat::R16Uint);
    }

    /// An image may decode before the material naming it, so the material's own
    /// event has to sweep the slots it just claimed.
    #[test]
    fn a_material_added_after_its_image_still_gets_it_promoted() {
        let mut app = promotion_app();
        let image = raw_image(&mut app, &[0x00, 0x80], TextureFormat::R16Uint);
        app.update();
        assert_eq!(format_of(&app, &image), TextureFormat::R16Uint);

        let _material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                normal_map_texture: Some(image.clone()),
                ..default()
            });
        app.update();

        assert_eq!(format_of(&app, &image), TextureFormat::R16Unorm);
    }
}
