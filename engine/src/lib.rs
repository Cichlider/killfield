//! Killfield engine — Rust port of `killfield/src/`.
//!
//! Ported module by module against the JS reference. Every module carries a
//! differential test against dumped JS state; see `tools/difftest/`.

pub mod ballistics;
pub mod chain;
pub mod collect;
pub mod constants;
pub mod directional;
pub mod field;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
pub mod game;
pub mod laika;
pub mod maze;
pub mod obs;
pub mod reward;
pub mod risk;
pub mod rng;
pub mod sandbox;
pub mod score;
pub mod semantic_obs;
pub mod teacher;
pub mod tuning;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
