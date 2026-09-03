//! Killfield engine — Rust port of `killfield/src/`.
//!
//! Ported module by module against the historical JS reference. The finished
//! engine is now maintained through Rust unit tests and deterministic seeds.

pub mod ballistics;
pub mod chain;
pub mod constants;
pub mod duel;
pub mod duel_obs;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi_duel;
pub mod field;
pub mod game;
pub mod laika;
pub mod maze;
pub mod range;
pub mod range_obs;
pub mod risk;
pub mod rng;
pub mod sandbox;
pub mod score;
pub mod teacher;
pub mod tuning;

#[cfg(any(target_arch = "wasm32", test))]
pub mod wasm;
