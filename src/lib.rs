//! kute: records, transcribes and summarizes calls, locally.

pub mod audio;
pub mod capture;
pub mod cli;
pub mod config;
pub mod daemon;
pub mod detect;
pub mod dialog;
pub mod doctor;
pub mod import;
pub mod index;
pub mod ipc;
pub mod jobs;
pub mod live;
pub mod merge;
pub mod models;
pub mod notify;
pub mod pipeline;
pub mod pw;
pub mod recorder;
pub mod render;
pub mod service;
pub mod split;
pub mod store;
pub mod summarize;
#[doc(hidden)]
pub mod testing;
pub mod time;
pub mod transcribe;
pub mod tray;
pub mod worker;
