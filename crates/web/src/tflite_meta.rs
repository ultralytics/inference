// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Reader for the Ultralytics metadata embedded in a single-file `.tflite`.
//!
//! A LiteRT export appends a small zip (one `metadata.json` entry) after the
//! TFLite flatbuffer; the runtime reads the flatbuffer by offset and ignores the
//! trailing zip. Mirrors the ONNX [`onnx_meta`](crate::onnx_meta) path. Pure
//! `&[u8]` helpers with no wasm/JS types.

use serde_json::Value;

/// Upper bound on the inflated `metadata.json` (a few KB in practice), guarding
/// against a zip bomb in a user-supplied model file.
const MAX_METADATA_BYTES: usize = 4 << 20;

/// Read a little-endian `u16` at `pos`.
fn le16(buf: &[u8], pos: usize) -> Option<usize> {
    Some(u16::from_le_bytes([*buf.get(pos)?, *buf.get(pos + 1)?]) as usize)
}

/// Read a little-endian `u32` at `pos`.
fn le32(buf: &[u8], pos: usize) -> Option<usize> {
    Some(u32::from_le_bytes([
        *buf.get(pos)?,
        *buf.get(pos + 1)?,
        *buf.get(pos + 2)?,
        *buf.get(pos + 3)?,
    ]) as usize)
}

/// Extract and decompress the `metadata.json` entry from the zip appended to a
/// single-file `.tflite`. Returns `None` if there is no trailing zip or no such
/// entry (e.g. a plain TFLite model exported without Ultralytics metadata).
fn extract_metadata_json(buf: &[u8]) -> Option<String> {
    // Locate the End Of Central Directory record (signature `PK\x05\x06`). It
    // lives in the last 22 bytes plus an optional comment (max 64 KiB), so scan
    // backwards from the tail.
    let eocd_sig = [0x50, 0x4b, 0x05, 0x06];
    let scan_start = buf.len().saturating_sub(65_557);
    let eocd = (scan_start..buf.len().checked_sub(21)?)
        .rev()
        .find(|&i| buf[i..i + 4] == eocd_sig)?;
    let entries = le16(buf, eocd + 10)?;
    let cd_offset = le32(buf, eocd + 16)?;

    // Walk the central directory for the `metadata.json` entry (signature
    // `PK\x01\x02`), then read its data via the local file header.
    let mut pos = cd_offset;
    for _ in 0..entries {
        if buf.get(pos..pos + 4)? != [0x50, 0x4b, 0x01, 0x02] {
            return None;
        }
        let method = le16(buf, pos + 10)?;
        let comp_size = le32(buf, pos + 20)?;
        let name_len = le16(buf, pos + 28)?;
        let extra_len = le16(buf, pos + 30)?;
        let comment_len = le16(buf, pos + 32)?;
        let local_off = le32(buf, pos + 42)?;
        let name = buf.get(pos + 46..pos + 46 + name_len)?;
        if name == b"metadata.json" {
            // The local header repeats the name/extra fields; the data follows.
            let l_name = le16(buf, local_off + 26)?;
            let l_extra = le16(buf, local_off + 28)?;
            let data = local_off + 30 + l_name + l_extra;
            let comp = buf.get(data..data + comp_size)?;
            // Cap the (de)compressed size both ways so a crafted model file cannot
            // force a huge allocation. Ultralytics metadata is a few KB.
            return match method {
                0 if comp.len() <= MAX_METADATA_BYTES => String::from_utf8(comp.to_vec()).ok(),
                8 => miniz_oxide::inflate::decompress_to_vec_with_limit(comp, MAX_METADATA_BYTES)
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok()),
                _ => None,
            };
        }
        pos += 46 + name_len + extra_len + comment_len;
    }
    None
}

/// Flatten Ultralytics `metadata.json` into the `key: value` lines the shared
/// [`ModelMetadata::from_yaml_str`](ultralytics_inference::metadata::ModelMetadata::from_yaml_str)
/// parser consumes (the same shape the ONNX path rebuilds from `metadata_props`).
/// Class names become an unquoted `names:` block (one `id: name` per line, so
/// apostrophes need no escaping); `imgsz`/`kpt_shape` become inline lists; the
/// nested `args` object is dropped (the parser reads none of it).
fn metadata_lines_from_json(json: &str) -> Option<String> {
    let value: Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object()?;
    let mut lines = Vec::new();
    for (key, val) in obj {
        match key.as_str() {
            "names" => {
                if let Some(names) = val.as_object() {
                    lines.push("names:".to_string());
                    for (id, name) in names {
                        lines.push(format!("  {id}: {}", name.as_str().unwrap_or_default()));
                    }
                }
            }
            "imgsz" | "kpt_shape" => {
                if let Some(arr) = val.as_array() {
                    let nums: Vec<String> = arr
                        .iter()
                        .filter_map(Value::as_i64)
                        .map(|n| n.to_string())
                        .collect();
                    lines.push(format!("{key}: [{}]", nums.join(", ")));
                }
            }
            // `args` is a nested object the metadata parser does not read; skip it.
            "args" => {}
            _ => match val {
                Value::String(s) => lines.push(format!("{key}: {s}")),
                Value::Bool(b) => lines.push(format!("{key}: {b}")),
                Value::Number(n) => lines.push(format!("{key}: {n}")),
                _ => {}
            },
        }
    }
    Some(lines.join("\n"))
}

/// Read the Ultralytics metadata from a single-file `.tflite` and return it as
/// the `key: value` text the shared metadata parser accepts. Returns `None` when
/// the model carries no embedded `metadata.json`.
pub(crate) fn metadata_text(buf: &[u8]) -> Option<String> {
    metadata_lines_from_json(&extract_metadata_json(buf)?)
}
