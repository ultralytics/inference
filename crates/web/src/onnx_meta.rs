// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Minimal ONNX `ModelProto` reader for the Ultralytics `metadata_props`.
//!
//! `ort-web` does not implement ONNX session metadata retrieval, so the browser
//! path reads the `metadata_props` (key/value pairs such as `task`, `names`,
//! `imgsz`) straight from the model protobuf. Only field 14 is decoded; the large
//! graph fields are skipped without being parsed. These helpers are pure
//! (`&[u8] -> map`) and carry no wasm/JS types.

use std::collections::HashMap;

/// Read a protobuf base-128 varint at `pos`, advancing it. Returns `None` on a
/// truncated/oversized value.
fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

/// Read the next protobuf field at `pos`, advancing it. Returns the field number
/// and, for length-delimited fields (wire type 2), the payload bytes; varint and
/// fixed-width fields are skipped and yield `None` payload. Returns `None` at the
/// end of the buffer or on a malformed field.
fn read_field<'a>(buf: &'a [u8], pos: &mut usize) -> Option<(u64, Option<&'a [u8]>)> {
    let tag = read_varint(buf, pos)?;
    let payload = match tag & 7 {
        0 => {
            read_varint(buf, pos)?;
            None
        }
        1 => {
            *pos += 8;
            None
        }
        5 => {
            *pos += 4;
            None
        }
        2 => {
            let len = read_varint(buf, pos)? as usize;
            let sub = buf.get(*pos..*pos + len)?;
            *pos += len;
            Some(sub)
        }
        _ => return None,
    };
    Some((tag >> 3, payload))
}

/// Extract `ModelProto.metadata_props` (field 14, repeated
/// `StringStringEntryProto`) into a key/value map. Other fields (including the
/// large graph) are skipped without being decoded.
pub(crate) fn parse_metadata_props(buf: &[u8]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut pos = 0;
    while let Some((field, payload)) = read_field(buf, &mut pos) {
        if field == 14
            && let Some(sub) = payload
            && let Some((key, value)) = parse_string_string_entry(sub)
        {
            map.insert(key, value);
        }
    }
    map
}

/// Parse a `StringStringEntryProto` (field 1 = key, field 2 = value).
fn parse_string_string_entry(buf: &[u8]) -> Option<(String, String)> {
    let mut pos = 0;
    let mut key = None;
    let mut value = None;
    while let Some((field, payload)) = read_field(buf, &mut pos) {
        if let Some(sub) = payload {
            let text = String::from_utf8_lossy(sub).into_owned();
            match field {
                1 => key = Some(text),
                2 => value = Some(text),
                _ => {}
            }
        }
    }
    Some((key?, value.unwrap_or_default()))
}
