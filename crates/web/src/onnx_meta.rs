// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Minimal ONNX `ModelProto` reader for the Ultralytics `metadata_props`, plus the
//! float32 I/O rewrite for float16 models.
//!
//! `ort-web` does not implement ONNX session metadata retrieval, so the browser
//! path reads the `metadata_props` (key/value pairs such as `task`, `names`,
//! `imgsz`) straight from the model protobuf. Only field 14 is decoded; the large
//! graph fields are skipped without being parsed. These helpers are pure
//! (`&[u8] -> map`) and carry no wasm/JS types.

use std::collections::{BTreeMap, HashSet};

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

/// A decoded protobuf field value.
enum Value<'a> {
    /// A varint (wire type 0).
    Varint(u64),
    /// A length-delimited payload (wire type 2).
    Bytes(&'a [u8]),
    /// A skipped fixed-width value (wire types 1 and 5).
    Fixed,
}

/// Iterate the protobuf fields of `buf` as `(field number, value, raw field bytes)`,
/// stopping at the end of the buffer or on a malformed field.
fn fields(buf: &[u8]) -> impl Iterator<Item = (u64, Value<'_>, &[u8])> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let start = pos;
        let tag = read_varint(buf, &mut pos)?;
        let value = match tag & 7 {
            0 => Value::Varint(read_varint(buf, &mut pos)?),
            1 => {
                pos += 8;
                Value::Fixed
            }
            5 => {
                pos += 4;
                Value::Fixed
            }
            2 => {
                let len = read_varint(buf, &mut pos)? as usize;
                let sub = buf.get(pos..pos + len)?;
                pos += len;
                Value::Bytes(sub)
            }
            _ => return None,
        };
        Some((tag >> 3, value, buf.get(start..pos)?))
    })
}

/// The first length-delimited `field` payload in `buf`.
fn bytes_field(buf: &[u8], field: u64) -> Option<&[u8]> {
    fields(buf).find_map(|(f, value, _)| match value {
        Value::Bytes(sub) if f == field => Some(sub),
        _ => None,
    })
}

/// Extract `ModelProto.metadata_props` (field 14, repeated
/// `StringStringEntryProto`) into a key/value map. Other fields (including the
/// large graph) are skipped without being decoded.
///
/// Ordered by key, so the text built from it in `build_metadata` is stable
/// across runs rather than following a hash order that varies per build.
pub(crate) fn parse_metadata_props(buf: &[u8]) -> BTreeMap<String, String> {
    fields(buf)
        .filter_map(|(field, value, _)| match value {
            Value::Bytes(sub) if field == 14 => parse_string_string_entry(sub),
            _ => None,
        })
        .collect()
}

/// Parse a `StringStringEntryProto` (field 1 = key, field 2 = value).
fn parse_string_string_entry(buf: &[u8]) -> Option<(String, String)> {
    let text = |field| bytes_field(buf, field).map(|s| String::from_utf8_lossy(s).into_owned());
    Some((text(1)?, text(2).unwrap_or_default()))
}

/// ONNX `TensorProto.DataType` of a float32 tensor.
const FLOAT: u64 = 1;
/// ONNX `TensorProto.DataType` of a float16 tensor.
const FLOAT16: u64 = 10;

/// Rewrite a model whose graph inputs or outputs are float16 (e.g. a `quantize=16` export
/// made on a GPU) to take and return float32, converting each such tensor with a `Cast` node
/// at the graph boundary while the graph itself stays float16.
///
/// `ort-web` cannot create or read float16 tensors, so this is what lets such a model
/// run in the browser. Returns `None` when no input or output is float16, so other
/// models are used as-is without a copy.
pub(crate) fn with_f32_io(model: &[u8]) -> Option<Vec<u8>> {
    let graph = bytes_field(model, 7)?;
    // `GraphProto.initializer` (5) and `sparse_initializer` (15, named by its `values` tensor)
    // names; an input backed by one is a weight, not fed at runtime.
    let initializers: HashSet<&[u8]> = fields(graph)
        .filter_map(|(field, value, _)| match value {
            Value::Bytes(tensor) if field == 5 => bytes_field(tensor, 8),
            Value::Bytes(sparse) if field == 15 => bytes_field(bytes_field(sparse, 1)?, 8),
            _ => None,
        })
        .collect();
    // Every value name in the graph (node outputs, inputs, outputs, value infos and
    // initializers), so a rename cannot clash with one; each rename joins it as it is made.
    let mut taken: HashSet<Vec<u8>> = initializers.iter().map(|n| n.to_vec()).collect();
    for (field, value, _) in fields(graph) {
        match (field, value) {
            (1, Value::Bytes(node)) => taken.extend(fields(node).filter_map(|(f, v, _)| match v {
                Value::Bytes(output) if f == 2 => Some(output.to_vec()),
                _ => None,
            })),
            (11..=13, Value::Bytes(info)) => taken.extend(bytes_field(info, 1).map(<[u8]>::to_vec)),
            _ => {}
        }
    }
    // `Cast` nodes into the graph go first and out of it last, keeping the node list
    // topologically sorted; every other graph field is copied unchanged.
    let (mut head, mut body, mut tail) = (Vec::new(), Vec::new(), Vec::new());
    for (field, value, raw) in fields(graph) {
        // `GraphProto.input` (11) and `GraphProto.output` (12).
        let rewritten = match value {
            Value::Bytes(info)
                if field == 12
                    || field == 11
                        && !bytes_field(info, 1).is_some_and(|n| initializers.contains(n)) =>
            {
                f32_value_info(info, &mut taken)
            }
            _ => None,
        };
        let Some((name, renamed, info)) = rewritten else {
            body.extend_from_slice(raw);
            continue;
        };
        write_bytes(&mut body, field, &info);
        if field == 11 {
            write_bytes(&mut head, 1, &cast_node(&renamed, name, FLOAT16));
        } else {
            write_bytes(&mut tail, 1, &cast_node(name, &renamed, FLOAT));
        }
    }
    if head.is_empty() && tail.is_empty() {
        return None;
    }
    head.extend(body);
    head.extend(tail);
    let mut out = Vec::with_capacity(model.len() + 256);
    for (field, _, raw) in fields(model) {
        if field == 7 {
            write_bytes(&mut out, 7, &head);
        } else {
            out.extend_from_slice(raw);
        }
    }
    Some(out)
}

/// For a float16 tensor `ValueInfoProto`, return its name, a `{name}_f32` rename not in
/// `taken` (and add it there), and a float32 `ValueInfoProto` of the same shape under that
/// rename; `None` otherwise.
fn f32_value_info<'a>(
    info: &'a [u8],
    taken: &mut HashSet<Vec<u8>>,
) -> Option<(&'a [u8], Vec<u8>, Vec<u8>)> {
    let name = bytes_field(info, 1)?;
    // `ValueInfoProto.type` (2) -> `TypeProto.tensor_type` (1).
    let tensor = bytes_field(bytes_field(info, 2)?, 1)?;
    if !fields(tensor).any(|(f, value, _)| f == 1 && matches!(value, Value::Varint(FLOAT16))) {
        return None;
    }
    // `TypeProto.Tensor`: the float32 `elem_type` (1) plus the original shape.
    let mut tensor_f32 = Vec::new();
    write_int(&mut tensor_f32, 1, FLOAT);
    for (_, _, raw) in fields(tensor).filter(|&(f, ..)| f != 1) {
        tensor_f32.extend_from_slice(raw);
    }
    let mut type_proto = Vec::new();
    write_bytes(&mut type_proto, 1, &tensor_f32);
    let mut renamed = [name, b"_f32"].concat();
    while !taken.insert(renamed.clone()) {
        renamed.push(b'_');
    }
    let mut out = Vec::new();
    write_bytes(&mut out, 1, &renamed);
    write_bytes(&mut out, 2, &type_proto);
    Some((name, renamed, out))
}

/// A `Cast` `NodeProto` from `input` to `output` with element type `to`.
fn cast_node(input: &[u8], output: &[u8], to: u64) -> Vec<u8> {
    // `AttributeProto`: name (1), i (3), and type (20) = INT (2).
    let mut attr = Vec::new();
    write_bytes(&mut attr, 1, b"to");
    write_int(&mut attr, 3, to);
    write_int(&mut attr, 20, 2);
    // `NodeProto`: input (1), output (2), op_type (4), attribute (5).
    let mut node = Vec::new();
    write_bytes(&mut node, 1, input);
    write_bytes(&mut node, 2, output);
    write_bytes(&mut node, 4, b"Cast");
    write_bytes(&mut node, 5, &attr);
    node
}

/// Append `v` as a protobuf base-128 varint.
fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Append varint field `field` holding `v`.
fn write_int(out: &mut Vec<u8>, field: u64, v: u64) {
    write_varint(out, field << 3);
    write_varint(out, v);
}

/// Append length-delimited field `field` holding `bytes`.
fn write_bytes(out: &mut Vec<u8>, field: u64, bytes: &[u8]) {
    write_varint(out, (field << 3) | 2);
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}
