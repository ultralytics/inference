// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use half::f16;

use crate::{InferenceError, Result};

const FLOAT: u64 = 1;
const FLOAT16: u64 = 10;
const LENGTH_DELIMITED: u8 = 2;
const MAX_VARINT_BYTES: usize = 10;

pub fn promote_fp16_to_fp32(model: &[u8]) -> Result<Option<Vec<u8>>> {
    let mut output = Vec::with_capacity(model.len().saturating_mul(2));
    let mut promoted = 0;
    rewrite_model(model, &mut output, &mut promoted)?;
    Ok((promoted != 0).then_some(output))
}

fn rewrite_model(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    let mut graph = false;
    let mut quantize = false;
    let initial_promoted = *promoted;
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            7 => {
                graph = true;
                write_message(output, field.number, |output| {
                    rewrite_graph(field.message()?, output, promoted)
                })?;
            }
            20 => write_message(output, field.number, |output| {
                rewrite_training_info(field.message()?, output, promoted)
            })?,
            25 => write_message(output, field.number, |output| {
                rewrite_function(field.message()?, output, promoted)
            })?,
            14 if find_bytes(field.message()?, 1)? == Some(b"quantize".as_slice()) => {
                quantize = true;
                write_message(output, field.number, |output| {
                    rewrite_metadata_quantize(field.message()?, output)
                })?;
            }
            _ => output.extend_from_slice(field.encoded),
        }
    }
    if !graph {
        return Err(model_error("ONNX model has no inference graph"));
    }
    if *promoted != initial_promoted && !quantize {
        write_message(output, 14, |output| {
            write_bytes_field(output, 1, b"quantize")?;
            write_bytes_field(output, 2, b"32")
        })?;
    }
    Ok(())
}

fn rewrite_metadata_quantize(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == 2 {
            write_bytes_field(output, field.number, b"32")?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_graph(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            1 => write_message(output, field.number, |output| {
                rewrite_node(field.message()?, output, promoted)
            })?,
            5 => write_message(output, field.number, |output| {
                rewrite_tensor(field.message()?, output, promoted)
            })?,
            15 => write_message(output, field.number, |output| {
                rewrite_sparse_tensor(field.message()?, output, promoted)
            })?,
            11..=13 => write_message(output, field.number, |output| {
                rewrite_value_info(field.message()?, output, promoted)
            })?,
            _ => output.extend_from_slice(field.encoded),
        }
    }
    Ok(())
}

fn rewrite_node(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    let cast = find_bytes(input, 4)? == Some(b"Cast".as_slice());
    for field in Fields::new(input) {
        let field = field?;
        if field.number == 5 {
            write_message(output, field.number, |output| {
                rewrite_attribute(field.message()?, output, promoted, cast)
            })?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_attribute(
    input: &[u8],
    output: &mut Vec<u8>,
    promoted: &mut usize,
    cast: bool,
) -> Result<()> {
    let cast_target = cast && find_bytes(input, 1)? == Some(b"to".as_slice());
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            3 if cast_target && field.varint()? == FLOAT16 => {
                write_varint_field(output, field.number, FLOAT);
                *promoted += 1;
            }
            5 | 10 => write_message(output, field.number, |output| {
                rewrite_tensor(field.message()?, output, promoted)
            })?,
            6 | 11 => write_message(output, field.number, |output| {
                rewrite_graph(field.message()?, output, promoted)
            })?,
            22 | 23 => write_message(output, field.number, |output| {
                rewrite_sparse_tensor(field.message()?, output, promoted)
            })?,
            14 | 15 => write_message(output, field.number, |output| {
                rewrite_type(field.message()?, output, promoted)
            })?,
            _ => output.extend_from_slice(field.encoded),
        }
    }
    Ok(())
}

fn rewrite_tensor(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    if find_varint(input, 2)? != Some(FLOAT16) {
        output.extend_from_slice(input);
        return Ok(());
    }
    if find_bytes(input, 13)?.is_some() || find_varint(input, 14)? == Some(1) {
        return Err(model_error(
            "FP32 inference does not support ONNX FP16 external tensor data",
        ));
    }
    let int32_data = has_field(input, 5)?;
    if int32_data && has_field(input, 9)? {
        return Err(model_error(
            "ONNX FP16 tensor has both int32_data and raw_data",
        ));
    }

    let mut wrote_int32_data = false;
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            2 => write_varint_field(output, field.number, FLOAT),
            5 if !wrote_int32_data => {
                write_message(output, 9, |output| rewrite_int32_data(input, output))?;
                wrote_int32_data = true;
            }
            5 => {}
            9 => write_message(output, field.number, |output| {
                let data = field.message()?;
                if !data.len().is_multiple_of(2) {
                    return Err(model_error("ONNX FP16 tensor has an invalid byte length"));
                }
                for value in data.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes([value[0], value[1]]);
                    output.extend_from_slice(&f16::from_bits(bits).to_f32().to_le_bytes());
                }
                Ok(())
            })?,
            _ => output.extend_from_slice(field.encoded),
        }
    }
    *promoted += 1;
    Ok(())
}

fn rewrite_int32_data(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number != 5 {
            continue;
        }
        if field.wire == 0 {
            write_fp16_bits(output, field.varint()?)?;
        } else {
            let data = field.message()?;
            let mut position = 0;
            while position < data.len() {
                write_fp16_bits(output, read_varint(data, &mut position)?)?;
            }
        }
    }
    Ok(())
}

fn write_fp16_bits(output: &mut Vec<u8>, bits: u64) -> Result<()> {
    let bits = u16::try_from(bits)
        .map_err(|_| model_error("ONNX FP16 int32_data value exceeds 16 bits"))?;
    output.extend_from_slice(&f16::from_bits(bits).to_f32().to_le_bytes());
    Ok(())
}

fn rewrite_sparse_tensor(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == 1 {
            write_message(output, field.number, |output| {
                rewrite_tensor(field.message()?, output, promoted)
            })?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_value_info(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == 2 {
            write_message(output, field.number, |output| {
                rewrite_type(field.message()?, output, promoted)
            })?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_type(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            1 | 8 => write_message(output, field.number, |output| {
                rewrite_tensor_type(field.message()?, output, promoted)
            })?,
            4 | 9 => write_message(output, field.number, |output| {
                rewrite_nested_type(field.message()?, output, promoted, 1)
            })?,
            5 => write_message(output, field.number, |output| {
                rewrite_nested_type(field.message()?, output, promoted, 2)
            })?,
            _ => output.extend_from_slice(field.encoded),
        }
    }
    Ok(())
}

fn rewrite_tensor_type(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == 1 && field.varint()? == FLOAT16 {
            write_varint_field(output, field.number, FLOAT);
            *promoted += 1;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_nested_type(
    input: &[u8],
    output: &mut Vec<u8>,
    promoted: &mut usize,
    type_field: u32,
) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == type_field {
            write_message(output, field.number, |output| {
                rewrite_type(field.message()?, output, promoted)
            })?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_training_info(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        if matches!(field.number, 1 | 2) {
            write_message(output, field.number, |output| {
                rewrite_graph(field.message()?, output, promoted)
            })?;
        } else {
            output.extend_from_slice(field.encoded);
        }
    }
    Ok(())
}

fn rewrite_function(input: &[u8], output: &mut Vec<u8>, promoted: &mut usize) -> Result<()> {
    for field in Fields::new(input) {
        let field = field?;
        match field.number {
            7 => write_message(output, field.number, |output| {
                rewrite_node(field.message()?, output, promoted)
            })?,
            11 => write_message(output, field.number, |output| {
                rewrite_attribute(field.message()?, output, promoted, false)
            })?,
            12 => write_message(output, field.number, |output| {
                rewrite_value_info(field.message()?, output, promoted)
            })?,
            _ => output.extend_from_slice(field.encoded),
        }
    }
    Ok(())
}

fn find_varint(input: &[u8], number: u32) -> Result<Option<u64>> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == number {
            return field.varint().map(Some);
        }
    }
    Ok(None)
}

fn find_bytes(input: &[u8], number: u32) -> Result<Option<&[u8]>> {
    for field in Fields::new(input) {
        let field = field?;
        if field.number == number {
            return field.message().map(Some);
        }
    }
    Ok(None)
}

fn has_field(input: &[u8], number: u32) -> Result<bool> {
    for field in Fields::new(input) {
        if field?.number == number {
            return Ok(true);
        }
    }
    Ok(false)
}

fn write_message(
    output: &mut Vec<u8>,
    number: u32,
    write: impl FnOnce(&mut Vec<u8>) -> Result<()>,
) -> Result<()> {
    write_varint(output, u64::from(number) << 3 | u64::from(LENGTH_DELIMITED));
    let length = output.len();
    output.resize(length + MAX_VARINT_BYTES, 0);
    let payload = output.len();
    write(output)?;
    let end = output.len();
    let mut encoded = [0; MAX_VARINT_BYTES];
    let encoded_len = encode_varint((end - payload) as u64, &mut encoded);
    output.copy_within(payload..end, length + encoded_len);
    output.truncate(end - (MAX_VARINT_BYTES - encoded_len));
    output[length..length + encoded_len].copy_from_slice(&encoded[..encoded_len]);
    Ok(())
}

fn write_varint_field(output: &mut Vec<u8>, number: u32, value: u64) {
    write_varint(output, u64::from(number) << 3);
    write_varint(output, value);
}

fn write_bytes_field(output: &mut Vec<u8>, number: u32, value: &[u8]) -> Result<()> {
    write_message(output, number, |output| {
        output.extend_from_slice(value);
        Ok(())
    })
}

fn write_varint(output: &mut Vec<u8>, value: u64) {
    let mut encoded = [0; MAX_VARINT_BYTES];
    let length = encode_varint(value, &mut encoded);
    output.extend_from_slice(&encoded[..length]);
}

const fn encode_varint(mut value: u64, output: &mut [u8; MAX_VARINT_BYTES]) -> usize {
    let mut length = 0;
    loop {
        output[length] = (value.to_le_bytes()[0] & 0x7f) | if value >= 0x80 { 0x80 } else { 0 };
        length += 1;
        value >>= 7;
        if value == 0 {
            return length;
        }
    }
}

fn read_varint(input: &[u8], position: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let byte = *input
            .get(*position)
            .ok_or_else(|| model_error("truncated ONNX protobuf"))?;
        *position += 1;
        if shift == 63 && byte > 1 {
            return Err(model_error("invalid ONNX protobuf varint"));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
    }
    Err(model_error("invalid ONNX protobuf varint"))
}

struct Field<'a> {
    number: u32,
    wire: u8,
    encoded: &'a [u8],
    value: &'a [u8],
}

impl<'a> Field<'a> {
    fn message(&self) -> Result<&'a [u8]> {
        if self.wire != LENGTH_DELIMITED {
            return Err(model_error("invalid ONNX protobuf field type"));
        }
        Ok(self.value)
    }

    fn varint(&self) -> Result<u64> {
        if self.wire != 0 {
            return Err(model_error("invalid ONNX protobuf field type"));
        }
        let mut position = 0;
        read_varint(self.value, &mut position)
    }
}

struct Fields<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Fields<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    fn next_field(&mut self) -> Result<Option<Field<'a>>> {
        if self.position == self.input.len() {
            return Ok(None);
        }
        let start = self.position;
        let key = read_varint(self.input, &mut self.position)?;
        let number = u32::try_from(key >> 3).map_err(|_| model_error("invalid ONNX field"))?;
        if number == 0 {
            return Err(model_error("invalid ONNX field number"));
        }
        let wire = (key & 7) as u8;
        let value_start;
        let end;
        match wire {
            0 => {
                value_start = self.position;
                read_varint(self.input, &mut self.position)?;
                end = self.position;
            }
            1 => {
                value_start = self.position;
                end = self
                    .position
                    .checked_add(8)
                    .ok_or_else(|| model_error("invalid ONNX field length"))?;
                self.position = end;
            }
            LENGTH_DELIMITED => {
                let length = usize::try_from(read_varint(self.input, &mut self.position)?)
                    .map_err(|_| model_error("invalid ONNX field length"))?;
                value_start = self.position;
                end = self
                    .position
                    .checked_add(length)
                    .ok_or_else(|| model_error("invalid ONNX field length"))?;
                self.position = end;
            }
            5 => {
                value_start = self.position;
                end = self
                    .position
                    .checked_add(4)
                    .ok_or_else(|| model_error("invalid ONNX field length"))?;
                self.position = end;
            }
            _ => return Err(model_error("unsupported ONNX protobuf wire type")),
        }
        if end > self.input.len() {
            return Err(model_error("truncated ONNX protobuf"));
        }
        Ok(Some(Field {
            number,
            wire,
            encoded: &self.input[start..end],
            value: &self.input[value_start..end],
        }))
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<Field<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_field().transpose()
    }
}

fn model_error(message: &str) -> InferenceError {
    InferenceError::ModelLoadError(format!("Failed to promote ONNX model to FP32: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_fp16_tensor_and_preserves_unknown_fields() {
        let mut tensor = Vec::new();
        write_varint_field(&mut tensor, 2, FLOAT16);
        write_message(&mut tensor, 9, |output| {
            output.extend_from_slice(&f16::from_f32(1.5).to_bits().to_le_bytes());
            output.extend_from_slice(&f16::from_f32(-2.0).to_bits().to_le_bytes());
            Ok(())
        })
        .unwrap();
        write_varint_field(&mut tensor, 99, 7);

        let mut graph = Vec::new();
        write_message(&mut graph, 5, |output| {
            output.extend_from_slice(&tensor);
            Ok(())
        })
        .unwrap();
        let mut model = Vec::new();
        write_message(&mut model, 7, |output| {
            output.extend_from_slice(&graph);
            Ok(())
        })
        .unwrap();

        let promoted = promote_fp16_to_fp32(&model).unwrap().unwrap();
        let graph = find_bytes(&promoted, 7).unwrap().unwrap();
        let tensor = find_bytes(graph, 5).unwrap().unwrap();
        assert_eq!(find_varint(tensor, 2).unwrap(), Some(FLOAT));
        assert_eq!(find_varint(tensor, 99).unwrap(), Some(7));
        assert_eq!(
            find_bytes(tensor, 9).unwrap().unwrap(),
            [0, 0, 192, 63, 0, 0, 0, 192]
        );
        let metadata = find_bytes(&promoted, 14).unwrap().unwrap();
        assert_eq!(
            find_bytes(metadata, 1).unwrap(),
            Some(b"quantize".as_slice())
        );
        assert_eq!(find_bytes(metadata, 2).unwrap(), Some(b"32".as_slice()));
        assert!(promote_fp16_to_fp32(&promoted).unwrap().is_none());
    }

    #[test]
    fn promotes_packed_fp16_int32_data() {
        let mut tensor = Vec::new();
        write_varint_field(&mut tensor, 2, FLOAT16);
        write_message(&mut tensor, 5, |output| {
            write_varint(output, u64::from(f16::from_f32(1.5).to_bits()));
            write_varint(output, u64::from(f16::from_f32(-2.0).to_bits()));
            Ok(())
        })
        .unwrap();
        let mut graph = Vec::new();
        write_message(&mut graph, 5, |output| {
            output.extend_from_slice(&tensor);
            Ok(())
        })
        .unwrap();
        let mut model = Vec::new();
        write_message(&mut model, 7, |output| {
            output.extend_from_slice(&graph);
            Ok(())
        })
        .unwrap();

        let promoted = promote_fp16_to_fp32(&model).unwrap().unwrap();
        let tensor = find_bytes(find_bytes(&promoted, 7).unwrap().unwrap(), 5)
            .unwrap()
            .unwrap();
        assert!(!has_field(tensor, 5).unwrap());
        assert_eq!(
            find_bytes(tensor, 9).unwrap().unwrap(),
            [0, 0, 192, 63, 0, 0, 0, 192]
        );
    }
}
