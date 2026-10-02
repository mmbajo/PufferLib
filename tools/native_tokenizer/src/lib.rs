//! Length-safe C interface to the pinned Hugging Face tokenizer implementation.
//! Handles own all model data; returned buffers must use the corresponding free.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{ptr, slice, str};
use tokenizers::Tokenizer;

pub struct PufTokenizer(Tokenizer);

#[repr(C)]
pub struct PufTokenEncoding {
    ids: *mut u32,
    type_ids: *mut u32,
    attention_mask: *mut u32,
    length: usize,
}

// Catch Rust panics so they cannot unwind through C/C++ callers. Callers must
// still pass valid pointer/length pairs; arbitrary invalid addresses are UB.
unsafe fn call(error: *mut *mut c_char, f: impl FnOnce() -> Result<(), String>) -> i32 {
    if !error.is_null() {
        *error = ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("Native tokenizer panic".into()));
    match result {
        Ok(()) => 0,
        Err(message) => {
            if !error.is_null() {
                *error = CString::new(message.replace('\0', "\\0")).unwrap().into_raw();
            }
            -1
        }
    }
}

unsafe fn utf8<'a>(data: *const c_char, length: usize) -> Result<&'a str, String> {
    if length == 0 {
        return Ok("");
    }
    if data.is_null() {
        return Err("NULL UTF-8 buffer with nonzero length".into());
    }
    str::from_utf8(slice::from_raw_parts(data.cast(), length)).map_err(|e| e.to_string())
}

unsafe fn tokenizer<'a>(handle: *const PufTokenizer) -> Result<&'a Tokenizer, String> {
    handle.as_ref().map(|h| &h.0).ok_or_else(|| "NULL tokenizer".into())
}

unsafe fn output<'a, T>(out: *mut T) -> Result<&'a mut T, String> {
    out.as_mut().ok_or_else(|| "NULL output pointer".into())
}

fn prepare(mut tokenizer: Tokenizer) -> Result<PufTokenizer, String> {
    // The application assembles/truncates sequences and pads model batches.
    // Keep normalization, pre-tokenization, added tokens and post-processing.
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    Ok(PufTokenizer(tokenizer))
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_from_file(
    path: *const c_char, length: usize, out: *mut *mut PufTokenizer, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        let out = output(out)?;
        *out = ptr::null_mut();
        let model = Tokenizer::from_file(utf8(path, length)?).map_err(|e| e.to_string())?;
        *out = Box::into_raw(Box::new(prepare(model)?));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_from_json(
    json: *const c_char, length: usize, out: *mut *mut PufTokenizer, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        let out = output(out)?;
        *out = ptr::null_mut();
        let model = Tokenizer::from_bytes(utf8(json, length)?.as_bytes()).map_err(|e| e.to_string())?;
        *out = Box::into_raw(Box::new(prepare(model)?));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_free(handle: *mut PufTokenizer) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

fn leak_u32(values: &[u32]) -> *mut u32 {
    Box::into_raw(values.to_vec().into_boxed_slice()).cast()
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_encode(
    handle: *const PufTokenizer, text: *const c_char, text_length: usize,
    pair: *const c_char, pair_length: usize, has_pair: i32, add_special_tokens: i32,
    out: *mut PufTokenEncoding, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        let out = output(out)?;
        *out = PufTokenEncoding {
            ids: ptr::null_mut(), type_ids: ptr::null_mut(),
            attention_mask: ptr::null_mut(), length: 0,
        };
        let model = tokenizer(handle)?;
        let text = utf8(text, text_length)?;
        let encoded = if has_pair != 0 {
            model.encode((text, utf8(pair, pair_length)?), add_special_tokens != 0)
        } else {
            model.encode(text, add_special_tokens != 0)
        }.map_err(|e| e.to_string())?;
        out.ids = leak_u32(encoded.get_ids());
        out.type_ids = leak_u32(encoded.get_type_ids());
        out.attention_mask = leak_u32(encoded.get_attention_mask());
        out.length = encoded.len();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_encoding_free(out: *mut PufTokenEncoding) {
    if let Some(out) = out.as_mut() {
        for buffer in [&mut out.ids, &mut out.type_ids, &mut out.attention_mask] {
            if !buffer.is_null() {
                drop(Box::from_raw(ptr::slice_from_raw_parts_mut(*buffer, out.length)));
                *buffer = ptr::null_mut();
            }
        }
        out.length = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_decode(
    handle: *const PufTokenizer, ids: *const u32, length: usize,
    skip_special_tokens: i32, out: *mut *mut c_char, out_length: *mut usize,
    error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        let out = output(out)?;
        let out_length = output(out_length)?;
        *out = ptr::null_mut();
        *out_length = 0;
        let ids = if length == 0 { &[] } else {
            if ids.is_null() { return Err("NULL token IDs with nonzero length".into()); }
            slice::from_raw_parts(ids, length)
        };
        let decoded = tokenizer(handle)?.decode(ids, skip_special_tokens != 0)
            .map_err(|e| e.to_string())?.into_bytes().into_boxed_slice();
        *out_length = decoded.len();
        *out = Box::into_raw(decoded).cast();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_string_free(string: *mut c_char) {
    if !string.is_null() { drop(CString::from_raw(string)); }
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_bytes_free(bytes: *mut c_char, length: usize) {
    if !bytes.is_null() {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(bytes.cast::<u8>(), length)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_token_id(
    handle: *const PufTokenizer, token: *const c_char, length: usize,
    id: *mut u32, found: *mut i32, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        let id = output(id)?;
        let found = output(found)?;
        let result = tokenizer(handle)?.token_to_id(utf8(token, length)?);
        *found = i32::from(result.is_some());
        *id = result.unwrap_or(0);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_vocab_size(
    handle: *const PufTokenizer, with_added_tokens: i32,
    size: *mut usize, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        *output(size)? = tokenizer(handle)?.get_vocab_size(with_added_tokens != 0);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_has_id(
    handle: *const PufTokenizer, id: u32, found: *mut i32, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        *output(found)? = i32::from(tokenizer(handle)?.id_to_token(id).is_some());
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn puf_tokenizer_max_id(
    handle: *const PufTokenizer, id: *mut u32, error: *mut *mut c_char,
) -> i32 {
    call(error, || {
        *output(id)? = tokenizer(handle)?.get_vocab(true).values().copied().max()
            .ok_or_else(|| "Empty tokenizer vocabulary".to_string())?;
        Ok(())
    })
}
