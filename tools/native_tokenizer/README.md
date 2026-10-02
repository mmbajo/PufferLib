# Native tokenizer bridge

This static library loads a supplied Hugging Face `tokenizer.json` and runs its
normalizer, pre-tokenizer, model, added tokens, post-processor and decoder. It
uses Hugging Face `tokenizers` **0.23.2**, pinned with all transitive dependencies
in `Cargo.lock`. BPE, WordPiece and Unigram are supported by that implementation.
The runtime uses native code and does not load Python or download model files.

Build with Rust/Cargo and a C compiler (tested with Rust 1.90.0 and GCC 11):

```sh
./tools/build_native_tokenizer.sh
g++ -std=c++17 your_program.cpp build/libpuffer_tokenizer.a -ldl -lpthread -lm -o your_program
```

`CARGO` can select a Cargo executable; ordinary `CARGO_HOME`, `RUSTUP_HOME`,
`RUSTC`, `CC`, and `CARGO_TARGET_DIR` overrides work. The script does not install
dependencies or toolchains. The first Cargo build fetches the locked crates;
subsequent builds can use Cargo's local cache. The optional first argument
changes the output library path. No CMake, ICU headers, Python or SentencePiece
installation is required. The Oniguruma regex dependency builds its native
implementation from the locked crate source.

The native/reference comparison can be run with a Python environment containing
`tokenizers==0.23.2` (Python is used only as the test oracle):

```sh
g++ -std=c++17 tests/test_native_tokenizer.cpp build/libpuffer_tokenizer.a \
    -ldl -lpthread -lm -o build/test_native_tokenizer
python tests/test_native_tokenizer.py --executable build/test_native_tokenizer
# Include an already downloaded Laya snapshot:
python tests/test_native_tokenizer.py --executable build/test_native_tokenizer \
    --laya-snapshot /path/to/laya/snapshot
```

Include `src/pretrained_tokenizer.h` from C or C++17. C++ callers can use
`pretrained::Tokenizer` with `Encode`, `EncodePair`, `Decode`, `TokenId`,
`HasId`, `MaxTokenId`, and `VocabSize`. `EncodeDetailed` and `EncodePairDetailed` also return
token type IDs and attention masks. UTF-8 buffers are length-delimited, including
embedded NUL. Malformed UTF-8, invalid tokenizer files and tokenizer errors are
reported as errors, not silently replaced. C++ errors throw `std::runtime_error`;
C callers receive a status code and owned error text. Free C results with the
matching functions in the header.

Loading clears any padding/truncation stored in `tokenizer.json`, matching the
default unpadded, untruncated high-level Hugging Face encoding path. Applications
control batch padding and sequence budgets. Special-token insertion is explicit
on every encode call; it follows the tokenizer's own post-processor. Decode can
preserve or skip special tokens. The model importer must also preserve its
tokenizer configuration/special-token metadata and check token IDs against the
embedding vocabulary; a tokenizer file does not establish model compatibility.
Serialized tokenizer features unsupported by the pinned library fail to load.
Decode follows the tokenizer JSON decoder; Python `AutoTokenizer` convenience
cleanup outside that decoder is not applied.

The bridge is covered by the repository's MIT license. Hugging Face tokenizers
is [Apache-2.0](https://github.com/huggingface/tokenizers/blob/v0.23.2/LICENSE).
Its license is included in `LICENSE.tokenizers`.
Its pinned crate checksum is
`7afbf6e88718afcc138bad01d6ccc3051dbbc3b2ce9793d8b8a3aeb610969cfc`.
Dependency licenses remain in the Cargo registry sources; a binary distributor
must include the applicable dependency notices and licenses.
