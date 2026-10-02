"""Compare the native tokenizer bridge with the Hugging Face Rust reference.

Run with Python's tokenizers package installed and --executable pointing to the
C++ test harness. --laya-snapshot optionally includes a local, pinned Laya
snapshot; ordinary tests generate small tokenizers without network access.
"""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

try:
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
except ImportError:
    Tokenizer = None


CORPUS = [
    "Hello world! A decision about café and Cafe\u0301.",
    "Choose up, down, left, or right. Numbers: 123.50 and -20.",
    "日本語の質問。 東京と大阪。", "😀🧪 👩🏽‍💻 Unicode text",
    "  Several spaces\tand\nnewlines. Fullwidth Ａ and mathematical 𝐀.",
    "[MASK] option marker and <decision> special token.",
] * 5
CASES = [
    ("", None), ("Hello WORLD café", None), ("Cafe\u0301 CAFE\u0301", ""),
    ("日本語の質問。 東京と大阪", None), ("😀🧪 👩🏽‍💻", None),
    ("  tabs\tand\nspaces  ", None), ("[CLS] [MASK] [SEP]", None),
    ("a<decision> b", None), ("embedded\0null", None),
    ("𝐀 fullwidthＡ Å", None), ("What's 123.50?", "Choice: up / DOWN"),
    ("日本語です。", "cafe\u0301 😀"), ("long text " * 300, None),
]
SPECIAL = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
LOOKUPS = SPECIAL + ["<decision>", "unknown-🧪-token", "\0"]


def tiny_tokenizer(family):
    if family == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(vocab_size=180, special_tokens=SPECIAL)
        tokenizer.decoder = decoders.WordPiece(prefix="##")
    elif family == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                     special_tokens=SPECIAL)
        tokenizer.decoder = decoders.ByteLevel()
    elif family == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")])
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        trainer = trainers.UnigramTrainer(vocab_size=160, special_tokens=SPECIAL, unk_token="[UNK]")
        tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    else:
        raise ValueError(family)
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    tokenizer.add_special_tokens([AddedToken("<decision>", lstrip=True, rstrip=True,
                                            normalized=False, special=True)])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                        ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )
    return tokenizer


@unittest.skipIf(Tokenizer is None, "native tokenizer parity needs the tokenizers package")
class NativeTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("PUFFER_TOKENIZER_TEST_BINARY")
        if not executable:
            raise unittest.SkipTest("set PUFFER_TOKENIZER_TEST_BINARY or pass --executable")
        cls.executable = str(Path(executable).resolve())
        if not Path(cls.executable).is_file():
            raise AssertionError(f"missing native tokenizer harness: {cls.executable}")

    def compare(self, tokenizer):
        records = [(text, pair, special) for text, pair in CASES for special in (False, True)]
        with tempfile.TemporaryDirectory(prefix="puffer-tokenizer-") as directory:
            directory = Path(directory)
            model, inputs, outputs = (directory / name for name in ("tokenizer.json", "inputs.bin", "outputs.bin"))
            tokenizer.save(str(model))
            # High-level requests own their padding and length budgets; saved
            # tokenizer training defaults must not silently truncate inputs.
            reference = Tokenizer.from_file(str(model))
            reference.no_padding()
            reference.no_truncation()
            with inputs.open("wb") as stream:
                stream.write(b"PUFTOK1\0")
                stream.write(struct.pack("<I", len(records)))
                stream.write(struct.pack("<I", len(LOOKUPS)))
                for text in LOOKUPS:
                    stream.write(struct.pack("<I", len(text.encode())))
                    stream.write(text.encode())
                for text, pair, special in records:
                    encoded = text.encode()
                    stream.write(struct.pack("<II", special, len(encoded)))
                    stream.write(encoded)
                    stream.write(struct.pack("<I", 0xFFFFFFFF if pair is None else len(pair.encode())))
                    if pair is not None:
                        stream.write(pair.encode())
            result = subprocess.run([self.executable, str(model), str(inputs), str(outputs)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with outputs.open("rb") as stream:
                self.assertEqual(struct.unpack("<I", stream.read(4))[0], len(records))
                self.assertEqual(struct.unpack("<II", stream.read(8)),
                                 (reference.get_vocab_size(False), reference.get_vocab_size(True)))
                self.assertEqual(struct.unpack("<I", stream.read(4))[0], len(LOOKUPS))
                for text in LOOKUPS:
                    found, value = struct.unpack("<II", stream.read(8))
                    self.assertEqual(value if found else None, reference.token_to_id(text))
                for text, pair, special in records:
                    count = struct.unpack("<I", stream.read(4))[0]
                    actual = [list(struct.unpack("<" + "I" * count, stream.read(4 * count)))
                              for _ in range(3)]
                    encoded = reference.encode(text, pair, add_special_tokens=special)
                    expected = [encoded.ids, encoded.type_ids, encoded.attention_mask]
                    with self.subTest(text=text[:60], pair=pair, special=special):
                        self.assertEqual(actual, expected)
                    for skip in (False, True):
                        size = struct.unpack("<I", stream.read(4))[0]
                        decoded = stream.read(size).decode()
                        self.assertEqual(decoded, reference.decode(encoded.ids, skip_special_tokens=skip))
                self.assertFalse(stream.read(1), "trailing native tokenizer test output")

    def test_wordpiece(self):
        self.compare(tiny_tokenizer("wordpiece"))

    def test_byte_level_bpe(self):
        self.compare(tiny_tokenizer("bpe"))

    def test_unigram(self):
        self.compare(tiny_tokenizer("unigram"))

    def test_saved_padding_and_truncation_do_not_change_requests(self):
        tokenizer = tiny_tokenizer("bpe")
        tokenizer.enable_padding(length=8, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        tokenizer.enable_truncation(max_length=8)
        self.compare(tokenizer)

    def test_actual_laya_tokenizer(self):
        snapshot = os.environ.get("PUFFER_LAYA_SNAPSHOT")
        if not snapshot:
            self.skipTest("pass --laya-snapshot to include the pinned public tokenizer")
        self.compare(Tokenizer.from_file(str(Path(snapshot) / "tokenizer/tokenizer.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--laya-snapshot")
    args, remaining = parser.parse_known_args()
    if args.executable:
        os.environ["PUFFER_TOKENIZER_TEST_BINARY"] = args.executable
    if args.laya_snapshot:
        os.environ["PUFFER_LAYA_SNAPSHOT"] = args.laya_snapshot
    unittest.main(argv=[__file__, *remaining], verbosity=2)
