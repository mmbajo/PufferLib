"""CPU-only exact resume and transactional snapshot checks, in C and C++."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DecisionSnakeStateTest(unittest.TestCase):
    def test_c_and_cpp_exact_continuation_and_corruption_rejection(self):
        outputs = []
        with tempfile.TemporaryDirectory(prefix="snake-state-tests-") as directory:
            engine = Path(directory) / "snake.o"
            compiler = shlex.split(os.environ.get("CC", "cc"))
            if not compiler or shutil.which(compiler[0]) is None:
                self.skipTest("C compiler unavailable")
            subprocess.run(compiler + ["-std=c11", "-O2", "-c",
                str(ROOT / "ocean/decision_snake/engine/snake.c"), "-o", str(engine)],
                cwd=ROOT, check=True, capture_output=True, text=True)
            for language, variable, default, standard in (
                ("c", "CC", "cc", "c11"), ("c++", "CXX", "c++", "c++17")
            ):
                with self.subTest(language=language):
                    compiler = shlex.split(os.environ.get(variable, default))
                    if not compiler or shutil.which(compiler[0]) is None:
                        self.skipTest(f"{language} compiler unavailable")
                    binary = Path(directory) / ("test-" + standard)
                    result = subprocess.run(compiler + ["-std=" + standard,
                        "-O2", "-Wall", "-Wextra", "-Wno-unused-parameter",
                        "-Wno-unused-function", "-Wno-missing-field-initializers",
                        "-I", str(ROOT / "src"), "-x", language,
                        str(ROOT / "tests/test_decision_snake_state.c"),
                        "-x", "none", str(engine), "-lm", "-o", str(binary)],
                        cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([str(binary)], cwd=ROOT,
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    outputs.append(result.stdout)
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
