"""Self-contained CPU checks for the native decision Snake environment.

Run with ``python3 -m unittest tests.test_decision_snake -v`` (or pytest).
The fixture was recorded from the original campaign engine before vendoring it.
There is no dependency on that checkout, Python packages, CUDA, or graphics.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/decision_snake_reference.json").read_text())


class DecisionSnakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="decision-snake-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.executables = {}
        engine = Path(cls.directory.name) / "snake-engine.o"
        command = shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-O2", "-c", str(ROOT / "ocean/decision_snake/engine/snake.c"),
            "-o", str(engine),
        ]
        if not command or shutil.which(command[0]) is None:
            raise unittest.SkipTest("C compiler unavailable")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
        for language, variable, default, standard in (
            ("C", "CC", "cc", "c11"),
            ("C++", "CXX", "c++", "c++17"),
        ):
            compiler = shlex.split(os.environ.get(variable, default))
            if not compiler or shutil.which(compiler[0]) is None:
                raise unittest.SkipTest(f"{language} compiler unavailable: {compiler}")
            output = Path(cls.directory.name) / ("snake-c" if language == "C" else "snake-cpp")
            command = compiler + [
                "-std=" + standard, "-O2", "-Wall", "-Wextra",
                "-Wno-unused-parameter", "-Wno-unused-function", "-Wno-missing-field-initializers",
                "-DPUF_HEADLESS", "-I", str(ROOT / "src"),
                "-x", "c" if language == "C" else "c++",
                str(ROOT / "tests/test_decision_snake.c"), "-x", "none", str(engine),
                "-lm", "-o", str(output),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
            cls.executables[language] = output

    def run_case(self, executable, case, mode):
        result = subprocess.run(
            [str(executable), mode, str(case["seed"]), str(case["max_steps"]), case["actions"]],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def check_trace(self, output, case, automatic):
        rows = output.splitlines()
        self.assertTrue(rows[-1].startswith("LIVE,"), rows[-1])
        live = list(map(int, rows.pop().split(",")[1:]))
        self.assertEqual(len(rows), case["rows"])
        self.assertEqual(list(map(int, rows[0].split(","))), case["first"])
        self.assertEqual(list(map(int, rows[-1].split(","))), case["final"])
        digest = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()
        self.assertEqual(digest, case["trace_sha256"], case["name"])
        # LIVE includes the post-step buffers that PufferLib actually consumes.
        # At a boundary the transition retains the final state; only puf_step
        # replaces live observations/masks with the next episode's reset state.
        episode_seed = (case["seed"] + int(automatic)) % 2**32
        self.assertEqual(live[:4], [episode_seed, (episode_seed + 1) % 2**32, 1, case["final"][6]])
        expected = FIXTURE["resets"][str(episode_seed)] if automatic else case["final"]
        self.assertEqual(live[4:], expected[7:])

    def test_explicit_step_matches_original_campaign_traces(self):
        for language, executable in self.executables.items():
            for case in FIXTURE["cases"]:
                with self.subTest(language=language, case=case["name"]):
                    self.check_trace(self.run_case(executable, case, "explicit"), case, False)

    def test_autoreset_preserves_final_transition_and_episode_metrics(self):
        for language, executable in self.executables.items():
            for case in FIXTURE["cases"]:
                with self.subTest(language=language, case=case["name"]):
                    self.check_trace(self.run_case(executable, case, "automatic"), case, True)

    def test_interleaved_environments_do_not_change_seeded_games(self):
        for language, executable in self.executables.items():
            for case in FIXTURE["cases"]:
                with self.subTest(language=language, case=case["name"]):
                    self.check_trace(self.run_case(executable, case, "interleaved"), case, True)

    def test_explicit_reset_replays_the_same_episode(self):
        for language, executable in self.executables.items():
            for case in FIXTURE["cases"]:
                with self.subTest(language=language, case=case["name"]):
                    output = self.run_case(executable, case, "repeat")
                    first, second = output.split("REPEAT\n")
                    self.assertEqual(first, second)
                    self.check_trace(first, case, False)

    def test_rejected_actions_leave_transition_and_rng_unchanged(self):
        for language, executable in self.executables.items():
            for case in FIXTURE["cases"]:
                with self.subTest(language=language, case=case["name"]):
                    self.check_trace(self.run_case(executable, case, "reject"), case, False)


if __name__ == "__main__":
    unittest.main()
