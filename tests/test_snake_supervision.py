"""Replay generated labels through the C engine and check board-disjoint splits."""
import _ctypes
import ctypes
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ('up', 'down', 'left', 'right')


class SnakeSupervisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='snake-supervision-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        cls.generator = cls.work / 'generate'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
            str(ROOT / 'tools/generate_snake_supervision.c'), '-o', str(cls.generator)], check=True)
        library = cls.work / 'snake.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-shared', '-fPIC',
            str(ROOT / 'ocean/decision_snake/engine/snake.c'), '-o', str(library)], check=True)
        cls.engine = ctypes.CDLL(str(library))
        cls.addClassCleanup(_ctypes.dlclose, cls.engine._handle)
        cls.engine.ib_snake_create.argtypes = [ctypes.c_uint32, ctypes.c_int]
        cls.engine.ib_snake_create.restype = ctypes.c_void_p
        cls.engine.ib_snake_destroy.argtypes = [ctypes.c_void_p]
        cls.engine.ib_snake_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
        cls.engine.ib_snake_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cls.engine.ib_snake_observe.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        for name in ('legal_actions', 'terminated', 'truncated', 'outcome', 'score'):
            function = getattr(cls.engine, 'ib_snake_' + name)
            function.argtypes = [ctypes.c_void_p]
        cls.engine.ib_snake_score.restype = ctypes.c_double
        cls.env = cls.engine.ib_snake_create(73, 64)
        cls.addClassCleanup(cls.engine.ib_snake_destroy, cls.env)
        cls.paths = [cls.work / 'train.jsonl', cls.work / 'heldout.jsonl']
        subprocess.run([str(cls.generator), *map(str, cls.paths), '32', '16', '73'],
            check=True, capture_output=True, text=True)
        cls.rows = [[json.loads(line) for line in path.read_text().splitlines()]
            for path in cls.paths]

    def replay(self, row, continuation=()):
        self.assertEqual(self.engine.ib_snake_reset(self.env, row['reset_seed'], 64), 0)
        for action in [*row['history'], *continuation]:
            self.assertEqual(self.engine.ib_snake_step(self.env, action), 0)

    def labels(self, row):
        self.replay(row)
        board = (ctypes.c_int32 * 100)()
        self.assertEqual(self.engine.ib_snake_observe(self.env, board, 100), 0)
        board = list(board)
        self.assertEqual(board, row['board'])
        head = tuple(map(int, re.search(r'Head=\((\d+),(\d+)\)', row['state']).groups()))
        food = tuple(map(int, re.search(r'Food=\((\d+),(\d+)\)', row['state']).groups()))
        body = re.search(r'Body neck-to-tail=(.*?)\.', row['state'])[1]
        body = [tuple(map(int, point)) for point in re.findall(r'\((\d+),(\d+)\)', body)]
        self.assertEqual(head, divmod(board.index(1), 10))
        self.assertEqual(food, divmod(board.index(-1), 10))
        self.assertEqual(body, [divmod(board.index(i), 10) for i in range(2, max(board) + 1)])
        neighbors = dict(re.findall(r'\b(up|down|left|right)=(\w+);', row['state']))
        self.assertEqual(set(neighbors), set(DIRECTIONS))
        for direction, (dr, dc) in zip(DIRECTIONS, ((-1, 0), (1, 0), (0, -1), (0, 1))):
            r, c = head[0] + dr, head[1] + dc
            kind = 'wall'
            if 0 <= r < 10 and 0 <= c < 10:
                cell = board[r * 10 + c]
                kind = {-1: 'food', 0: 'empty', 2: 'neck', max(board): 'tail'}.get(cell, 'body')
            self.assertEqual(neighbors[direction], kind)
        initial_score = self.engine.ib_snake_score(self.env)
        legal = self.engine.ib_snake_legal_actions(self.env)
        distances, collisions = {}, {}
        for first in range(4):
            if not legal & (1 << first):
                continue
            self.replay(row, (first,))
            collisions[first] = int(self.engine.ib_snake_outcome(self.env) == -1)
            self.assertEqual(collisions[first],
                int(neighbors[DIRECTIONS[first]] in ('wall', 'body')))
            distances[first] = 5
            frontier = [(first,)]
            while frontier:
                path = frontier.pop(0)
                self.replay(row, path)
                if self.engine.ib_snake_score(self.env) > initial_score:
                    distances[first] = len(path)
                    break
                if (len(path) < 4 and not self.engine.ib_snake_terminated(self.env)
                    and not self.engine.ib_snake_truncated(self.env)):
                    mask = self.engine.ib_snake_legal_actions(self.env)
                    frontier.extend((*path, action) for action in range(4) if mask & (1 << action))
        return distances, collisions

    def test_exact_labels_replay_and_unique_choice_answers(self):
        for rows in self.rows:
            for i in range(0, len(rows), 3):
                group = rows[i:i + 3]
                distances, collisions = self.labels(group[0])
                for row in group:
                    question = row['questions']['q']
                    self.assertEqual(list(row['questions']), ['q'])
                    self.assertEqual(row['board'], group[0]['board'])
                    self.assertEqual(row['state'], group[0]['state'])
                    self.assertLessEqual(len(row['history']), 60)
                    if question['type'] == 'choice':
                        offered = [DIRECTIONS.index(name) for name in question['criteria']]
                        values = [distances[action] for action in offered]
                        self.assertLessEqual(min(values), 4)
                        self.assertEqual(values.count(min(values)), 1)
                        self.assertEqual(row['target'], values.index(min(values)))
                    elif question['type'] == 'noul':
                        move = re.search(r'attempting (\w+) now', question['instructions'])[1]
                        self.assertEqual(row['target'], collisions[DIRECTIONS.index(move)])
                        self.assertNotIn('criteria', question)
                    else:
                        move = re.search(r'first moving (\w+),', question['instructions'])[1]
                        distance = distances[DIRECTIONS.index(move)]
                        target = distance - 1 if distance <= 2 else 2 if distance <= 4 else 3
                        self.assertEqual(row['target'], target)
                        self.assertEqual(question['criteria'],
                            ['1 move', '2 moves', '3-4 moves', 'not within 4 moves'])

    def test_direct_board_disjointness_and_class_counts(self):
        all_boards = []
        for rows, expected in zip(self.rows, (32, 16)):
            self.assertEqual(len(rows), expected * 3)
            counts = {'choice': [0] * 3, 'noul': [0] * 2, 'score': [0] * 4}
            candidate_counts = []
            for i in range(0, len(rows), 3):
                board = rows[i]['board']
                self.assertFalse(any(board == previous for previous in all_boards))
                all_boards.append(board)
                for row in rows[i:i + 3]:
                    question = row['questions']['q']
                    counts[question['type']][row['target']] += 1
                    if question['type'] == 'choice':
                        candidate_counts.append(len(question['criteria']))
            self.assertEqual(counts['noul'], [expected // 2] * 2)
            self.assertEqual(counts['score'], [expected // 4] * 4)
            self.assertEqual(candidate_counts.count(2), expected // 2)
            self.assertEqual(candidate_counts.count(3), expected // 2)
            self.assertTrue(all(counts['choice']))

    def test_generation_is_deterministic(self):
        repeated = [self.work / 'repeat-train.jsonl', self.work / 'repeat-heldout.jsonl']
        subprocess.run([str(self.generator), *map(str, repeated), '32', '16', '73'],
            check=True, capture_output=True, text=True)
        for before, after in zip(self.paths, repeated):
            self.assertEqual(before.read_bytes(), after.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
