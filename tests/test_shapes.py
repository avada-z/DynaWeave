import argparse
import unittest

import torch

from experiments.shape_scenes import HELD_OUT_PAIRS, ShapeScenes, held_out
from dynaweave.model import check


class ShapeTests(unittest.TestCase):
    def test_generator_is_repeatable(self):
        first = next(iter(ShapeScenes(size=128, seed=13)))
        second = next(iter(ShapeScenes(size=128, seed=13)))
        self.assertEqual(first['caption'], second['caption'])
        self.assertTrue(torch.equal(first['image'], second['image']))
        self.assertEqual(first['image'].shape, (3, 128, 128))
        self.assertEqual(first['image'].dtype, torch.uint8)

    def test_held_out_combinations(self):
        stream = iter(ShapeScenes(size=64, objects=(1, 1), exclude=HELD_OUT_PAIRS))
        for _ in range(100):
            caption = next(stream)['caption']
            for colour, cell in HELD_OUT_PAIRS:
                self.assertFalse(f' {colour} ' in caption and caption.endswith(f'at the {cell}'))
        images, captions = held_out(8, size=64, objects=(1, 1), pairs=HELD_OUT_PAIRS)
        self.assertEqual(images.shape, (8, 3, 64, 64))
        for caption, (colour, cell) in zip(captions, HELD_OUT_PAIRS):
            self.assertIn(f' {colour} ', caption)
            self.assertTrue(caption.endswith(f'at the {cell}'))

    def test_autoencoder_flow_and_larger_canvas(self):
        torch.set_num_threads(2)
        check(argparse.Namespace())


if __name__ == '__main__':
    unittest.main()
