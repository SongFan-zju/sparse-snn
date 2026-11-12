"""Portable checkpoints and deterministic continuation of the actual public runner."""
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.config import DEFAULT_CONFIG, DEFAULT_SPLIT, recipe
from src.data import partitions
from src.models.factory import backbone, convert, install_qat, load_model, activation_sites
from src.runtime import configure, cpu_state, save
from src.training import train


class PublicPipelineTests(unittest.TestCase):
    def setUp(self):
        configure('cpu', 2026100321, 2)

    def test_fixed_partitions(self):
        roles = partitions('cifar100', 50000, DEFAULT_SPLIT, 2026100321)
        self.assertEqual({k: len(v) for k, v in roles.items()},
                         dict(training=42000, calibration=1000, selection=2000, confirmation=5000))
        self.assertEqual(len(np.unique(np.concatenate(list(roles.values())))), 50000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            bad = {k: v.tolist() for k, v in roles.items()}
            bad['training'][0] = bad['calibration'][0]
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                partitions('cifar100', 50000, path, 2026100321)

    def test_checkpoint_alone_restores_each_snn_recipe(self):
        experiments = json.loads(DEFAULT_CONFIG.read_text())['experiments']
        with tempfile.TemporaryDirectory() as tmp:
            for name in experiments:
                cfg = recipe(name)
                if cfg['stage'] != 'snn':
                    continue
                with self.subTest(name=name):
                    cfg['base_width'] = 2
                    source = backbone(cfg)
                    thresholds = {n: torch.ones(c) for n, c in activation_sites(source, cfg)}
                    source = install_qat(source, thresholds, cfg)
                    model = convert(source, cfg, training=True).eval()
                    x = torch.randn(2, 3, 32, 32)
                    # Grad-enabled forward exercises the training operator's hard path.
                    expected = model(x)
                    expected.sum().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.requires_grad))
                    path = Path(tmp) / 'only_checkpoint.pt'
                    save(path, dict(model=cpu_state(model), config=cfg))
                    restored = load_model(path, cfg).eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(restored(x), expected.detach(), rtol=0, atol=0)
                    if cfg['timesteps'] == 2:
                        with self.assertRaises(ValueError):
                            load_model(path, dict(cfg, timesteps=4))

    def test_interrupted_training_resumes_exactly_without_test_data(self):
        cfg = recipe('c100-ann')
        cfg.update(dataset='cifar10', base_width=2, epochs=2, batch_size=5,
                   training_examples=10, eval_epochs=[])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / 'cifar-10-batches-py'
            data.mkdir()
            rng = np.random.default_rng(cfg['seed'])
            for i in range(1, 6):
                with (data / f'data_batch_{i}').open('wb') as handle:
                    pickle.dump(dict(data=rng.integers(0, 256, (2, 3072), dtype=np.uint8),
                                     labels=[2 * (i - 1), 2 * (i - 1) + 1]), handle)
            args = SimpleNamespace(data=str(data), source=None, teacher=None, thresholds=None,
                                   split_file=str(DEFAULT_SPLIT), output=str(folder / 'complete'),
                                   resume=False, data_parallel=False, checkpoint_every=1)
            with patch('src.training.report'):
                train(cfg, args, torch.device('cpu'))
            expected = torch.load(Path(args.output) / 'latest.pt', weights_only=False)
            configure('cpu', cfg['seed'], 2)
            args.output = str(folder / 'interrupted')

            class Interrupted(Exception):
                pass

            def stop_after_first_epoch(row, *unused):
                if row.get('epoch') == 1:
                    raise Interrupted()

            with patch('src.training.report', side_effect=stop_after_first_epoch):
                with self.assertRaises(Interrupted):
                    train(cfg, args, torch.device('cpu'))
            # Fresh process state must be replaced by the saved random streams.
            torch.rand(7)
            args.resume = True
            with patch('src.training.report'):
                train(cfg, args, torch.device('cpu'))
            actual = torch.load(Path(args.output) / 'latest.pt', weights_only=False)
            self.assertEqual(actual['steps'], expected['steps'])
            self.assertEqual(actual['epoch'], 2)
            for key in expected['model']:
                torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(actual['rng']['order'], expected['rng']['order']))
            self.assertTrue(torch.equal(actual['rng']['augment'], expected['rng']['augment']))
            self.assertEqual(actual['rng']['mix'], expected['rng']['mix'])
            self.assertFalse((data / 'test_batch').exists())
            args.resume = False
            with self.assertRaises(FileExistsError):
                train(cfg, args, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
