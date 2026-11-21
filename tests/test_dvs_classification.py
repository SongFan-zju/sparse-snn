import torch
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.dvs_classification import DVSFrames, make_ann, install_qcfs, EventResNet
from src.conversion.qcfs_ecl_resnet18 import ECLResNet18


def check_constant_event_stream_matches_existing_conversion(architecture, L=4):
    torch.set_num_threads(2)
    torch.manual_seed(2026100321)
    ann = install_qcfs(make_ann(architecture), L=L).eval()
    image = torch.randn(2, 2, 8, 8)
    stream = image[:, None].expand(-1, 4, -1, -1, -1)
    with torch.no_grad():
        reference = ECLResNet18(ann, timesteps=4)(image)
        for method in ('ecl', 'triangular', 'density'):
            assert torch.equal(EventResNet(ann, 4, method)(stream), reference)


def test_event_order_used_and_density_has_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(2026100321)
    ann = install_qcfs(make_ann('resnet20'), L=4).eval()
    for node in ann.modules():
        if hasattr(node, 'thresh'):
            node.thresh.data.fill_(.5)
    # Source weights must be shared; the only arm difference is backward.
    model = EventResNet(ann, 4, 'density').train()
    baseline = EventResNet(ann, 4, 'triangular').train()
    stream = torch.randn(2, 4, 2, 8, 8, requires_grad=True)
    before = {k: v.clone() for k, v in model.named_buffers()}
    output = model(stream)
    assert torch.equal(output, baseline(stream))
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert stream.grad is not None and torch.isfinite(stream.grad).all()
    assert torch.count_nonzero(model.source.conv1.weight.grad)
    assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())


class DVSConversionTests(unittest.TestCase):
    def test_L2_T4_conversion_for_both_backbones(self):
        for architecture in ['resnet18','resnet20']:
            check_constant_event_stream_matches_existing_conversion(architecture,L=2)

    def test_completed_relu_reuse_preserves_weights_and_rejects_wrong_data(self):
        from scripts.train_dvs import load_relu_reference, SEED
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'relu'
            folder.mkdir()
            model = make_ann('resnet20')
            state = dict(model=model.state_dict(),epoch=300,epochs=300,seed=SEED,examples=2700000)
            torch.save(state,folder/'last.pt')
            config = dict(architecture='resnet20',manifest_sha256='fixed-data',dataset='CIFAR10-DVS',num_classes=10)
            (root/'config.json').write_text(json.dumps(dict(config,L=4)))
            args = SimpleNamespace(relu_source=folder/'last.pt',relu_epochs=300)
            restored = make_ann('resnet20')
            reference = load_relu_reference(restored,args,dict(config,L=2))
            self.assertEqual(reference['new_ReLU_training_examples'],0)
            self.assertTrue(all(torch.equal(value,restored.state_dict()[key]) for key,value in model.state_dict().items()))
            with self.assertRaises(ValueError):
                load_relu_reference(restored,args,dict(config,manifest_sha256='different-data'))

    def test_original_gesture_mapping_merges_roll_directions(self):
        from scripts.prepare_dvs128 import load_gesture_names
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'gesture_mapping.csv'
            rows = [f'action_{label},{label}' for label in range(1,12)]
            rows.insert(8,'forearm_roll_backward,8')
            path.write_text('action,label\n'+'\n'.join(rows)+'\n')
            names = load_gesture_names(path)
            self.assertEqual(len(names),11)
            self.assertIn('action_8',names[7])
            self.assertIn('forearm_roll_backward',names[7])
            self.assertEqual(names[10],'action_11')

    def test_official_gesture_dataset_retains_handed_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root/'sample.npz',frames=np.ones((4,2,128,128),dtype=np.uint32))
            manifest = dict(dataset='DVS128 Gesture',status='complete',official_split=True,
                            classes=[str(i) for i in range(11)],T=4,
                            samples=[dict(path='sample.npz',target=10,split='train')])
            (root/'manifest.json').write_text(json.dumps(manifest))
            dataset = DVSFrames(root/'manifest.json','train',training=True)
            self.assertTrue(dataset.official_split)
            self.assertEqual(dataset.num_classes,11)
            with patch('torch.rand',side_effect=AssertionError('Gesture must not horizontally flip')):
                image,target = dataset[0]
            self.assertEqual(target,10)
            self.assertEqual(image.shape,(4,2,48,48))

    def test_gesture_head_and_conversion_accept_eleventh_class(self):
        torch.set_num_threads(2)
        ann = install_qcfs(make_ann('resnet20',num_classes=11),L=4).eval()
        model = EventResNet(ann,4,'density').train()
        logits = model(torch.randn(2,4,2,8,8))
        self.assertEqual(logits.shape,(2,11))
        torch.nn.functional.cross_entropy(logits,torch.tensor([10,0])).backward()
        self.assertTrue(bool(model.source.conv1.weight.grad.isfinite().all()))

    def test_resnet18_conversion(self):
        check_constant_event_stream_matches_existing_conversion('resnet18')

    def test_resnet20_conversion(self):
        check_constant_event_stream_matches_existing_conversion('resnet20')

    def test_density_gradient(self):
        test_event_order_used_and_density_has_gradients()


if __name__ == '__main__':
    unittest.main()
