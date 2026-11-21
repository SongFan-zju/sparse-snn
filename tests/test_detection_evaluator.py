"""Integration checks for the external author's VOC07 AP implementation."""
from pathlib import Path
import os
import pickle
import tempfile
import unittest

import numpy as np

from src.detection_fast_snn import upstream_runtime


class DetectionEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(os.environ.get('FAST_SNN_UPSTREAM',
                  'artifacts/fast_snn_detection_20261117/upstream/object detection'))
        if not (path/'models/yolov2_r34.py').is_file():
            raise unittest.SkipTest('Set FAST_SNN_UPSTREAM to an external Fast-SNN checkout')
        cls.runtime=upstream_runtime(path)

    def test_empty_class_has_zero_ap_and_perfect_detection_keeps_one_ap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'test.txt').write_text('000001\n')
            (folder/'det.txt').write_text('')
            record={'000001':[dict(name='dog',difficult=0,bbox=[1,1,11,11])]}
            with (folder/'annots.pkl').open('wb') as stream:
                pickle.dump(record,stream)
            evaluator=self.runtime.evaluator.VOCAPIEvaluator.__new__(
                self.runtime.evaluator.VOCAPIEvaluator)
            evaluator.imgsetpath=str(folder/'test.txt')
            rec,prec,ap=evaluator.voc_eval(str(folder/'det.txt'),'dog',str(folder))
            self.assertEqual(ap,0.)
            self.assertEqual((np.asarray(rec).size,np.asarray(prec).size),(0,0))
            (folder/'det.txt').write_text('000001 0.900 1.0 1.0 11.0 11.0\n')
            rec,prec,ap=evaluator.voc_eval(str(folder/'det.txt'),'dog',str(folder))
            self.assertAlmostEqual(ap,1.)
            self.assertEqual((rec[-1],prec[-1]),(1.,1.))


if __name__=='__main__':
    unittest.main()
