import unittest
from src.dimp_evaluation import overlap_and_distance, sot_metrics


class SOTMetricTests(unittest.TestCase):
    def test_xywh_overlap_and_missing_prediction(self):
        self.assertEqual(overlap_and_distance([1,2,3,4], [1,2,3,4]), (1.,0.))
        self.assertEqual(overlap_and_distance(None, [1,2,3,4])[0], 0.)
        self.assertEqual(overlap_and_distance([20,20,3,4], [1,2,3,4])[0], 0.)

    def test_unscored_initial_and_occlusion_frames_are_excluded(self):
        rows = [dict(score=False),dict(score=True,iou=1.,center_error=0.),
                dict(score=False),dict(score=True,iou=0.,center_error=float('inf'))]
        values = sot_metrics(rows)
        self.assertEqual(values, dict(total=2,ao=50.,sr50=50.,sr75=50.,precision20=50.))


if __name__ == '__main__':
    unittest.main()
