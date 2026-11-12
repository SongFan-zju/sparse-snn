import unittest
import torch

from src.conversion.general_phase_adjoint import hard_sequence
from src.conversion.integer_cap_phase import integer_cap_hard_sequence, IntegerCapPhaseSequence
from src.conversion.joint_phase_adjoint import JointPhaseEvent


class IntegerCapTests(unittest.TestCase):
    def test_m1_matches_original_events_and_t2_boundary_gradients(self):
        currents = torch.tensor([[.5, 1.5, 0., 1., .3, -.001, 2., 1.],
                                 [.5, -1.5, 0., -1., -.3, .001, -2., -1.001]], dtype=torch.float64)
        threshold = torch.tensor(1., dtype=torch.float64)
        gradient = torch.arange(16, dtype=torch.float64).reshape(2, 8) / 9
        reference = currents.clone().requires_grad_()
        expected = torch.stack(JointPhaseEvent.apply(reference[0], reference[1], threshold))
        expected_vjp, = torch.autograd.grad(expected, reference, gradient)
        actual = currents.clone().requires_grad_()
        events = IntegerCapPhaseSequence(threshold, 2, 1)(actual.flatten()).reshape_as(actual)
        vjp, = torch.autograd.grad(events, actual, gradient)
        torch.testing.assert_close(events, expected, rtol=0, atol=0)
        torch.testing.assert_close(vjp, expected_vjp, rtol=0, atol=0)
        torch.testing.assert_close(integer_cap_hard_sequence(currents, threshold, 1),
                                   hard_sequence(currents, threshold), rtol=0, atol=0)

    def test_capacity_and_retraction_constraints(self):
        currents = torch.tensor([[20., -20., 2., 0.], [-40., -10., -20., 20.],
                                 [30., 20., 8., -30.], [-50., -50., -50., -50.]])
        unit = torch.tensor(.25)
        events = integer_cap_hard_sequence(currents, unit, 3)
        counts = (events / unit).round()
        self.assertTrue((counts.abs() <= 3).all())
        self.assertTrue((counts.cumsum(0) >= 0).all())
        self.assertTrue((counts == 3).any() and (counts == -3).any())
        bits = (counts.abs().long() & 1) + 2 * ((counts.abs().long() >> 1) & 1)
        torch.testing.assert_close(events, counts.sign() * bits * unit, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
