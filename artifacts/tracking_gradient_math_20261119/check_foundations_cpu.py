"""Small CPU math checks; no dataset, model update, or CUDA initialization."""
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.conversion.bounded_phase import _vjp, bounded_phase_events
from src.tracking_vistream import _TriangleEvent

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def cdf(z):
    z = z.clamp(-1, 1)
    return torch.where(z <= 0, .5 * (z + 1).square(), 1 - .5 * (1 - z).square())


def phase_integrals(x, q0, c0, L, loss=None):
    # Every possible change lies at k-c0-q0-prefix(x), k=1,...,L.
    # Midpoint trajectories are piecewise constant; exact CDF cell masses.
    candidates = (torch.arange(1, L+1)[:, None] - c0 - q0 - x.cumsum(0)[None]).flatten()
    internal = candidates[(candidates > -1) & (candidates < 1)]
    boundaries = torch.cat([x.new_tensor([-1., 1.]), internal]).unique(sorted=True)
    z = (boundaries[:-1] + boundaries[1:]) / 2
    mass = cdf(boundaries[1:]) - cdf(boundaries[:-1])
    q = q0 + z
    count = torch.full_like(z, float(c0))
    output = []
    for current in x:
        q = q + current
        event = ((q >= 1) & (count < L)).to(x.dtype) - ((q < 0) & (count > 0)).to(x.dtype)
        q, count = q-event, count+event
        output.append(event)
    output = torch.stack(output)
    mean = (output * mass).sum(1)
    expected_loss = None if loss is None else (loss(output) * mass).sum()
    return mean, expected_loss


def actual_jacobian(x, q0=.5, c0=0, L=7):
    shape = (len(x), 1)
    q, c, theta = x.new_tensor([q0]), x.new_tensor([c0]), x.new_tensor(1.)
    return torch.stack([_vjp(x.reshape(shape), q, c, theta, L, torch.eye(len(x))[t].reshape(shape))[:, 0]
                        for t in range(len(x))])


def triangle(x, q0=.5, c0=0, L=7):
    q, c = x.new_tensor(q0), x.new_tensor(float(c0))
    values = []
    for current in x:
        q = q + current
        event = _TriangleEvent.apply(q, c, L)
        q, c = q-event, c+event.detach()
        values.append(event)
    return torch.stack(values)


def run():
    gen = torch.Generator().manual_seed(2026100321)
    errors = []
    h = 1e-6
    for _ in range(40):
        x = torch.randn(4, generator=gen) * .8
        q0 = float(torch.rand((), generator=gen) * 2 - .5)
        c0 = int(torch.randint(8, (), generator=gen))
        cols = []
        for s in range(4):
            direction = torch.eye(4)[s] * h
            plus, _ = phase_integrals(x+direction, q0, c0, 7)
            minus, _ = phase_integrals(x-direction, q0, c0, 7)
            cols.append((plus-minus)/(2*h))
        finite_difference = torch.stack(cols, 1)
        errors.append(float((actual_jacobian(x, q0, c0) - finite_difference).abs().max()))

    # One binary output, convex loss: mean-Jacobian * hard adjoint has wrong
    # sign relative to the exact expected downstream loss derivative.
    x = torch.zeros((1, 1), requires_grad=True)
    y = bounded_phase_events(x, torch.tensor([.5]), torch.tensor([0.]), torch.tensor(1.), 7)
    surrogate_loss = .5 * (y[0, 0]-.25).square()
    surrogate_grad, = torch.autograd.grad(surrogate_loss, x)
    loss = lambda output: .5 * (output[0]-.25).square()
    mean, expected = phase_integrals(torch.zeros(1), .5, 0, 7, loss)
    _, plus = phase_integrals(torch.tensor([h]), .5, 0, 7, loss)
    _, minus = phase_integrals(torch.tensor([-h]), .5, 0, 7, loss)
    exact_gradient = (plus-minus)/(2*h)
    assert abs(float(exact_gradient)-.125) < 1e-8
    assert abs(float(surrogate_grad)-(-.125)) < 1e-12
    currents = torch.tensor([.4, -.6], requires_grad=True)
    jd = actual_jacobian(currents.detach())
    jt = torch.autograd.functional.jacobian(triangle, currents)
    adjoints = [torch.tensor([1., 1.]), torch.tensor([1., -1.]), torch.tensor([.3, 1.])]
    vjps = []
    for g in adjoints:
        vd, vt = jd.T@g, jt.T@g
        vjps.append(dict(output_adjoint=g.tolist(), density=vd.tolist(), triangular=vt.tolist(),
                         cosine=float(torch.dot(vd, vt)/(vd.norm()*vt.norm()))))
    base = currents.detach()
    phase_mean, _ = phase_integrals(base, .5, 0, 7)
    g = torch.tensor([.3, 1.])
    vd, vt = jd.T@g, jt.T@g
    step = 1e-3
    density_updated, _ = phase_integrals(base-step*vd, .5, 0, 7)
    triangular_updated, _ = phase_integrals(base-step*vt, .5, 0, 7)
    shifted, _ = phase_integrals(base+step*torch.tensor([1., -1.]), .5, 0, 7)
    assert float(g@density_updated) < float(g@phase_mean) < float(g@triangular_updated)
    assert abs(float(shifted.sum()-phase_mean.sum())) < 1e-12
    t1_errors=[]
    for c0 in (0, 3, 7):
        for q0 in (-.2, .5, 1.2):
            for current in (-.85, -.16, .22, .77, 1.4):
                xx=torch.tensor([current],requires_grad=True)
                t1d=actual_jacobian(xx.detach(),q0,c0)
                t1t=torch.autograd.functional.jacobian(lambda value: triangle(value,q0,c0),xx)
                t1_errors.append(float((t1d-t1t).abs().max()))
    def field_at(value, density):
        jj=actual_jacobian(value) if density else torch.autograd.functional.jacobian(triangle,value)
        return jj.T@torch.ones(2)
    fields={}
    for density in (False,True):
        columns=[]
        for s in range(2):
            direction=torch.eye(2)[s]*h
            columns.append((field_at(base+direction,density)-field_at(base-direction,density))/(2*h))
        matrix=torch.stack(columns,1)
        fields['density' if density else 'triangular']=dict(field_derivative=matrix.tolist(),
            cross_partial_0_by_1=float(matrix[0,1]),cross_partial_1_by_0=float(matrix[1,0]),
            cross_partial_difference=float(matrix[0,1]-matrix[1,0]))
    assert abs(fields['density']['cross_partial_difference'])<1e-8
    assert abs(fields['triangular']['cross_partial_difference']+.9)<1e-8
    result = dict(seed=2026100321, device='cpu', threads=2, cuda_initialized=torch.cuda.is_initialized(),
        positive_mechanism=dict(T=2,L=7,q0=.5,c0=0,currents=base.tolist(),
            phase_mean=phase_mean.tolist(), cumulative_mean=float(phase_mean.sum()),
            cumulative_exact_gradient=(jd.T@torch.ones(2)).tolist(),
            cumulative_triangular_gradient=(jt.T@torch.ones(2)).tolist(),
            total_current_preserving_direction=[1.,-1.],
            exact_cumulative_directional_derivative=float((jd.T@torch.ones(2))@torch.tensor([1.,-1.])),
            triangular_cumulative_directional_derivative=float((jt.T@torch.ones(2))@torch.tensor([1.,-1.])),
            actual_cumulative_change_under_time_redistribution=float(shifted.sum()-phase_mean.sum()),
            linear_time_readout=g.tolist(), step=step,readout_before=float(g@phase_mean),
            readout_after_density_gradient_step=float(g@density_updated),
            readout_after_triangular_gradient_step=float(g@triangular_updated),
            density_descent_directional_derivative=-float(vd@vd),
            triangular_descent_directional_derivative=-float(vd@vt),
            t1_shape_matched_cases=len(t1_errors),t1_max_difference=max(t1_errors)),
        local_integrability=fields,
        exact_local_mean_check=dict(cases=40, T=4, L=7, theta=1,
                                   integration='Exact triangular-CDF phase cell masses',
                                   largest_absolute_vjp_fd_error=max(errors)),
        downstream_loss_counterexample=dict(T=1,L=7,q0=.5,c0=0,current=0,hard_y=float(y),
            loss='.5*(y-.25)^2', phase_mean=float(mean), expected_loss=float(expected),
            hard_loss=float(surrogate_loss), exact_expected_loss_derivative=float(exact_gradient),
            actual_bounded_surrogate_derivative=float(surrogate_grad),
            expected_loss_gradient_sign_is_opposite=True,
            note='T1 triangular SG is identical here: counterexample diagnoses both SGs, not their performance difference.'),
        temporal_coupling=dict(T=2,L=7,q0=.5,c0=0,currents=currents.detach().tolist(),
            hard_forward=triangle(currents).detach().tolist(), density_jacobian=jd.tolist(), triangular_jacobian=jt.tolist(),
            density_singular_values=torch.linalg.svdvals(jd).tolist(),
            triangular_singular_values=torch.linalg.svdvals(jt).tolist(),
            density_condition_number=float(torch.linalg.cond(jd)),
            triangular_condition_number=float(torch.linalg.cond(jt)), vjps=vjps),
        scope='Local exact phase integral and synthetic loss only; no model, dataset, GPU, update, or tracking accuracy test.')
    assert max(errors) < 1e-7
    assert max(t1_errors)<1e-12
    assert not result['cuda_initialized']
    target = Path(__file__).with_name('foundations_cpu_checks.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
