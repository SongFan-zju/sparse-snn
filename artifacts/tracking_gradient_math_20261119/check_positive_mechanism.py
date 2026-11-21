"""CPU2 check of trajectory-gradient error decomposition for signed ECL."""
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.conversion.triangular_density_horizon import triangle_density_partition_vjp,triangle_density_partition_mean
from src.conversion.ecl_triangular_surrogate import TriangularSequence
from src.conversion.general_phase_adjoint import hard_sequence

torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
theta=torch.tensor(1.)

def phase_cdf(a):
    v=a-.5
    if v<=-1:return 0.
    if v<=0:return .5*(v+1)**2
    if v<1:return 1-.5*(1-v)**2
    return 1.

def exact_phase_integrals(xs):
    total=0.;boundaries=[-.5,1.5];T=len(xs)
    for x in xs:
        total+=x
        for k in range(1,T+1):
            for a in (k-total,k-total-1e-3):
                if -.5<a<1.5:boundaries.append(a)
    boundaries=sorted(set(boundaries));mean=[0.]*T;loss=0.
    for left,right in zip(boundaries,boundaries[1:]):
        mass=phase_cdf(right)-phase_cdf(left)
        q=(left+right)/2;c=0;ys=[]
        for x in xs:
            q+=x;e=int(q>1)-int(q<=-1e-3 and c>=1)
            q-=e;c+=e;ys.append(e)
        for t,y in enumerate(ys):mean[t]+=mass*y
        loss+=mass*math.log1p(math.exp(-sum(ys)/T))
    return torch.tensor(mean),loss

def ordinary(v):
    node=TriangularSequence(theta);node.timesteps=len(v)
    return node(v)

T1=[]
for value in [-1.1,-.2,.17,.8,1.7]:
    x=torch.tensor([[value]],requires_grad=True)
    jd=triangle_density_partition_vjp(x,theta,torch.ones_like(x))
    jt,=torch.autograd.grad(ordinary(x).sum(),x)
    assert torch.allclose(jd,jt,atol=1e-14,rtol=0)
    T1.append(dict(input=value,density=float(jd),triangle=float(jt)))

xs=[.28,-.43,.36,.11]
x=torch.tensor(xs)[:,None].requires_grad_(True);T=len(xs)
y=hard_sequence(x,theta)
g=torch.full_like(x,-.5/T) # hard terminal mean=0; softplus(-mean) adjoint
assert float(y.sum())==0
jd=triangle_density_partition_vjp(x,theta,g).flatten()
jt,=torch.autograd.grad((ordinary(x)*g).sum(),x)
jt=jt.flatten()
eps=1e-6;reference=[];mean_reference=[]
for s in range(T):
    plus=xs.copy();minus=xs.copy();plus[s]+=eps;minus[s]-=eps
    mp,lp=exact_phase_integrals(plus);mm,lm=exact_phase_integrals(minus)
    reference.append((lp-lm)/(2*eps))
    mean_reference.append(float(((mp-mm)/(2*eps)*g.flatten()).sum()))
reference=torch.tensor(reference);mean_reference=torch.tensor(mean_reference)
mu,lbar=exact_phase_integrals(xs)
actual_mu=triangle_density_partition_mean(x,theta).flatten()
assert torch.allclose(mu,actual_mu,atol=1e-13,rtol=0)
assert torch.allclose(jd,mean_reference,atol=1e-8,rtol=0)
R=reference-jd;D=jd-jt
left=float((reference-jt).square().sum()-(reference-jd).square().sum())
right=float(D.square().sum()+2*(R*D).sum())
assert abs(left-right)<1e-12
assert left>0
out=dict(scope='Actual classification/DVS signed-ECL Torch operators; CPU scalar currents only; no network accuracy claim',
    single_tick_equivalence=T1,
    nonlinear_terminal_readout=dict(inputs=xs,hard_events=y.flatten().tolist(),
        phase_mean_events=mu.tolist(),loss='softplus(-sum(events)/T)',
        hard_output_adjoint=g.flatten().tolist(),density_VJP=jd.tolist(),triangle_VJP=jt.tolist(),
        exact_phase_expected_loss_gradient=reference.tolist(),
        R_task_nonlinearity=R.tolist(),D_trajectory_Jacobian_error=D.tolist(),
        density_error_norm=float(R.norm()),triangle_error_norm=float((reference-jt).norm()),
        squared_error_improvement=left,decomposition_rhs=right,
        phase_mean_VJP_finite_difference_max_abs_error=float((jd-mean_reference).abs().max())),
    cuda_initialized=torch.cuda.is_initialized(),status='passed')
assert not out['cuda_initialized']
Path(__file__).with_name('mechanism_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
