"""Finite CPU checks for the common count-boundary theorem; no training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.conversion.triangular_density_horizon import (
    triangle_density_partition_mean, triangle_density_partition_vjp,
)

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def cdf(z):
    z=z.clamp(-1,1)
    return torch.where(z<=0,.5*(z+1).square(),1-.5*(1-z).square())


def unified_mean(x,theta,q0=.5,c0=0,cap=None,epsilon=.001,reduce_zero=False):
    K = c0+len(x) if cap is None else cap
    levels = torch.arange(1,K+1)
    bounds = torch.where(levels<=c0,x.new_tensor(-torch.inf),x.new_tensor(torch.inf))
    total = x.new_tensor(0.)
    previous_count = x.new_tensor(float(c0))
    values=[]
    for tick in x:
        total=total+tick/theta
        p=levels-c0-q0-total
        lower=torch.cat([x.new_tensor([-torch.inf]),bounds[:-1]])
        higher=torch.cat([bounds[1:],x.new_tensor([torch.inf])])
        if reduce_zero:
            assert epsilon==0
            bounds=torch.minimum(higher,torch.maximum(lower,p))
        else:
            bounds=torch.minimum(higher,torch.minimum(torch.maximum(bounds,p-epsilon/theta),
                                                       torch.maximum(lower,p)))
        current_count=(1-cdf(bounds)).sum()
        values.append(theta*(current_count-previous_count))
        previous_count=current_count
    return torch.stack(values)


def exact_cells_mean(x,theta,q0=.5,c0=0,cap=None,epsilon=.001):
    # All candidate phase boundaries, including both sides of the dead zone.
    K=c0+len(x) if cap is None else cap
    levels=torch.arange(1,K+1)[:,None]
    positive=levels-c0-q0-x.cumsum(0)[None]/theta
    boundaries=torch.cat([positive.flatten(),(positive-epsilon/theta).flatten()])
    interior=boundaries[(boundaries>-1)&(boundaries<1)]
    cuts=torch.cat([x.new_tensor([-1.,1.]),interior]).unique(sorted=True)
    mid=(cuts[:-1]+cuts[1:])/2
    mass=cdf(cuts[1:])-cdf(cuts[:-1])
    membrane=theta*(q0+mid)
    count=torch.full_like(mid,float(c0))
    values=[]
    for tick in x:
        membrane=membrane+tick
        positive=(membrane>theta)
        if cap is not None: positive=positive&(count<cap)
        negative=(membrane<=-epsilon)&(count>=1)
        event=positive.to(x.dtype)-negative.to(x.dtype)
        charge=theta*event
        membrane,count=membrane-charge,count+event
        values.append((charge*mass).sum())
    return torch.stack(values)


def run():
    gen=torch.Generator().manual_seed(2026100321)
    errors=[]
    h=1e-6
    for _ in range(10):
        theta=float(.25+torch.rand((),generator=gen)*1.5)
        x=torch.randn(4,generator=gen)*theta*.7
        mean=unified_mean(x,theta)
        actual_mean=triangle_density_partition_mean(x[:,None],torch.tensor(theta))[:,0]
        cells=exact_cells_mean(x,theta)
        g=torch.randn(4,generator=gen)
        actual_vjp=triangle_density_partition_vjp(x[:,None],torch.tensor(theta),g[:,None])[:,0]
        fd=[]
        for s in range(4):
            direction=torch.eye(4)[s]*h
            fd.append(float(g@(exact_cells_mean(x+direction,theta)-exact_cells_mean(x-direction,theta))/(2*h)))
        xx=x.detach().requires_grad_(True)
        analytic,=torch.autograd.grad(unified_mean(xx,theta),xx,g)
        errors.append(dict(theta=theta,T=4,epsilon=.001,
            code_mean_vs_unified=float((actual_mean-mean).abs().max()),
            code_mean_vs_exact_cells=float((actual_mean-cells).abs().max()),
            code_vjp_vs_unified_autograd=float((actual_vjp-analytic).abs().max()),
            code_vjp_vs_exact_cells_fd=float((actual_vjp-torch.tensor(fd)).abs().max())))
    bounded=[]
    # A few fixed states, including saturation; no scan or training.
    for q0,c0,cap,epsilon in [(.5,0,7,0.),(-.2,3,7,0.),(1.2,7,7,0.),(.3,2,3,.001)]:
        x=torch.randn(4,generator=gen)*.6
        mean=unified_mean(x,1.,q0,c0,cap,epsilon)
        cells=exact_cells_mean(x,1.,q0,c0,cap,epsilon)
        row=dict(q0=q0,c0=c0,cap=cap,epsilon=epsilon,
                 unified_vs_exact_cells=float((mean-cells).abs().max()))
        if epsilon==0:
            simplified=unified_mean(x,1.,q0,c0,cap,epsilon,True)
            row['three_term_vs_reduced_two_term']=float((mean-simplified).abs().max())
        bounded.append(row)
    # The earlier credit-return example, now the real classification epsilon.
    x=torch.tensor([.4,-.6]);theta=torch.tensor(1.)
    jac=torch.stack([triangle_density_partition_vjp(x[:,None],theta,torch.eye(2)[t,:,None])[:,0]
                     for t in range(2)])
    g=torch.tensor([.3,1.]); v=jac.T@g; tri=x.new_tensor([.3,.3]); eta=.001
    mean=triangle_density_partition_mean(x[:,None],theta)[:,0]
    step_density=exact_cells_mean(x-eta*v,1.)
    step_tri=exact_cells_mean(x-eta*tri,1.)
    positive=dict(T=2,epsilon=.001,currents=x.tolist(),mean=mean.tolist(),
        actual_classification_jacobian=jac.tolist(),sum_readout_gradient=(jac.T@torch.ones(2)).tolist(),
        linear_readout=g.tolist(),before=float(g@mean),
        after_density_step=float(g@step_density),after_triangle_step=float(g@step_tri))
    assert positive['after_density_step']<positive['before']<positive['after_triangle_step']
    assert max(row['code_vjp_vs_exact_cells_fd'] for row in errors)<1e-7
    assert max(row['code_vjp_vs_unified_autograd'] for row in errors)<1e-10
    assert max(row['code_mean_vs_exact_cells'] for row in errors)<1e-10
    assert max(row['unified_vs_exact_cells'] for row in bounded)<1e-10
    assert all(row.get('three_term_vs_reduced_two_term',0)==0 for row in bounded)
    result=dict(status='passed',seed=2026100321,device='cpu',threads=2,
        cuda_initialized=torch.cuda.is_initialized(),classification_real_code_cases=errors,
        finite_cap_cases=bounded,positive_classification_example=positive,
        scope='Ten fixed-seed T4 checks and a few local states; no dataset/model/GPU/training/production-code changes.')
    assert not result['cuda_initialized']
    Path(__file__).with_name('unified_count_boundary_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
