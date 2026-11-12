"""Generate paper figures from explicit input/output paths."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.conversion.recursive_phase_adjoint import recursive_phase_mean, recursive_phase_vjp
from src.runtime import save


def atomic_json(path, value):
    save(path, value, json_format=True)


def corruption(phase_path, baseline_path, output):
    phase_rows = json.loads(Path(phase_path).read_text())['metrics']
    baseline = json.loads(Path(baseline_path).read_text())['metrics']
    names = list(phase_rows['corruptions'])
    if set(names) != set(baseline['corruptions']):
        raise ValueError('The two reports must cover the same corruptions')
    changes = [np.mean([x['top1'] for x in phase_rows['corruptions'][name]]) -
               np.mean([x['top1'] for x in baseline['corruptions'][name]]) for name in names]
    mean = phase_rows['top1'] - baseline['top1']
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.barh(range(len(names)), changes, color=['#286f9c' if x >= 0 else '#be6b40' for x in changes])
        ax.set_yticks(range(len(names)), [name.replace('_', ' ').capitalize() for name in names])
        ax.invert_yaxis()
        ax.axvline(0, color='#444444', linewidth=.8)
        ax.axvline(mean, color='#444444', linestyle='--', label=f'Overall mean: {mean:+.2f} pp')
        ax.set_xlabel('Phase accuracy minus baseline accuracy (percentage points)')
        ax.legend(frameon=False)
        fig.tight_layout()
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def phase(output):
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    theta = torch.tensor(1., dtype=torch.float64)
    cases = [('Early charge', [1.6, -.4]), ('Late charge', [-.4, 1.6])]
    records = []
    for name, values in cases:
        currents = torch.tensor(values, dtype=torch.float64).reshape(2, 1)
        means = recursive_phase_mean(currents, theta)
        jacobian = torch.stack([
            recursive_phase_vjp(currents, theta, weight[:, None])[:, 0]
            for weight in torch.eye(2, dtype=torch.float64)
        ])
        records.append(dict(name=name, input_current=values,
                            mean_current=currents.mean().item(),
                            mean_charge=means[:, 0].tolist(),
                            total_mean_charge=means.sum().item(),
                            local_mean_jacobian=jacobian.tolist(),
                            mean_current_total_prediction=2 * currents.mean().clamp(0, 1).item()))
    torch.testing.assert_close(torch.tensor(records[0]['mean_charge']), torch.tensor([1., .2]))
    torch.testing.assert_close(torch.tensor(records[1]['mean_charge']), torch.tensor([0., 1.]))
    assert records[0]['local_mean_jacobian'] == [[0., 0.], [1., 1.]]
    assert records[1]['local_mean_jacobian'] == [[0., 0.], [0., 0.]]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4),
                             gridspec_kw={'width_ratios': [1.2, 1.2, .9]})
    for row, record in enumerate(records):
        color = ['#2274A5', '#D47828'][row]
        for col, key in enumerate(['input_current', 'mean_charge']):
            ax = axes[row, col]
            values = record[key]
            ax.axhline(0, color='#aaaaaa', linewidth=.8)
            ax.bar([1, 2], values, color=color, width=.58)
            ax.set_xticks([1, 2], ['Step 1', 'Step 2'])
            ax.set_xlim(.4, 2.6)
            ax.set_ylim(-.85, 2.25)
            ax.set_yticks([0, 1])
            for t, value in enumerate(values, 1):
                ax.text(t, value + (.08 if value >= 0 else -.09), f'{value:.1f}',
                        ha='center', va='bottom' if value >= 0 else 'top')
            if row == 0:
                ax.set_title(['Input current', 'Expected emitted charge'][col], pad=14)
            if col == 0:
                ax.set_ylabel(record['name'] + '\n(threshold units)')
                ax.text(.5, .93, 'Mean input = 0.6', transform=ax.transAxes, ha='center')
            else:
                ax.text(.5, .93, f"Total = {record['total_mean_charge']:.2f}",
                        transform=ax.transAxes, ha='center', fontweight='bold')
        ax = axes[row, 2]
        matrix = np.asarray(record['local_mean_jacobian'])
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:.0f}', ha='center', va='center',
                        color='white' if matrix[i, j] > .5 else '#333333', fontsize=14)
        ax.set_xticks([0, 1], [r'$I_1$', r'$I_2$'])
        ax.set_yticks([0, 1], [r'$\mu_1$', r'$\mu_2$'])
        if row == 0:
            ax.set_title(r'Local Jacobian $\partial\mu_t/\partial I_j$', pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('The same mean input can have different finite-horizon gradients', fontsize=13)
    fig.text(.5, .025,
             'T = 2, fixed threshold = 1, shared initial phase uniform on [0, 1).\n'
             'Analytic neuron example; this figure does not establish an accuracy gain.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .09, 1, .95), h_pad=1.6, w_pad=2)
    fig.savefig(out / 'arrival_order.pdf', bbox_inches='tight')
    fig.savefig(out / 'arrival_order.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    atomic_json(out / 'data.json', dict(
        status='analytic_example_verified', cases=records, timesteps=2, threshold=1.,
        randomness='None: two prescribed input sequences, exact analytic means and VJPs.',
        interpretation='A scalar function of final mean input gives the same prediction for both orders. The finite-horizon mean and its temporal Jacobian differ because late charge has only one remaining emission opportunity.',
        mean_current_approximation='T * clamp(mean(I), 0, theta) = 1.2 for both sequences; an illustrative rate approximation, not an implementation or benchmark of DSR.',
        deployment='The illustrated mean is used only to define the training Jacobian. Deployment uses hard signed ECL with initial membrane theta/2.',
        evidence_limit='This is a mechanism illustration, not a new accuracy result, proof of broad novelty, or proof of an unbiased whole-network loss gradient.',
        source='src/plots.py'))
    print(out)
