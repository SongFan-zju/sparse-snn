"""Dense PyTorch inference timing and signed-event activity (not sparse-kernel energy)."""
import statistics
import time
import torch
from src.conversion.qcfs_ecl_resnet18 import DualThresholdSequence

WARMUP_PASSES, TIMED_PASSES = 3, 5

@torch.inference_mode()
def event_statistics(output, unit, max_count):
    """Counts include both signs; sign bits themselves are not magnitude events."""
    scaled = output.detach()/unit
    rounded = scaled.round()
    counts = rounded.to(torch.int64)
    magnitude = counts.abs()
    bit_events = (magnitude & 1) + ((magnitude >> 1) & 1)
    values = torch.stack((
        torch.count_nonzero(counts), magnitude.sum(), bit_events.sum(),
        (counts > 0).sum(), (counts < 0).sum(),
        (magnitude > max_count).sum(),
    ))
    error = (scaled-rounded).abs().amax()
    return values, error

@torch.inference_mode()
def collect_activity(model, batches, spec):
    nodes = {n: m for n, m in model.named_modules() if isinstance(m, DualThresholdSequence)}
    if len(nodes) != spec['event_nodes']:
        raise RuntimeError(f'Unexpected event node count: {len(nodes)}')
    totals, errors, elements, calls = {}, {}, {}, {}
    classifier_calls = [0]

    def hook_for(name):
        def hook(node, args, output):
            unit = node.unit if spec['max_count'] == 3 else node.threshold
            values, error = event_statistics(output, unit, spec['max_count'])
            totals[name] = totals.get(name, torch.zeros_like(values)) + values
            errors[name] = torch.maximum(errors.get(name, torch.zeros_like(error)), error)
            elements[name] = elements.get(name, 0) + output.numel()
            calls[name] = calls.get(name, 0) + 1
        return hook

    def classifier_hook(module, args, output):
        classifier_calls[0] += 1

    handles = [node.register_forward_hook(hook_for(name)) for name, node in nodes.items()]
    handles.append(model.source.fc.register_forward_hook(classifier_hook))
    try:
        for batch in batches:
            logits = model(batch)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Nonfinite inference output')
    finally:
        for handle in handles: handle.remove()
    examples = sum(len(batch) for batch in batches)
    per_node = {}
    keys = ('nonzero_signed_packets', 'absolute_unit_counts', 'magnitude_bit_events',
            'positive_packets', 'negative_packets')
    for name in nodes:
        values = totals[name].cpu().tolist()
        error = float(errors[name].cpu())
        if values[5] or not 0. <= error <= 1e-4 or calls[name] != len(batches):
            raise RuntimeError(f'Invalid event counts or node calls at{name}')
        row = dict(zip(keys, values[:5]))
        row.update(neuron_steps=elements[name], forward_calls=calls[name],
                   max_integer_rounding_error=error)
        if spec['max_count'] == 1 and not values[0] == values[1] == values[2]:
            raise RuntimeError('M1 packet, unit and bit counts must agree')
        per_node[name] = row
    if classifier_calls[0] != len(batches):
        raise RuntimeError('Expected one classifier call per model forward')
    aggregate = {key: sum(row[key] for row in per_node.values()) for key in keys}
    aggregate['neuron_steps'] = sum(elements.values())
    return dict(examples=examples, model_forward_calls=len(batches),
        classifier_calls=classifier_calls[0], nodes=per_node, total=aggregate,
        per_image={key: aggregate[key]/examples for key in keys},
        nonzero_packet_fraction=aggregate['nonzero_signed_packets']/aggregate['neuron_steps'])

@torch.inference_mode()
def measure(model, batches, device):
    if device.type == "cpu":
        return measure_cpu(model, batches)
    for _ in range(WARMUP_PASSES):
        for batch in batches:
            model(batch)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = []
    for _ in range(TIMED_PASSES):
        trial = []
        for batch in batches:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            logits = model(batch)
            end.record()
            del logits
            trial.append((start, end))
        events.append(trial)
    torch.cuda.synchronize(device)
    batch_ms = [[start.elapsed_time(end) for start, end in trial] for trial in events]
    pass_ms = [sum(trial) for trial in batch_ms]
    examples = sum(len(batch) for batch in batches)
    mean_ms = statistics.mean(pass_ms)
    return dict(warmup_passes=WARMUP_PASSES, timed_passes=TIMED_PASSES,
        examples_per_pass=examples, batch_sizes=[len(x) for x in batches],
        batch_milliseconds=batch_ms, pass_milliseconds=pass_ms,
        mean_milliseconds_per_image=mean_ms/examples,
        median_milliseconds_per_image=statistics.median(pass_ms)/examples,
        images_per_second=1000*examples/mean_ms,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))

@torch.inference_mode()
def measure_cpu(model, batches):
    for _ in range(WARMUP_PASSES):
        for batch in batches:
            model(batch)
    times = []
    for _ in range(TIMED_PASSES):
        trial = []
        for batch in batches:
            start = time.perf_counter()
            model(batch)
            trial.append(1000 * (time.perf_counter() - start))
        times.append(trial)
    examples = sum(len(batch) for batch in batches)
    passes = [sum(trial) for trial in times]
    return dict(warmup_passes=WARMUP_PASSES, timed_passes=TIMED_PASSES,
                examples_per_pass=examples, batch_sizes=[len(x) for x in batches],
                batch_milliseconds=times, pass_milliseconds=passes,
                mean_milliseconds_per_image=statistics.mean(passes) / examples,
                median_milliseconds_per_image=statistics.median(passes) / examples,
                images_per_second=1000 * examples / statistics.mean(passes))
