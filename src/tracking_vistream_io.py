"""MOT16 public-DPM IO and feature-injected author ViStream/UniTrack tracking.

The association classes/functions are loaded from the supplied author tree;
no author file is changed and no AppearanceModel/backbone is constructed.
GT is restricted to training ID targets and official TrackEval scoring.
"""
from __future__ import annotations

import ast
from collections import OrderedDict, deque
import configparser
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import scipy
import scipy.linalg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch.nn import functional as F
from torchvision import ops

TRAIN_SEQUENCES = ('MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09')
EVAL_SEQUENCES = ('MOT16-10', 'MOT16-11', 'MOT16-13')
ALL_SEQUENCES = TRAIN_SEQUENCES + EVAL_SEQUENCES
IMAGE_SIZE = (1088, 608)  # width, height; author MOT configuration
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def sequence_path(root, name):
    root = Path(root)
    candidates = (root / 'train' / name, root / 'images/train' / name,
                  root / name, root / 'MOT16/train' / name)
    for path in candidates:
        if (path / 'seqinfo.ini').is_file():
            return path
    raise FileNotFoundError(f'MOT16 train sequence {name} missing under {root}')


def sequence_info(root, name):
    if name not in ALL_SEQUENCES:
        raise ValueError('This protocol uses the seven MOT16 train sequences only')
    path = sequence_path(root, name)
    parser = configparser.ConfigParser()
    parser.read(path / 'seqinfo.ini')
    section = parser['Sequence']
    return dict(name=name, path=str(path), length=int(section['seqLength']),
                frame_rate=int(section['frameRate']), width=int(section['imWidth']),
                height=int(section['imHeight']), image_dir=section.get('imDir', 'img1'),
                image_extension=section.get('imExt', '.jpg'))


def _table(path, minimum_columns):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, minimum_columns), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=',', dtype=np.float64, ndmin=2)
    if rows.shape[1] < minimum_columns or not np.isfinite(rows).all():
        raise ValueError(f'Invalid MOT table: {path}')
    return rows


def letterbox_preprocess(image_bgr, image_size=IMAGE_SIZE):
    """Author INTER_AREA letterbox and RGB/ImageNet normalization; FP32 CHW."""
    width, height = image_size
    h, w = image_bgr.shape[:2]
    ratio = min(float(height) / h, float(width) / w)
    new_size = (round(w * ratio), round(h * ratio))
    padw, padh = (width - new_size[0]) / 2, (height - new_size[1]) / 2
    top, bottom = round(padh - .1), round(padh + .1)
    left, right = round(padw - .1), round(padw + .1)
    image = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    image = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float() / 255.
    image = (image - image.new_tensor(MEAN)[:, None, None]) / image.new_tensor(STD)[:, None, None]
    return image, dict(ratio=ratio, padw=padw, padh=padh, image_size=tuple(image_size))


def public_observations(rows, transform):
    """Raw det.txt tlwh -> normalized letterbox center-xywh + unchanged score."""
    result = np.empty((len(rows), 5), dtype=np.float32)
    if len(rows):
        width, height = transform['image_size']
        ratio = transform['ratio']
        result[:, 0] = ((rows[:, 2] + rows[:, 4] / 2) * ratio + transform['padw']) / width
        result[:, 1] = ((rows[:, 3] + rows[:, 5] / 2) * ratio + transform['padh']) / height
        result[:, 2] = rows[:, 4] * ratio / width
        result[:, 3] = rows[:, 5] * ratio / height
        result[:, 4] = rows[:, 6]
    return result


@dataclass
class MOTFrame:
    sequence: str
    frame_index: int  # one-based official frame number
    image: torch.Tensor
    image_bgr: np.ndarray
    observations: np.ndarray
    public_detections: np.ndarray
    transform: dict


class MOT16Sequence:
    def __init__(self, root, name, image_size=IMAGE_SIZE, training_targets=False):
        self.info = sequence_info(root, name)
        self.root, self.name, self.path = Path(root), name, Path(self.info['path'])
        self.image_size = tuple(image_size)
        # There is deliberately no caller-supplied detections/GT override path.
        detections = _table(self.path / 'det/det.txt', 7)
        if len(detections):
            if not np.all(detections[:, 1] == -1):
                raise ValueError('Expected public DPM det.txt identity column -1, not GT/track identities')
            if not np.all((detections[:, 0] >= 1) & (detections[:, 0] <= len(self)) &
                          (detections[:, 0] == detections[:, 0].astype(int))):
                raise ValueError('Detections contain invalid official frame numbers')
            if not np.all(detections[:, 4:6] > 0):
                raise ValueError('Nonpositive public detection sizes')
        self._detections = {int(frame): detections[detections[:, 0] == frame]
                            for frame in np.unique(detections[:, 0])}
        self.detection_count = len(detections)
        self._gt = None
        if training_targets:
            if name not in TRAIN_SEQUENCES:
                raise ValueError('GT training labels are forbidden for held-out sequences')
            rows = _table(self.path / 'gt/gt.txt', 9)
            rows = rows[(rows[:, 6] != 0) & (rows[:, 7] == 1)]
            self._gt = {int(frame): rows[rows[:, 0] == frame] for frame in np.unique(rows[:, 0])}

    def __len__(self):
        return self.info['length']

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        frame = index + 1
        path = self.path / self.info['image_dir'] / f'{frame:06d}{self.info["image_extension"]}'
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise FileNotFoundError(f'Missing/corrupt official frame {path}')
        if image_bgr.shape[:2] != (self.info['height'], self.info['width']):
            raise ValueError(f'Frame dimensions disagree with seqinfo.ini: {path}')
        image, transform = letterbox_preprocess(image_bgr, self.image_size)
        detections = self._detections.get(frame, np.empty((0, 7), dtype=np.float64))
        return MOTFrame(self.name, frame, image, image_bgr,
                        public_observations(detections, transform), detections.copy(), transform)

    def training_gt(self, frame_index):
        if self._gt is None or self.name not in TRAIN_SEQUENCES:
            raise RuntimeError('Open a training sequence with training_targets=True to access GT IDs')
        return self._gt.get(frame_index, np.empty((0, 9), dtype=np.float64)).copy()


def training_identity_map(root):
    """Contiguous classifier targets, unique across the four training videos."""
    keys = []
    for name in TRAIN_SEQUENCES:
        rows = _table(sequence_path(root, name) / 'gt/gt.txt', 9)
        valid = rows[(rows[:, 6] != 0) & (rows[:, 7] == 1)]
        keys.extend((name, int(identity)) for identity in np.unique(valid[:, 1]))
    return {key: index for index, key in enumerate(keys)}


def training_id_targets(sequence, frame, identity_map, minimum_iou=.5, confidence_threshold=.5):
    """Match real public detections to train GT; unmatched targets are -1.

    Returned IDs align with ALL frame observations; select with roi_features'
    returned indices. GT boxes never replace detector boxes or tracker input.
    """
    if frame.sequence != sequence.name or sequence.name not in TRAIN_SEQUENCES:
        raise ValueError('Training targets require the matching training video')
    gt = sequence.training_gt(frame.frame_index)
    targets = np.full(len(frame.public_detections), -1, dtype=np.int64)
    selected = np.flatnonzero(frame.observations[:, 4] > confidence_threshold)
    if len(selected) and len(gt):
        boxes = frame.public_detections[selected, 2:6].copy()
        truth = gt[:, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
        truth[:, 2:] += truth[:, :2]
        overlaps = box_iou_numpy(boxes, truth, inclusive=False)
        costs = np.where(overlaps >= minimum_iou, 1 - overlaps, np.inf)
        _, assignment, _ = scipy_lapjv(costs, extend_cost=True, cost_limit=1 - minimum_iou + 1e-8)
        for local, matched in enumerate(assignment):
            if matched >= 0:
                targets[selected[local]] = identity_map[(sequence.name, int(gt[matched, 1]))]
    return torch.from_numpy(targets)


def roi_features(feature_tensor, observations, image_size=IMAGE_SIZE, feat_size=(4, 10),
                 confidence_threshold=.5):
    """Differentiable AUTHOR ROIAlign; returns un-detached ROI and det indices.

    feature_tensor is the already computed [1,C,Hf,Wf] cumulative frame feature.
    No backbone call, feature normalization, or extra frame/view is performed.
    """
    if feature_tensor.ndim != 4 or feature_tensor.shape[0] != 1:
        raise ValueError('One frame feature tensor [1,C,H,W] is required')
    observations = np.asarray(observations, dtype=np.float32).reshape(-1, 5)
    indices = np.flatnonzero(observations[:, 4] > confidence_threshold)
    boxes = feature_tensor.new_tensor(observations[indices, :4])
    center, size = boxes[:, :2].clone(), boxes[:, 2:].clone()
    boxes[:, :2], boxes[:, 2:] = center - size / 2, center + size / 2
    # Normalized input coordinates become feature-map coordinates directly.
    boxes[:, [0, 2]] *= feature_tensor.shape[-1]
    boxes[:, [1, 3]] *= feature_tensor.shape[-2]
    roi = ops.roi_align(feature_tensor, [boxes], output_size=feat_size,
                        spatial_scale=1., sampling_ratio=-1, aligned=False)
    return roi, torch.as_tensor(indices, dtype=torch.long, device=feature_tensor.device)


def box_iou_numpy(boxes, query_boxes, inclusive=True):
    """cython_bbox uses inclusive (+1) pixel extents; TrackEval remains untouched."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    query_boxes = np.asarray(query_boxes, dtype=np.float64).reshape(-1, 4)
    extra = 1. if inclusive else 0.
    wh = np.maximum(0., np.minimum(boxes[:, None, 2:], query_boxes[None, :, 2:]) -
                    np.maximum(boxes[:, None, :2], query_boxes[None, :, :2]) + extra)
    intersection = wh.prod(2)
    a = np.maximum(0., boxes[:, 2:] - boxes[:, :2] + extra).prod(1)
    b = np.maximum(0., query_boxes[:, 2:] - query_boxes[:, :2] + extra).prod(1)
    union = a[:, None] + b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def scipy_lapjv(cost, extend_cost=True, cost_limit=np.inf):
    """LAP cost-limit objective using dummy unmatched cost_limit/2 endpoints.

    This preserves optional matching (unlike Hungarian then filtering). Equal
    cost assignments can have a different tie order from native lap.lapjv.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    x, y = np.full(n, -1, dtype=int), np.full(m, -1, dtype=int)
    if not n or not m:
        return 0., x, y
    if not extend_cost or not np.isfinite(cost_limit):
        raise ValueError('Tracker fallback expects extend_cost=True and finite cost_limit')
    augmented = np.full((n + m, n + m), np.inf)
    augmented[:n, :m] = np.where(np.isfinite(cost) & (cost <= cost_limit), cost, np.inf)
    augmented[np.arange(n), m + np.arange(n)] = cost_limit / 2
    augmented[n + np.arange(m), np.arange(m)] = cost_limit / 2
    augmented[n:, m:] = 0.
    rows, columns = linear_sum_assignment(augmented)
    valid = (rows < n) & (columns < m)
    x[rows[valid]], y[columns[valid]] = columns[valid], rows[valid]
    return float(cost[rows[valid], columns[valid]].sum()), x, y


class _LegacyNumpy:
    """Local author-module compatibility; do not mutate global numpy aliases."""
    def __getattr__(self, name):
        aliases = dict(float=np.float64, int=int, bool=bool)
        return aliases[name] if name in aliases else getattr(np, name)


class _DeviceAdapter(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda' and not node.args and not node.keywords:
            return ast.copy_location(ast.Call(func=ast.Attribute(value=node.func.value, attr='to', ctx=ast.Load()),
                args=[ast.Name(id='association_device', ctx=ast.Load())], keywords=[]), node)
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'torch.cuda.empty_cache':
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def _definitions(path, namespace, *, assignments=False, adapt_device=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Author source missing: {path}')
    accepted = (ast.FunctionDef, ast.ClassDef) + ((ast.Assign, ast.AnnAssign) if assignments else ())
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, accepted)]
    tree = ast.Module(body=nodes, type_ignores=[])
    if adapt_device:
        tree = _DeviceAdapter().visit(tree)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)


class _NoSecondBackbone(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError('Second backbone disabled: supply feature_tensor to the tracker')


def tracker_options(frame_rate=30, image_size=IMAGE_SIZE, device='cpu'):
    """Retained author no-motion-fusion configuration; raw DPM score >0.5."""
    return types.SimpleNamespace(device=str(device), frame_rate=frame_rate, track_buffer=30,
        conf_thres=.5, min_box_area=200, feat_size=[4, 10], img_size=list(image_size),
        nms_thres=.4, iou_thres=.5, dup_iou_thres=.15, confirm_iou_thres=.7,
        prop_flag=False, use_kalman=True, asso_with_motion=False, motion_lambda=1., motion_gated=False)


class FeatureBoxTracker:
    """Original BoxAssociationTracker with its backbone replaced by frame features."""
    def __init__(self, upstream, opt=None, association_device=None):
        self.upstream = Path(upstream)
        self.opt = opt or tracker_options()
        namespace = dict(__name__='vistream_author_box_adapter', torch=torch, ops=ops, F=F,
            np=_LegacyNumpy(), scipy=scipy, cdist=cdist, deque=deque, OrderedDict=OrderedDict,
            association_device=torch.device(association_device or self.opt.device),
            AppearanceModel=_NoSecondBackbone,
            lap=types.SimpleNamespace(lapjv=scipy_lapjv), bbox_ious=box_iou_numpy)
        _definitions(self.upstream / 'utils/box.py', namespace)
        _definitions(self.upstream / 'core/motion/kalman_filter.py', namespace, assignments=True)
        namespace['kalman_filter'] = types.SimpleNamespace(chi2inv95=namespace['chi2inv95'])
        _definitions(self.upstream / 'core/association/matching.py', namespace, adapt_device=True)
        namespace['matching'] = types.SimpleNamespace(**{key: value for key, value in namespace.items()
                                                       if callable(value)})
        _definitions(self.upstream / 'tracker/mot/basetrack.py', namespace)
        _definitions(self.upstream / 'tracker/mot/multitracker.py', namespace, adapt_device=True)
        _definitions(self.upstream / 'tracker/mot/box.py', namespace)
        self._namespace = namespace
        self.sequence = None
        self._core = None
        self._next_frame = 1

    def reset_sequence(self, info, reset_backbone=None):
        if reset_backbone is not None:
            reset_backbone()
        self.sequence = info['name']
        self.opt.frame_rate = info['frame_rate']
        self.opt.dataLen = info['length']
        self._namespace['BaseTrack']._count = 0
        self._core = self._namespace['BoxAssociationTracker'](self.opt)
        def extract_emb(this, unused_image, obs):
            feat = this._feature_tensor
            scale = [feat.shape[-1] / this.opt.img_size[0], feat.shape[-2] / this.opt.img_size[1]]
            boxes = self._namespace['scale_box'](scale, obs).to(feat)
            return ops.roi_align(feat, [boxes[:, :4]], this.opt.feat_size).detach().cpu()
        self._core.extract_emb = types.MethodType(extract_emb, self._core)
        self._next_frame = 1

    @torch.no_grad()
    def update(self, feature_tensor, frame):
        if self._core is None or frame.sequence != self.sequence or frame.frame_index != self._next_frame:
            raise ValueError('Reset once at the sequence boundary, then process every frame in order')
        if feature_tensor.ndim != 4 or feature_tensor.shape[0] != 1:
            raise ValueError('Tracker requires an already computed [1,C,H,W] frame feature')
        if tuple(frame.transform['image_size']) != tuple(self.opt.img_size):
            raise ValueError('Frame preprocessing and tracker image sizes differ')
        self._core._feature_tensor = feature_tensor.detach()
        try:
            tracks = self._core.update(None, frame.image_bgr, frame.observations)
        finally:
            self._core._feature_tensor = None
        self._next_frame += 1
        return tracks

    def report(self):
        return dict(author_association='Original BoxAssociationTracker/AssociationTracker class bodies',
            roi='torchvision roi_align, output4x10, aligned=False, sampling_ratio=-1',
            second_backbone=False, association_device=str(self._namespace['association_device']),
            matching='SciPy linear_sum_assignment with LAP cost-limit dummy endpoints',
            matching_limit='Equal-cost ties may differ from native lap.lapjv',
            iou='NumPy float64 inclusive (+1), matching cython_bbox pixel convention',
            motion='Author Kalman prediction/IoU recovery retained; no motion fusion/gating',
            score='Untransformed public DPM score; strict > author conf_thres',
            conf_thres=self.opt.conf_thres, no_detection_frames='Backbone must advance on every frame')


def tracking_rows(tracks, min_box_area=200):
    rows = []
    for track in tracks:
        x, y, w, h = map(float, track.tlwh)
        if w * h > min_box_area and w / h <= 1.6:
            rows.append((int(track.track_id), x, y, w, h))
    return rows


def write_mot_results(path, frame_results, expected_length, sequence):
    """Require complete frame coverage even when empty frames write no MOT rows."""
    path = Path(path)
    frame_results = list(frame_results)
    if [frame for frame, _ in frame_results] != list(range(1, expected_length + 1)):
        raise ValueError('Full official sequence coverage is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for frame, tracks in frame_results:
            for identity, x, y, w, h in tracks:
                handle.write(f'{frame},{identity},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n')
    coverage = dict(sequence=sequence, frames_processed=expected_length, first_frame=1,
                    last_frame=expected_length, full_sequence=True, detections='public DPM det/det.txt',
                    ground_truth_detection_input=False)
    path.with_suffix('.coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')


@torch.no_grad()
def run_sequence(backbone, sequence, tracker, output_path, device='cpu'):
    """Exactly one backbone call per frame, including no-detection frames."""
    tracker.reset_sequence(sequence.info, reset_backbone=backbone.reset)
    results = []
    started = time.monotonic()
    for index in range(len(sequence)):
        frame = sequence[index]
        feature_tensor = backbone(frame.image.unsqueeze(0).to(device))
        tracks = tracker.update(feature_tensor, frame)
        results.append((frame.frame_index, tracking_rows(tracks, tracker.opt.min_box_area)))
        if frame.frame_index == 1 or frame.frame_index % 100 == 0 or frame.frame_index == len(sequence):
            print(json.dumps(dict(sequence=sequence.name, frame=frame.frame_index, total_frames=len(sequence),
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
    write_mot_results(output_path, results, len(sequence), sequence.name)
    return dict(sequence=sequence.name, frames=len(sequence), backbone_calls=len(sequence),
                reset_at_sequence_boundary=True, public_dpm=True, gt_detection_input=False)


def _trackeval_modules(upstream):
    """Load bundled official metrics without importing unrelated optional datasets."""
    path = Path(upstream).resolve() / 'eval/trackeval'
    prefix = '_vistream_trackeval_' + hashlib.sha256(str(path).encode()).hexdigest()[:10]
    if prefix in sys.modules and hasattr(sys.modules[prefix], 'Evaluator'):
        return sys.modules[prefix]
    for suffix, directory in (('', path), ('.datasets', path / 'datasets'), ('.metrics', path / 'metrics')):
        module = types.ModuleType(prefix + suffix)
        module.__path__ = [str(directory)]
        module.__file__ = str(directory / '__init__.py')
        sys.modules[module.__name__] = module
    root = sys.modules[prefix]
    root.datasets, root.metrics = sys.modules[prefix + '.datasets'], sys.modules[prefix + '.metrics']
    def load(suffix):
        name = prefix + '.' + suffix
        spec = importlib.util.spec_from_file_location(name, path.joinpath(*suffix.split('.')).with_suffix('.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if getattr(module, 'np', None) is np:
            module.np = _LegacyNumpy()
        return module
    root._timing = load('_timing')
    root.utils = load('utils')
    load('metrics._base_metric')
    for name, cls in (('hota', 'HOTA'), ('clear', 'CLEAR'), ('identity', 'Identity'), ('count', 'Count')):
        setattr(root.metrics, cls, getattr(load('metrics.' + name), cls))
    load('datasets._base_dataset')
    root.datasets.MotChallenge2DBox = load('datasets.mot_challenge_2d_box').MotChallenge2DBox
    root.Evaluator = load('eval').Evaluator
    return root


def evaluate_trackeval(upstream, data_root, tracker_directory, output_directory=None):
    """Bundled TrackEval HOTA/IDF1/MOTA on all held-out TRAIN-video frames.

    This is not the MOT16 private-test leaderboard. GT is used here only by
    TrackEval's official preprocessing and metric computation.
    """
    tracker_directory = Path(tracker_directory).resolve()
    lengths = {}
    train_root = None
    for name in EVAL_SEQUENCES:
        info = sequence_info(data_root, name)
        lengths[name] = info['length']
        parent = Path(info['path']).parent
        if train_root is not None and parent != train_root:
            raise ValueError('Evaluation sequences must share their official train directory')
        train_root = parent
        coverage = json.loads((tracker_directory / f'{name}.coverage.json').read_text())
        if not (coverage.get('sequence') == name and coverage.get('full_sequence') and
                coverage.get('frames_processed') == info['length'] and coverage.get('first_frame') == 1 and
                coverage.get('last_frame') == info['length'] and not coverage.get('ground_truth_detection_input', True)):
            raise ValueError(f'Incomplete/non-public-detection tracking coverage: {name}')
    te = _trackeval_modules(upstream)
    output_directory = Path(output_directory or tracker_directory / 'trackeval').resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation_config = te.Evaluator.get_default_eval_config()
    evaluation_config.update(USE_PARALLEL=False, PLOT_CURVES=False, PRINT_CONFIG=False,
                             TIME_PROGRESS=False, LOG_ON_ERROR=str(output_directory / 'error.log'))
    dataset_config = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(GT_FOLDER=str(train_root), TRACKERS_FOLDER=str(tracker_directory.parent),
        TRACKERS_TO_EVAL=[tracker_directory.name], TRACKER_SUB_FOLDER='', OUTPUT_FOLDER=str(output_directory),
        BENCHMARK='MOT16', SPLIT_TO_EVAL='train', SKIP_SPLIT_FOL=True, SEQ_INFO=lengths,
        PRINT_CONFIG=False, DO_PREPROC=True)
    evaluator = te.Evaluator(evaluation_config)
    dataset = te.datasets.MotChallenge2DBox(dataset_config)
    results, messages = evaluator.evaluate([dataset], [te.metrics.HOTA(), te.metrics.CLEAR(), te.metrics.Identity()])
    result = results[dataset.get_name()][tracker_directory.name]
    metrics = {}
    for name in (*EVAL_SEQUENCES, 'COMBINED_SEQ'):
        row = result[name]['pedestrian']
        metrics[name] = dict(HOTA=100 * float(np.mean(row['HOTA']['HOTA'])),
                             IDF1=100 * float(row['Identity']['IDF1']), MOTA=100 * float(row['CLEAR']['MOTA']))
    report = dict(benchmark='MOT16 official train held-out videos; not private test',
        sequences=list(EVAL_SEQUENCES), official_sequence_lengths=lengths,
        total_frames=sum(lengths.values()), metrics_percent=metrics,
        detections='public DPM', gt_detection_input=False, official_trackeval_preprocessing=True,
        evaluator_source=str(Path(upstream) / 'eval/trackeval'))
    (output_directory / 'heldout_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


@torch.no_grad()
def evaluate_tracking(model, data_root, sequences, output, upstream,
                      image_size=IMAGE_SIZE, device='cuda:0'):
    """Run all three complete held-out videos once and return official metrics."""
    if set(sequences) != set(EVAL_SEQUENCES) or len(sequences) != len(EVAL_SEQUENCES):
        raise ValueError('Fixed held-out evaluation requires exactly MOT16-10/11/13')
    output = Path(output)
    tracks_directory = output / 'tracks'
    tracker = FeatureBoxTracker(upstream, tracker_options(image_size=image_size, device=device))
    previous_training = model.training
    model.to(device).eval()
    sequences_run = []
    try:
        for name in EVAL_SEQUENCES:
            sequence = MOT16Sequence(data_root, name, image_size=image_size, training_targets=False)
            sequences_run.append(run_sequence(model, sequence, tracker, tracks_directory / f'{name}.txt', device))
    finally:
        model.train(previous_training)
    report = evaluate_trackeval(upstream, data_root, tracks_directory, output / 'trackeval')
    report.update(status='complete', image_size=list(image_size), sequence_runs=sequences_run,
                  tracker=tracker.report(), teacher=False, prediction_fusion=False,
                  backbone_calls=sum(row['backbone_calls'] for row in sequences_run))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
