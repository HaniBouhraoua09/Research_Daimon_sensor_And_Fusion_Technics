"""
tactile_ros_nodes.py
====================================================================
ROS2 (rclpy) nodes for the multi-modal tactile material-classification
experiment. This is the "ROS node that simulates the experiment" required
by the final assignment.

The pipeline mirrors the COGAR design-pattern architecture described in the
paper:

    TactileReplayPublisher   (Sensor-Device + Publish-Subscribe)
        - replays the REAL recorded test trials from disk
        - selects the peak-deformation frame (same rule as the offline pipeline)
        - injects SYNTHETIC perturbations under explicit assumptions:
              * relative Gaussian noise   (domain randomisation)
              * modality dropout          (simulated sensor failure)
        - publishes each modality on its own topic:
              /tactile/rawimg  /tactile/depth  /tactile/deformation  /tactile/shear

    FusionClassifierSubscriber   (Computational pipeline)
        - subscribes to the four modality topics
        - re-assembles one synchronised 4-modality sample per trial
        - runs ALL seven trained models (4 baselines + 3 fusion strategies)
        - publishes the per-model predictions on /tactile/predictions
        - also stores them internally so the notebook can read them directly

Message transport uses only std_msgs (Float32MultiArray / Int32MultiArray),
so NO custom interface package needs to be built. Trial id, ground-truth
label and tensor shape are carried in the MultiArray layout.

ASSUMPTIONS (must be stated for the assignment):
  A1. The recorded contact data are the ground truth; robustness is probed by
      adding zero-mean Gaussian noise whose std is proportional to each
      modality's own std:  sigma = noise_level * (std(modality_sample) + eps).
      noise_level is therefore a dimensionless "relative noise" knob.
  A2. Modality dropout sets a whole modality channel to zero, emulating a dead
      sensor stream while the others keep working.
  A3. One physical specimen per class (inherited from the dataset); the noise
      sweep is applied identically to every model, so the *relative* fusion
      comparison is fair.

Author: Student 8314923
"""

import os
import glob
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension

import torch

# Reuse the EXACT project code so the notebook stays coherent with training.
from models import build_model
from dataset_loader import MATERIAL_NAME_FIXES, EXCLUDE_TRIALS


# Fixed order of the four modality topics and the seven models.
MODALITIES = ['rawimg', 'depth', 'deformation', 'shear']
MODEL_ORDER = ['rawimg', 'depth', 'deformation', 'shear',
               'early_fusion', 'late_fusion', 'hybrid_fusion']

TOPICS = {m: f'/tactile/{m}' for m in MODALITIES}
PRED_TOPIC = '/tactile/predictions'

_EPS = 1e-6


# ====================================================================
# helpers to pack / unpack a modality tensor into a Float32MultiArray
# ====================================================================
def pack_modality(arr, trial_id, label):
    """arr: float32 (C, H, W). Carry trial_id + label + shape in the layout."""
    c, h, w = arr.shape
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label='trial_id', size=int(trial_id), stride=c * h * w),
        MultiArrayDimension(label='gt_label', size=int(label),    stride=c * h * w),
        MultiArrayDimension(label='C', size=int(c), stride=c * h * w),
        MultiArrayDimension(label='H', size=int(h), stride=h * w),
        MultiArrayDimension(label='W', size=int(w), stride=w),
    ]
    msg.layout.data_offset = 0
    msg.data = arr.astype(np.float32).reshape(-1).tolist()
    return msg


def unpack_modality(msg):
    """Return (trial_id, label, np.float32 array of shape (C, H, W))."""
    dims = {d.label: d.size for d in msg.layout.dim}
    trial_id = dims['trial_id']
    label = dims['gt_label']
    c, h, w = dims['C'], dims['H'], dims['W']
    arr = np.asarray(msg.data, dtype=np.float32).reshape(c, h, w)
    return trial_id, label, arr


# ====================================================================
# peak-frame extraction (identical rule to dataset_loader.TactileDataset)
# ====================================================================
def load_peak_sample(trial_dir):
    """Load a trial and return the peak-deformation frame of all 4 modalities.

    Returns a dict of channel-first float32 arrays:
        rawimg (1,H,W) in [0,1], depth (1,H,W), deformation (2,H,W), shear (2,H,W)
    """
    rawimg = np.load(os.path.join(trial_dir, 'rawimg.npy'))       # (T,H,W)
    depth  = np.load(os.path.join(trial_dir, 'depth.npy'))        # (T,H,W)
    defo   = np.load(os.path.join(trial_dir, 'deformation.npy'))  # (T,H,W,2)
    shear  = np.load(os.path.join(trial_dir, 'shear.npy'))        # (T,H,W,2)

    defo_mag = np.linalg.norm(defo, axis=-1)                      # (T,H,W)
    peak = int(defo_mag.reshape(len(defo_mag), -1).max(axis=1).argmax())

    return {
        'rawimg':      (rawimg[peak].astype(np.float32) / 255.0)[None, ...],   # (1,H,W)
        'depth':        depth[peak].astype(np.float32)[None, ...],             # (1,H,W)
        'deformation':  defo[peak].astype(np.float32).transpose(2, 0, 1),      # (2,H,W)
        'shear':        shear[peak].astype(np.float32).transpose(2, 0, 1),     # (2,H,W)
    }


# ====================================================================
# PUBLISHER  (Sensor-Device)
# ====================================================================
class TactileReplayPublisher(Node):
    """Replays recorded test trials and publishes the 4 modalities as topics.

    Synthetic perturbations are controlled by ROS2 parameters so they can be
    changed live from the notebook:
        noise_level (double)  - relative Gaussian noise std (0.0 = clean)
        seed        (int)     - RNG seed for the noise realisation
        dropout     (string)  - comma-separated modalities to zero out, e.g. "shear"
    """

    def __init__(self, label_map):
        super().__init__('tactile_replay_publisher')
        self.label_map = label_map
        self.declare_parameter('noise_level', 0.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('dropout', '')
        self._pubs = {m: self.create_publisher(Float32MultiArray, TOPICS[m], 10)
                      for m in MODALITIES}

    # ---- parameter convenience accessors ----
    @property
    def noise_level(self):
        return float(self.get_parameter('noise_level').value)

    @property
    def seed(self):
        return int(self.get_parameter('seed').value)

    @property
    def dropout_set(self):
        raw = str(self.get_parameter('dropout').value)
        return {x.strip() for x in raw.split(',') if x.strip()}

    def label_of(self, trial_dir):
        mat_folder = os.path.basename(os.path.dirname(trial_dir)).split('=')[1]
        mat = MATERIAL_NAME_FIXES.get(mat_folder, mat_folder)
        return self.label_map[mat]

    def _perturb(self, arr, rng):
        """Apply relative Gaussian noise (A1) and modality dropout (A2)."""
        nl = self.noise_level
        if nl > 0.0:
            sigma = nl * (float(arr.std()) + _EPS)
            arr = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        return arr

    def publish_trial(self, trial_dir, trial_id):
        """Publish one trial's 4 modalities (peak frame). Returns the gt label."""
        label = self.label_of(trial_dir)
        sample = load_peak_sample(trial_dir)
        # Deterministic per (seed, trial) noise so a sweep is reproducible.
        rng = np.random.default_rng(self.seed * 100000 + trial_id)
        drop = self.dropout_set
        for m in MODALITIES:
            arr = sample[m]
            if m in drop:
                arr = np.zeros_like(arr)          # simulated dead sensor (A2)
            else:
                arr = self._perturb(arr, rng)     # relative Gaussian noise (A1)
            self._pubs[m].publish(pack_modality(arr, trial_id, label))
        return label


# ====================================================================
# SUBSCRIBER / CLASSIFIER  (Computational pipeline)
# ====================================================================
class FusionClassifierSubscriber(Node):
    """Subscribes to the 4 modality topics, runs all 7 models per trial."""

    def __init__(self, models, num_classes, device='cpu'):
        super().__init__('fusion_classifier')
        self.models = models                 # {name: torch.nn.Module (eval)}
        self.num_classes = num_classes
        self.device = device
        self._buffer = {}                    # trial_id -> {'label':int, modality:tensor}
        self.results = []                    # list of dicts (one per completed trial)
        self._pred_pub = self.create_publisher(Int32MultiArray, PRED_TOPIC, 10)
        self._subs = [
            self.create_subscription(
                Float32MultiArray, TOPICS[m],
                self._make_cb(m), 10)
            for m in MODALITIES
        ]

    def reset(self):
        self._buffer.clear()
        self.results.clear()

    def _make_cb(self, modality):
        def _cb(msg):
            trial_id, label, arr = unpack_modality(msg)
            slot = self._buffer.setdefault(trial_id, {'label': label})
            slot[modality] = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1,C,H,W)
            if all(m in slot for m in MODALITIES):
                self._classify(trial_id, slot)
                del self._buffer[trial_id]
        return _cb

    @torch.no_grad()
    def _classify(self, trial_id, slot):
        batch = {m: slot[m] for m in MODALITIES}
        row = {'trial_id': int(trial_id), 'true': int(slot['label'])}
        preds = []
        for name in MODEL_ORDER:
            if name in self.models:
                p = int(self.models[name](batch).argmax(dim=1).item())
            else:
                p = -1
            row[name] = p
            preds.append(p)
        self.results.append(row)
        # Publish predictions too (architectural fidelity / ros2 topic echo).
        out = Int32MultiArray()
        out.data = [int(trial_id), int(slot['label'])] + preds
        self._pred_pub.publish(out)


# ====================================================================
# convenience: load all checkpoints into eval-mode models
# ====================================================================
def load_models(checkpoint_dir, num_classes, model_order=MODEL_ORDER, device='cpu'):
    models = {}
    for name in model_order:
        ckpt = os.path.join(checkpoint_dir, f'{name}_best.pt')
        if not os.path.exists(ckpt):
            print(f'  [skip] {name}: no checkpoint at {ckpt}')
            continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net = build_model(name, num_classes).to(device)
        net.load_state_dict(state['state_dict'])
        net.eval()
        models[name] = net
        print(f'  loaded {name} (val_acc={state.get("val_acc", "?")})')
    if not models:
        raise RuntimeError(f'No checkpoints found in {checkpoint_dir}')
    return models


# ====================================================================
# build the TEST split (identical seed -> identical trials as the paper)
# ====================================================================
def build_test_trials(dataset_root, class_subset=None, test_size=0.2,
                      val_size=0.15, seed=42):
    """Return (test_trial_dirs, label_map). Mirrors dataset_loader.build_splits
    but lets you restrict to a class subset (e.g. Experiment 1: apple/orange/kiwi)."""
    from sklearn.model_selection import train_test_split

    material_dirs = sorted(glob.glob(os.path.join(dataset_root, 'material=*')))
    materials_raw = [os.path.basename(d).split('=')[1] for d in material_dirs]
    materials = [MATERIAL_NAME_FIXES.get(m, m) for m in materials_raw]

    keep = set(class_subset) if class_subset else set(materials)
    label_map = {mat: i for i, mat in enumerate(sorted(keep))}

    all_trials, all_labels = [], []
    for mat_dir, mat in zip(material_dirs, materials):
        if mat not in keep:
            continue
        for t in sorted(glob.glob(os.path.join(mat_dir, 'trial_*'))):
            num = int(os.path.basename(t).split('_')[1])
            if (mat, num) in EXCLUDE_TRIALS:
                continue
            all_trials.append(t)
            all_labels.append(label_map[mat])

    trainval, test, trainval_y, _ = train_test_split(
        all_trials, all_labels, test_size=test_size,
        stratify=all_labels, random_state=seed)
    return test, label_map
