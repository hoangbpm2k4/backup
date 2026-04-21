# Copyright      2021  Piotr Żelasko
# Copyright      2024  Xiaomi Corporation     (Author: Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dataset import HubertAsrDataset
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    """
    DataModule for multi-speaker ASR experiments.

    Loads Vietnamese multi-speaker CutSets (single/overlap2/overlap3)
    with augmentation variants (clean/rir/noise/rir_noise).

    Expected manifest directory layout:
        {manifest_dir}/train.all.cuts_augmented.jsonl.gz   (train: all types + augmentations)
        {manifest_dir}/val.all.cuts_augmented.jsonl.gz     (validation)
        {manifest_dir}/test.all.cuts_augmented.jsonl.gz    (test)

    Or individual files:
        {manifest_dir}/train.single.cuts_clean.jsonl.gz
        {manifest_dir}/train.overlap2.cuts_clean.jsonl.gz
        ...
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="When True, use all augmented data. When False, use clean-only subset.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/lhotse_cutsets"),
            help="Path to directory with valid/test cuts (and train if --train-manifest-dir not set).",
        )
        group.add_argument(
            "--train-manifest-dir",
            type=Path,
            default=None,
            help="Path to directory with train cuts. "
                 "When set, train cuts come from here while val/test cuts "
                 "still come from --manifest-dir.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--do-normalize",
            type=str2bool,
            default=True,
            help="whether to normalize the data",
        )
        group.add_argument(
            "--ssa-single-target",
            type=str2bool,
            default=False,
            help="If True, sample one target speaker per utterance (SSA-style).",
        )
        group.add_argument(
            "--ssa-target-policy",
            type=str,
            default="random",
            choices=["random", "round_robin"],
            help="Target-speaker sampling policy when --ssa-single-target=True.",
        )
        group.add_argument(
            "--augment-types",
            type=str,
            default="clean",
            help="Comma-separated augmentation types to load. "
            "Options: clean,noise,rir,rir_noise. "
            "Example: --augment-types 'clean,noise,rir'",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        do_normalize: bool,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = HubertAsrDataset(
            do_normalize=do_normalize,
            num_speakers=self.args.num_speakers,
            ssa_single_target=self.args.ssa_single_target,
            ssa_target_policy=self.args.ssa_target_policy,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.num_workers > 0,
            pin_memory=True,
            prefetch_factor=8 if self.args.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet, do_normalize: bool) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = HubertAsrDataset(
            do_normalize=do_normalize,
            num_speakers=self.args.num_speakers,
            ssa_single_target=self.args.ssa_single_target,
            ssa_target_policy=self.args.ssa_target_policy,
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet, do_normalize: bool) -> DataLoader:
        logging.debug("About to create test dataset")
        test = HubertAsrDataset(
            do_normalize=do_normalize,
            num_speakers=self.args.num_speakers,
            ssa_single_target=self.args.ssa_single_target,
            ssa_target_policy=self.args.ssa_target_policy,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @property
    def _train_dir(self) -> Path:
        """Return the directory to use for train cuts."""
        d = getattr(self.args, "train_manifest_dir", None)
        return d if d is not None else self.args.manifest_dir

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        """Load clean-only train cuts (single + overlap2 + overlap3)."""
        logging.info("About to get train clean cuts")
        d = self._train_dir
        parts = []
        for name in ["train.single.cuts_clean.jsonl.gz",
                      "train.overlap2.cuts_clean.jsonl.gz",
                      "train.overlap3.cuts_clean.jsonl.gz"]:
            p = d / name
            if p.exists():
                parts.append(load_manifest_lazy(p))
                logging.info(f"  Loaded {name}")
        if not parts:
            raise FileNotFoundError(f"No train clean cuts found in {d}")
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + p
        return combined

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        """Load train cuts with configurable augmentation types.

        Uses --augment-types to control which variants to load.
        E.g. 'clean,noise,rir' loads clean + noise + rir for each mix type.
        """
        aug_types = [s.strip() for s in self.args.augment_types.split(",")]
        logging.info(f"About to get all training cuts (augment_types={aug_types})")
        d = self._train_dir
        parts = []

        # Train data with selected augmentations
        for mix in ["single", "overlap2", "overlap3"]:
            for aug in aug_types:
                p = d / f"train.{mix}.cuts_{aug}.jsonl.gz"
                if p.exists():
                    parts.append(load_manifest_lazy(p))
                    logging.info(f"  Loaded {p.name}")

        # Add val/test clean data to training only when train and val share the
        # same manifest dir (legacy behaviour).  When --train-manifest-dir is
        # set to a separate directory (e.g. a trainonly dataset) the val cuts
        # live in --manifest-dir and must NOT bleed into training.
        separate_train_dir = getattr(self.args, "train_manifest_dir", None) is not None
        if not separate_train_dir:
            for split in ["val", "test"]:
                p = d / f"{split}.all.cuts_clean.jsonl.gz"
                if p.exists():
                    parts.append(load_manifest_lazy(p))
                    logging.info(f"  Loaded {p.name} (added to training)")

        if not parts:
            raise FileNotFoundError(f"No train cuts found in {d}")
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + p
        logging.info(f"  Total training parts: {len(parts)}")
        return combined

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        """Load validation cuts (clean only, ~10K subset for fast eval)."""
        logging.info("About to get validation cuts (~10K subset)")
        d = self.args.manifest_dir
        clean = d / "val.all.cuts_clean.jsonl.gz"
        if clean.exists():
            cuts = load_manifest_lazy(clean)
            return cuts.subset(first=10000)
        parts = []
        for mix in ["single", "overlap2", "overlap3"]:
            p = d / f"val.{mix}.cuts_clean.jsonl.gz"
            if p.exists():
                parts.append(load_manifest_lazy(p))
        if not parts:
            raise FileNotFoundError(f"No val cuts found in {d}")
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + p
        return combined.subset(first=10000)

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        """Return empty CutSet (no separate dev-other for Vietnamese)."""
        return CutSet()

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        """Return empty CutSet (user has separate test set)."""
        return CutSet()

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        """Return empty CutSet (no separate test-other for Vietnamese)."""
        return CutSet()
