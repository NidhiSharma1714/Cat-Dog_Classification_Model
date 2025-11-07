"""Continuous learning helper for cat/dog classifier.

This module provides a small utility that watches incoming prediction
requests, keeps a rolling buffer of the newly-seen images and fine-tunes
the previously trained TensorFlow/Keras model on top of that buffer.  Once
the model has been updated we immediately regenerate the TFLite artefact so
future inferences pick up the fresh weights.

The implementation intentionally keeps the surface area tiny so both the
CLI `predict.py` script and the FastAPI application can opt into the same
behaviour.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

# TensorFlow is optional for pure inference deployments.  We only import it
# lazily when continuous learning is actually invoked.
try:  # pragma: no cover - import guard
    import tensorflow as tf  # type: ignore
except ImportError:  # pragma: no cover - import guard
    tf = None  # type: ignore


@dataclass
class ContinuousLearningConfig:
    keras_model_path: Path = Path("saved_model") / "cat_dog_classifier.h5"
    tflite_model_path: Path = Path("cat_dog_classifier.tflite")
    buffer_dir: Path = Path("dataset") / "continuous_buffer"
    min_buffer_size: int = 1
    max_buffer_per_class: int = 128
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    image_size: tuple[int, int] = (150, 150)
    augment_factor: int = 3  # how many synthetic batches per update


class ContinuousLearner:
    """Handles incremental fine-tuning on newly observed samples."""

    def __init__(
        self,
        config: ContinuousLearningConfig | None = None,
        *,
        on_model_updated: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or ContinuousLearningConfig()
        self._lock = threading.Lock()
        self._on_model_updated = on_model_updated

        for label in ("cat", "dog"):
            (self.config.buffer_dir / label).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_with_image(self, image_path: Path, predicted_label: str) -> bool:
        """Persist the new image and (optionally) fine-tune the model."""

        with Image.open(image_path).convert("RGB") as img:
            return self.update_with_pil_image(img, predicted_label)

    def update_with_pil_image(self, image: Image.Image, predicted_label: str) -> bool:
        """Version that accepts an already loaded PIL image."""

        label = predicted_label.strip().lower()
        if label not in {"cat", "dog"}:
            raise ValueError(f"Unsupported label '{predicted_label}'. Expected 'Cat' or 'Dog'.")

        saved_path = self._store_image(image, label)
        print(f"[ContinuousLearning] Stored sample at {saved_path}")

        return self._maybe_train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _store_image(self, image: Image.Image, label: str) -> Path:
        label_dir = self.config.buffer_dir / label
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{int(time.time() * 1e6)}.png"
        target_path = label_dir / filename
        image.save(target_path)

        self._enforce_buffer_limit(label_dir)
        return target_path

    def _enforce_buffer_limit(self, label_dir: Path) -> None:
        files = sorted(label_dir.glob("*.png"), key=os.path.getmtime)
        overflow = len(files) - self.config.max_buffer_per_class
        for _ in range(max(0, overflow)):
            oldest = files.pop(0)
            try:
                oldest.unlink()
                print(f"[ContinuousLearning] Dropped stale buffer sample {oldest}")
            except FileNotFoundError:
                pass

    def _buffer_size(self) -> int:
        total = 0
        for label in ("cat", "dog"):
            total += len(list((self.config.buffer_dir / label).glob("*.png")))
        return total

    def _maybe_train(self) -> bool:
        if tf is None:
            print("[ContinuousLearning] TensorFlow not available; skipping update.")
            return False

        if self._buffer_size() < self.config.min_buffer_size:
            return False

        with self._lock:
            return self._train_and_refresh()

    def _train_and_refresh(self) -> bool:
        config = self.config

        if not config.keras_model_path.exists():
            print(f"[ContinuousLearning] Keras model not found at {config.keras_model_path}; skipping.")
            return False

        print("[ContinuousLearning] Fine-tuning model with buffered samples...")

        model = tf.keras.models.load_model(config.keras_model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            zoom_range=0.1,
            horizontal_flip=True,
        )

        train_flow = datagen.flow_from_directory(
            str(config.buffer_dir),
            target_size=config.image_size,
            batch_size=config.batch_size,
            class_mode="binary",
            shuffle=True,
        )

        if train_flow.samples == 0:
            print("[ContinuousLearning] Buffer empty at fit time; skipping training.")
            return False

        steps = max(1, min(train_flow.samples // config.batch_size * config.augment_factor, config.augment_factor))

        model.fit(
            train_flow,
            epochs=config.epochs,
            steps_per_epoch=steps,
            verbose=0,
        )

        model.save(config.keras_model_path)
        self._export_tflite(model)
        self._clear_buffer()

        print("[ContinuousLearning] Model updated and artefacts refreshed.")

        if self._on_model_updated:
            try:
                self._on_model_updated()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[ContinuousLearning] on_model_updated callback failed: {exc}")

        return True

    def _export_tflite(self, model: "tf.keras.Model") -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(self.config.tflite_model_path, "wb") as f:
            f.write(tflite_model)

    def _clear_buffer(self) -> None:
        for label in ("cat", "dog"):
            label_dir = self.config.buffer_dir / label
            for path in label_dir.glob("*.png"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue


# Singleton-style convenience accessor so callers can share one instance
_GLOBAL_LEARNER: Optional[ContinuousLearner] = None


def get_continuous_learner(on_model_updated: Optional[Callable[[], None]] = None) -> ContinuousLearner:
    global _GLOBAL_LEARNER
    if _GLOBAL_LEARNER is None:
        _GLOBAL_LEARNER = ContinuousLearner(on_model_updated=on_model_updated)
    elif on_model_updated and _GLOBAL_LEARNER._on_model_updated is None:
        _GLOBAL_LEARNER._on_model_updated = on_model_updated
    return _GLOBAL_LEARNER


def reset_continuous_learner() -> None:
    """Useful for tests to clear the singleton state."""

    global _GLOBAL_LEARNER
    _GLOBAL_LEARNER = None


