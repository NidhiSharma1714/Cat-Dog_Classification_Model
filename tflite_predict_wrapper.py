import numpy as np

# --------------------------------------------
#  ✅ Robust TensorFlow Lite Import Handling
# --------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Using tflite-runtime")
except ImportError:
    try:
        # For TensorFlow <= 2.15
        from tensorflow.lite.python.interpreter import Interpreter
        print("✅ Using TensorFlow Lite (python.interpreter)")
    except ImportError:
        try:
            # For TensorFlow 2.16–2.18+ (internal reorg)
            from tensorflow.lite.python import interpreter as tflite_interpreter
            Interpreter = tflite_interpreter.Interpreter
            print("✅ Using TensorFlow Lite (new path: tensorflow.lite.python.interpreter)")
        except Exception as e:
            raise ImportError(
                "❌ Could not import TensorFlow Lite Interpreter. "
                "Try installing tflite-runtime or use a compatible TensorFlow version.\n"
                f"Original error: {e}"
            )


class TFLiteWrapper:
    """
    Simple TFLite wrapper that:
    - Loads a TFLite model
    - Exposes `model_input_hw` = (H, W, C)
    - Predicts on a batch np.array images with shape (N,H,W,C)
    - Returns probabilities with shape (N, nb_classes)
      * If model outputs single sigmoid (N,1), returns (N,2) as [1-p, p]
      * If model outputs multi-dim, returns softmaxed probabilities (if necessary)
    """

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Infer model input H,W,C from input_details
        in_shape = self.input_details[0]["shape"]  # could be [1,H,W,C] or [H,W,C]
        if len(in_shape) == 4:
            _, h, w, c = in_shape
        elif len(in_shape) == 3:
            h, w, c = in_shape
        else:
            # fallback - try last 3 dims
            h, w, c = in_shape[-3], in_shape[-2], in_shape[-1]
        self.model_input_hw = (int(h), int(w), int(c))

        # number of output features (last dim)
        out_shape = self.output_details[0]["shape"]
        self.nb_output_features = int(out_shape[-1]) if len(out_shape) >= 1 else 1

        print("TFLite model loaded.")
        print(" - input_detail:", self.input_details[0])
        print(" - output_detail:", self.output_details[0])
        print(f" - model_input_hw (H,W,C): {self.model_input_hw}")
        print(f" - nb_output_features: {self.nb_output_features}")

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Run inference and return probabilities (N, nb_classes).
        Args:
            images: np.ndarray, shape (N,H,W,C) or (H,W,C) or (N, H, W, C)
                    expected values in [0,1] float if model is float32.
        Returns:
            probs: np.ndarray, shape (N, nb_classes) where nb_classes is either
                   inferred 2 for binary-sigmoid or the model output dim for multi-class.
        """
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")

        # Ensure batch dimension
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)

        N, h, w, c = images.shape
        # Basic shape check - warn if caller passed non-matching sizes (we won't resize here)
        exp_h, exp_w, exp_c = self.model_input_hw
        if (h, w, c) != (exp_h, exp_w, exp_c):
            # it's okay if test/test_predict resizes; but warn to avoid silent errors
            print(f"Warning: input images shape {(h,w,c)} != model expected {(exp_h,exp_w,exp_c)}. "
                  "Make sure you resized inputs before calling predict.")

        # Prepare input
        input_index = self.input_details[0]["index"]
        # Cast to float32 as your TFLite model is float
        inp = images.astype(np.float32)

        # Try to set whole batch at once; if interpreter expects fixed batch of 1 it may still work or raise
        try:
            self.interpreter.set_tensor(input_index, inp)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        except Exception:
            # fallback: per-sample inference
            outputs = []
            for i in range(N):
                self.interpreter.set_tensor(input_index, np.expand_dims(inp[i], axis=0))
                self.interpreter.invoke()
                out_i = self.interpreter.get_tensor(self.output_details[0]["index"])
                outputs.append(out_i[0])
            output_data = np.stack(outputs, axis=0)

        out = np.array(output_data)

        # Normalize output shape
        if out.ndim == 1:
            out = np.expand_dims(out, axis=1)  # (N,) -> (N,1)
        if out.ndim == 2 and out.shape[0] != N and out.shape[1] == N:
            # strange transposed shape; transpose back
            out = out.T

        # Case 1: single sigmoid output per sample (N,1) -> convert to (N,2): [1-p, p]
        if out.shape[1] == 1:
            p = out[:, 0]
            probs = np.stack([1.0 - p, p], axis=1).astype(np.float32)
            return probs

        # Case 2: multi-output. If sums aren't ~1, apply softmax
        sums = out.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-3):
            exps = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            return probs.astype(np.float32)

        return out.astype(np.float32)
