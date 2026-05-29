import random
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier


class DigitClassificationInterface(ABC):
    """
    Abstract class that defines inference protocol 
    for all digit classification models.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Args:
            image (np.ndarray): A numpy array of shape (28, 28, 1) and dtype float32.

        Returns:
            int: The predicted digit class in [0, 9].
        """

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented.")


class CNNModel(DigitClassificationInterface):
    """
    Args:
        model (nn.Module): Trained PyTorch model expecting input (N, 1, 28, 28) 
                           and returning (N, 10) logits.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> int:
        # (28, 28, 1) -> (1, 1, 28, 28)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        logits = self._model(tensor)
        
        return int(logits.argmax(dim=1).item())


class RandomForestModel(DigitClassificationInterface):
    """
    Args:
        model (RandomForestClassifier): Fitted sklearn classifier trained on 784-d input.
    """

    def __init__(self, model: RandomForestClassifier):
        self._model = model

    def predict(self, image: np.ndarray) -> int:
        # (28, 28, 1) -> (1, 784)
        flat = image.reshape(1, -1)
        return int(self._model.predict(flat)[0])


class RandomModel(DigitClassificationInterface):
    """
    Args:
        seed (int | None): Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed

    def predict(self, image: np.ndarray) -> int:
        crop = image[9:19, 9:19, 0]
        pixel_sum = int(crop.sum())
        base = self._seed if self._seed is not None else pixel_sum
        return random.Random(base + pixel_sum).randint(0, 9)


_REGISTRY: dict[str, type[DigitClassificationInterface]] = {
    "cnn":  CNNModel,
    "rf":   RandomForestModel,
    "rand": RandomModel,
}

class DigitClassifier:
    """
    Facade for digit classification.
    Performs inference based on the selected algorithm's name.
    
    Supported algorithms:
        - cnn  (convolutional model)
        - rf   (random forest)
        - rand (random digit generator)
        
    Args:
        algorithm (str): The name of the algorithm to use.
        **model_kwargs: Keyword arguments passed to the backend constructor.
    """

    def __init__(self, algorithm: str, **model_kwargs):
        if algorithm not in _REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from: {sorted(_REGISTRY)}."
            )
        self._model = _REGISTRY[algorithm](**model_kwargs)

    def predict(self, image: np.ndarray) -> int:
        """
        Performs a digit classification based on the selected algorithm.
        
        Args:
            image (np.ndarray): A numpy array of shape (28, 28, 1) and dtype float32.

        Returns:
            int: The predicted digit class in [0, 9].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected (28, 28, 1), got {image.shape}.")

        return self._model.predict(image)
