import numpy as np
from typing import Tuple, List, Union


def generate_test_data(
    shape: Union[Tuple[int, int], Tuple[int, int, int, int]],
    data_range: Tuple[float, float] = (0.0, 1.0),
    is_image_like: bool = False,
) -> np.ndarray:
    if is_image_like:
        data = np.random.randint(0, 256, size=shape, dtype=np.uint8).astype(np.float32) / 255.0
    else:
        data = np.random.uniform(low=data_range[0], high=data_range[1], size=shape).astype(np.float32)
    return data


def generate_batch_test_data(
    batch_size: int,
    data_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    data_range: Tuple[float, float] = (0.0, 1.0),
    is_image_like: bool = False,
) -> List[np.ndarray]:
    batch_data = []
    for _ in range(batch_size):
        if len(data_shape) == 2:
            data = generate_test_data(shape=data_shape, data_range=data_range, is_image_like=is_image_like)
        else:
            data = generate_test_data(shape=(1,) + data_shape, data_range=data_range, is_image_like=is_image_like)
        batch_data.append(data)
    return batch_data
