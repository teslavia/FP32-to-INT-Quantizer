import numpy as np
import os
from typing import Union, List, Tuple


def save_quant_data(
    quant_data: Union[np.ndarray, List[np.ndarray]],
    save_path: str,
    quant_bit: int = 8,
    int_min: int = -128,
    int_max: int = 127,
) -> None:
    if isinstance(quant_data, (list, tuple)):
        quant_data = np.concatenate([data.flatten() for data in quant_data])
    else:
        quant_data = quant_data.flatten()

    if quant_bit == 4:
        quant_data = np.clip(quant_data, int_min, int_max)
        quant_data_uint4 = quant_data + 8
        if len(quant_data_uint4) % 2 != 0:
            quant_data_uint4 = np.append(quant_data_uint4, 0)
        packed_data = (quant_data_uint4[::2] << 4) | quant_data_uint4[1::2]
        packed_data = packed_data.astype(np.uint8)
    else:
        packed_data = quant_data.astype(np.int8)

    with open(save_path, "wb") as f:
        f.write(packed_data.tobytes())
    print(f"Quantized data saved to: {save_path} (Size: {os.path.getsize(save_path)} bytes)")


def load_quant_data(load_path: str, target_shape: Tuple[int, ...], quant_bit: int = 8) -> np.ndarray:
    with open(load_path, "rb") as f:
        data_bytes = f.read()

    if quant_bit == 4:
        packed_data = np.frombuffer(data_bytes, dtype=np.uint8)
        quant_data_uint4 = np.empty(len(packed_data) * 2, dtype=np.uint8)
        quant_data_uint4[::2] = (packed_data >> 4) & 0x0F
        quant_data_uint4[1::2] = packed_data & 0x0F
        quant_data = quant_data_uint4 - 8
        quant_data = quant_data[: np.prod(target_shape)]
    else:
        quant_data = np.frombuffer(data_bytes, dtype=np.int8)
        if len(quant_data) != np.prod(target_shape):
            raise ValueError(f"Data length {len(quant_data)} does not match target shape {target_shape}")

    quant_data = quant_data.reshape(target_shape)
    print(f"Quantized data loaded, shape: {quant_data.shape}")
    return quant_data


def save_quant_params(quant_params: dict, save_path: str) -> None:
    np.save(save_path, quant_params)
    print(f"Quantization parameters saved to: {save_path}")


def load_quant_params(load_path: str) -> dict:
    quant_params = np.load(load_path, allow_pickle=True).item()
    print(f"Quantization parameters loaded (Precision: INT{quant_params['quant_bit']})")
    return quant_params
