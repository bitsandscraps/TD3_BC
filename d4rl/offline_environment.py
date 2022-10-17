from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import gym
import h5py
import numpy as np
from numpy.typing import NDArray
import requests
from tqdm import tqdm

from d4rl.infos import DATASET_URLS, REF_MAX_SCORE, REF_MIN_SCORE


DATA_ROOT = Path('/home/wisrl/offline/data')


def compute_timeouts(terminals: NDArray[np.bool_],
                     max_episode_steps: int) -> NDArray[np.bool_]:
    if max_episode_steps <= 0:
        raise ValueError(
            f'Invalid max_episode_steps argument: {max_episode_steps}')
    step = 0
    timeouts = np.zeros_like(terminals)
    for index, terminal in enumerate(terminals):
        step += 1
        if terminal:
            step = 0
        elif step == max_episode_steps:
            timeouts[index] = True
            step = 0
    return timeouts


def download(url: str, **kwargs) -> Generator[bytes, None, None]:
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with tqdm(total=total,
              unit='B',
              unit_scale=True,
              unit_divisor=1024,
              **kwargs) as bar:
        for chunk in response.iter_content(chunk_size=128):
            bar.update(len(chunk))
            yield chunk


def get_dataset_path(key: str) -> Path:
    url = DATASET_URLS[key]
    dataset_name = url.split('/')[-1]
    DATA_ROOT.mkdir(exist_ok=True)
    dataset_path = DATA_ROOT / dataset_name
    if not dataset_path.is_file():
        try:
            with open(dataset_path, 'wb') as fd:
                for chunk in download(url, desc=dataset_name, leave=False):
                    fd.write(chunk)
        except BaseException:
            dataset_path.unlink(missing_ok=True)
            raise
    return dataset_path


def get_keys_from_h5file(h5file: h5py.File) -> List[str]:
    keys = []

    def visitor(name: str, node) -> None:
        if isinstance(node, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def sanity_check_1d(key: str,
                    data_dict: Dict[str, np.ndarray],
                    num_samples: int) -> Dict[str, np.ndarray]:
    if data_dict[key].shape == (num_samples, 1):
        data_dict[key] = data_dict[key].ravel()
    assert data_dict[key].shape == (num_samples,), \
        f'{key.title()} has wrong shape: {data_dict[key].shape}'
    return data_dict


def sanity_check_nd(key: str,
                    data_dict: Dict[str, np.ndarray],
                    num_samples: int,
                    true_shape: Optional[Tuple[int, ...]],
                    ) -> Dict[str, np.ndarray]:
    shape = data_dict[key].shape
    if true_shape is None:
        check = (shape[0] == num_samples)
    else:
        true_shape = (num_samples,) + true_shape
        check = (shape == true_shape)
    key = key.title()
    assert check, f'{key} shape does not match env: {shape} vs {true_shape}'
    return data_dict


class OfflineEnv(gym.Env):
    max_episode_steps: int

    def __init__(self, dataset_key: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.ref_max_score = REF_MAX_SCORE[dataset_key]
        self.ref_min_score = REF_MIN_SCORE[dataset_key]

    def load_data(self, max_episode_steps: int = 0):
        data_dict = {}
        with h5py.File(get_dataset_path(self.dataset_key)) as dataset_file:
            for key in tqdm(get_keys_from_h5file(dataset_file),
                            desc='load data file',
                            leave=False):
                data_dict[key] = dataset_file[key][()]      # type: ignore
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, f'Dataset is missing key {key}'
        num_samples = data_dict['observations'].shape[0]
        sanity_check_nd('observations', data_dict,
                        num_samples, self.observation_space.shape)
        sanity_check_nd('actions', data_dict,
                        num_samples, self.action_space.shape)
        sanity_check_1d('rewards', data_dict, num_samples)
        sanity_check_1d('terminals', data_dict, num_samples)
        if 'timeouts' not in data_dict:
            data_dict['timeouts'] = compute_timeouts(data_dict['terminals'],
                                                     max_episode_steps)
        data_dict['timeouts'][-1] = True
        sanity_check_1d('timeouts', data_dict, num_samples)
        return data_dict

    def get_normalized_score(self, score):
        denominator = self.ref_max_score - self.ref_min_score
        return (score - self.ref_min_score) / denominator * 100
