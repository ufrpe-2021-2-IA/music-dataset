import requests
import pathlib
import typing

import numpy as np
import pandas as pd

_BASE_URL = 'https://raw.githubusercontent.com/ufrpe-2021-2-IA/music-dataset/feat/summary-statistics/gtzan/processed'
_SCENARIOS = ['raw', 'min_max', 'standardized']


def download_dataset(scenario: str,
                     target_directory='.'):
    if scenario not in _SCENARIOS:
        raise ValueError(f'Unrecognized scenario "{scenario}"')

    url = f'{_BASE_URL}/mgd_{scenario}.csv'
    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError("Couldn't download dataset.")

    save_path = pathlib.Path(target_directory).absolute()

    if not save_path.exists():
        save_path.mkdir(parents=True)

    save_path = save_path.joinpath(f'mgd_{scenario}.csv')

    with open(save_path, 'w+') as f:
        f.write(r.text)


def download_experiments(scenario: str,
                         target_directory='.'):
    if scenario not in _SCENARIOS:
        raise ValueError(f'Unrecognized scenario "{scenario}"')

    save_path = pathlib.Path(
        target_directory).absolute().joinpath(f'{scenario}/')
    down_url = f'{_BASE_URL}/{scenario}/'

    # Criar diretório
    save_path.mkdir(exist_ok=True, parents=True)

    r = requests.get(f'{down_url}/train.csv')
    with open(save_path.joinpath('train.csv'), 'w+') as f:
        f.write(r.text)

    r = requests.get(f'{down_url}/test.csv')
    with open(save_path.joinpath('test.csv'), 'w+') as f:
        f.write(r.text)

    for i in range(1, 6):
        for j in range(1, 11):
            path = f'experiment-{i}/fold-{j}.csv'
            url = f'{down_url}/{path}'
            target_path = save_path.joinpath(path)
            r = requests.get(url)

            # Criar diretório
            target_path.parent.mkdir(exist_ok=True, parents=True)

            with open(target_path, 'w+') as f:
                f.write(r.text)


def prepare_sklearn(csv_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    return df.iloc[:, 2:].to_numpy(), df['label'].to_numpy()
