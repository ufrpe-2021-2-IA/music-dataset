"""
Módulo utilitário, contém funções que auxiliam no
processamento do GTZAN e MSD.
"""

import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds


# Set GTZAN download URL
_DOWNLOAD_URL = 'NONE'


def load_gtzan(gtzan_dir) -> tf.data.Dataset:
    gtzan_dir_path = pathlib.Path(gtzan_dir)

    if not gtzan_dir_path.exists():
        gtzan_dir_path.mkdir(parents=True, exist_ok=True)

    original_url = tfds.audio.gtzan.gtzan._DOWNLOAD_URL
    tfds.audio.gtzan.gtzan._DOWNLOAD_URL = _DOWNLOAD_URL

    builder = tfds.builder('gtzan',
                           data_dir=str(gtzan_dir_path),
                           version='1.0.0')

    download_dir = gtzan_dir_path.joinpath('download')
    builder.download_and_prepare(download_dir=download_dir)

    tfds.audio.gtzan.gtzan._DOWNLOAD_URL = original_url
    return builder.as_dataset(split='train', shuffle_files=False, as_supervised=True)
