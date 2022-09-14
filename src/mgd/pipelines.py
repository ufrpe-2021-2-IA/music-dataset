"""
Módulo que define as pipelines para obtenção do MGD.
"""

import pathlib

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from mgd import dataset_processing as processing
from mgd.genres import Genres


def gtzan_collect_genres(gtzan_dir: str, save_result=False) -> tf.data.Dataset:
    save_path = pathlib.Path(gtzan_dir)
    save_path = save_path.joinpath('remapped_genres')

    if save_path.exists():
        return tf.data.Dataset.load(str(save_path))

    gtzan = processing.load_gtzan(gtzan_dir)

    classes = tf.constant([1, 4, 7, 9], dtype=tf.int64)
    new_classes = tf.constant([0, 1, 3, 4], dtype=tf.int64)
    mapper = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(classes, new_classes),
        default_value=5)

    def _filter(audio, label):
        return tf.reduce_any(tf.math.equal(label, classes))

    gtzan = gtzan.filter(_filter)

    def _map(audio, label):
        return audio, mapper.lookup(label)

    gtzan = gtzan.map(_map)

    if save_result:
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=False)
            gtzan.save(str(save_path))

    return gtzan


def gtzan_to_df(gtzan: tf.data.Dataset) -> pd.DataFrame:
    df = pd.DataFrame(list(gtzan.as_numpy_iterator()))
    df = df.rename(columns={0: 'audio', 1: 'label'})
    df['src'] = 'GTZAN'

    return df
