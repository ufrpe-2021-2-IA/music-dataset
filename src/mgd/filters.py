"""
Módulo com as definições dos pipes.
"""

import typing
import pathlib

import pandas as pd
import numpy as np
import librosa
import sklearn


class AudioFeatures(typing.NamedTuple):
    """
    Representa as características de uma música.
    """
    mfcc: np.ndarray
    sf: np.ndarray
    sc: np.ndarray
    sr: np.ndarray
    tonnetz: np.ndarray


class NormalizedAudioFeatures(typing.NamedTuple):
    """
    Representa as características médias normalizadas de uma música.
    """
    mfcc: np.ndarray
    sf: np.ndarray
    sc: np.ndarray
    sr: np.ndarray
    tonnetz: np.ndarray


def load_audio(path: str | pathlib.Path,
               sample_rate=22050,
               to_mono=True,
               dtype=np.float32) -> np.ndarray:
    """
    Carrega uma música usando o `librosa`.

    Parameters
    ----------
    path: str ou pathlib.Path
        caminho para um arquivo de música

    sample_rate: inteiro, opcional
        sample_rate

    to_mono: booleano, opcional
        se devemos converter para um áudio com único canal

    dtype: tipo de dado, opcional
        indica o tipo dos resultados da array

    Return
    ------
    y: np.ndarray de tipo `dtype` representando o áudio passado

    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    if not path.exists():
        raise ValueError(f'O caminho "{str(path)}" não existe.')

    y, _ = librosa.load(str(path), sr=sample_rate, mono=to_mono, dtype=dtype)

    return y


def extract_features(audio_seq: np.ndarray,
                     sample_rate=22050,
                     n_mfcc=13) -> AudioFeatures:
    """
    Recebe uma música (waveform de float) como entrada
        e extrai suas características por frame.
    """

    # Calcular o valor de cada característica utilizando os frames padrão
    mfcc = librosa.feature.mfcc(y=audio_seq, sr=sample_rate, n_mfcc=n_mfcc)
    sf = librosa.feature.spectral_flatness(y=audio_seq)
    sc = librosa.feature.spectral_centroid(y=audio_seq, sr=sample_rate)
    sr = librosa.feature.spectral_rolloff(y=audio_seq, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=audio_seq, sr=sample_rate)

    return AudioFeatures(mfcc=mfcc, sf=sf, sc=sc, sr=sr, tonnetz=tonnetz)


def normalize_features(features: AudioFeatures) -> NormalizedAudioFeatures:
    """
    Recebe as características de uma música por frame e retorna a média
        normalizada dessas características.
    """

    return NormalizedAudioFeatures(mfcc=_standard_scaler(features.mfcc),
                                   sf=_min_max_scaler(features.sf),
                                   sc=_min_max_scaler(features.sc),
                                   sr=_min_max_scaler(features.sr),
                                   tonnetz=_standard_scaler(features.tonnetz))


def _standard_scaler(value: np.ndarray) -> np.ndarray:
    scaler = sklearn.preprocessing.StandardScaler()
    result = scaler.fit_transform(X=value).mean(axis=-1)
    return result


def _min_max_scaler(value: np.ndarray,
                    range=(-1.0, 1.0)) -> np.ndarray:
    return sklearn.preprocessing.minmax_scale(value,
                                              feature_range=range,
                                              axis=-1).mean(axis=-1)
