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


def normalize_features(features: AudioFeatures,
                       norm_agg: typing.Literal['baseline',
                                                'scenario1',
                                                'scenario2',
                                                'scenario3',
                                                'scenario4'] = 'scenario1') -> NormalizedAudioFeatures:
    """
    Recebe as características de uma música por frame e retorna a média
        normalizada dessas características.

    Parameters
    ----------
    features: AudioFeatures
        contém as características raw extraídas

    norm_agg: 'baseline', 'scenario1', 'scenario2', 'scenario3'
        indica a estratégia de normalização e agregação a ser utilizada.
            - baseline = média dos frames, sem agregação
            - scenario1 = standard score (mfcc, tonnetz) + min_max(sf,sc,sr) + média dos frames
            - scenario2 = min_max + média dos frames
            - scenario3 = standard score + média dos frames
            - scenario4 = l2-norm + média dos frames
    """

    def default(x):
        return x.mean(axis=-1)

    mfcc = default
    sf = default
    sc = default
    sr = default
    tonnetz = default

    if norm_agg == 'scenario1':
        mfcc = _standard_scaler
        sf = _min_max_scaler
        sc = _min_max_scaler
        sr = _min_max_scaler
        tonnetz = _standard_scaler
    elif norm_agg == 'scenario2':
        mfcc = _min_max_scaler
        sf = _min_max_scaler
        sc = _min_max_scaler
        sr = _min_max_scaler
        tonnetz = _min_max_scaler
    elif norm_agg == 'scenario3':
        mfcc = _standard_scaler
        sf = _standard_scaler
        sc = _standard_scaler
        sr = _standard_scaler
        tonnetz = _standard_scaler
    elif norm_agg == 'scenario4':
        mfcc = _l2_normalize
        sf = _l2_normalize
        sc = _l2_normalize
        sr = _standard_scaler
        tonnetz = _l2_normalize

    return NormalizedAudioFeatures(mfcc=mfcc(features.mfcc),
                                   sf=sf(features.sf),
                                   sc=sc(features.sc),
                                   sr=sr(features.sr),
                                   tonnetz=tonnetz(features.tonnetz))


def _standard_scaler(value: np.ndarray) -> np.ndarray:
    scaler = sklearn.preprocessing.StandardScaler()
    result = scaler.fit_transform(X=value).mean(axis=-1)
    return result


def _min_max_scaler(value: np.ndarray,
                    range=(-1.0, 1.0)) -> np.ndarray:
    return sklearn.preprocessing.minmax_scale(value,
                                              feature_range=range,
                                              axis=-1).mean(axis=-1)


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    scaler = sklearn.preprocessing.Normalizer(norm='l2')
    result = scaler.fit_transform(X=value).mean(axis=-1)
    return result
