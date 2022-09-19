"""
Módulo com as definições dos pipes.
"""

import pathlib
import typing

import librosa
import numpy as np
import pandas as pd
import sklearn
from scipy import stats


class SummaryStatistics(typing.NamedTuple):
    mean: np.ndarray
    std: np.ndarray
    skew: np.ndarray
    kurtosis: np.ndarray
    median: np.ndarray
    min: np.ndarray
    max: np.ndarray


class AudioFeatures(typing.NamedTuple):
    """
    Representa as características de uma música.
    """
    mfcc: np.ndarray
    sf: np.ndarray
    sc: np.ndarray
    sr: np.ndarray
    tonnetz: np.ndarray


class SummarizedAudioFeatures(typing.NamedTuple):
    """
    Representa as características sonoras de uma música em termos de 
        meidadas estatística.
    """
    mfcc: SummaryStatistics
    sf: SummaryStatistics
    sc: SummaryStatistics
    sr: SummaryStatistics
    tonnetz: SummaryStatistics


def load_audio(path: typing.Union[str, pathlib.Path],
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


def summarize_features(features: AudioFeatures) -> SummarizedAudioFeatures:
    """
    Recebe as características de uma música por frame e retorna a média
        normalizada dessas características.

    Parameters
    ----------
    features: AudioFeatures
        contém as características raw extraídas
    """

    def _stats(feature: np.ndarray, axis=1) -> SummaryStatistics:
        return SummaryStatistics(mean=feature.mean(axis=axis),
                                 std=feature.std(axis=axis),
                                 skew=stats.skew(feature, axis=axis),
                                 kurtosis=stats.kurtosis(feature, axis=axis),
                                 median=np.median(feature, axis=axis),
                                 min=feature.min(axis=axis),
                                 max=feature.max(axis=axis))

    return SummarizedAudioFeatures(mfcc=_stats(features.mfcc),
                                   sf=_stats(features.sf),
                                   sc=_stats(features.sc),
                                   sr=_stats(features.sr),
                                   tonnetz=_stats(features.tonnetz))
