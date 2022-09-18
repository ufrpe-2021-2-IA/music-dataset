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
                                                'scenario4',
                                                'scenario5'] = 'scenario1') -> NormalizedAudioFeatures:
    """
    Recebe as características de uma música por frame e retorna a média
        normalizada dessas características.

    Parameters
    ----------
    features: AudioFeatures
        contém as características raw extraídas

    norm_agg: 'baseline', 'scenario1', 'scenario2', 'scenario3'
        indica a estratégia de normalização e agregação (média) a ser utilizada.
            - baseline = sem normalização (mfcc, tonnetz, sf, sc, src) + média dos frames
            - scenario1 = standard score (mfcc, tonnetz) + min_max (sf, sc, sr) + média dos frames
            - scenario2 = standard score (mfcc, tonnetz, sf, sc, sr) + média dos frames
            - scenario3 = min_max (mfcc, tonnetz, sf, sc, sr) + média dos frames
            - scenario4 = min_max (mfcc, tonnetz) + standard score (sf, sc, sr) + média dos frames
            - scenario5 = l2-norm (mfcc, tonnetz) + min_max (sf, sc, sr) + média dos frames
            - scenario6 = l2-norm (mfcc, tonnetz) + standard score (sf, sc, sr) + média dos frames
    """

    def default(x):
        """
        x tem que ter shape (n_samples, n_features)
        """

        return x.mean(axis=0)

    mfcc = default
    sf = default
    sc = default
    sr = default
    tonnetz = default

    if norm_agg == 'baseline':
        mfcc = default
        sf = default
        sc = default
        sr = default
        tonnetz = default
    elif norm_agg == 'scenario1':
        mfcc = _standard_scaler
        sf = _min_max_scaler
        sc = _min_max_scaler
        sr = _min_max_scaler
        tonnetz = _standard_scaler
    elif norm_agg == 'scenario2':
        mfcc = _standard_scaler
        sf = _standard_scaler
        sc = _standard_scaler
        sr = _standard_scaler
        tonnetz = _standard_scaler
    elif norm_agg == 'scenario3':
        mfcc = _min_max_scaler
        sf = _min_max_scaler
        sc = _min_max_scaler
        sr = _min_max_scaler
        tonnetz = _min_max_scaler
    elif norm_agg == 'scenario4':
        mfcc = _min_max_scaler
        sf = _standard_scaler
        sc = _standard_scaler
        sr = _standard_scaler
        tonnetz = _min_max_scaler
    elif norm_agg == 'scenario5':
        mfcc = _l2_normalize
        sf = _min_max_scaler
        sc = _min_max_scaler
        sr = _min_max_scaler
        tonnetz = _l2_normalize
    elif norm_agg == 'scenario6':
        mfcc = _l2_normalize
        sf = _standard_scaler
        sc = _standard_scaler
        sr = _standard_scaler
        tonnetz = _l2_normalize
    else:
        print(f'[WARNING] Unrecognized scenario "{norm_agg}", using default.')

    return NormalizedAudioFeatures(mfcc=mfcc(features.mfcc.T),
                                   sf=sf(features.sf.T),
                                   sc=sc(features.sc.T),
                                   sr=sr(features.sr.T),
                                   tonnetz=tonnetz(features.tonnetz.T))


def _standard_scaler(value: np.ndarray) -> np.ndarray:
    """
    value tem que ter shape (n_samples, n_features)
    """

    scaler = sklearn.preprocessing.StandardScaler()
    result = scaler.fit_transform(X=value).mean(axis=0)
    return result


def _min_max_scaler(value: np.ndarray,
                    feature_range=(-1.0, 1.0)) -> np.ndarray:
    """
    value tem que ter shape (n_samples, n_features)
    """

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)
    result = scaler.fit_transform(X=value).mean(axis=0)
    return result


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    """
    value tem que ter shape (n_samples, n_features)
    """

    scaler = sklearn.preprocessing.Normalizer(norm='l2')
    result = scaler.fit_transform(X=value).mean(axis=0)
    return result
