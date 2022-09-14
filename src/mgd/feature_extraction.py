"""
Módulo com funções e classes para a extração de características da música.
"""

import typing

import numpy as np
import librosa


class AudioFeatures(typing.NamedTuple):
    mfccs: np.ndarray
    sf: float
    sc: float
    sr: float
    tonnetz: np.ndarray


def extract_from_audio(audio_seq,
                       sample_rate=22050,
                       n_mfcc=13,
                       ) -> AudioFeatures:
    pass
