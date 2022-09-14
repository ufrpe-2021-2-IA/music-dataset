"""
Módulo que define as pipelines para obtenção do MGD.
"""

import pathlib
import typing

from mgd import filters
from mgd import genre


class Example(typing.NamedTuple):
    features: filters.NormalizedAudioFeatures
    label: genre.Genre
    song_src: str
    song_id: str


def extract_examples_from_dir(dir: str | pathlib.Path,
                              genre: genre.Genre,
                              src_dataset: str) -> typing.List[Example]:
    """
    Cria exemplos de treinamento/avaliação a partir de um diretório
        que contém arquivos de áudio de um dado dataset para um 
        dado gênero.
    """
    path = dir

    if isinstance(path, str):
        path = pathlib.Path(path)

    examples: typing.List[Example] = []
    src_dataset = src_dataset.upper()

    for p in path.iterdir():
        if not p.is_file():
            # Caso não seja um arquivo, ir para o próximo
            continue

        audio_seq = filters.load_audio(p)
        features = filters.extract_features(audio_seq)
        normalized_features = filters.normalize_features(features)

        examples.append(Example(features=normalized_features,
                                label=genre,
                                song_src=src_dataset,
                                song_id=p.name))

    return examples
