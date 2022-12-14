"""
Módulo que define as pipelines para obtenção do MGD.
"""

import pathlib
import typing

from mgd import filters


class Example(typing.NamedTuple):
    features: filters.SummarizedAudioFeatures
    label: str
    song_id: str


def extract_examples_from_dir(dir: typing.Union[str, pathlib.Path]) -> typing.List[Example]:
    """
    Cria exemplos de treinamento/avaliação a partir de um diretório
        que contém arquivos de áudio de um dado dataset para um 
        dado gênero.
    """
    path = dir

    if isinstance(path, str):
        path = pathlib.Path(path)

    examples: typing.List[Example] = []
    genre = path.name

    for p in path.iterdir():
        if not p.is_file():
            # Caso não seja um arquivo, ir para o próximo
            continue

        audio_seq = filters.load_audio(p)
        features = filters.extract_features(audio_seq)
        summarized_features = filters.summarize_features(features)

        examples.append(Example(features=summarized_features,
                                label=genre,
                                song_id=p.name))

    return examples
