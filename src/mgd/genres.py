from __future__ import annotations

import enum


class Genres(enum.Enum):
    CLASSIC = (0, 'Clássica')
    HIP_HOP = (1, 'Hip-Hop')
    ELECTRONIC = (2, 'Eletrônica')
    POP = (3, 'Pop')
    ROCK = (4, 'Rock')

    @classmethod
    def is_genre_gtzan(cls, value: str | int) -> bool:
        if isinstance(value, str):
            return value in list(_GTZAN_STR_MAPPER.keys())

        # Valor numérico das strings no GTZAN
        return value in list(_GTZAN_INT_MAPPER.keys())

    @classmethod
    def from_gtzan_genre(cls, value: str | int) -> Genres:
        if isinstance(value, str):
            return _GTZAN_STR_MAPPER[value]

        # Valor numérico das strings no GTZAN
        return _GTZAN_INT_MAPPER[value]


_GTZAN_STR_MAPPER = {
    'classical': Genres.CLASSIC,
    'hiphop': Genres.HIP_HOP,
    'pop': Genres.POP,
    'rock': Genres.ROCK
}

_GTZAN_INT_MAPPER = {
    1: Genres.CLASSIC,
    4: Genres.HIP_HOP,
    7: Genres.POP,
    9: Genres.ROCK
}
