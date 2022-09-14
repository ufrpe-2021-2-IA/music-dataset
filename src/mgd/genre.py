from __future__ import annotations

import enum


class Genre(enum.Enum):
    CLASSIC = (0, 'Clássica')
    HIP_HOP = (1, 'Hip-Hop')
    ELECTRONIC = (2, 'Eletrônica')
    POP = (3, 'Pop')
    ROCK = (4, 'Rock')

    @classmethod
    def from_number(cls, v: int) -> Genres:
        if v == 0:
            return cls.CLASSIC

        if v == 1:
            return cls.HIP_HOP

        if v == 2:
            return cls.ELECTRONIC

        if v == 3:
            return cls.POP

        if v == 4:
            return cls.ROCK

        raise ValueError("Valor não reconhecido")
