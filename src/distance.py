# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Zkopírujte sem své řešení z Cvičení 01.

    Jak DBSCAN (vyhledávání sousedů v epsilon-okolí), tak spektrální shlukování
    (sestavení afinitní matice) v tomto cvičení volají metodu ``calculate()``
    této hierarchie tříd - bez jejího doplnění se cvičení nespustí.
"""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Abstraktní základ pro metriky vzdálenosti.

    Každá konkrétní metrika dědí od této třídy a implementuje metodu
    ``calculate``. Metoda ``create_distance_matrix`` je sdílená a volá
    ``calculate`` v cyklu — není třeba ji přepisovat.
    """

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí ``True``, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá vzdálenost mezi dvěma body.

        Úkol:
            Implementujte výpočet vzdálenosti specifický pro danou metriku.

        Parameters
        ----------
        point_a:
            První bod — pole tvaru ``(n_příznaků,)``.
        point_b:
            Druhý bod — pole tvaru ``(n_příznaků,)``.

        Returns
        -------
        float
            Vzdálenost mezi ``point_a`` a ``point_b``.
        """
        # assert: Ověřte, že oba vstupy jsou 1D vektory stejné délky
        raise NotImplementedError(
            "Úkol: implementujte calculate() — zkopírujte řešení z Cvičení 01."
        )

    def create_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vytvoří čtvercovou matici vzdáleností mezi všemi dvojicemi bodů.

        Matice je symetrická s nulovou diagonálou.
        Volá ``self.calculate`` pro každou dvojici — implementace metriky
        není třeba zde duplikovat.

        Parameters
        ----------
        data:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice vzdáleností tvaru ``(n_bodů, n_bodů)``.
        """
        # assert: Ověřte, že data jsou 2D matice s alespoň 2 body
        assert data.ndim == 2
        assert data.shape[0] >= 2
        n: int = data.shape[0]
        matrix: np.ndarray = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist: float = self.calculate(data[i], data[j])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix


class EuclideanDistance(Distance):
    """ Třída pro výpočet Euklidovské vzdálenosti mezi dvěma vektory.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def is_metric(self) -> bool:
        """Vrací True, protože Euklidovská vzdálenost splňuje definici metriky."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Spočte Euklidovskou vzdálenost mezi 1D vektory x a y."""
        return float(np.sqrt(np.sum((x - y) ** 2)))


class ManhattanDistance(Distance):
    """Manhattanská (L1, taxicab) vzdálenost mezi dvěma vektory."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        raise NotImplementedError(
            "Úkol: Implementujte ManhattanDistance.is_metric()"
            "Vrátí True, pokud vzdálenost splňuje axiomy metriky."
        )

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá manhattanskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Manhattanská vzdálenost mezi x a y.
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        raise NotImplementedError(
            "Úkol: Implementujte výpočet manhattanskou vzdálenost mezi dvěma vektory x a y."
            )


class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost (1 - kosinová podobnost) mezi dvěma vektory.
    """

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika (porušuje trojúhelníkovou nerovnost)."""
        raise NotImplementedError(
            "Úkol: Implementujte CosineCoeficient.is_metric()"
            "Vrátí True, pokud vzdálenost splňuje axiomy metriky."
        )

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá kosinovou vzdálenost (1 - kosinový koeficient podobnosti)
        mezi dvěma 1D vektory příznaků x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Kosinová vzdálenost mezi x a y (1 - kosinová podobnost).
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        # assert  Ověřte, že x ani y nejsou nulové vektory (dělení nulou při normalizaci)
        raise NotImplementedError(
            "Úkol: Implementujte výpočet kosinovou vzdálenost/koeficient mezi dvěma vektory x a y."
        )
