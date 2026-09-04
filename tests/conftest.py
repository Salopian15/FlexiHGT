"""Shared fixtures: a small in-memory taxonomy standing in for ete3."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytest

from flexihgt.taxonomy import TaxonomyProvider, build_index

# A miniature NCBI tree: (taxid, parent, rank, name).
# Drosophilidae is the in-group throughout; Bacillaceae and Culicidae are
# out-groups at family level, and 32630 is the synthetic-construct clade.
NODES: List[Tuple[int, int, str, str]] = [
    (1, 1, 'no rank', 'root'),
    (131567, 1, 'no rank', 'cellular organisms'),
    (2, 131567, 'superkingdom', 'Bacteria'),
    (1239, 2, 'phylum', 'Bacillota'),
    (91061, 1239, 'class', 'Bacilli'),
    (1385, 91061, 'order', 'Bacillales'),
    (186817, 1385, 'family', 'Bacillaceae'),
    (1386, 186817, 'genus', 'Bacillus'),
    (1390, 1386, 'species', 'Bacillus amyloliquefaciens'),
    (1423, 1386, 'species', 'Bacillus subtilis'),
    (2759, 131567, 'superkingdom', 'Eukaryota'),
    (33208, 2759, 'kingdom', 'Metazoa'),
    (6656, 33208, 'phylum', 'Arthropoda'),
    (50557, 6656, 'class', 'Insecta'),
    (7147, 50557, 'order', 'Diptera'),
    (7214, 7147, 'family', 'Drosophilidae'),
    (7215, 7214, 'genus', 'Drosophila'),
    (7227, 7215, 'species', 'Drosophila melanogaster'),
    (7240, 7215, 'species', 'Drosophila simulans'),
    (7157, 7147, 'family', 'Culicidae'),
    (7164, 7157, 'genus', 'Anopheles'),
    (7165, 7164, 'species', 'Anopheles gambiae'),
    (28384, 1, 'no rank', 'other sequences'),
    (81077, 28384, 'no rank', 'artificial sequences'),
    (32630, 81077, 'species', 'synthetic construct'),
]

# 9999 is a retired taxid that NCBI merged into D. melanogaster (7227).
MERGED: Dict[int, int] = {9999: 7227}


class FakeTaxonomy(TaxonomyProvider):
    """In-memory :class:`TaxonomyProvider` over :data:`NODES`."""

    def __init__(self) -> None:
        self.parents = {taxid: parent for taxid, parent, _, _ in NODES}
        self.ranks = {taxid: rank for taxid, _, rank, _ in NODES}
        self.names = {taxid: name for taxid, _, _, name in NODES}
        self.updated = False

    def _lineage(self, taxid: int) -> List[int]:
        if taxid not in self.parents:
            return []
        lineage = [taxid]
        while lineage[-1] != 1:
            lineage.append(self.parents[lineage[-1]])
        return list(reversed(lineage))

    def translate_merged(self, taxids: Sequence[int]) -> Tuple[List[int], Dict[int, int]]:
        merged = {t: MERGED[t] for t in taxids if t in MERGED}
        live = [t for t in taxids if t not in MERGED]
        return live, merged

    def get_lineage_translator(self, taxids: Sequence[int]) -> Dict[int, List[int]]:
        return {t: self._lineage(t) for t in taxids if self._lineage(t)}

    def get_rank(self, taxids: Sequence[int]) -> Dict[int, str]:
        return {t: self.ranks[t] for t in taxids if t in self.ranks}

    def get_taxid_translator(self, taxids: Sequence[int]) -> Dict[int, str]:
        return {t: self.names[t] for t in taxids if t in self.names}

    def update_database(self) -> None:
        self.updated = True


@pytest.fixture()
def provider() -> FakeTaxonomy:
    return FakeTaxonomy()


@pytest.fixture()
def index(provider: FakeTaxonomy):
    """Index covering every leaf taxon used in the tests."""
    return build_index(provider, [7227, 7240, 7165, 1390, 1423, 32630, 9999])
