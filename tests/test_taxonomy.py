"""Tests for taxonomy resolution."""

from __future__ import annotations

import pytest

from flexihgt.taxonomy import TaxonomyIndex, build_index


def test_lineage_and_alignment(index):
    info = index.get(7227)
    assert info is not None
    assert info.name == 'Drosophila melanogaster'
    assert info.rank == 'species'
    assert info.alignment['family'] == 7214
    assert info.alignment['superkingdom'] == 2759


def test_merged_taxid_is_resolved_not_dropped(index):
    """Regression: retired taxids used to be silently discarded.

    ``lineages`` is only fetched for live taxids, so the old
    ``lineages.get(orig_id)`` lookup returned empty for exactly the merged
    taxids the code was trying to rescue, and ``continue`` dropped them.
    """
    info = index.get(9999)
    assert info is not None
    assert info.taxid == 7227
    assert info.original_taxid == 9999
    assert info.name == 'Drosophila melanogaster'
    assert index.share_rank(9999, 7227, 'species')


def test_ancestor_names_are_available(index):
    """Regression: donor taxonomy was empty because only leaves were indexed."""
    lineage = index.lineage_names(1390)
    assert lineage['family'] == 'Bacillaceae'
    assert lineage['genus'] == 'Bacillus'
    assert lineage['superkingdom'] == 'Bacteria'
    # Ordered from most to least specific.
    assert list(lineage)[0] == 'species'
    assert list(lineage)[-1] == 'superkingdom'


def test_species_name_resolves_for_non_species_ranks(provider):
    """A hit annotated to genus still gets a usable label."""
    idx = build_index(provider, [1386])
    assert idx.species_name(1386) == 'Bacillus'


def test_share_rank(index):
    assert index.share_rank(7227, 7240, 'family')      # both Drosophilidae
    assert not index.share_rank(7227, 7165, 'family')  # Drosophilidae vs Culicidae
    assert index.share_rank(7227, 7165, 'order')       # both Diptera
    assert not index.share_rank(7227, 1390, 'superkingdom')


def test_share_rank_with_unknown_taxid_is_false(index):
    assert not index.share_rank(7227, 12345678, 'family')
    assert not index.share_rank(None, 7227, 'family')


def test_synthetic_detected_by_lineage(index):
    assert index.is_synthetic(32630)
    assert not index.is_synthetic(7227)
    assert not index.is_synthetic(1390)


@pytest.mark.parametrize('raw,expected', [
    ('7227', 7227), (7227, 7227), (7227.0, 7227), (' 7227 ', 7227),
    ('nan', None), ('', None), (None, None), ('not-a-taxid', None),
])
def test_taxid_coercion(raw, expected):
    assert TaxonomyIndex._as_int(raw) == expected


def test_build_index_raises_when_nothing_resolves(provider):
    """A broken taxonomy DB must fail loudly, not yield "0 HGT events"."""
    with pytest.raises(RuntimeError):
        build_index(provider, [99999991, 99999992])


def test_build_index_raises_on_no_usable_taxids(provider):
    with pytest.raises(RuntimeError):
        build_index(provider, ['', None, 'nan'])


# ------------------------------------------------------------- ete3 backend
class FakeNCBITaxa:
    """Minimal ete3 stand-in exposing the sqlite handle FlexiHGT reads."""

    def __init__(self, db=None, lineage_calls=None):
        self.db = db
        self.lineage_calls = lineage_calls if lineage_calls is not None else []

    def get_lineage_translator(self, taxids):
        self.lineage_calls.append(list(taxids))
        return {t: [1, t] for t in taxids}

    def get_rank(self, taxids):
        return {t: 'species' for t in taxids}

    def get_taxid_translator(self, taxids):
        return {t: f'taxon {t}' for t in taxids}


def _ete3_like_db():
    """In-memory database with ete3's `species` table layout."""
    import sqlite3

    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE species (taxid INT PRIMARY KEY, parent INT, '
               'spname VARCHAR(50), common VARCHAR(50), rank VARCHAR(50), track TEXT)')
    # ete3 stores `track` leaf-first, ending at the root.
    db.executemany(
        'INSERT INTO species VALUES (?,?,?,?,?,?)',
        [
            (7227, 7215, 'Drosophila melanogaster', '', 'species', '7227,7215,7214,1'),
            (1390, 1386, 'Bacillus amyloliquefaciens', '', 'species', '1390,1386,186817,1'),
        ],
    )
    return db


def test_bulk_lineage_query_returns_root_first_lineages():
    from flexihgt.taxonomy import Ete3Taxonomy

    provider = Ete3Taxonomy(FakeNCBITaxa(db=_ete3_like_db()))
    lineages = provider.get_lineage_translator([7227, 1390])
    assert lineages[7227] == [1, 7214, 7215, 7227]
    assert lineages[1390] == [1, 186817, 1386, 1390]


def test_bulk_lineage_query_avoids_the_per_taxid_api():
    """The whole point is to stop calling ete3 once per taxid."""
    from flexihgt.taxonomy import Ete3Taxonomy

    calls = []
    provider = Ete3Taxonomy(FakeNCBITaxa(db=_ete3_like_db(), lineage_calls=calls))
    provider.get_lineage_translator([7227, 1390])
    assert calls == []


def test_bulk_lineage_falls_back_when_schema_is_unexpected():
    import sqlite3

    from flexihgt.taxonomy import Ete3Taxonomy

    db = sqlite3.connect(':memory:')          # no `species` table at all
    calls = []
    provider = Ete3Taxonomy(FakeNCBITaxa(db=db, lineage_calls=calls))
    assert provider.get_lineage_translator([7227]) == {7227: [1, 7227]}
    assert calls == [[7227]]                  # ete3's own API was used instead


def test_ete3_provider_construction_does_not_import_ete3():
    """The CLI builds a provider before knowing whether the run needs one."""
    from flexihgt.taxonomy import Ete3Taxonomy, get_provider

    assert isinstance(get_provider('ete3'), Ete3Taxonomy)


def test_get_provider_rejects_unknown_backend():
    from flexihgt.taxonomy import get_provider

    with pytest.raises(ValueError):
        get_provider('blast')


# ----------------------------------------------------------- taxopy backend
class FakeTaxDb:
    """Stub exposing the three dictionaries TaxopyTaxonomy reads."""

    taxid2parent = {1: 1, 7214: 1, 7215: 7214, 7227: 7215, 1386: 1}
    taxid2rank = {1: 'no rank', 7214: 'family', 7215: 'genus', 7227: 'species'}
    taxid2name = {1: 'root', 7214: 'Drosophilidae', 7215: 'Drosophila',
                  7227: 'Drosophila melanogaster'}
    oldtaxid2newtaxid = {9999: 7227}


def test_taxopy_backend_resolves_lineages():
    from flexihgt.taxonomy import TaxopyTaxonomy

    provider = TaxopyTaxonomy(taxdb=FakeTaxDb())
    assert provider.get_lineage_translator([7227])[7227] == [1, 7214, 7215, 7227]
    assert provider.get_rank([7227]) == {7227: 'species'}
    assert provider.get_taxid_translator([7214]) == {7214: 'Drosophilidae'}


def test_taxopy_backend_handles_merged_taxids():
    from flexihgt.taxonomy import TaxopyTaxonomy

    live, merged = TaxopyTaxonomy(taxdb=FakeTaxDb()).translate_merged([7227, 9999])
    assert live == [7227]
    assert merged == {9999: 7227}


def test_taxopy_backend_builds_a_usable_index():
    from flexihgt.taxonomy import TaxopyTaxonomy

    idx = build_index(TaxopyTaxonomy(taxdb=FakeTaxDb()), [7227, 9999])
    assert idx.lineage_names(7227)['family'] == 'Drosophilidae'
    assert idx.get(9999).taxid == 7227      # merged taxid survives
