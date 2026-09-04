"""Tests for the out-of-core (DuckDB) scoring engine."""

from __future__ import annotations

import pytest

from flexihgt import aggregate, core
from flexihgt.core import AUTO_ENGINE_BYTES, HGTDetect, HGTParameters, select_engine
from flexihgt.taxonomy import build_index

duckdb_required = pytest.mark.skipif(
    not aggregate.available(), reason='DuckDB is not installed'
)

# Deliberately varied: mixed sides, outgroup-only, ingroup-only, repeated
# species, a synthetic hit and a self-hit.
HITS = """\
g1\tb1\t1e-120\t420\t210\t62.0\t1390
g1\tb2\t1e-110\t410\t205\t60.0\t1423
g1\tb2b\t1e-105\t400\t205\t59.0\t1390
g1\tf1\t1e-3\t95\t180\t28.0\t7240
g1\tsyn\t1e-99\t400\t200\t99.0\t32630
g1\tself\t0.0\t900\t200\t99.0\t7227
g2\tf2\t1e-150\t500\t260\t88.0\t7240
g2\tb3\t1e-4\t100\t120\t30.0\t1390
g3\tb4\t1e-130\t440\t215\t65.0\t1390
g3\tb5\t1e-125\t430\t212\t64.0\t1423
g4\tf3\t1e-90\t380\t200\t80.0\t7240
g5\tb6\t1e-60\t250\t150\t45.0\t1390
g5\tf4\t1e-55\t240\t150\t44.0\t7240
"""

TAXIDS = ['1390', '1423', '7240', '7227', '32630']


@pytest.fixture()
def hits_file(tmp_path):
    path = tmp_path / 'hits.tsv'
    path.write_text(HITS, encoding='utf-8')
    return path


@pytest.fixture()
def index(provider):
    return build_index(provider, TAXIDS)


def make_table(hits_file, index, rank='family', boundary_taxid=7227):
    return aggregate.HitTable(
        hits_file,
        aggregate.taxonomy_table(index, TAXIDS, rank),
        query_boundary=index.rank_taxid(boundary_taxid, rank),
        query_species=index.rank_taxid(boundary_taxid, 'species'),
    )


# ------------------------------------------------------------ engine choice
def test_small_tables_use_pandas(hits_file):
    assert select_engine('auto', hits_file) == 'pandas'


def test_engine_can_be_forced(hits_file):
    assert select_engine('pandas', hits_file) == 'pandas'


def test_unknown_engine_rejected(hits_file):
    with pytest.raises(ValueError, match='Unknown engine'):
        select_engine('spark', hits_file)


@duckdb_required
def test_duckdb_can_be_forced(hits_file):
    assert select_engine('duckdb', hits_file) == 'duckdb'


def test_auto_falls_back_when_duckdb_is_missing(hits_file, monkeypatch, caplog):
    """A huge table without DuckDB must warn, not crash."""
    monkeypatch.setattr(aggregate, 'available', lambda: False)
    monkeypatch.setattr(core, 'AUTO_ENGINE_BYTES', 0)
    with caplog.at_level('WARNING'):
        assert select_engine('auto', hits_file) == 'pandas'
    assert 'DuckDB is not installed' in caplog.text


def test_forcing_duckdb_without_it_installed_is_an_error(hits_file, monkeypatch):
    monkeypatch.setattr(aggregate, 'available', lambda: False)
    with pytest.raises(RuntimeError, match='needs DuckDB'):
        select_engine('duckdb', hits_file)


# ---------------------------------------------------------------- filtering
@duckdb_required
def test_synthetic_and_self_hits_are_excluded(hits_file, index):
    with make_table(hits_file, index) as table:
        top = table.top_hits(['g1'], n=10)['g1']
    subjects = {h['subject_id'] for side in top.values() for h in side}
    assert 'syn' not in subjects            # synthetic construct
    assert 'self' not in subjects           # the query organism's own species
    assert {'b1', 'b2', 'b2b', 'f1'} <= subjects


@duckdb_required
def test_genes_without_outgroup_hits_are_not_scored(hits_file, index):
    with make_table(hits_file, index) as table:
        assert 'g4' not in table.aggregate().index   # in-group hits only
        assert table.gene_count() == 5              # but it is still counted


@duckdb_required
def test_distinct_species_counted_once(hits_file, index):
    with make_table(hits_file, index) as table:
        row = table.aggregate().loc['g1']
    # b1 and b2b are both taxid 1390, so the out-group has two species not three.
    assert row['out_species'] == 2
    assert row['recipient_species'] == 1


@duckdb_required
def test_last_taxid_of_a_multi_taxid_field(tmp_path, index):
    path = tmp_path / 'h.tsv'
    path.write_text('g1\tb1\t1e-90\t300\t100\t50.0\t7240;1390\n', encoding='utf-8')
    with make_table(path, index) as table:
        assert table.aggregate().loc['g1']['max_outgroup_bitscore'] == 300


@duckdb_required
def test_best_outgroup_hit(hits_file, index):
    with make_table(hits_file, index) as table:
        assert table.best_outgroup_hits(['g1'])['g1'] == ('1390', 'b1')


# --------------------------------------------------------------- equivalence
@duckdb_required
@pytest.mark.parametrize('rank', ['family', 'order', 'superkingdom'])
@pytest.mark.parametrize('out_pct', [0.0, 0.5])
def test_duckdb_and_pandas_engines_agree(tmp_path, hits_file, provider, rank, out_pct):
    """The two engines must produce byte-identical results.

    This is the guard on the whole out-of-core path: the SQL aggregation and
    the pandas aggregation are independent implementations of the same six
    numbers, and only this test keeps them honest.
    """
    params = HGTParameters(
        query_taxid=7227, tax_level=rank, out_pct=out_pct, ai_threshold=10,
    )
    fasta = tmp_path / 'q.faa'
    fasta.write_text(''.join(f'>g{i}\nMKV\n' for i in range(1, 6)), encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'stub')

    original = core.run_search
    core.run_search = lambda *a, **k: hits_file
    try:
        outputs = {}
        for engine in ('pandas', 'duckdb'):
            detector = HGTDetect(params, taxonomy=provider)
            out = tmp_path / f'{engine}.tsv'
            detector.run_analysis(fasta, db, output_file=out, engine=engine, use_cache=False)
            outputs[engine] = (
                out.read_text(encoding='utf-8'),
                out.with_name(f'{engine}_top_hits.tsv').read_text(encoding='utf-8'),
            )
    finally:
        core.run_search = original

    assert outputs['pandas'][0] == outputs['duckdb'][0], f'results differ at {rank}/{out_pct}'
    assert outputs['pandas'][1] == outputs['duckdb'][1], f'top hits differ at {rank}/{out_pct}'
    assert 'g1' in outputs['pandas'][0]


@duckdb_required
def test_scan_taxids(hits_file):
    assert sorted(aggregate.scan_taxids(hits_file)) == sorted(set(TAXIDS))


@duckdb_required
def test_taxonomy_table_shape(index):
    table = aggregate.taxonomy_table(index, TAXIDS, 'family')
    assert list(table.columns) == ['taxid', 'species', 'boundary', 'synthetic']
    assert bool(table.loc[table['taxid'] == 32630, 'synthetic'].iloc[0]) is True
    assert int(table.loc[table['taxid'] == 1390, 'boundary'].iloc[0]) == 186817


@duckdb_required
def test_hit_table_requires_a_boundary(hits_file, index):
    with pytest.raises(ValueError, match='per-genome query table or a single boundary'):
        aggregate.HitTable(hits_file, aggregate.taxonomy_table(index, TAXIDS, 'family'))


def test_auto_threshold_is_documented_in_gb():
    assert AUTO_ENGINE_BYTES == 2 * 1024 ** 3


@duckdb_required
def test_is_recipient_is_never_null(hits_file, index):
    """A genome with no ancestor at the in-group rank must not erase the table.

    ``t.boundary = NULL`` is NULL in SQL, and a NULL satisfies neither
    ``FILTER (WHERE is_recipient)`` nor ``FILTER (WHERE NOT is_recipient)``, so
    every one of that genome's rows dropped out of both sides of the aggregate
    and it silently reported no candidates. Reachable per genome, where the
    boundary is a column rather than a constant.
    """
    queries = aggregate.query_table([(gene, None, None) for gene in
                                     ('g1', 'g2', 'g3', 'g4', 'g5')])
    with aggregate.HitTable(
        hits_file,
        aggregate.taxonomy_table(index, TAXIDS, 'family'),
        queries=queries,
    ) as table:
        nulls = table._con.execute(
            'SELECT count(*) FROM hits WHERE is_recipient IS NULL'
        ).fetchone()[0]
        assert nulls == 0
        frame = table.aggregate()
    # Nothing is in-group, so every gene keeps an out-group aggregate.
    assert not frame.empty
    assert frame['max_recipient_bitscore'].isna().all()


@duckdb_required
def test_quotes_in_the_path_do_not_break_the_query(tmp_path, index):
    directory = tmp_path / "o'brien"
    directory.mkdir()
    path = directory / 'hits.tsv'
    path.write_text(HITS, encoding='utf-8')
    with make_table(path, index) as table:
        assert table.gene_count() > 0
