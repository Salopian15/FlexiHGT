"""End-to-end test of run_analysis with the search step stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

from flexihgt import core
from flexihgt.core import HGTDetect, HGTParameters

# g1: strong bacterial signal against a weak fly hit -> candidate.
# g2: best hit is a fly -> not a candidate.
# g3: only synthetic-construct hits -> filtered out.
# g4: bacteria only, no in-group hits at all -> candidate (regression case).
HITS = """\
g1\tbact_a\t1e-120\t420\t210\t62.0\t1390
g1\tbact_b\t1e-110\t410\t205\t60.0\t1423
g1\tfly_a\t1e-3\t95\t180\t28.0\t7240
g2\tfly_b\t1e-150\t500\t260\t88.0\t7240
g2\tbact_c\t1e-4\t100\t120\t30.0\t1390
g3\tsynth\t1e-99\t400\t200\t99.0\t32630
g4\tbact_d\t1e-130\t440\t215\t65.0\t1390
g4\tbact_e\t1e-125\t430\t212\t64.0\t1423
"""

FASTA = '>g1\nMKVLW\n>g2\nMKVLA\n>g3\nMKVLC\n>g4\nMKVLD\n>g5\nMKVLE\n'


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    fasta = tmp_path / 'proteome.faa'
    fasta.write_text(FASTA, encoding='utf-8')
    hits = tmp_path / 'proteome.hits.tsv'
    hits.write_text(HITS, encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'stub')

    # Stub the search: the tools are not installed in CI.
    monkeypatch.setattr(core, 'run_search', lambda *a, **k: hits)
    return fasta, db, tmp_path


def test_full_pipeline(workspace, provider):
    fasta, db, tmp_path = workspace
    detector = HGTDetect(
        HGTParameters(query_taxid=7227, tax_level='family', out_pct=0.5),
        taxonomy=provider,
    )
    out = tmp_path / 'results.tsv'
    result = detector.run_analysis(fasta, db, output_file=out, taxonomy_dump=tmp_path / 'tax.tsv')

    assert sorted(result.candidates['gene']) == ['g1', 'g4']
    assert result.genes_total == 5      # g5 is in the FASTA but has no hits
    assert result.genes_with_hits == 4
    assert result.gene_errors == []

    rows = out.read_text(encoding='utf-8').splitlines()
    assert len(rows) == 3               # header + two candidates
    header = rows[0].split('\t')
    g1 = dict(zip(header, rows[1].split('\t')))
    assert g1['Gene/Protein'] == 'g1'
    assert 'family: Bacillaceae' in g1['Donor taxonomy']
    assert g1['Donor name'] == 'Bacillus amyloliquefaciens'
    assert g1['No recipient hits'] == 'no'

    g4 = dict(zip(header, rows[2].split('\t')))
    assert g4['No recipient hits'] == 'yes'
    assert g4['Recipient species'] == '0'

    top = result.top_hits_path.read_text(encoding='utf-8')
    assert 'Bacillus amyloliquefaciens' in top
    assert 'Drosophila simulans' in top
    assert (tmp_path / 'tax.tsv').exists()


def test_pipeline_rejects_unresolvable_query_taxid(workspace, provider):
    fasta, db, tmp_path = workspace
    detector = HGTDetect(HGTParameters(query_taxid=99999999), taxonomy=provider)
    with pytest.raises(RuntimeError, match='could not be resolved'):
        detector.run_analysis(fasta, db, output_file=tmp_path / 'r.tsv')


def test_pipeline_rejects_missing_input(tmp_path, provider):
    detector = HGTDetect(HGTParameters(query_taxid=7227), taxonomy=provider)
    with pytest.raises(FileNotFoundError):
        detector.run_analysis(tmp_path / 'nope.faa', tmp_path / 'db.dmnd')


def test_pipeline_writes_empty_results_without_crashing(workspace, provider):
    """A run with no candidates still produces well-formed output files."""
    fasta, db, tmp_path = workspace
    detector = HGTDetect(
        HGTParameters(query_taxid=7227, ai_threshold=1e9),  # impossible threshold
        taxonomy=provider,
    )
    out = tmp_path / 'results.tsv'
    result = detector.run_analysis(fasta, db, output_file=out)
    assert len(result) == 0
    assert out.read_text(encoding='utf-8').strip() == '\t'.join(detector.RESULT_COLUMNS)


def test_default_output_name_includes_tax_level(workspace, provider, monkeypatch):
    fasta, db, tmp_path = workspace
    monkeypatch.chdir(tmp_path)
    detector = HGTDetect(
        HGTParameters(query_taxid=7227, tax_level='order', out_pct=0.5),
        taxonomy=provider,
    )
    result = detector.run_analysis(fasta, db)
    assert result.results_path == Path('proteome_order_HGT.tsv')
    assert result.results_path.exists()
    assert result.top_hits_path.name == 'proteome_order_HGT_top_hits.tsv'


# ------------------------------------------------------------------- caching
def _detector(provider):
    return HGTDetect(
        HGTParameters(query_taxid=7227, tax_level='family', out_pct=0.5),
        taxonomy=provider,
    )


def test_annotation_cache_is_reused(workspace, provider, monkeypatch):
    """A second run must not re-parse or re-annotate unchanged search results."""
    # The cache is a parquet file, so without a parquet engine there is nothing
    # to reuse and _save_annotation_cache logs and moves on by design.
    pytest.importorskip('pyarrow')
    fasta, db, tmp_path = workspace
    detector = _detector(provider)
    out = tmp_path / 'r.tsv'

    first = detector.run_analysis(fasta, db, output_file=out)
    cache, meta = HGTDetect.annotation_cache_paths(tmp_path / 'proteome.hits.tsv')
    assert cache.exists() and meta.exists()

    calls = []
    original = HGTDetect.annotate_hits
    monkeypatch.setattr(
        HGTDetect, 'annotate_hits',
        lambda self, h, t: (calls.append(1), original(self, h, t))[1],
    )
    second = _detector(provider).run_analysis(fasta, db, output_file=out)
    assert calls == []                                   # cache hit
    assert list(second.candidates['gene']) == list(first.candidates['gene'])


def test_annotation_cache_is_invalidated_by_tax_level(workspace, provider):
    fasta, db, tmp_path = workspace
    out = tmp_path / 'r.tsv'
    _detector(provider).run_analysis(fasta, db, output_file=out)
    # A different in-group boundary changes the annotation, so it must rebuild.
    at_order = HGTDetect(
        HGTParameters(query_taxid=7227, tax_level='order', out_pct=0.5),
        taxonomy=provider,
    )
    assert at_order._load_annotation_cache(tmp_path / 'proteome.hits.tsv') is None


def test_no_cache_disables_it(workspace, provider):
    fasta, db, tmp_path = workspace
    _detector(provider).run_analysis(fasta, db, output_file=tmp_path / 'r.tsv', use_cache=False)
    cache, meta = HGTDetect.annotation_cache_paths(tmp_path / 'proteome.hits.tsv')
    assert not cache.exists() and not meta.exists()


def test_rescore_only_skips_the_search(workspace, provider, monkeypatch):
    fasta, _db, tmp_path = workspace

    def explode(*_a, **_k):
        raise AssertionError('the search must not run under rescore_only')

    monkeypatch.setattr(core, 'run_search', explode)
    result = _detector(provider).run_analysis(
        fasta, None, output_file=tmp_path / 'r.tsv', rescore_only=True
    )
    assert sorted(result.candidates['gene']) == ['g1', 'g4']


def test_rescore_only_requires_existing_hits(tmp_path, provider):
    fasta = tmp_path / 'other.faa'
    fasta.write_text(FASTA, encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='rescore-only'):
        _detector(provider).run_analysis(fasta, None, rescore_only=True)


def test_database_required_without_rescore_only(workspace, provider):
    fasta, _db, tmp_path = workspace
    with pytest.raises(ValueError, match='database is required'):
        _detector(provider).run_analysis(fasta, None, output_file=tmp_path / 'r.tsv')


# ------------------------------------------------------------ taxon exclusion
def test_search_excludes_at_species_rank(tmp_path, provider, monkeypatch):
    """A strain-level query taxid must exclude the whole species clade."""
    fasta = tmp_path / 'p.faa'
    fasta.write_text(FASTA, encoding='utf-8')
    hits = tmp_path / 'p.hits.tsv'
    hits.write_text(HITS, encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'stub')

    recorded = {}

    def fake_search(*_a, **kwargs):
        recorded.update(kwargs)
        return hits

    monkeypatch.setattr(core, 'run_search', fake_search)
    # 7240 (D. simulans) is already a species, so it resolves to itself.
    detector = HGTDetect(HGTParameters(query_taxid=7240, out_pct=0.5), taxonomy=provider)
    detector.run_analysis(fasta, db, output_file=tmp_path / 'r.tsv', use_cache=False)
    assert recorded['exclude_taxid'] == 7240
    assert recorded['evalue'] == detector.params.evalue
