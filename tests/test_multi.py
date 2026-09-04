"""Tests for multi-genome runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from flexihgt import core, multi, multi_cli
from flexihgt.core import HGTDetect, HGTParameters
from flexihgt.manifest import GenomeRecord, read_manifest
from flexihgt.multi import MultiGenomeDetect


def _duckdb_available():
    from flexihgt import aggregate

    return aggregate.available()

# fly (7227, Drosophilidae) and bug (7165, Culicidae) are both Diptera.
# The cross hits are the point: moz_x is a Culicidae hit on a *fly* gene and
# fly_x is a Drosophilidae hit on a *bug* gene, so the same subject taxon lands
# on opposite sides of the split depending on which genome asked.
HITS = """\
fly|g1\tbact_a\t1e-120\t420\t210\t62.0\t1390
fly|g1\tbact_b\t1e-110\t410\t205\t60.0\t1423
fly|g1\tfly_a\t1e-3\t95\t180\t28.0\t7240
fly|g1\tmoz_x\t1e-5\t110\t175\t30.0\t7165
fly|g2\tfly_b\t1e-150\t500\t260\t88.0\t7240
fly|g2\tbact_c\t1e-4\t100\t120\t30.0\t1390
bug|g1\tbact_d\t1e-130\t440\t215\t65.0\t1390
bug|g1\tbact_e\t1e-125\t430\t212\t64.0\t1423
bug|g1\tfly_x\t1e-2\t92\t170\t26.0\t7240
bug|g1\tmoz_a\t1e-40\t250\t190\t70.0\t7165
"""


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Two genomes, a manifest, and a stubbed search returning HITS."""
    (tmp_path / 'fly.faa').write_text('>g1\nMKVLW\n>g2\nMKVLA\n>g3\nMKVLQ\n', encoding='utf-8')
    (tmp_path / 'bug.faa').write_text('>g1\nMKVLC\n', encoding='utf-8')
    manifest = tmp_path / 'genomes.tsv'
    manifest.write_text(
        'genome_id\ttaxid\tfasta\nfly\t7227\tfly.faa\nbug\t7165\tbug.faa\n',
        encoding='utf-8',
    )

    hits = tmp_path / 'work' / 'combined_query.hits.tsv'
    hits.parent.mkdir(parents=True, exist_ok=True)
    hits.write_text(HITS, encoding='utf-8')
    monkeypatch.setattr(multi, 'run_search', lambda *a, **k: hits)

    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'stub')
    return manifest, db, tmp_path


def detector(provider, **overrides):
    params = dict(tax_level='family', out_pct=0.5)
    params.update(overrides)
    return MultiGenomeDetect(HGTParameters(**params), taxonomy=provider)


# ------------------------------------------------------------ construction
def test_query_taxid_must_be_unset(provider):
    with pytest.raises(ValueError, match='query_taxid must be unset'):
        MultiGenomeDetect(HGTParameters(query_taxid=7227), taxonomy=provider)


# -------------------------------------------------------------- annotation
def test_genome_column_derived_from_query_ids(provider, workspace):
    _, _, tmp_path = workspace
    hits = HGTDetect(HGTParameters()).load_hits(tmp_path / 'work' / 'combined_query.hits.tsv')
    annotated = MultiGenomeDetect.add_genome_column(hits)
    assert list(annotated['_genome']) == ['fly'] * 6 + ['bug'] * 4
    assert str(annotated['_genome'].dtype) == 'category'


def test_in_group_boundary_is_per_genome(provider, workspace, index):
    """The same hit taxon is in-group for one genome and out-group for another."""
    _, _, tmp_path = workspace
    md = detector(provider)
    hits = md.detector.load_hits(tmp_path / 'work' / 'combined_query.hits.tsv')
    records = [
        GenomeRecord('fly', 7227, tmp_path / 'fly.faa'),
        GenomeRecord('bug', 7165, tmp_path / 'bug.faa'),
    ]
    annotated = md.annotate_hits(hits, index, records)
    by_subject = {s: bool(v) for s, v in zip(annotated['sseqid'], annotated['_is_recipient'])}

    # Drosophilidae: in-group on a fly gene, out-group on a bug gene.
    assert by_subject['fly_a'] is True       # 7240 on fly|g1
    assert by_subject['fly_x'] is False      # 7240 on bug|g1
    # Culicidae: the mirror image.
    assert by_subject['moz_x'] is False      # 7165 on fly|g1
    # Bacteria are out-group for both.
    assert by_subject['bact_a'] is False
    assert by_subject['bact_d'] is False


def test_tax_level_applies_to_every_genome(provider, workspace, index):
    _, _, tmp_path = workspace
    md = detector(provider, tax_level='order')      # Diptera covers both genomes
    hits = md.detector.load_hits(tmp_path / 'work' / 'combined_query.hits.tsv')
    records = [
        GenomeRecord('fly', 7227, tmp_path / 'fly.faa'),
        GenomeRecord('bug', 7165, tmp_path / 'bug.faa'),
    ]
    annotated = md.annotate_hits(hits, index, records)
    by_subject = {s: bool(v) for s, v in zip(annotated['sseqid'], annotated['_is_recipient'])}
    # Diptera covers both families, so the cross hits flip to in-group.
    assert by_subject['moz_x'] is True
    assert by_subject['fly_x'] is True
    assert by_subject['fly_a'] is True
    assert by_subject['bact_a'] is False


def test_self_hits_are_dropped_per_genome(provider, tmp_path, index):
    """Each genome's own species is removed, not one global taxon."""
    (tmp_path / 'fly.faa').write_text('>g1\nMKV\n', encoding='utf-8')
    (tmp_path / 'bug.faa').write_text('>g1\nMKV\n', encoding='utf-8')
    hits_path = tmp_path / 'h.tsv'
    hits_path.write_text(
        'fly|g1\tself_fly\t0.0\t900\t100\t99.0\t7227\n'   # fly's own species
        'fly|g1\tmoz\t1e-50\t300\t100\t50.0\t7165\n'      # kept for the fly
        'bug|g1\tself_moz\t0.0\t900\t100\t99.0\t7165\n'   # mosquito's own species
        'bug|g1\tfly\t1e-50\t300\t100\t50.0\t7227\n',     # kept for the mosquito
        encoding='utf-8',
    )
    md = detector(provider)
    hits = md.detector.load_hits(hits_path)
    records = [
        GenomeRecord('fly', 7227, tmp_path / 'fly.faa'),
        GenomeRecord('bug', 7165, tmp_path / 'bug.faa'),
    ]
    annotated = md.annotate_hits(hits, index, records)
    assert sorted(annotated['sseqid']) == ['fly', 'moz']


# ------------------------------------------------------------------- runs
def test_full_multi_genome_run(provider, workspace):
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest)
    result = detector(provider).run(records, db, output_dir=tmp_path / 'out')

    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['fly'].genes_total == 3          # g3 has no hits
    assert outcomes['fly'].genes_with_hits == 2
    assert outcomes['fly'].candidates == 1           # g1 only; g2's best hit is a fly
    assert outcomes['bug'].candidates == 1
    assert result.total_candidates == 2
    assert result.total_errors == 0

    combined = (tmp_path / 'out' / 'combined_family_HGT.tsv').read_text(encoding='utf-8')
    rows = combined.splitlines()
    assert rows[0].startswith('Genome\tGene/Protein')
    assert len(rows) == 3                            # header + one row per genome
    assert {line.split('\t')[0] for line in rows[1:]} == {'fly', 'bug'}
    assert 'Bacillaceae' in combined


def test_per_genome_files_are_written(provider, workspace):
    manifest, db, tmp_path = workspace
    result = detector(provider).run(read_manifest(manifest), db, output_dir=tmp_path / 'out')
    genome_dir = tmp_path / 'out' / 'genomes'
    assert (genome_dir / 'fly_family_HGT.tsv').exists()
    assert (genome_dir / 'fly_family_HGT_top_hits.tsv').exists()
    assert (genome_dir / 'bug_family_HGT.tsv').exists()
    assert result.per_genome_dir == genome_dir


def test_no_per_genome_leaves_only_the_combined_table(provider, workspace):
    manifest, db, tmp_path = workspace
    result = detector(provider).run(
        read_manifest(manifest), db, output_dir=tmp_path / 'out', per_genome=False
    )
    assert not (tmp_path / 'out' / 'genomes').exists()
    assert result.per_genome_dir is None
    assert result.combined_path.exists()
    # No scratch directories left behind.
    assert [p.name for p in sorted((tmp_path / 'out').iterdir()) if p.is_dir()] == ['work']


def test_summary_reports_every_genome(provider, workspace):
    manifest, db, tmp_path = workspace
    result = detector(provider).run(read_manifest(manifest), db, output_dir=tmp_path / 'out')
    rows = result.summary_path.read_text(encoding='utf-8').splitlines()
    assert rows[0].split('\t') == MultiGenomeDetect.SUMMARY_COLUMNS
    summary = {line.split('\t')[0]: line.split('\t') for line in rows[1:]}
    assert summary['fly'][1] == '7227'
    assert summary['fly'][2] == 'ok'
    assert summary['bug'][2] == 'ok'


def test_genome_with_no_hits_is_reported_not_dropped(provider, workspace):
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest) + [
        GenomeRecord('lonely', 1390, tmp_path / 'bug.faa'),
    ]
    result = detector(provider).run(records, db, output_dir=tmp_path / 'out')
    lonely = {o.genome_id: o for o in result.outcomes}['lonely']
    assert lonely.status == 'ok'
    assert lonely.genes_with_hits == 0
    assert lonely.candidates == 0


def test_unresolvable_taxid_skips_one_genome_only(provider, workspace):
    """One bad taxid in a manifest of thousands must not abort the run."""
    manifest, db, tmp_path = workspace
    (tmp_path / 'ghost.faa').write_text('>g1\nMKV\n', encoding='utf-8')
    records = read_manifest(manifest) + [
        GenomeRecord('ghost', 99999999, tmp_path / 'ghost.faa'),
    ]
    result = detector(provider).run(records, db, output_dir=tmp_path / 'out')
    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['ghost'].status == 'skipped'
    assert 'could not be resolved' in outcomes['ghost'].note
    assert outcomes['fly'].status == 'ok'
    assert result.total_candidates == 2


def test_all_taxids_unresolvable_is_fatal(provider, workspace, tmp_path):
    manifest, db, _ = workspace
    (tmp_path / 'ghost.faa').write_text('>g1\nMKV\n', encoding='utf-8')
    records = [GenomeRecord('ghost', 99999999, tmp_path / 'ghost.faa')]
    with pytest.raises(RuntimeError, match='None of the manifest taxids'):
        detector(provider).run(records, db, output_dir=tmp_path / 'out')


def test_rescore_only_skips_the_search(provider, workspace, monkeypatch):
    manifest, _db, tmp_path = workspace

    def explode(*_a, **_k):
        raise AssertionError('the search must not run under rescore_only')

    monkeypatch.setattr(multi, 'run_search', explode)
    result = detector(provider).run(
        read_manifest(manifest), None,
        output_dir=tmp_path / 'out', work_dir=tmp_path / 'work', rescore_only=True,
    )
    assert result.total_candidates == 2


def test_rescore_only_requires_existing_hits(provider, workspace, tmp_path):
    manifest, _db, _ = workspace
    with pytest.raises(FileNotFoundError, match='rescore-only'):
        detector(provider).run(
            read_manifest(manifest), None,
            output_dir=tmp_path / 'out', work_dir=tmp_path / 'empty', rescore_only=True,
        )


def test_database_required_without_rescore_only(provider, workspace, tmp_path):
    manifest, _db, _ = workspace
    with pytest.raises(ValueError, match='database is required'):
        detector(provider).run(read_manifest(manifest), None, output_dir=tmp_path / 'out')


def test_search_is_issued_once_without_taxon_exclude(provider, workspace, monkeypatch):
    """One search for all genomes; the per-genome taxon cannot be excluded there."""
    manifest, db, tmp_path = workspace
    hits = tmp_path / 'work' / 'combined_query.hits.tsv'
    calls = []

    def record_call(*args, **kwargs):
        calls.append(kwargs)
        return hits

    monkeypatch.setattr(multi, 'run_search', record_call)
    detector(provider).run(read_manifest(manifest), db, output_dir=tmp_path / 'out')
    assert len(calls) == 1
    assert calls[0]['exclude_taxid'] is None


def test_scores_match_an_equivalent_single_genome_run(provider, workspace):
    """A genome scored in a multi run must match scoring it on its own."""
    manifest, db, tmp_path = workspace
    multi_result = detector(provider).run(
        read_manifest(manifest), db, output_dir=tmp_path / 'out'
    )
    multi_rows = [
        line for line in
        (tmp_path / 'out' / 'genomes' / 'fly_family_HGT.tsv').read_text(encoding='utf-8').splitlines()
    ]

    # The same fly hits, run through the untouched single-genome pipeline.
    single_hits = tmp_path / 'fly_only.hits.tsv'
    single_hits.write_text(
        ''.join(
            line.replace('fly|', '') + '\n'
            for line in HITS.splitlines() if line.startswith('fly|')
        ),
        encoding='utf-8',
    )
    single = HGTDetect(
        HGTParameters(query_taxid=7227, tax_level='family', out_pct=0.5),
        taxonomy=provider,
    )
    original = core.run_search
    core.run_search = lambda *a, **k: single_hits
    try:
        single.run_analysis(
            tmp_path / 'fly.faa', db, output_file=tmp_path / 'single.tsv', use_cache=False
        )
    finally:
        core.run_search = original

    single_rows = (tmp_path / 'single.tsv').read_text(encoding='utf-8').splitlines()
    # Same header, and the same candidate with the same scores (bar the gene
    # name, which is namespaced in the multi run).
    assert multi_rows[0] == single_rows[0]
    assert len(multi_rows) == len(single_rows) == 2
    assert multi_rows[1].split('\t')[1:] == single_rows[1].split('\t')[1:]
    assert multi_rows[1].split('\t')[0] == 'fly|g1'
    assert single_rows[1].split('\t')[0] == 'g1'
    assert multi_result.total_candidates == 2


def test_empty_run_writes_headers(provider, workspace):
    manifest, db, tmp_path = workspace
    result = detector(provider, ai_threshold=1e9).run(
        read_manifest(manifest), db, output_dir=tmp_path / 'out'
    )
    assert result.total_candidates == 0
    combined = result.combined_path.read_text(encoding='utf-8').strip()
    assert combined.startswith('Genome\tGene/Protein')
    assert len(combined.splitlines()) == 1


# -------------------------------------------------------------------- CLI
def test_cli_parses_a_run(workspace, monkeypatch):
    manifest, db, tmp_path = workspace
    monkeypatch.setattr(multi_cli, 'check_environment', lambda *a, **k: True)

    recorded = {}

    def fake_run(self, records, db_path=None, **kwargs):
        recorded['genomes'] = [r.genome_id for r in records]
        recorded.update(kwargs)
        return multi.MultiRunResult(outcomes=[], combined_path=Path('c.tsv'))

    monkeypatch.setattr(multi_cli.MultiGenomeDetect, 'run', fake_run)
    code = multi_cli.main([
        str(manifest), '-db', str(db), '-o', str(tmp_path / 'out'),
        '--genomes', 'fly', '--block_size', '6', '--no-per-genome',
    ])
    assert code == multi_cli.EXIT_OK
    assert recorded['genomes'] == ['fly']
    assert recorded['tuning'].block_size == 6
    assert recorded['per_genome'] is False


def test_cli_reports_a_bad_manifest(tmp_path):
    manifest = tmp_path / 'm.tsv'
    manifest.write_text('nonsense\n1\n', encoding='utf-8')
    assert multi_cli.main([str(manifest), '-db', 'db.dmnd']) == multi_cli.EXIT_USAGE


def test_cli_returns_partial_when_a_genome_is_skipped(workspace, monkeypatch):
    manifest, db, tmp_path = workspace
    monkeypatch.setattr(multi_cli, 'check_environment', lambda *a, **k: True)
    monkeypatch.setattr(
        multi_cli.MultiGenomeDetect, 'run',
        lambda self, *a, **k: multi.MultiRunResult(
            outcomes=[multi.GenomeOutcome('x', 1, status='skipped', note='bad taxid')],
            combined_path=Path('c.tsv'), summary_path=Path('s.tsv'),
        ),
    )
    assert multi_cli.main([str(manifest), '-db', str(db)]) == multi_cli.EXIT_PARTIAL


def test_cli_rescore_only_needs_no_database(workspace, monkeypatch):
    manifest, _db, tmp_path = workspace
    seen = {}

    def fake_check(method, input_file=None, need_search_tool=True, **kwargs):
        seen['need_search_tool'] = need_search_tool
        return True

    monkeypatch.setattr(multi_cli, 'check_environment', fake_check)
    monkeypatch.setattr(
        multi_cli.MultiGenomeDetect, 'run',
        lambda self, *a, **k: multi.MultiRunResult(outcomes=[], combined_path=Path('c.tsv')),
    )
    assert multi_cli.main([str(manifest), '--rescore-only']) == multi_cli.EXIT_OK
    assert seen['need_search_tool'] is False


# ----------------------------------------------------------------- resume
def test_resume_skips_genomes_already_written(provider, workspace):
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest)
    out = tmp_path / 'out'
    detector(provider).run(records, db, output_dir=out)

    marker = out / 'genomes' / 'fly_family_HGT.tsv'
    before = marker.stat().st_mtime_ns
    result = detector(provider).run(records, db, output_dir=out, resume=True)

    assert marker.stat().st_mtime_ns == before          # not rewritten
    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['fly'].note == 'reused existing results'
    assert outcomes['fly'].candidates == 1


def test_resume_still_scores_new_genomes(provider, workspace):
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest)
    out = tmp_path / 'out'
    detector(provider).run(records[:1], db, output_dir=out)       # fly only
    result = detector(provider).run(records, db, output_dir=out, resume=True)
    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['fly'].note == 'reused existing results'
    assert outcomes['bug'].note == ''
    assert result.total_candidates == 2


def test_gene_ids_come_from_the_hit_table(provider, workspace, monkeypatch):
    """Proteomes are read once for the combined FASTA, not again per genome."""
    from flexihgt import multi as multi_module

    manifest, db, tmp_path = workspace
    calls = []
    original = multi_module.iter_protein_ids
    monkeypatch.setattr(
        multi_module, 'iter_protein_ids',
        lambda record: (calls.append(record.genome_id), original(record))[1],
    )
    detector(provider).run(read_manifest(manifest), db, output_dir=tmp_path / 'out')
    assert calls == []


# ------------------------------------------------------------ duckdb engine
@pytest.mark.skipif(not _duckdb_available(), reason='DuckDB is not installed')
def test_engines_agree_for_a_multi_genome_run(provider, workspace):
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest)
    outputs = {}
    for engine in ('pandas', 'duckdb'):
        result = detector(provider).run(
            records, db, output_dir=tmp_path / engine, engine=engine
        )
        outputs[engine] = result.combined_path.read_text(encoding='utf-8')
    assert outputs['pandas'] == outputs['duckdb']
    assert 'fly' in outputs['pandas'] and 'bug' in outputs['pandas']


@pytest.mark.skipif(not _duckdb_available(), reason='DuckDB is not installed')
def test_duckdb_summary_counts_match(provider, workspace):
    manifest, db, tmp_path = workspace
    result = detector(provider).run(
        read_manifest(manifest), db, output_dir=tmp_path / 'out', engine='duckdb'
    )
    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['fly'].genes_total == 3
    assert outcomes['fly'].genes_with_hits == 2
    assert outcomes['fly'].candidates == 1
    assert outcomes['bug'].candidates == 1


@pytest.mark.skipif(not _duckdb_available(), reason='DuckDB is not installed')
def test_resume_is_honoured_by_the_duckdb_engine(provider, workspace):
    """Regression: --resume was accepted and then ignored out of core.

    The flag exists for interrupted runs over thousands of genomes -- exactly
    the runs large enough to select this engine -- and results the user had
    already accepted were silently rewritten.
    """
    manifest, db, tmp_path = workspace
    records = read_manifest(manifest)
    out = tmp_path / 'out'
    detector(provider).run(records, db, output_dir=out, engine='duckdb')

    marker = out / 'genomes' / 'fly_family_HGT.tsv'
    before = marker.stat().st_mtime_ns
    result = detector(provider).run(
        records, db, output_dir=out, engine='duckdb', resume=True
    )

    assert marker.stat().st_mtime_ns == before          # not rewritten
    outcomes = {o.genome_id: o for o in result.outcomes}
    assert outcomes['fly'].note == 'reused existing results'
    assert outcomes['fly'].candidates == 1


def test_genome_without_the_requested_rank_is_skipped(provider, index):
    """No ancestor at tax_level means nothing can be in-group for that genome.

    Every one of its genes would then score as if it had no recipient hits and
    be reported as a candidate. Ranks are patchy in NCBI taxonomy -- the
    bacterium here has no kingdom -- so one such genome must be skipped and
    reported, not abort the manifest or quietly produce nonsense.
    """
    records = [
        GenomeRecord('fly', 7227, Path('fly.faa')),
        GenomeRecord('bacterium', 1390, Path('b.faa')),
    ]
    usable, skipped = detector(provider, tax_level='kingdom')._partition_records(
        records, index
    )
    assert [record.genome_id for record in usable] == ['fly']
    assert skipped['bacterium'].status == 'skipped'
    assert 'kingdom' in skipped['bacterium'].note


def test_run_fails_when_no_genome_has_the_requested_rank(provider, workspace):
    manifest, db, tmp_path = workspace
    with pytest.raises(RuntimeError, match='None of the manifest taxids'):
        detector(provider, tax_level='tribe').run(
            read_manifest(manifest), db, output_dir=tmp_path / 'out'
        )
