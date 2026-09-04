"""Tests for search invocation, result reuse and database resolution."""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest

from flexihgt import search
from flexihgt.search import SearchError, SearchTuning, resolve_database, run_search


@pytest.fixture()
def query(tmp_path) -> Path:
    path = tmp_path / 'query.faa'
    path.write_text('>g1\nMKV\n', encoding='utf-8')
    return path


@pytest.fixture()
def diamond_db(tmp_path) -> Path:
    path = tmp_path / 'db.dmnd'
    path.write_bytes(b'not really a database')
    return path


def fake_runner(recorder, payload='g1\ts1\t1e-90\t300\t100\t50.0\t1390\n'):
    """Replace ``_run`` with a stub that records argv and writes an output file."""
    def _run(cmd, tool):
        cmd = [str(part) for part in cmd]
        recorder.append((tool, cmd))
        # diamond: '-o <path>';  mmseqs easy-search: '<query> <db> <path> <tmp>'
        out = Path(cmd[cmd.index('-o') + 1]) if '-o' in cmd else Path(cmd[4])
        out.write_text(payload, encoding='utf-8')
    return _run


# ------------------------------------------------------------ db resolution
def test_resolve_database_accepts_bare_prefix(tmp_path, diamond_db):
    # DIAMOND accepts `db` or `db.dmnd`; the old strict exists() check did not.
    assert resolve_database(tmp_path / 'db', 'diamond') == tmp_path / 'db'
    assert resolve_database(diamond_db, 'diamond') == diamond_db


def test_resolve_database_accepts_mmseqs_prefix(tmp_path):
    (tmp_path / 'targetDB.dbtype').write_bytes(b'\x00')
    assert resolve_database(tmp_path / 'targetDB', 'mmseqs') == tmp_path / 'targetDB'


def test_resolve_database_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_database(tmp_path / 'nope', 'diamond')


# -------------------------------------------------------------- invocation
def test_diamond_command_uses_parameters(monkeypatch, tmp_path, query, diamond_db):
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    out = tmp_path / 'hits.tsv'
    run_search(query, diamond_db, method='diamond', exclude_taxid=7227,
               max_hits=42, tuning=SearchTuning(threads=3), output_file=out)

    tool, cmd = calls[0]
    assert tool == 'diamond'
    # max_hits and threads are honoured, not hardcoded.
    assert cmd[cmd.index('--max-target-seqs') + 1] == '42'
    assert cmd[cmd.index('--threads') + 1] == '3'
    assert cmd[cmd.index('--taxon-exclude') + 1] == '7227'
    # The variadic --outfmt field list must come last, or diamond swallows the
    # option that follows it.
    assert cmd[cmd.index('--outfmt'):] == ['--outfmt', '6', *search.HIT_COLUMNS]
    assert cmd[-1] == 'staxids'
    assert out.exists()


def test_mmseqs_is_actually_run(monkeypatch, tmp_path, query):
    """Regression: -s mmseqs silently ran DIAMOND instead."""
    (tmp_path / 'targetDB.dbtype').write_bytes(b'\x00')
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    run_search(query, tmp_path / 'targetDB', method='mmseqs', exclude_taxid=7227,
               output_file=tmp_path / 'hits.tsv')

    tool, cmd = calls[0]
    assert tool == 'mmseqs'
    assert cmd[:2] == ['mmseqs', 'easy-search']
    assert 'taxid' in cmd[cmd.index('--format-output') + 1]
    assert cmd[cmd.index('--taxon-list') + 1] == '!7227'


def test_unknown_method_raises(monkeypatch, query, diamond_db, tmp_path):
    monkeypatch.setattr(search, '_run', fake_runner([]))
    with pytest.raises(ValueError):
        run_search(query, diamond_db, method='blast', output_file=tmp_path / 'x.tsv')


def test_missing_executable_is_reported_clearly(monkeypatch, query, diamond_db, tmp_path):
    def explode(cmd, tool):
        raise FileNotFoundError(tool)

    monkeypatch.setattr(search, '_run', explode)
    with pytest.raises(FileNotFoundError):
        run_search(query, diamond_db, output_file=tmp_path / 'x.tsv')


class FakePopen:
    """Stands in for a search tool: yields output lines, then an exit code."""

    def __init__(self, lines, returncode=0):
        self.stdout = _LineStream(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode


class _LineStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_nonzero_exit_includes_tool_output(monkeypatch, tmp_path, query, diamond_db):
    lines = ['Opening the database...\n', 'Error: database was built with an older version\n']
    monkeypatch.setattr(
        search.subprocess, 'Popen', lambda *a, **k: FakePopen(lines, returncode=3)
    )
    with pytest.raises(SearchError, match='older version'):
        run_search(query, diamond_db, output_file=tmp_path / 'x.tsv')


def test_tool_output_is_streamed_as_it_arrives(monkeypatch, tmp_path, query, diamond_db, caplog):
    """A multi-hour search must not be silent until it finishes."""
    lines = ['Processing query block 1\n', '\n', 'Processing query block 2\n']

    def fake_popen(cmd, **_kwargs):
        Path(cmd[cmd.index('-o') + 1]).write_text('g1\ts1\t1e-9\t99\t10\t50.0\t1390\n')
        return FakePopen(lines)

    monkeypatch.setattr(search.subprocess, 'Popen', fake_popen)
    with caplog.at_level('INFO'):
        run_search(query, diamond_db, output_file=tmp_path / 'x.tsv')
    assert '[diamond] Processing query block 1' in caplog.text
    assert '[diamond] Processing query block 2' in caplog.text


def test_missing_tool_is_reported_clearly(monkeypatch, tmp_path, query, diamond_db):
    def explode(*_a, **_k):
        raise FileNotFoundError('diamond')

    monkeypatch.setattr(search.subprocess, 'Popen', explode)
    with pytest.raises(SearchError, match='not found on PATH'):
        run_search(query, diamond_db, output_file=tmp_path / 'x.tsv')


# ------------------------------------------------------------------- reuse
def test_results_are_reused_when_inputs_match(monkeypatch, tmp_path, query, diamond_db):
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    out = tmp_path / 'hits.tsv'
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out)
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out)
    assert len(calls) == 1


def test_results_are_not_reused_after_a_settings_change(monkeypatch, tmp_path, query, diamond_db):
    """Regression: any non-empty file at the expected path was reused."""
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    out = tmp_path / 'hits.tsv'
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out)
    run_search(query, diamond_db, exclude_taxid=7240, output_file=out)  # different taxon
    assert len(calls) == 2


def test_orphan_results_without_provenance_are_not_reused(monkeypatch, tmp_path, query, diamond_db):
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    out = tmp_path / 'hits.tsv'
    out.write_text('g1\ts1\t1e-5\t50\t10\t50.0\t1390\n', encoding='utf-8')  # stale, no sidecar
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out)
    assert len(calls) == 1


def test_force_reruns(monkeypatch, tmp_path, query, diamond_db):
    calls = []
    monkeypatch.setattr(search, '_run', fake_runner(calls))
    out = tmp_path / 'hits.tsv'
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out)
    run_search(query, diamond_db, exclude_taxid=7227, output_file=out, force=True)
    assert len(calls) == 2


def test_failed_search_leaves_no_partial_output(monkeypatch, tmp_path, query, diamond_db):
    def explode(cmd, tool):
        Path(cmd[cmd.index('-o') + 1]).write_text('half a table\n', encoding='utf-8')
        raise SearchError('killed')

    monkeypatch.setattr(search, '_run', explode)
    out = tmp_path / 'hits.tsv'
    with pytest.raises(SearchError):
        run_search(query, diamond_db, output_file=out)
    assert not out.exists()


def test_sidecar_records_provenance(monkeypatch, tmp_path, query, diamond_db):
    monkeypatch.setattr(search, '_run', fake_runner([]))
    out = tmp_path / 'hits.tsv'
    run_search(query, diamond_db, exclude_taxid=7227, max_hits=99, output_file=out)
    recorded = json.loads((tmp_path / 'hits.tsv.meta.json').read_text(encoding='utf-8'))
    assert recorded['exclude_taxid'] == 7227
    assert recorded['max_hits'] == 99
    assert recorded['method'] == 'diamond'


# ------------------------------------------------- database taxonomy check
def _probe_result(rows, returncode=0, stdout=''):
    """Stub diamond blastp: write probe output, then report an exit code."""
    def run(cmd, **_kwargs):
        if '-o' in cmd:
            pathlib.Path(cmd[cmd.index('-o') + 1]).write_text(
                ''.join(f'{row}\n' for row in rows), encoding='utf-8'
            )

        class Completed:
            pass

        Completed.returncode = returncode
        Completed.stdout = stdout
        return Completed
    return run


def test_taxonomy_probe_passes_when_taxids_come_back(monkeypatch, diamond_db, query):
    monkeypatch.setattr(
        search.subprocess, 'run', _probe_result(['g1\t1390', 'g2\t7240'])
    )
    assert search.check_database_taxonomy(diamond_db, 'diamond', query) is None


def test_taxonomy_probe_catches_a_db_without_taxonomy(monkeypatch, diamond_db, query):
    """Failing here costs a second; failing after the search costs hours."""
    monkeypatch.setattr(search.subprocess, 'run', _probe_result(['g1\t', 'g2\t']))
    problem = search.check_database_taxonomy(diamond_db, 'diamond', query)
    assert problem is not None
    assert '--taxonmap' in problem


def test_taxonomy_probe_is_inconclusive_without_hits(monkeypatch, diamond_db, query, caplog):
    """No hits from a handful of sequences proves nothing either way."""
    monkeypatch.setattr(search.subprocess, 'run', _probe_result([]))
    with caplog.at_level('INFO'):
        assert search.check_database_taxonomy(diamond_db, 'diamond', query) is None
    assert 'no hits' in caplog.text


def test_taxonomy_probe_reads_diamonds_own_error(monkeypatch, diamond_db, query):
    """DIAMOND rejects the taxid output field up front on an untaxed database.

    Verified against diamond v2.1.13: it exits 1 in milliseconds with this
    message rather than returning empty taxids, so treating a non-zero exit as
    "inconclusive" would swallow the exact case worth catching.
    """
    message = (
        'Error: Options require taxonomy information included in the database. '
        'Please use the respective options to build this information into the '
        'database when running diamond makedb: taxonomy mapping information '
        '(--taxonmap option)'
    )
    monkeypatch.setattr(
        search.subprocess, 'run', _probe_result([], returncode=1, stdout=message)
    )
    problem = search.check_database_taxonomy(diamond_db, 'diamond', query)
    assert problem is not None
    assert 'has no taxonomy' in problem
    assert '--taxonmap' in problem


def test_taxonomy_probe_skipped_for_unrelated_failures(monkeypatch, diamond_db, query):
    """A database that cannot be searched at all is a different problem."""
    monkeypatch.setattr(
        search.subprocess, 'run',
        _probe_result([], returncode=1, stdout='Error: Invalid input file format'),
    )
    assert search.check_database_taxonomy(diamond_db, 'diamond', query) is None


@pytest.mark.parametrize('message,expected', [
    ('Options require taxonomy information ... --taxonmap option', True),
    ('taxonomy information not included in the database', True),
    ('Error: Invalid input file format', False),
    ('', False),
])
def test_missing_taxonomy_error_detection(message, expected):
    assert search._is_missing_taxonomy_error(message) is expected


def test_taxonomy_probe_survives_a_missing_binary(monkeypatch, diamond_db, query):
    def explode(*_a, **_k):
        raise OSError('no diamond')

    monkeypatch.setattr(search.subprocess, 'run', explode)
    assert search.check_database_taxonomy(diamond_db, 'diamond', query) is None


def test_taxonomy_probe_needs_a_query_file(diamond_db, tmp_path):
    assert search.check_database_taxonomy(diamond_db, 'diamond', None) is None
    assert search.check_database_taxonomy(diamond_db, 'diamond', tmp_path / 'gone.faa') is None


def test_probe_takes_only_the_first_few_sequences(tmp_path):
    fasta = tmp_path / 'many.faa'
    fasta.write_text(''.join(f'>g{i}\nMKV\n' for i in range(50)), encoding='utf-8')
    probe = search._first_sequences(fasta, 5)
    assert probe.count('>') == 5
    assert probe.startswith('>g0\n')


def test_probe_handles_a_short_file(tmp_path):
    fasta = tmp_path / 'few.faa'
    fasta.write_text('>g1\nMKV\n>g2\nMKA\n', encoding='utf-8')
    assert search._first_sequences(fasta, 5).count('>') == 2


def test_mmseqs_taxonomy_check(tmp_path):
    db = tmp_path / 'targetDB'
    Path(f'{db}.dbtype').write_bytes(b'\x00')
    problem = search.check_database_taxonomy(db, 'mmseqs')
    assert problem is not None and 'createtaxdb' in problem

    Path(f'{db}_mapping').write_bytes(b'\x00')
    assert search.check_database_taxonomy(db, 'mmseqs') is None
