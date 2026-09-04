"""Tests for the environment checks and the CLI wiring."""

from __future__ import annotations

from flexihgt import cli, utils


# ------------------------------------------------------------------- checks
def test_python_dependency_check_uses_import_names(monkeypatch):
    """Regression: it probed for a module named 'biopython' (it installs as 'Bio').

    Keys must be importable module names, not distribution names.
    """
    import importlib.util as iu

    assert 'biopython' not in utils.PYTHON_DEPENDENCIES
    for module in utils.PYTHON_DEPENDENCIES:
        assert iu.find_spec(module) is not None or module == 'ete3'

    monkeypatch.setattr(utils.importlib.util, 'find_spec', lambda name: object())
    assert utils.check_python_dependencies() is True


def test_python_dependency_check_names_the_missing_package(monkeypatch, caplog):
    monkeypatch.setattr(
        utils.importlib.util, 'find_spec',
        lambda name: None if name == 'ete3' else object(),
    )
    with caplog.at_level('ERROR'):
        assert utils.check_python_dependencies() is False
    assert 'ete3' in caplog.text


def test_search_tool_check_reports_missing(monkeypatch):
    monkeypatch.setattr(utils.shutil, 'which', lambda _: None)
    assert utils.check_search_tool('diamond') is False


def test_search_tool_check_passes(monkeypatch):
    monkeypatch.setattr(utils.shutil, 'which', lambda _: '/usr/bin/diamond')
    monkeypatch.setattr(utils.subprocess, 'run', lambda *a, **k: None)
    assert utils.check_search_tool('diamond') is True


def test_taxonomy_db_path_is_posix(monkeypatch, tmp_path):
    """Regression: the old Windows probe path could never match."""
    path = utils.ete3_database_path()
    assert '\\' not in str(path)
    assert path.name == 'taxa.sqlite'


def test_protein_fasta_accepted(tmp_path):
    """Regression: protein input was validated against 'ACGT' and always failed."""
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQ\n', encoding='utf-8')
    assert utils.sniff_fasta(fasta) is True


def test_invalid_residues_rejected(tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV!!!123\n', encoding='utf-8')
    assert utils.sniff_fasta(fasta) is False


def test_missing_header_rejected(tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('MKVLWAALL\n', encoding='utf-8')
    assert utils.sniff_fasta(fasta) is False


def test_nucleotide_input_warns_but_passes(tmp_path, caplog):
    fasta = tmp_path / 'n.fasta'
    fasta.write_text('>g1\nACGTACGTACGT\n', encoding='utf-8')
    with caplog.at_level('WARNING'):
        assert utils.sniff_fasta(fasta) is True
    assert 'nucleotide' in caplog.text


def test_check_environment_reports_all_failures(monkeypatch, caplog):
    monkeypatch.setattr(utils, 'check_search_tool', lambda _: False)
    monkeypatch.setattr(utils, 'check_taxonomy_database', lambda: False)
    with caplog.at_level('ERROR'):
        assert utils.check_environment('diamond') is False
    assert 'diamond executable' in caplog.text
    assert 'NCBI taxonomy database' in caplog.text


# ---------------------------------------------------------------------- CLI
def test_parser_defaults():
    args = cli.build_parser().parse_args(['in.faa', '-q', '7227', '-db', 'db.dmnd'])
    assert args.tax_level == 'family'
    assert args.search == 'diamond'
    assert args.max_hits == 200
    assert args.force is False


def test_update_only_exits_without_running_analysis(monkeypatch):
    """Regression: --update called a nonexistent ete3 attribute and then exited 1.

    It also could not run standalone, because the check guarding it
    (`if not args.input_file`) tested a required positional argument.
    """
    called = {}
    monkeypatch.setattr(cli, '_update_taxonomy', lambda: called.setdefault('updated', True))
    assert cli.main(['--update-only']) == cli.EXIT_OK
    assert called['updated']


def test_update_failure_is_reported(monkeypatch):
    def explode():
        raise RuntimeError('no network')

    monkeypatch.setattr(cli, '_update_taxonomy', explode)
    assert cli.main(['--update-only']) == cli.EXIT_FAILED


def test_missing_input_file_exits_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)
    code = cli.main([str(tmp_path / 'nope.faa'), '-q', '7227', '-db', 'db'])
    assert code == cli.EXIT_USAGE


def test_bad_threshold_exits_usage(tmp_path, monkeypatch):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)
    code = cli.main([str(fasta), '-q', '7227', '-db', 'db', '--HGTIndex', '9'])
    assert code == cli.EXIT_USAGE


def test_partial_failure_exit_code(tmp_path, monkeypatch):
    """A run where some genes errored must not look successful."""
    import pandas as pd

    from flexihgt.core import AnalysisResult

    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'x')

    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)
    monkeypatch.setattr(
        cli.HGTDetect, 'run_analysis',
        lambda self, *a, **k: AnalysisResult(
            candidates=pd.DataFrame(columns=['gene', 'scores', 'taxonomy', 'top_hits']),
            genes_total=1,
            gene_errors=[('g1', 'boom')],
        ),
    )
    code = cli.main([str(fasta), '-q', '7227', '-db', str(db)])
    assert code == cli.EXIT_PARTIAL


def test_search_failure_exit_code(tmp_path, monkeypatch):
    from flexihgt.search import SearchError

    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'x')

    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)

    def explode(self, *a, **k):
        raise SearchError('diamond died')

    monkeypatch.setattr(cli.HGTDetect, 'run_analysis', explode)
    assert cli.main([str(fasta), '-q', '7227', '-db', str(db)]) == cli.EXIT_FAILED


# ------------------------------------------------------------ new CLI flags
def test_tuning_flags_parse():
    args = cli.build_parser().parse_args([
        'in.faa', '-q', '7227', '-db', 'db.dmnd',
        '--evalue', '1e-10', '--block_size', '6', '--index_chunks', '1',
        '--tmpdir', '/scratch', '--memory_limit', '64G', '--no-cache',
    ])
    assert args.evalue == 1e-10
    assert args.block_size == 6
    assert args.index_chunks == 1
    assert args.tmpdir == '/scratch'
    assert args.memory_limit == '64G'
    assert args.no_cache is True


def test_rescore_only_does_not_require_a_database(monkeypatch, tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)

    recorded = {}

    def fake_run(self, *_a, **kwargs):
        recorded.update(kwargs)
        import pandas as pd

        from flexihgt.core import AnalysisResult
        return AnalysisResult(
            candidates=pd.DataFrame(columns=['gene', 'scores', 'taxonomy', 'top_hits'])
        )

    monkeypatch.setattr(cli.HGTDetect, 'run_analysis', fake_run)
    assert cli.main([str(fasta), '-q', '7227', '--rescore-only']) == cli.EXIT_OK
    assert recorded['rescore_only'] is True


def test_tuning_reaches_run_analysis(monkeypatch, tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    db = tmp_path / 'db.dmnd'
    db.write_bytes(b'x')
    monkeypatch.setattr(cli, 'check_environment', lambda *a, **k: True)

    recorded = {}

    def fake_run(self, *_a, **kwargs):
        recorded.update(kwargs)
        import pandas as pd

        from flexihgt.core import AnalysisResult
        return AnalysisResult(
            candidates=pd.DataFrame(columns=['gene', 'scores', 'taxonomy', 'top_hits'])
        )

    monkeypatch.setattr(cli.HGTDetect, 'run_analysis', fake_run)
    cli.main([str(fasta), '-q', '7227', '-db', str(db), '--block_size', '6', '--threads', '8'])
    assert recorded['tuning'].block_size == 6
    assert recorded['tuning'].threads == 8


def test_rescore_only_skips_the_search_tool_check(monkeypatch, tmp_path):
    """No point demanding DIAMOND when the search is not going to run."""
    seen = {}

    def fake_check(method, input_file=None, need_search_tool=True, **kwargs):
        seen['need_search_tool'] = need_search_tool
        return False

    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    monkeypatch.setattr(cli, 'check_environment', fake_check)
    cli.main([str(fasta), '-q', '7227', '--rescore-only'])
    assert seen['need_search_tool'] is False


def test_unknown_taxonomy_backend_is_rejected():
    import pytest

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['in.faa', '-q', '1', '-db', 'd', '--taxonomy-backend', 'nope'])


# --------------------------------------------------------- taxonomy backend
def test_taxopy_backend_does_not_require_ete3(monkeypatch):
    """Regression: --taxonomy-backend taxopy still demanded ete3 and its db.

    The whole point of the backend is not needing either, so the checks it was
    meant to pass could only be got past with --skip-checks.
    """
    monkeypatch.setattr(
        utils.importlib.util, 'find_spec',
        lambda name: None if name == 'ete3' else object(),
    )
    assert utils.check_python_dependencies('taxopy') is True
    assert utils.check_python_dependencies('ete3') is False


def test_taxopy_backend_skips_the_ete3_database_check(monkeypatch):
    def explode():
        raise AssertionError('the ete3 database is irrelevant to a taxopy run')

    monkeypatch.setattr(utils, 'check_taxonomy_database', explode)
    monkeypatch.setattr(utils.importlib.util, 'find_spec', lambda name: object())
    assert utils.check_environment('diamond', need_search_tool=False,
                                   taxonomy_backend='taxopy') is True


def test_taxopy_dumps_are_checked_when_a_directory_is_given(tmp_path, caplog):
    with caplog.at_level('ERROR'):
        assert utils.check_taxopy_database(str(tmp_path)) is False
    assert 'nodes.dmp' in caplog.text

    for name in utils.TAXOPY_DUMPS:
        (tmp_path / name).write_text('', encoding='utf-8')
    assert utils.check_taxopy_database(str(tmp_path)) is True


def test_taxopy_without_a_directory_is_not_blocked():
    """taxopy downloads its own copy, so there is nothing to verify."""
    assert utils.check_taxopy_database(None) is True


def test_update_is_refused_for_a_backend_it_cannot_update(monkeypatch, caplog):
    monkeypatch.setattr(cli, '_update_taxonomy', lambda: pytest.fail('must not run'))
    with caplog.at_level('ERROR'):
        assert cli.main(['--update-only', '--taxonomy-backend', 'taxopy']) == cli.EXIT_USAGE
    assert 'ete3' in caplog.text
