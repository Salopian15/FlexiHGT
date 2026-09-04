"""Tests for config-file support and the manifest builder."""

from __future__ import annotations

import gzip
import json
import sys

import pytest

from flexihgt import cli, multi_cli
from flexihgt.config import ConfigError, apply_config, load_config
from flexihgt.manifest import ManifestError, build_manifest, read_manifest

toml_required = pytest.mark.skipif(
    sys.version_info < (3, 11), reason='tomllib needs Python 3.11+'
)


# ------------------------------------------------------------------- config
def test_loads_json(tmp_path):
    path = tmp_path / 'c.json'
    path.write_text(json.dumps({'tax_level': 'phylum', 'AI': 30}), encoding='utf-8')
    assert load_config(path) == {'tax_level': 'phylum', 'AI': 30}


@toml_required
def test_loads_toml(tmp_path):
    path = tmp_path / 'c.toml'
    path.write_text('tax_level = "phylum"\nAI = 30\n', encoding='utf-8')
    assert load_config(path) == {'tax_level': 'phylum', 'AI': 30}


@toml_required
def test_flexihgt_section_is_unwrapped(tmp_path):
    """A config may live alongside other tools' settings."""
    path = tmp_path / 'c.toml'
    path.write_text('[other]\nx = 1\n\n[flexihgt]\ntax_level = "order"\n', encoding='utf-8')
    assert load_config(path) == {'tax_level': 'order'}


def test_flag_spellings_are_normalised(tmp_path):
    path = tmp_path / 'c.json'
    path.write_text(json.dumps({'--tax-level': 'order', 'max-hits': 50}), encoding='utf-8')
    assert load_config(path) == {'tax_level': 'order', 'max_hits': 50}


def test_missing_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match='not found'):
        load_config(tmp_path / 'nope.json')


def test_unsupported_format_is_an_error(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('a: 1\n', encoding='utf-8')
    with pytest.raises(ConfigError, match='Unsupported config format'):
        load_config(path)


def test_malformed_json_is_an_error(tmp_path):
    path = tmp_path / 'c.json'
    path.write_text('{not json', encoding='utf-8')
    with pytest.raises(ConfigError, match='Could not parse'):
        load_config(path)


def test_config_fills_in_defaults():
    parser = cli.build_parser()
    argv = ['in.faa', '-q', '7227', '-db', 'db.dmnd']
    args = parser.parse_args(argv)
    apply_config(parser, args, {'tax_level': 'phylum', 'max_hits': 42}, argv)
    assert args.tax_level == 'phylum'
    assert args.max_hits == 42


def test_command_line_beats_config():
    """An explicit flag must win, however the config was written."""
    parser = cli.build_parser()
    argv = ['in.faa', '-q', '7227', '-db', 'db.dmnd', '-t', 'order', '--max_hits=7']
    args = parser.parse_args(argv)
    apply_config(parser, args, {'tax_level': 'phylum', 'max_hits': 42}, argv)
    assert args.tax_level == 'order'
    assert args.max_hits == 7


def test_unknown_config_option_is_rejected():
    parser = cli.build_parser()
    argv = ['in.faa', '-q', '7227', '-db', 'db.dmnd']
    args = parser.parse_args(argv)
    with pytest.raises(ConfigError, match='Unknown option'):
        apply_config(parser, args, {'nonsense': 1}, argv)


def test_cli_reports_a_bad_config(tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1\nMKV\n', encoding='utf-8')
    bad = tmp_path / 'c.json'
    bad.write_text('{oops', encoding='utf-8')
    code = cli.main([str(fasta), '-q', '7227', '-db', 'db', '--config', str(bad)])
    assert code == cli.EXIT_USAGE


# --------------------------------------------------------- manifest builder
def test_builds_from_uniprot_style_filenames(tmp_path):
    (tmp_path / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    (tmp_path / 'UP000005640_9606.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    records, unresolved = build_manifest(tmp_path, tmp_path / 'm.tsv')
    assert unresolved == []
    assert sorted(r.taxid for r in records) == [7227, 9606]
    # The written manifest round-trips through the reader.
    assert len(read_manifest(tmp_path / 'm.tsv')) == 2


def test_builds_from_ncbi_datasets_layout(tmp_path):
    data = tmp_path / 'ncbi_dataset' / 'data'
    (data / 'GCF_000001215.4').mkdir(parents=True)
    (data / 'GCF_000001215.4' / 'protein.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    (data / 'assembly_data_report.jsonl').write_text(
        json.dumps({'accession': 'GCF_000001215.4', 'organism': {'taxId': 7227}}) + '\n',
        encoding='utf-8',
    )
    records, unresolved = build_manifest(tmp_path)
    assert unresolved == []
    assert records[0].genome_id == 'GCF_000001215.4'
    assert records[0].taxid == 7227


def test_builds_from_assembly_report(tmp_path):
    (tmp_path / 'GCF_000001215.4_assembly_report.txt').write_text(
        '# Assembly name:  Release_6\n# Taxid:          7227\n# Date: 2014\n',
        encoding='utf-8',
    )
    (tmp_path / 'GCF_000001215.4.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    records, _ = build_manifest(tmp_path)
    assert records[0].taxid == 7227


def test_taxid_map_overrides_everything(tmp_path):
    (tmp_path / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    override = tmp_path / 'map.tsv'
    override.write_text('UP000000803_7227\t9999\n', encoding='utf-8')
    records, _ = build_manifest(tmp_path, taxid_map=override)
    assert records[0].taxid == 9999


def test_unresolvable_files_are_reported_not_dropped_silently(tmp_path, caplog):
    (tmp_path / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    (tmp_path / 'mystery.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    with caplog.at_level('WARNING'):
        records, unresolved = build_manifest(tmp_path)
    assert len(records) == 1
    assert [path.name for path, _ in unresolved] == ['mystery.faa']
    assert 'no taxid' in caplog.text


def test_gzipped_proteomes_are_found(tmp_path):
    path = tmp_path / 'UP000000803_7227.faa.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write('>p1\nMKV\n')
    records, _ = build_manifest(tmp_path)
    assert records[0].fasta.name.endswith('.faa.gz')


def test_empty_directory_is_an_error(tmp_path):
    with pytest.raises(ManifestError, match='No proteome files'):
        build_manifest(tmp_path)


def test_not_a_directory_is_an_error(tmp_path):
    path = tmp_path / 'f.txt'
    path.write_text('x', encoding='utf-8')
    with pytest.raises(ManifestError, match='Not a directory'):
        build_manifest(path)


def test_cli_make_manifest(tmp_path):
    (tmp_path / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    out = tmp_path / 'm.tsv'
    assert multi_cli.main([str(tmp_path), '--make-manifest', str(out)]) == multi_cli.EXIT_OK
    assert 'genome_id\ttaxid\tfasta' in out.read_text(encoding='utf-8')


def test_cli_make_manifest_flags_gaps(tmp_path):
    (tmp_path / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')
    (tmp_path / 'mystery.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    out = tmp_path / 'm.tsv'
    # Partial, because a proteome was left out of the manifest.
    assert multi_cli.main([str(tmp_path), '--make-manifest', str(out)]) == multi_cli.EXIT_PARTIAL


def test_repeated_stems_get_distinct_genome_ids(tmp_path):
    """Regression: three files sharing a stem produced '<id>_1' twice.

    The counter was kept against the disambiguated id rather than the base, so
    two genomes ended up sharing a namespace in the combined FASTA and were
    silently merged.
    """
    for parent in ('a', 'b', 'c'):
        directory = tmp_path / parent / 'UP000000803_7227'
        directory.mkdir(parents=True)
        (directory / 'UP000000803_7227.fasta').write_text('>p1\nMKV\n', encoding='utf-8')

    records, _ = build_manifest(tmp_path)
    ids = [record.genome_id for record in records]
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_config_values_are_converted(tmp_path):
    """A config file must go through the same type= as the command line."""
    parser = cli.build_parser()
    args = parser.parse_args(['p.faa', '-q', '7227', '-db', 'db'])
    apply_config(parser, args, {'max_hits': '42', 'evalue': '1e-9'}, ['p.faa'])
    assert args.max_hits == 42
    assert args.evalue == 1e-9


def test_config_rejects_a_value_outside_choices():
    parser = cli.build_parser()
    args = parser.parse_args(['p.faa', '-q', '7227', '-db', 'db'])
    with pytest.raises(ConfigError, match='tax_level'):
        apply_config(parser, args, {'tax_level': 'Family'}, ['p.faa'])


def test_config_rejects_an_unconvertible_value():
    parser = cli.build_parser()
    args = parser.parse_args(['p.faa', '-q', '7227', '-db', 'db'])
    with pytest.raises(ConfigError, match='max_hits'):
        apply_config(parser, args, {'max_hits': 'lots'}, ['p.faa'])


def test_config_rejects_a_non_boolean_flag():
    parser = cli.build_parser()
    args = parser.parse_args(['p.faa', '-q', '7227', '-db', 'db'])
    with pytest.raises(ConfigError, match='flag'):
        apply_config(parser, args, {'force': 'yes'}, ['p.faa'])
