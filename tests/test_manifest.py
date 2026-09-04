"""Tests for genome manifests and the combined query FASTA."""

from __future__ import annotations

import gzip
import json

import pytest

from flexihgt.manifest import (
    GENOME_SEPARATOR,
    GenomeRecord,
    ManifestError,
    iter_protein_ids,
    manifest_signature,
    namespaced,
    read_manifest,
    select_genomes,
    split_query_id,
    write_combined_fasta,
)


@pytest.fixture()
def genomes(tmp_path):
    """Two small proteomes plus a manifest referring to them by relative path."""
    (tmp_path / 'fly.faa').write_text('>p1 a fly protein\nMKVLW\n>p2\nMKVLA\n', encoding='utf-8')
    (tmp_path / 'bug.faa').write_text('>q1\nMKVLC\n', encoding='utf-8')
    manifest = tmp_path / 'genomes.tsv'
    manifest.write_text(
        'genome_id\ttaxid\tfasta\n'
        'fly\t7227\tfly.faa\n'
        'bug\t1390\tbug.faa\n',
        encoding='utf-8',
    )
    return manifest, tmp_path


# --------------------------------------------------------------- parsing
def test_reads_a_tsv_manifest(genomes):
    manifest, tmp_path = genomes
    records = read_manifest(manifest)
    assert [r.genome_id for r in records] == ['fly', 'bug']
    assert [r.taxid for r in records] == [7227, 1390]
    # Relative paths resolve against the manifest's directory.
    assert records[0].fasta == (tmp_path / 'fly.faa').resolve()


def test_reads_a_csv_manifest(tmp_path):
    (tmp_path / 'a.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    manifest = tmp_path / 'm.csv'
    manifest.write_text('taxid,fasta\n7227,a.faa\n', encoding='utf-8')
    records = read_manifest(manifest)
    assert records[0].taxid == 7227
    assert records[0].genome_id == 'a'          # defaults to the FASTA stem


def test_column_aliases_are_accepted(tmp_path):
    (tmp_path / 'a.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    manifest = tmp_path / 'm.tsv'
    manifest.write_text('assembly\ttax_id\tproteome\nGCA_1\t7227\ta.faa\n', encoding='utf-8')
    assert read_manifest(manifest)[0].genome_id == 'GCA_1'


def test_comment_and_blank_lines_ignored(tmp_path):
    (tmp_path / 'a.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    manifest = tmp_path / 'm.tsv'
    manifest.write_text(
        '## downloaded 2026-07-01\ntaxid\tfasta\n\n7227\ta.faa\n\n', encoding='utf-8'
    )
    assert len(read_manifest(manifest)) == 1


def test_missing_columns_reported(tmp_path):
    manifest = tmp_path / 'm.tsv'
    manifest.write_text('name\tsomething\nfly\t1\n', encoding='utf-8')
    with pytest.raises(ManifestError, match='missing required column'):
        read_manifest(manifest)


def test_all_problems_reported_together(tmp_path):
    """A 5,000-row manifest should not be debugged one error per run."""
    (tmp_path / 'ok.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    manifest = tmp_path / 'm.tsv'
    manifest.write_text(
        'genome_id\ttaxid\tfasta\n'
        'a\t7227\tok.faa\n'
        'b\tnot-a-taxid\tok.faa\n'
        'c\t7227\tmissing.faa\n'
        'a\t1390\tok.faa\n'
        'bad|id\t1390\tok.faa\n',
        encoding='utf-8',
    )
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(manifest)
    message = str(excinfo.value)
    assert 'invalid taxid' in message
    assert 'FASTA not found' in message
    assert 'duplicate genome id' in message
    assert 'may not contain' in message
    assert message.startswith('4 problem(s)')


def test_negative_taxid_rejected(tmp_path):
    (tmp_path / 'a.faa').write_text('>p1\nMKV\n', encoding='utf-8')
    manifest = tmp_path / 'm.tsv'
    manifest.write_text('taxid\tfasta\n-1\ta.faa\n', encoding='utf-8')
    with pytest.raises(ManifestError, match='invalid taxid'):
        read_manifest(manifest)


def test_empty_manifest_rejected(tmp_path):
    manifest = tmp_path / 'm.tsv'
    manifest.write_text('', encoding='utf-8')
    with pytest.raises(ManifestError, match='empty'):
        read_manifest(manifest)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(ManifestError, match='not found'):
        read_manifest(tmp_path / 'nope.tsv')


# ------------------------------------------------------------- selection
def test_select_by_comma_list(genomes):
    records = read_manifest(genomes[0])
    assert [r.genome_id for r in select_genomes(records, 'bug')] == ['bug']


def test_select_by_file(genomes, tmp_path):
    records = read_manifest(genomes[0])
    ids = tmp_path / 'ids.txt'
    ids.write_text('# batch 3\nfly\n', encoding='utf-8')
    assert [r.genome_id for r in select_genomes(records, str(ids))] == ['fly']


def test_select_none_returns_everything(genomes):
    records = read_manifest(genomes[0])
    assert len(select_genomes(records, None)) == 2


def test_select_unknown_id_is_an_error(genomes):
    records = read_manifest(genomes[0])
    with pytest.raises(ManifestError, match='not in the manifest'):
        select_genomes(records, 'nosuchgenome')


# ------------------------------------------------------------ namespacing
def test_round_trip():
    assert split_query_id(namespaced('fly', 'p1')) == ('fly', 'p1')


def test_protein_ids_may_contain_the_separator():
    """UniProt-style ids like sp|P12345|NAME must survive namespacing."""
    query = namespaced('fly', 'sp|P12345|NAME')
    assert split_query_id(query) == ('fly', 'sp|P12345|NAME')


def test_unnamespaced_id_yields_no_genome():
    assert split_query_id('p1') == ('', 'p1')


# ---------------------------------------------------------- combined FASTA
def test_combined_fasta_namespaces_every_header(genomes, tmp_path):
    manifest, _ = genomes
    records = read_manifest(manifest)
    out = tmp_path / 'combined.faa'
    counts = write_combined_fasta(records, out)

    assert counts == {'fly': 2, 'bug': 1}
    headers = [l for l in out.read_text(encoding='utf-8').splitlines() if l.startswith('>')]
    assert headers == ['>fly|p1', '>fly|p2', '>bug|q1']
    # The description after the id is dropped, as the search tool would anyway.
    assert 'a fly protein' not in out.read_text(encoding='utf-8')


def test_combined_fasta_handles_missing_trailing_newline(tmp_path):
    """Without care the next header gets glued onto the previous sequence."""
    (tmp_path / 'a.faa').write_text('>p1\nMKV', encoding='utf-8')      # no newline
    (tmp_path / 'b.faa').write_text('>q1\nMKA\n', encoding='utf-8')
    records = [
        GenomeRecord('a', 7227, tmp_path / 'a.faa'),
        GenomeRecord('b', 1390, tmp_path / 'b.faa'),
    ]
    out = tmp_path / 'combined.faa'
    write_combined_fasta(records, out)
    assert out.read_text(encoding='utf-8').splitlines() == ['>a|p1', 'MKV', '>b|q1', 'MKA']


def test_combined_fasta_reads_gzipped_proteomes(tmp_path):
    path = tmp_path / 'a.faa.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write('>p1\nMKV\n')
    records = [GenomeRecord('a', 7227, path)]
    out = tmp_path / 'combined.faa'
    assert write_combined_fasta(records, out) == {'a': 1}
    assert '>a|p1' in out.read_text(encoding='utf-8')


def test_combined_fasta_is_not_rebuilt_when_unchanged(genomes, tmp_path):
    """Rewriting it would bump the mtime and invalidate the cached search."""
    records = read_manifest(genomes[0])
    out = tmp_path / 'combined.faa'
    write_combined_fasta(records, out)
    first_mtime = out.stat().st_mtime_ns

    counts = write_combined_fasta(records, out)
    assert out.stat().st_mtime_ns == first_mtime
    assert counts == {'fly': 2, 'bug': 1}


def test_combined_fasta_rebuilt_when_the_genome_set_changes(genomes, tmp_path):
    records = read_manifest(genomes[0])
    out = tmp_path / 'combined.faa'
    write_combined_fasta(records, out)
    write_combined_fasta(records[:1], out)
    headers = [l for l in out.read_text(encoding='utf-8').splitlines() if l.startswith('>')]
    assert headers == ['>fly|p1', '>fly|p2']


def test_combined_fasta_force_rebuilds(genomes, tmp_path):
    records = read_manifest(genomes[0])
    out = tmp_path / 'combined.faa'
    write_combined_fasta(records, out)
    sidecar = out.with_name(out.name + '.manifest.json')
    recorded = json.loads(sidecar.read_text(encoding='utf-8'))
    assert recorded['signature'] == manifest_signature(records)
    write_combined_fasta(records, out, force=True)
    assert out.exists()


def test_iter_protein_ids(genomes):
    records = read_manifest(genomes[0])
    assert list(iter_protein_ids(records[0])) == ['fly|p1', 'fly|p2']


def test_separator_is_a_pipe():
    assert GENOME_SEPARATOR == '|'
