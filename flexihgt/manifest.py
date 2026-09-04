"""Genome manifests for multi-genome runs.

A manifest names the proteomes to analyse and the NCBI taxid each one belongs
to, replacing the single ``-q/--query_tax`` of a one-proteome run.  Nothing in
the single-genome pipeline uses this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Separates the genome id from the protein id in a combined query FASTA.
#: Split on the *first* occurrence only, so protein ids may contain it
#: (``sp|P12345|NAME`` is very common).
GENOME_SEPARATOR = '|'

#: Accepted spellings for each manifest column, lower-cased.
_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    'genome_id': ('genome_id', 'genome', 'id', 'name', 'label', 'assembly'),
    'taxid': ('taxid', 'tax_id', 'query_taxid', 'ncbi_taxid', 'taxon'),
    'fasta': ('fasta', 'proteome', 'path', 'file', 'faa'),
}


class ManifestError(ValueError):
    """Raised when a manifest cannot be used, listing every problem found."""


@dataclass(frozen=True)
class GenomeRecord:
    """One proteome to analyse."""

    genome_id: str
    taxid: int
    fasta: Path


def _open_text(path: Path) -> io.TextIOBase:
    """Open a FASTA, transparently handling gzip."""
    if path.suffix.lower() == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def _normalise_header(fieldnames: Optional[Sequence[str]]) -> Dict[str, str]:
    """Map each canonical column onto the actual header name used."""
    if not fieldnames:
        return {}
    lowered = {name.strip().lower().lstrip('#').strip(): name for name in fieldnames if name}
    resolved: Dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                resolved[canonical] = lowered[alias]
                break
    return resolved


def read_manifest(path: Path) -> List[GenomeRecord]:
    """Parse a TSV/CSV manifest of genomes.

    Required columns are ``taxid`` and ``fasta``; ``genome_id`` defaults to the
    FASTA's stem.  Relative FASTA paths resolve against the manifest's own
    directory, so a manifest can travel with its data.

    Every problem is collected and reported together -- fixing a 5,000-row
    manifest one error per run would be miserable.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f'Manifest not found: {path}')

    text = path.read_text(encoding='utf-8')
    if not text.strip():
        raise ManifestError(f'Manifest is empty: {path}')

    # Blank and comment lines are dropped before parsing, so their positions are
    # kept alongside: an error message pointing at the wrong line of a 5,000-row
    # manifest is worse than no line number at all.
    kept = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.startswith('##')
    ]
    if not kept:
        raise ManifestError(f'Manifest has no content rows: {path}')
    rows = [line for _, line in kept]
    line_numbers = [number for number, _ in kept[1:]]      # data rows only
    # Sniff the delimiter from the header, not from line 0 -- a leading comment
    # would otherwise decide it.
    delimiter = '\t' if '\t' in rows[0] else ','
    reader = csv.DictReader(rows, delimiter=delimiter)
    columns = _normalise_header(reader.fieldnames)

    missing = [name for name in ('taxid', 'fasta') if name not in columns]
    if missing:
        raise ManifestError(
            f'Manifest {path} is missing required column(s): {", ".join(missing)}. '
            f'Found: {", ".join(reader.fieldnames or ["<none>"])}'
        )

    records: List[GenomeRecord] = []
    problems: List[str] = []
    seen: Dict[str, int] = {}

    for index, row in enumerate(reader):
        # A quoted field spanning newlines would consume more than one physical
        # line; fall back to the last known position rather than index off the
        # end, since the number is only ever used in a message.
        line_number = line_numbers[index] if index < len(line_numbers) else len(text.splitlines())
        raw_fasta = (row.get(columns['fasta']) or '').strip()
        raw_taxid = (row.get(columns['taxid']) or '').strip()
        if not raw_fasta and not raw_taxid:
            continue                                    # blank row

        fasta = Path(raw_fasta)
        if not fasta.is_absolute():
            fasta = (path.parent / fasta).resolve()

        genome_id = (row.get(columns['genome_id'], '') or '').strip() if 'genome_id' in columns else ''
        if not genome_id:
            genome_id = fasta.name.split('.')[0]

        if not raw_fasta:
            problems.append(f'line {line_number}: no FASTA path')
            continue
        if not fasta.is_file():
            problems.append(f'line {line_number}: FASTA not found: {fasta}')
        try:
            taxid = int(float(raw_taxid))
            if taxid <= 0:
                raise ValueError
        except ValueError:
            problems.append(f'line {line_number}: invalid taxid {raw_taxid!r}')
            continue
        if not genome_id:
            # Reachable via the fallback: a file named '.faa' has an empty stem,
            # which the old ``genome_id.split()[0]`` raised IndexError on.
            problems.append(f'line {line_number}: genome id is empty')
            continue
        # ``split() != [id]`` catches every kind of whitespace, including a
        # trailing tab the strip() above cannot see inside a quoted field.
        if GENOME_SEPARATOR in genome_id or genome_id.split() != [genome_id]:
            problems.append(
                f'line {line_number}: genome id {genome_id!r} may not contain '
                f'{GENOME_SEPARATOR!r} or whitespace'
            )
            continue
        if genome_id in seen:
            problems.append(
                f'line {line_number}: duplicate genome id {genome_id!r} '
                f'(first seen on line {seen[genome_id]})'
            )
            continue

        seen[genome_id] = line_number
        records.append(GenomeRecord(genome_id=genome_id, taxid=taxid, fasta=fasta))

    if problems:
        raise ManifestError(
            f'{len(problems)} problem(s) in {path}:\n  ' + '\n  '.join(problems)
        )
    if not records:
        raise ManifestError(f'Manifest contains no usable rows: {path}')

    logger.info('Manifest %s: %d genome(s)', path, len(records))
    return records


def select_genomes(records: Sequence[GenomeRecord], selection: Optional[str]) -> List[GenomeRecord]:
    """Restrict ``records`` to a comma-separated list of ids, or a file of ids.

    Lets a scheduler split one manifest across a job array without editing it.
    """
    if not selection:
        return list(records)

    candidate = Path(selection)
    if candidate.is_file():
        wanted = {
            line.strip() for line in candidate.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')
        }
    else:
        wanted = {part.strip() for part in selection.split(',') if part.strip()}

    known = {record.genome_id for record in records}
    unknown = sorted(wanted - known)
    if unknown:
        raise ManifestError(
            f'{len(unknown)} selected genome id(s) are not in the manifest: '
            + ', '.join(unknown[:5]) + ('...' if len(unknown) > 5 else '')
        )
    selected = [record for record in records if record.genome_id in wanted]
    logger.info('Selected %d of %d genome(s)', len(selected), len(records))
    return selected


#: Proteome file extensions recognised when scanning a directory.
FASTA_SUFFIXES = ('.faa', '.fa', '.fasta', '.fas', '.pep')

#: UniProt reference proteomes are named ``UP000000803_7227.fasta`` -- the taxid
#: is the trailing underscore-separated field of the stem.
_UNIPROT_STEM = __import__('re').compile(r'^UP\d+_(\d+)$')


def _fasta_files(directory: Path) -> List[Path]:
    files = [
        path for path in sorted(directory.rglob('*'))
        if path.is_file()
        and (path.suffix.lower() in FASTA_SUFFIXES
             or (path.suffix.lower() == '.gz'
                 and Path(path.stem).suffix.lower() in FASTA_SUFFIXES))
    ]
    return files


def _stem(path: Path) -> str:
    """Filename without any FASTA/gzip suffixes."""
    name = path.name
    if name.lower().endswith('.gz'):
        name = name[:-3]
    for suffix in FASTA_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _taxids_from_ncbi_datasets(root: Path) -> Dict[str, int]:
    """Accession -> taxid, from an NCBI datasets download.

    ``ncbi_dataset/data/assembly_data_report.jsonl`` carries one JSON object per
    assembly, each with its accession and organism taxId.
    """
    mapping: Dict[str, int] = {}
    for report in root.rglob('assembly_data_report.jsonl'):
        try:
            for line in report.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                accession = record.get('accession')
                organism = record.get('organism') or {}
                taxid = organism.get('taxId') or record.get('taxId')
                if accession and taxid:
                    mapping[str(accession)] = int(taxid)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning('Could not read %s: %s', report, exc)
    return mapping


def _taxids_from_assembly_reports(root: Path) -> Dict[str, int]:
    """Accession -> taxid, from ``*_assembly_report.txt`` headers."""
    mapping: Dict[str, int] = {}
    for report in root.rglob('*_assembly_report.txt'):
        accession, taxid = None, None
        try:
            for line in report.read_text(encoding='utf-8', errors='replace').splitlines():
                if not line.startswith('#'):
                    break
                key, separator, value = line.lstrip('# ').partition(':')
                if not separator:
                    continue
                key, value = key.strip().lower(), value.strip()
                if key == 'taxid':
                    taxid = int(value)
                elif key in ('assembly name', 'assembly accession', 'refseq assembly accession'):
                    accession = accession or value
        except (OSError, ValueError) as exc:
            logger.warning('Could not read %s: %s', report, exc)
            continue
        if taxid:
            mapping[report.name.replace('_assembly_report.txt', '')] = taxid
            if accession:
                mapping[accession] = taxid
    return mapping


def _read_taxid_map(path: Path) -> Dict[str, int]:
    """User-supplied ``key<TAB>taxid`` overrides, keyed by id or filename."""
    mapping: Dict[str, int] = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.replace(',', '\t').split('\t')
        if len(parts) < 2:
            continue
        try:
            mapping[parts[0].strip()] = int(float(parts[1].strip()))
        except ValueError:
            continue
    return mapping


def build_manifest(
    directory: Path,
    destination: Optional[Path] = None,
    taxid_map: Optional[Path] = None,
) -> Tuple[List[GenomeRecord], List[Tuple[Path, str]]]:
    """Scan a directory of proteomes and work out each one's taxid.

    Writing a manifest by hand is fine for ten genomes and miserable for
    thousands.  Taxids are looked for, in order of preference, in:

    1. a user-supplied ``--taxid-map`` file (always wins),
    2. an NCBI datasets ``assembly_data_report.jsonl``,
    3. NCBI ``*_assembly_report.txt`` headers,
    4. a UniProt-style ``UP000000803_7227.fasta`` filename.

    Returns the records it resolved and the files it could not, so the gaps can
    be filled in by hand rather than silently dropped.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ManifestError(f'Not a directory: {directory}')

    files = _fasta_files(directory)
    if not files:
        raise ManifestError(
            f'No proteome files found under {directory} '
            f'(looked for {", ".join(FASTA_SUFFIXES)}, optionally gzipped)'
        )

    overrides = _read_taxid_map(Path(taxid_map)) if taxid_map else {}
    datasets = _taxids_from_ncbi_datasets(directory)
    reports = _taxids_from_assembly_reports(directory)
    logger.info(
        'Scanning %s: %d proteome file(s), %d datasets record(s), %d assembly report(s)',
        directory, len(files), len(datasets), len(reports),
    )

    records: List[GenomeRecord] = []
    unresolved: List[Tuple[Path, str]] = []
    taken: set = set()

    for path in files:
        stem = _stem(path)
        # An NCBI datasets download puts each proteome under its accession.
        accession = path.parent.name
        taxid = (
            overrides.get(stem) or overrides.get(path.name) or overrides.get(accession)
            or datasets.get(accession) or datasets.get(stem)
            or reports.get(accession) or reports.get(stem)
        )
        if taxid is None:
            match = _UNIPROT_STEM.match(stem)
            if match:
                taxid = int(match.group(1))
        if taxid is None:
            unresolved.append((path, 'no taxid found'))
            continue

        genome_id = accession if accession not in ('.', directory.name) else stem
        genome_id = genome_id.replace(GENOME_SEPARATOR, '_')
        # Disambiguate against every id already issued, not just against the
        # base name: counting per base name hands out '<base>_1' twice as soon
        # as three files share a stem, and two genomes sharing an id are
        # silently merged into one namespace in the combined FASTA.
        if genome_id in taken:
            base, suffix = genome_id, 1
            while genome_id in taken:
                genome_id = f'{base}_{suffix}'
                suffix += 1
            logger.warning('Duplicate genome id %r; using %r for %s', base, genome_id, path)
        taken.add(genome_id)
        records.append(GenomeRecord(genome_id=genome_id, taxid=int(taxid), fasta=path.resolve()))

    if destination:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['genome_id', 'taxid', 'fasta'])
            for record in records:
                writer.writerow([record.genome_id, record.taxid, record.fasta])
        logger.info('Wrote manifest with %d genome(s) to %s', len(records), destination)

    if unresolved:
        logger.warning(
            '%d proteome(s) had no taxid and were left out; supply them with '
            '--taxid-map (e.g. %s)',
            len(unresolved), ', '.join(str(path.name) for path, _ in unresolved[:3]),
        )
    return records, unresolved


def namespaced(genome_id: str, protein_id: str) -> str:
    return f'{genome_id}{GENOME_SEPARATOR}{protein_id}'


def split_query_id(query_id: str) -> Tuple[str, str]:
    """Inverse of :func:`namespaced`; splits on the first separator only."""
    genome, separator, protein = str(query_id).partition(GENOME_SEPARATOR)
    if not separator:
        return '', str(query_id)
    return genome, protein


def iter_protein_ids(record: GenomeRecord) -> Iterator[str]:
    """Namespaced ids for one genome's proteins, streamed from its FASTA."""
    with _open_text(record.fasta) as handle:
        for line in handle:
            if not line.startswith('>'):
                continue
            parts = line[1:].split(None, 1)
            if parts:
                yield namespaced(record.genome_id, parts[0])


def manifest_signature(records: Sequence[GenomeRecord]) -> str:
    """Fingerprint of the genome set, used to avoid pointless FASTA rebuilds."""
    payload = json.dumps(
        [
            {
                'genome_id': record.genome_id,
                'taxid': record.taxid,
                'fasta': str(record.fasta),
                'size': record.fasta.stat().st_size if record.fasta.exists() else -1,
                'mtime': int(record.fasta.stat().st_mtime) if record.fasta.exists() else -1,
            }
            for record in records
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def write_combined_fasta(
    records: Sequence[GenomeRecord],
    destination: Path,
    force: bool = False,
) -> Dict[str, int]:
    """Concatenate every proteome into one FASTA with namespaced ids.

    A single large query file is what makes a multi-genome run affordable: one
    search pays the database load and index build once, instead of once per
    genome.

    The file is only rebuilt when the genome set changes.  Rewriting it
    needlessly would bump its mtime and invalidate the cached search results,
    which is the one thing worth protecting here.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_name(destination.name + '.manifest.json')
    signature = manifest_signature(records)

    if not force and destination.exists() and sidecar.exists():
        try:
            recorded = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            recorded = {}
        if recorded.get('signature') == signature:
            logger.info('Reusing combined query FASTA: %s', destination)
            return {gid: int(n) for gid, n in recorded.get('counts', {}).items()}

    counts: Dict[str, int] = {}
    partial = destination.with_name(destination.name + '.partial')
    with open(partial, 'w', encoding='utf-8') as out:
        for record in records:
            written = 0
            seen_ids: set = set()
            with _open_text(record.fasta) as handle:
                for line in handle:
                    if line.startswith('>'):
                        parts = line[1:].split(None, 1)
                        if not parts:
                            continue                    # header with no id
                        protein_id = parts[0]
                        if protein_id in seen_ids:
                            logger.warning(
                                'Duplicate protein id %r in %s; keeping both, '
                                'their scores will be indistinguishable',
                                protein_id, record.fasta,
                            )
                        seen_ids.add(protein_id)
                        out.write(f'>{namespaced(record.genome_id, protein_id)}\n')
                        written += 1
                    else:
                        # Some FASTA files lack a trailing newline; without this
                        # the next header would be glued onto the last sequence.
                        out.write(line if line.endswith('\n') else line + '\n')
            if not written:
                logger.warning('No sequences found in %s (%s)', record.fasta, record.genome_id)
            counts[record.genome_id] = written

    partial.replace(destination)
    sidecar.write_text(
        json.dumps({'signature': signature, 'counts': counts}, indent=2), encoding='utf-8'
    )
    total = sum(counts.values())
    logger.info(
        'Wrote combined query FASTA: %s (%d genome(s), %d sequence(s))',
        destination, len(counts), total,
    )
    return counts
