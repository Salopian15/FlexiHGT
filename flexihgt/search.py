"""Homology search backends (DIAMOND and MMseqs2).

Both backends emit the same seven columns in the same order, so everything
downstream is engine-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Canonical output columns, in order.  Referred to by name everywhere else so
#: that a change here does not require hunting down bare integer indices.
QSEQID = 'qseqid'
SSEQID = 'sseqid'
EVALUE = 'evalue'
BITSCORE = 'bitscore'
LENGTH = 'length'
PIDENT = 'pident'
STAXIDS = 'staxids'

HIT_COLUMNS: List[str] = [QSEQID, SSEQID, EVALUE, BITSCORE, LENGTH, PIDENT, STAXIDS]


class SearchError(RuntimeError):
    """Raised when the underlying search tool fails."""


#: How many trailing output lines to keep for the error message. The tools are
#: chatty, but the last few lines are where the diagnosis lives.
_ERROR_CONTEXT_LINES = 60


def _run(cmd: Sequence[str], tool: str) -> None:
    """Run a search tool, relaying its progress as it happens.

    The output is streamed line by line rather than captured and reported at
    the end.  A proteome search runs for hours, and buffering meant it printed
    nothing at all until it finished -- a slow run and a hung one looked
    identical.
    """
    logger.info('Running %s: %s', tool, ' '.join(str(part) for part in cmd))
    tail: Deque[str] = deque(maxlen=_ERROR_CONTEXT_LINES)
    try:
        process = subprocess.Popen(
            [str(part) for part in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,     # both streams, interleaved in order
            text=True,
            bufsize=1,                    # line buffered
        )
    except FileNotFoundError as exc:
        raise SearchError(
            f'{tool} was not found on PATH. Install it (e.g. '
            f'`conda install -c bioconda {tool}`) and try again.'
        ) from exc

    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            tail.append(line)
            logger.info('[%s] %s', tool, line)
    returncode = process.wait()

    if returncode != 0:
        # The tool's own diagnostics are far more useful than the exit code,
        # which is all the previous os.system() call surfaced.
        raise SearchError(
            f'{tool} failed with exit code {returncode}:\n'
            + ('\n'.join(tail) or '(no output)')
        )


#: Query sequences used to probe a database for taxonomy support.
_PROBE_SEQUENCES = 5


def check_database_taxonomy(
    db_path: Path,
    method: str,
    query_file: Optional[Path] = None,
) -> Optional[str]:
    """Confirm the database carries taxonomy, before the search consumes hours.

    A DIAMOND database built without ``--taxonmap`` returns hits with no taxid,
    every one of which FlexiHGT has to discard -- but only *after* the search
    has finished.  At proteome scale that wastes hours; across a manifest it
    wastes days.

    The check runs a real search with a handful of query sequences and looks at
    whether the taxid column comes back populated.  ``diamond dbinfo`` was the
    obvious candidate, but it reports no taxonomy fields at all (checked
    against v2.1.13), so it cannot answer the question.

    Returns a description of the problem, or ``None`` if the database looks
    usable.  Being unable to tell is never treated as a problem: a check that
    cannot reach a verdict must not block the run.
    """
    if method == 'diamond':
        return _probe_diamond_taxonomy(db_path, query_file)
    return _check_mmseqs_taxonomy(db_path)


def _probe_diamond_taxonomy(db_path: Path, query_file: Optional[Path]) -> Optional[str]:
    if query_file is None or not Path(query_file).is_file():
        return None

    probe = _first_sequences(Path(query_file), _PROBE_SEQUENCES)
    if not probe.strip():
        return None

    with tempfile.TemporaryDirectory(prefix='flexihgt-probe-') as tmp:
        tmp_dir = Path(tmp)
        query = tmp_dir / 'probe.faa'
        query.write_text(probe, encoding='utf-8')
        output = tmp_dir / 'probe.tsv'
        try:
            completed = subprocess.run(
                ['diamond', 'blastp', '-d', str(db_path), '-q', str(query),
                 '-o', str(output), '--max-target-seqs', '1', '--threads', '1',
                 '--quiet', '--outfmt', '6', QSEQID, STAXIDS],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug('Taxonomy probe could not run (%s); skipping the check', exc)
            return None

        if completed.returncode != 0:
            message = (completed.stdout or '').strip()
            if _is_missing_taxonomy_error(message):
                # DIAMOND validates the requested output fields before doing any
                # work, so a database without taxonomy fails here in
                # milliseconds -- which is the whole point of probing.
                return (
                    f'{db_path} has no taxonomy, so every hit would be discarded. '
                    'Rebuild it with `diamond makedb --taxonmap '
                    'prot.accession2taxid.gz --taxonnodes nodes.dmp --taxonnames '
                    'names.dmp`, or pass --skip-checks to search anyway.\n'
                    f'diamond said: {message.splitlines()[-1] if message else "(no output)"}'
                )
            # Any other failure is a different problem, and the real search will
            # report it properly.
            logger.debug('Taxonomy probe exited %d; skipping the check: %s',
                         completed.returncode, message)
            return None
        if not output.exists():
            return None

        rows = [line for line in output.read_text(encoding='utf-8').splitlines() if line.strip()]

    if not rows:
        # No hits from a handful of sequences proves nothing either way.
        logger.info('Taxonomy probe found no hits; continuing without a verdict')
        return None

    taxed = sum(1 for row in rows if len(row.split('\t')) > 1 and row.split('\t')[1].strip())
    if taxed == 0:
        return (
            f'{db_path} returned {len(rows)} hit(s), none carrying a taxid, so every '
            'hit would be discarded. Rebuild it with `diamond makedb --taxonmap '
            'prot.accession2taxid.gz --taxonnodes nodes.dmp --taxonnames names.dmp`, '
            'or pass --skip-checks to search anyway.'
        )
    logger.info('Database taxonomy confirmed (%d/%d probe hits carried a taxid)',
                taxed, len(rows))
    return None


def _is_missing_taxonomy_error(message: str) -> bool:
    """Does this tool error say the database lacks taxonomy?

    DIAMOND's wording is ``Options require taxonomy information included in the
    database ... --taxonmap``; matched loosely so a rephrasing in a future
    release still registers.
    """
    lowered = message.lower()
    return 'taxonomy' in lowered and (
        'taxonmap' in lowered or 'require' in lowered or 'not included' in lowered
    )


def _first_sequences(fasta: Path, count: int) -> str:
    """The first ``count`` records of a FASTA file, as text."""
    records: List[str] = []
    current: List[str] = []
    with open(fasta, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('>'):
                if current:
                    records.append(''.join(current))
                    if len(records) >= count:
                        break
                current = [line if line.endswith('\n') else line + '\n']
            elif current:
                current.append(line if line.endswith('\n') else line + '\n')
    if current and len(records) < count:
        records.append(''.join(current))
    return ''.join(records)


def _check_mmseqs_taxonomy(db_path: Path) -> Optional[str]:
    """MMseqs2 stores taxonomy alongside the database as sibling files."""
    markers = [Path(f'{db_path}_mapping'), Path(f'{db_path}_taxonomy')]
    if any(marker.exists() for marker in markers):
        return None
    if not Path(f'{db_path}.dbtype').exists():
        return None                       # not obviously an MMseqs2 database
    return (
        f'{db_path} has no taxonomy files ({", ".join(m.name for m in markers)}). '
        'Create them with `mmseqs createtaxdb`, or pass --skip-checks to search anyway.'
    )


def resolve_database(db_path: Path, method: str) -> Path:
    """Validate a database path, tolerating the usual suffix conventions.

    DIAMOND accepts ``db`` or ``db.dmnd``; MMseqs2 databases are a *prefix*
    with sibling files such as ``db.dbtype``.  The previous strict
    ``db_path.exists()`` check rejected both idioms.
    """
    candidates: List[Path]
    if method == 'diamond':
        candidates = [db_path, db_path.with_suffix(db_path.suffix + '.dmnd')]
    else:
        candidates = [db_path, Path(f'{db_path}.dbtype'), Path(f'{db_path}.index')]

    for candidate in candidates:
        if candidate.exists():
            return db_path
    raise FileNotFoundError(
        f'Database not found: {db_path} (looked for '
        + ', '.join(str(c) for c in candidates) + ')'
    )


def _signature(db_path: Path, method: str, exclude_taxid: Optional[int],
               max_hits: int, evalue: float, input_file: Path) -> Dict[str, object]:
    """Fingerprint of the inputs that determine a search result.

    Only settings that change the *hits* belong here -- thread count, block
    size and tmpdir affect speed, not output, so changing them must not
    invalidate a cached search.
    """
    resolved = db_path.resolve()
    try:
        db_stat = resolved.stat()
        db_size, db_mtime = db_stat.st_size, int(db_stat.st_mtime)
    except OSError:
        # MMseqs2 prefixes are not themselves files.
        db_size, db_mtime = -1, -1
    input_stat = input_file.stat()
    return {
        'method': method,
        'database': str(resolved),
        'database_size': db_size,
        'database_mtime': db_mtime,
        'exclude_taxid': exclude_taxid,
        'max_hits': max_hits,
        'evalue': evalue,
        'input': str(input_file.resolve()),
        'input_size': input_stat.st_size,
        'input_mtime': int(input_stat.st_mtime),
    }


def _sidecar(output_file: Path) -> Path:
    return output_file.with_suffix(output_file.suffix + '.meta.json')


def _reusable(output_file: Path, signature: Dict[str, object]) -> bool:
    """Only reuse results we can prove came from the same inputs.

    Previously *any* non-empty file at the expected path was reused, so a run
    against a different database -- or a job killed halfway through writing --
    would silently produce results attributed to the wrong search.
    """
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    sidecar = _sidecar(output_file)
    if not sidecar.exists():
        logger.info(
            'Ignoring %s: no %s provenance file, cannot confirm it matches this run',
            output_file, sidecar.name,
        )
        return False
    try:
        recorded = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        logger.warning('Ignoring unreadable provenance file %s', sidecar)
        return False
    if recorded != signature:
        differing = sorted(
            key for key in set(recorded) | set(signature)
            if recorded.get(key) != signature.get(key)
        )
        logger.info(
            'Ignoring %s: it was produced with different settings (%s)',
            output_file, ', '.join(differing),
        )
        return False
    return True


@dataclass
class SearchTuning:
    """Knobs that affect search *speed* but not which hits come back."""

    threads: int = 0
    #: DIAMOND -b. The main throughput lever; needs roughly 6x its value in GB
    #: of RAM. 0 leaves DIAMOND's own default in place.
    block_size: float = 0.0
    #: DIAMOND -c. Fewer index chunks means fewer passes over the database.
    index_chunks: int = 0
    #: Scratch space. Worth pointing at local NVMe when the working directory
    #: is on a network filesystem.
    tmpdir: Optional[Path] = None
    #: MMseqs2 --split-memory-limit, e.g. '64G'.
    memory_limit: Optional[str] = None


def run_search(
    input_file: Path,
    db_path: Path,
    *,
    method: str = 'diamond',
    exclude_taxid: Optional[int] = None,
    max_hits: int = 200,
    # Matches HGTParameters.evalue and the CLI default; a looser default here
    # only ever applied to callers that forgot to pass one, and silently
    # produced a much larger hit table than the same run through the CLI.
    evalue: float = 1e-5,
    tuning: Optional[SearchTuning] = None,
    output_file: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Run the configured search tool and return the path to its TSV output."""
    db_path = resolve_database(db_path, method)
    output_file = output_file or input_file.with_suffix('.hits.tsv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tuning = tuning or SearchTuning()

    signature = _signature(db_path, method, exclude_taxid, max_hits, evalue, input_file)
    if not force and _reusable(output_file, signature):
        logger.info('Reusing existing %s search results: %s', method, output_file)
        return output_file

    threads = tuning.threads or (os.cpu_count() or 1)
    scratch_root = str(tuning.tmpdir) if tuning.tmpdir else str(output_file.parent)
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename on success, so an interrupted run
    # never leaves a half-written table that looks complete.
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix='.flexihgt-') as tmp:
        tmp_dir = Path(tmp)
        partial = tmp_dir / output_file.name
        if method == 'diamond':
            _run_diamond(input_file, db_path, partial, tmp_dir, exclude_taxid,
                         max_hits, evalue, threads, tuning)
        elif method == 'mmseqs':
            _run_mmseqs(input_file, db_path, partial, tmp_dir, exclude_taxid,
                        max_hits, evalue, threads, tuning)
        else:
            raise ValueError(f'Unknown search method: {method}')
        shutil.move(str(partial), str(output_file))

    _sidecar(output_file).write_text(json.dumps(signature, indent=2), encoding='utf-8')
    logger.info('Finished %s search. Results saved to %s', method, output_file)
    return output_file


def _run_diamond(input_file: Path, db_path: Path, output_file: Path, tmp_dir: Path,
                 exclude_taxid: Optional[int], max_hits: int, evalue: float,
                 threads: int, tuning: SearchTuning) -> None:
    cmd: List[str] = [
        'diamond', 'blastp',
        '-d', str(db_path),
        '-q', str(input_file),
        '-o', str(output_file),
        '--max-target-seqs', str(max_hits),
        # Only the best HSP per query/subject pair can affect a max-bitscore or
        # min-e-value, so the rest is output we would parse and discard.
        '--max-hsps', '1',
        '-e', repr(evalue),
        '--threads', str(threads),
        '--tmpdir', str(tmp_dir),
    ]
    if exclude_taxid:
        cmd += ['--taxon-exclude', str(exclude_taxid)]
    if tuning.block_size:
        cmd += ['-b', repr(tuning.block_size)]
    if tuning.index_chunks:
        cmd += ['-c', str(tuning.index_chunks)]
    # --outfmt takes a variable-length field list, so it must come last.
    cmd += ['--outfmt', '6', *HIT_COLUMNS]
    _run(cmd, 'diamond')


def _run_mmseqs(input_file: Path, db_path: Path, output_file: Path, tmp_dir: Path,
                exclude_taxid: Optional[int], max_hits: int, evalue: float,
                threads: int, tuning: SearchTuning) -> None:
    """MMseqs2 equivalent of the DIAMOND search.

    Requires a taxonomy-aware target database (one created with
    ``mmseqs createtaxdb``); without it MMseqs2 cannot emit the ``taxid``
    column and the run fails loudly rather than producing untaxed hits.
    """
    mmseqs_tmp = tmp_dir / 'mmseqs_tmp'
    mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        'mmseqs', 'easy-search',
        str(input_file), str(db_path), str(output_file), str(mmseqs_tmp),
        '--format-mode', '0',
        '--format-output', 'query,target,evalue,bits,alnlen,pident,taxid',
        '--max-seqs', str(max_hits),
        '-e', repr(evalue),
        '--threads', str(threads),
    ]
    if exclude_taxid:
        # '!taxid' excludes that clade; hits are also filtered again after
        # loading, so a database without taxonomy support cannot slip through.
        cmd += ['--taxon-list', f'!{exclude_taxid}']
    if tuning.memory_limit:
        cmd += ['--split-memory-limit', tuning.memory_limit]
    _run(cmd, 'mmseqs')
