"""Dependency and input checks run before an analysis starts."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

#: Import name -> pip/conda package name.  Keys are *import* names, which are
#: not always the package name: the old code probed for a module called
#: ``biopython`` (it installs as ``Bio``), so that check always failed.
CORE_DEPENDENCIES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
}

#: Extra requirement per taxonomy backend.  Demanding ete3 regardless of
#: backend made ``--taxonomy-backend taxopy`` -- whose whole point is not
#: needing ete3 -- fail the checks it was supposed to pass.
BACKEND_DEPENDENCIES = {
    'ete3': {'ete3': 'ete3'},
    'taxopy': {'taxopy': 'taxopy'},
}

#: What a run needs with the default backend.
PYTHON_DEPENDENCIES = {**CORE_DEPENDENCIES, **BACKEND_DEPENDENCIES['ete3']}

#: Residues accepted in a protein FASTA: the 20 standard amino acids plus the
#: IUPAC ambiguity codes, stop/gap characters and the rarer selenocysteine and
#: pyrrolysine.  The previous check validated protein input against 'ACGT',
#: which rejected every real proteome.
PROTEIN_ALPHABET = set('ACDEFGHIKLMNPQRSTVWYBZXJUO*-.')
NUCLEOTIDE_ALPHABET = set('ACGTUNRYKMSWBDHVN-.')


def check_search_tool(method: str) -> bool:
    """Check that the search tool for ``method`` is installed and runnable."""
    tool = {'diamond': 'diamond', 'mmseqs': 'mmseqs'}.get(method)
    if tool is None:
        logger.error('Unknown search method: %s', method)
        return False
    if shutil.which(tool) is None:
        logger.error(
            '%s not found on PATH. Install it with `conda install -c bioconda %s`.',
            tool, tool,
        )
        return False
    try:
        subprocess.run(
            [tool, '--version'],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.error('%s is on PATH but could not be executed: %s', tool, exc)
        return False
    return True


def check_python_dependencies(taxonomy_backend: str = 'ete3') -> bool:
    """Check the importable Python dependencies, reporting all that are missing."""
    required = {**CORE_DEPENDENCIES, **BACKEND_DEPENDENCIES.get(taxonomy_backend, {})}
    missing = [
        package for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        logger.error('Missing Python package(s): %s. Install with `pip install %s`.',
                     ', '.join(missing), ' '.join(missing))
        return False
    return True


def ete3_database_path() -> Path:
    """Location of the ete3 taxonomy SQLite database."""
    try:
        from ete3.ncbi_taxonomy.ncbiquery import DEFAULT_TAXADB  # type: ignore

        return Path(DEFAULT_TAXADB)
    except Exception:  # noqa: BLE001 - ete3 missing or restructured
        # ete3's documented default. os.path.expanduser does not translate
        # backslashes, so the old '~\\.etetoolkit\\taxa.sqlite' probe never
        # matched on any platform.
        return Path(os.path.expanduser('~')) / '.etetoolkit' / 'taxa.sqlite'


def check_taxonomy_database() -> bool:
    path = ete3_database_path()
    if path.exists() and path.stat().st_size > 0:
        return True
    logger.error(
        'NCBI taxonomy database not found at %s. Run `flexihgt --update-only` '
        '(or `ete3 ncbiquery --update`) to download it.',
        path,
    )
    return False


#: Files taxopy reads a taxonomy from.
TAXOPY_DUMPS = ('nodes.dmp', 'names.dmp')


def check_taxopy_database(taxdb_dir: Optional[str] = None) -> bool:
    """Check the dumps taxopy needs, when we have been told where to look.

    With no ``--taxonomy-dir`` taxopy downloads its own copy on first use, so
    there is nothing to verify and the check must not block the run -- the same
    rule the database taxonomy probe follows.
    """
    if not taxdb_dir:
        return True
    directory = Path(taxdb_dir)
    if not directory.is_dir():
        logger.error('Taxonomy directory not found: %s', directory)
        return False
    missing = [name for name in TAXOPY_DUMPS if not (directory / name).is_file()]
    if missing:
        logger.error(
            '%s is missing %s. Download them from '
            'https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz and unpack '
            'them there.',
            directory, ', '.join(missing),
        )
        return False
    return True


def sniff_fasta(input_file: Path) -> bool:
    """Sanity-check that ``input_file`` looks like a protein FASTA file."""
    input_file = Path(input_file)
    if input_file.suffix.lower() not in {'.fasta', '.faa', '.fa', '.fas', '.pep'}:
        logger.warning('Unexpected FASTA extension %s; continuing anyway', input_file.suffix)

    try:
        text_lines = _first_lines(input_file, limit=200)
    except OSError as exc:
        logger.error('Could not read %s: %s', input_file, exc)
        return False

    if not text_lines:
        logger.error('Input file is empty: %s', input_file)
        return False
    if not text_lines[0].startswith('>'):
        logger.error('Invalid FASTA header format in %s (first line: %.60s)',
                     input_file, text_lines[0])
        return False

    residues = {
        char.upper()
        for line in text_lines
        if not line.startswith('>')
        for char in line.strip()
    }
    if not residues:
        logger.error('No sequence data found in %s', input_file)
        return False

    invalid = residues - PROTEIN_ALPHABET
    if invalid:
        logger.error('Invalid residue(s) in %s: %s', input_file, ''.join(sorted(invalid)))
        return False
    if residues <= NUCLEOTIDE_ALPHABET and len(residues) <= 6:
        logger.warning(
            '%s looks like nucleotide sequence. FlexiHGT expects protein sequences '
            'for a DIAMOND blastp search; translate it first or use a nucleotide '
            'workflow.',
            input_file,
        )
    return True


def _first_lines(path: Path, limit: int) -> List[str]:
    lines: List[str] = []
    with open(path, encoding='utf-8', errors='replace') as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            lines.append(line.rstrip('\n'))
    return lines


def check_environment(
    search_method: str,
    input_file: Optional[Path] = None,
    need_search_tool: bool = True,
    taxonomy_backend: str = 'ete3',
    taxonomy_dir: Optional[str] = None,
) -> bool:
    """Run every applicable check, reporting *all* problems in one pass.

    Returns ``True`` only when everything passed -- the previous version
    returned ``None`` from its checks, so the success branch was unreachable
    and it always exited with "please install the missing dependencies".

    Which taxonomy is checked follows ``taxonomy_backend``: a taxopy run has no
    reason to own an ete3 ``taxa.sqlite``, and demanding one made the backend
    unusable without ``--skip-checks``.
    """
    checks = [
        ('Python packages', check_python_dependencies(taxonomy_backend)),
    ]
    if taxonomy_backend == 'taxopy':
        checks.append(('taxopy taxonomy dumps', check_taxopy_database(taxonomy_dir)))
    else:
        checks.append(('NCBI taxonomy database', check_taxonomy_database()))
    if need_search_tool:
        checks.append((f'{search_method} executable', check_search_tool(search_method)))
    if input_file is not None:
        checks.append(('input FASTA', sniff_fasta(input_file)))

    failed = [name for name, ok in checks if not ok]
    if failed:
        logger.error('Environment check failed: %s', ', '.join(failed))
        return False
    logger.info('Environment checks passed')
    return True
