"""Command line interface for multi-genome runs (``flexihgt-multi``).

A separate entry point from ``flexihgt``: the single-genome command and its
behaviour are untouched.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .cli import EXIT_FAILED, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE
from .config import ConfigError, apply_config, load_config
from .core import ENGINES, HGTParameters
from .core import select_engine
from .manifest import (
    ManifestError,
    build_manifest,
    iter_protein_ids,
    read_manifest,
    select_genomes,
)
from .multi import MultiGenomeDetect
from .search import SearchError, SearchTuning, check_database_taxonomy, resolve_database
from .taxonomy import TAX_RANKS, get_provider
from .utils import check_environment

logger = logging.getLogger('flexihgt.multi')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='flexihgt-multi',
        description=(
            'Detect horizontal gene transfer across many proteomes from a single '
            'combined search.'
        ),
        epilog='Author: Jack A. Crosby, Aberystwyth University/Queens University Belfast',
    )
    parser.add_argument(
        'manifest',
        help='TSV/CSV listing the genomes: columns taxid and fasta, optionally genome_id. '
             'With --make-manifest, the directory of proteomes to scan instead',
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument(
        '-db', '--database',
        help='Path to the search database (required unless --rescore-only)',
    )

    scoring = parser.add_argument_group('scoring thresholds')
    scoring.add_argument('--bitscore_parameter', type=float, default=100,
                         help='Minimum out-group bitscore (default: %(default)s)')
    scoring.add_argument('--HGTIndex', type=float, default=0.5,
                         help='Minimum HGT index (default: %(default)s)')
    scoring.add_argument('--out_pct', type=float, default=0.8,
                         help='Minimum out-group species fraction (default: %(default)s)')
    scoring.add_argument('--AI', type=float, default=45,
                         help='Minimum Alien Index (default: %(default)s)')
    scoring.add_argument('-t', '--tax_level', default='family', choices=list(TAX_RANKS),
                         metavar='RANK',
                         help='Rank separating in-group from out-group, applied per '
                              'genome (default: %(default)s)')

    search = parser.add_argument_group('search')
    search.add_argument('-s', '--search', default='diamond',
                        choices=list(HGTParameters.SEARCH_METHODS),
                        help='Search method (default: %(default)s)')
    search.add_argument('--max_hits', type=int, default=200,
                        help='Maximum target sequences per query (default: %(default)s)')
    search.add_argument('--evalue', type=float, default=1e-5,
                        help='Search E-value cutoff (default: %(default)s)')
    search.add_argument('--threads', type=int, default=0,
                        help='Threads for the search tool (default: all available)')
    search.add_argument('--block_size', type=float, default=0,
                        help='DIAMOND -b. Needs roughly 6x this value in GB of RAM')
    search.add_argument('--index_chunks', type=int, default=0,
                        help='DIAMOND -c. Fewer chunks means fewer passes over the database')
    search.add_argument('--memory_limit', metavar='SIZE',
                        help="MMseqs2 --split-memory-limit, e.g. '64G'")
    search.add_argument('--tmpdir', metavar='PATH', help='Scratch directory for the search')
    search.add_argument('--force', action='store_true',
                        help='Rebuild the combined FASTA and re-run the search')
    search.add_argument('--rescore-only', action='store_true',
                        help='Skip the search and rescore the existing combined hit table')
    search.add_argument('--engine', default='auto', choices=list(ENGINES),
                        help='Scoring engine: pandas loads the hit table into memory, '
                             'duckdb streams it from disk (default: %(default)s, which '
                             'picks duckdb for tables over 2 GB)')

    output = parser.add_argument_group('output')
    output.add_argument('-o', '--outdir', default='flexihgt_multi',
                        help='Output directory (default: %(default)s)')
    output.add_argument('--work-dir', metavar='PATH',
                        help='Where the combined FASTA and hit table live '
                             '(default: <outdir>/work)')
    output.add_argument('--no-per-genome', action='store_true',
                        help='Write only the combined table, not one file per genome')
    output.add_argument('--top_hits', type=int, default=5,
                        help='Hits per side recorded per candidate (default: %(default)s)')
    output.add_argument('--taxonomy_dump', metavar='PATH',
                        help='Also write the resolved taxonomy to this TSV (for debugging)')
    output.add_argument('--resume', action='store_true',
                        help='Skip genomes whose results already exist, so an '
                             'interrupted run can be restarted')

    parser.add_argument('--genomes', metavar='IDS',
                        help='Restrict to these genome ids: a comma-separated list or a '
                             'file with one id per line. Useful for job arrays')
    parser.add_argument('--taxonomy-backend', default='ete3', choices=['ete3', 'taxopy'],
                        help='Taxonomy source (default: %(default)s)')
    parser.add_argument('--taxonomy-dir', metavar='PATH',
                        help='Directory holding nodes.dmp/names.dmp, for --taxonomy-backend taxopy')
    manifest_group = parser.add_argument_group('manifest building')
    manifest_group.add_argument('--make-manifest', metavar='PATH',
                                help='Scan the given directory of proteomes, write a '
                                     'manifest to this path, and exit')
    manifest_group.add_argument('--taxid-map', metavar='PATH',
                                help='TSV of "id<TAB>taxid" overrides for --make-manifest, '
                                     'for proteomes whose taxid cannot be inferred')

    parser.add_argument('--config', metavar='PATH',
                        help='TOML or JSON file of options. Command line flags win')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what the run would do, then stop')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip the dependency/environment checks, including the '
                             'pre-flight test that the database carries taxonomy')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity. Use WARNING for very large runs, where '
                             'one line per candidate is too much (default: %(default)s)')
    return parser


def _report_plan(records, db_path, params, args) -> None:
    """Describe what a multi-genome run would do, without doing it."""
    work_dir = Path(args.work_dir) if args.work_dir else Path(args.outdir) / 'work'
    hits_path = work_dir / 'combined_query.hits.tsv'
    proteins = 0
    for record in records:
        proteins += sum(1 for _ in iter_protein_ids(record))

    logger.info('--- plan ---')
    logger.info('Genomes:    %d', len(records))
    logger.info('Proteins:   %d', proteins)
    logger.info('Database:   %s', db_path)
    logger.info('Thresholds: bitscore>=%s, HGT index>=%s, out_pct>=%s, AI>=%s at %s level',
                params.bitscore_parameter, params.hgt_index, params.out_pct,
                params.ai_threshold, params.tax_level)
    if hits_path.exists():
        size = hits_path.stat().st_size
        logger.info('Search:     reuse %s (%.2f GB) if its settings match',
                    hits_path, size / 1024 ** 3)
        logger.info('Engine:     %s', select_engine(args.engine, hits_path))
    else:
        estimate = proteins * params.max_hits * 60 / 1024 ** 3
        logger.info('Search:     will run; hit table roughly %.1f GB '
                    '(%d proteins x %d hits)', estimate, proteins, params.max_hits)
        logger.info('Engine:     %s%s', args.engine,
                    ' (duckdb once over 2 GB)' if args.engine == 'auto' else '')
    if args.resume:
        genome_dir = Path(args.outdir) / 'genomes'
        done = len(list(genome_dir.glob('*_HGT.tsv'))) if genome_dir.is_dir() else 0
        logger.info('Resume:     %d genome(s) already scored, %d to do',
                    done, max(0, len(records) - done))
    logger.info('Output:     %s', Path(args.outdir).resolve())


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    if args.config:
        try:
            apply_config(parser, args, load_config(Path(args.config)), argv)
        except ConfigError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return EXIT_USAGE

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    try:
        if args.make_manifest:
            try:
                records, unresolved = build_manifest(
                    Path(args.manifest), Path(args.make_manifest),
                    Path(args.taxid_map) if args.taxid_map else None,
                )
            except ManifestError as exc:
                logger.error('%s', exc)
                return EXIT_USAGE
            logger.info('Wrote %d genome(s) to %s', len(records), args.make_manifest)
            return EXIT_PARTIAL if unresolved else EXIT_OK

        try:
            records = select_genomes(read_manifest(Path(args.manifest)), args.genomes)
        except ManifestError as exc:
            logger.error('%s', exc)
            return EXIT_USAGE

        if not args.database and not args.rescore_only:
            parser.error('-db/--database is required (unless --rescore-only is given)')

        try:
            params = HGTParameters(
                bitscore_parameter=args.bitscore_parameter,
                hgt_index=args.HGTIndex,
                out_pct=args.out_pct,
                ai_threshold=args.AI,
                tax_level=args.tax_level,
                search_method=args.search,
                query_taxid=None,          # supplied per genome by the manifest
                max_hits=args.max_hits,
                evalue=args.evalue,
                threads=args.threads,
                top_hits=args.top_hits,
            )
        except ValueError as exc:
            logger.error('Invalid parameters: %s', exc)
            return EXIT_USAGE

        if not args.skip_checks and not check_environment(
            params.search_method, records[0].fasta,
            need_search_tool=not args.rescore_only,
            taxonomy_backend=args.taxonomy_backend,
            taxonomy_dir=args.taxonomy_dir,
        ):
            return EXIT_USAGE

        db_path = None
        if args.database:
            try:
                db_path = resolve_database(Path(args.database), params.search_method)
            except FileNotFoundError as exc:
                logger.error('%s', exc)
                return EXIT_USAGE

        try:
            taxonomy = get_provider(args.taxonomy_backend, args.taxonomy_dir)
        except Exception as exc:  # noqa: BLE001 - missing optional backend
            logger.error('Could not initialise the %s taxonomy backend: %s',
                         args.taxonomy_backend, exc)
            return EXIT_USAGE

        if db_path is not None and not args.skip_checks:
            problem = check_database_taxonomy(
                db_path, params.search_method, records[0].fasta
            )
            if problem:
                # Failing here costs a second; failing after the search of a
                # whole manifest costs days.
                logger.error('%s', problem)
                return EXIT_USAGE

        logger.info('FlexiHGT %s starting (multi-genome, %d genome(s))',
                    __version__, len(records))
        logger.info('Parameters: %s', params)

        if args.dry_run:
            _report_plan(records, db_path, params, args)
            return EXIT_OK

        start_time = time.time()
        try:
            result = MultiGenomeDetect(params, taxonomy=taxonomy).run(
                records,
                db_path,
                output_dir=Path(args.outdir),
                work_dir=Path(args.work_dir) if args.work_dir else None,
                tuning=SearchTuning(
                    threads=args.threads,
                    block_size=args.block_size,
                    index_chunks=args.index_chunks,
                    tmpdir=Path(args.tmpdir) if args.tmpdir else None,
                    memory_limit=args.memory_limit,
                ),
                force=args.force,
                rescore_only=args.rescore_only,
                per_genome=not args.no_per_genome,
                engine=args.engine,
                resume=args.resume,
                taxonomy_dump=Path(args.taxonomy_dump) if args.taxonomy_dump else None,
            )
        except (SearchError, FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.error('Run failed: %s', exc, exc_info=args.log_level == 'DEBUG')
            return EXIT_FAILED
        except Exception:  # noqa: BLE001 - unexpected, show the traceback
            logger.exception('Run failed with an unexpected error')
            return EXIT_FAILED

        elapsed = time.time() - start_time
        logger.info(
            'Finished in %dh %dm %ds',
            int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60),
        )
        logger.info('%d candidate(s) across %d genome(s); combined table: %s',
                    result.total_candidates, len(result), result.combined_path)

        skipped = [o for o in result.outcomes if o.status == 'skipped']
        if skipped:
            logger.warning('%d genome(s) skipped, e.g. %s', len(skipped),
                           '; '.join(f'{o.genome_id}: {o.note}' for o in skipped[:3]))
        if result.failed or result.total_errors:
            logger.error(
                '%d genome(s) failed and %d gene(s) errored; see %s',
                len(result.failed), result.total_errors, result.summary_path,
            )
            return EXIT_PARTIAL
        if skipped:
            return EXIT_PARTIAL
        return EXIT_OK

    except KeyboardInterrupt:
        logger.warning('Process interrupted by user')
        return 130


if __name__ == '__main__':
    sys.exit(main())
