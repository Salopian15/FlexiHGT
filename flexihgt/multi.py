"""Multi-genome HGT detection.

Runs the existing single-genome pipeline over many proteomes at once, with a
per-genome in-group boundary instead of one global ``--query_tax``.

This module composes :class:`~flexihgt.core.HGTDetect` rather than changing it:
loading, scoring, candidate selection and output are the same tested code, so a
single-genome run behaves exactly as before.  Only two things are genuinely
new -- the search is issued once for every genome combined, and the in-group
test compares each hit against *its own* genome's boundary.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import (
    _IS_RECIPIENT,
    _IS_SYNTHETIC,
    _SPECIES,
    _TAXID,
    AnalysisResult,
    HGTDetect,
    HGTParameters,
    select_engine,
)
from .manifest import GenomeRecord, iter_protein_ids, split_query_id, write_combined_fasta
from .search import QSEQID, SearchTuning, run_search
from .taxonomy import TaxonomyIndex, TaxonomyProvider, build_index, write_index

logger = logging.getLogger(__name__)

#: Genome each hit's query protein came from.
_GENOME = '_genome'

@dataclass
class GenomeOutcome:
    """What happened to one genome."""

    genome_id: str
    taxid: int
    status: str = 'ok'                 # 'ok' | 'skipped' | 'failed'
    note: str = ''
    genes_total: int = 0
    genes_with_hits: int = 0
    candidates: int = 0
    errors: int = 0
    results_path: Optional[Path] = None


@dataclass
class MultiRunResult:
    """Outcome of a whole multi-genome run."""

    outcomes: List[GenomeOutcome] = field(default_factory=list)
    combined_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    hits_path: Optional[Path] = None
    per_genome_dir: Optional[Path] = None

    @property
    def total_candidates(self) -> int:
        return sum(outcome.candidates for outcome in self.outcomes)

    @property
    def total_errors(self) -> int:
        return sum(outcome.errors for outcome in self.outcomes)

    @property
    def failed(self) -> List[GenomeOutcome]:
        return [o for o in self.outcomes if o.status == 'failed']

    def __len__(self) -> int:
        return len(self.outcomes)


class MultiGenomeDetect:
    """Detect HGT across many genomes from a single combined search."""

    SUMMARY_COLUMNS = [
        'Genome', 'TaxID', 'Status', 'Genes', 'Genes with hits',
        'Candidates', 'Errors', 'Note',
    ]

    def __init__(
        self,
        params: HGTParameters,
        taxonomy: Optional[TaxonomyProvider] = None,
    ) -> None:
        if params.query_taxid is not None:
            raise ValueError(
                'query_taxid must be unset for a multi-genome run; the taxid of '
                'each genome comes from the manifest'
            )
        self.params = params
        # All scoring and output goes through the single-genome implementation.
        self.detector = HGTDetect(params, taxonomy=taxonomy)

    @property
    def taxonomy_provider(self) -> TaxonomyProvider:
        return self.detector.taxonomy

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------
    @staticmethod
    def add_genome_column(hits: pd.DataFrame) -> pd.DataFrame:
        """Derive the genome of each hit from its namespaced query id.

        Works on the categorical's *categories* and then remaps codes, so the
        per-row cost is a numpy gather rather than a Python split per hit.
        """
        column = hits[QSEQID]
        if not isinstance(column.dtype, pd.CategoricalDtype):
            column = column.astype('category')
        categories = column.cat.categories
        genome_per_category = pd.Index(
            [split_query_id(value)[0] for value in categories], dtype=object
        )
        genome_categories = pd.Index(genome_per_category.unique(), dtype=object)
        category_to_genome = genome_categories.get_indexer(genome_per_category)

        codes = column.cat.codes.to_numpy()
        genome_codes = np.where(codes >= 0, category_to_genome[codes.clip(min=0)], -1)
        hits[_GENOME] = pd.Categorical.from_codes(genome_codes, categories=genome_categories)
        return hits

    def annotate_hits(
        self,
        hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
        records: Sequence[GenomeRecord],
    ) -> pd.DataFrame:
        """Multi-genome equivalent of :meth:`HGTDetect.annotate_hits`.

        The in-group test becomes a comparison of two mapped columns -- the
        hit's ancestor at ``tax_level`` against its *own genome's* ancestor at
        the same rank -- rather than a comparison against one global taxid.
        Both sides are vectorised maps, so this costs no more than the
        single-genome version.
        """
        hits = self.add_genome_column(hits)
        hits[_TAXID] = self.detector._hit_taxids(hits)  # noqa: SLF001 - shared helper

        unique_taxids = list(hits[_TAXID].cat.categories)
        rank = self.params.tax_level
        hit_boundary = taxonomy.rank_taxid_map(unique_taxids, rank)
        synthetic_map = {taxid: taxonomy.is_synthetic(taxid) for taxid in unique_taxids}
        species_map = taxonomy.rank_taxid_map(unique_taxids, 'species')

        unknown = [taxid for taxid in unique_taxids if taxonomy.get(taxid) is None]
        if unknown:
            logger.warning(
                '%d taxid(s) could not be resolved and their hits will be ignored '
                '(e.g. %s)',
                len(unknown), ', '.join(sorted(unknown)[:5]),
            )
            for taxid in unknown:
                synthetic_map[taxid] = True

        genome_boundary = {
            record.genome_id: taxonomy.rank_taxid(record.taxid, rank)
            for record in records
        }
        genome_species = {
            record.genome_id: self._exclusion_species(record, taxonomy)
            for record in records
        }

        hits[_IS_SYNTHETIC] = hits[_TAXID].map(synthetic_map).astype(bool)
        hits[_SPECIES] = pd.to_numeric(
            hits[_TAXID].map(species_map), errors='coerce'
        ).astype('Int64')

        hit_side = pd.to_numeric(hits[_TAXID].map(hit_boundary), errors='coerce')
        query_side = pd.to_numeric(hits[_GENOME].map(genome_boundary), errors='coerce')
        # NaN on either side means "no ancestor at this rank", which is never a
        # match; the comparison already yields False there.
        hits[_IS_RECIPIENT] = (hit_side == query_side) & hit_side.notna()

        # Self-hits: the combined search cannot use --taxon-exclude, because the
        # taxon to exclude differs per genome. Filtering here instead is exact
        # and costs one vectorised comparison.
        excluded = pd.to_numeric(hits[_GENOME].map(genome_species), errors='coerce')
        self_hits = (
            pd.to_numeric(hits[_SPECIES], errors='coerce') == excluded
        ) & excluded.notna()
        if self_hits.any():
            logger.info('Dropping %d self-hit(s) across all genomes', int(self_hits.sum()))
            hits = hits[~self_hits]

        return hits

    @staticmethod
    def _exclusion_species(record: GenomeRecord, taxonomy: TaxonomyIndex) -> Optional[int]:
        """Species-rank ancestor of a genome's taxid, for self-hit removal."""
        species = taxonomy.rank_taxid(record.taxid, 'species')
        if species is not None:
            return species
        info = taxonomy.get(record.taxid)
        return info.taxid if info else record.taxid

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        records: Sequence[GenomeRecord],
        db_path: Optional[Path] = None,
        output_dir: Path = Path('flexihgt_multi'),
        work_dir: Optional[Path] = None,
        tuning: Optional[SearchTuning] = None,
        force: bool = False,
        rescore_only: bool = False,
        per_genome: bool = True,
        engine: str = 'auto',
        resume: bool = False,
        taxonomy_dump: Optional[Path] = None,
    ) -> MultiRunResult:
        """Search every genome at once, then score and report them separately."""
        if not records:
            raise ValueError('No genomes to analyse')

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(work_dir) if work_dir else output_dir / 'work'
        work_dir.mkdir(parents=True, exist_ok=True)

        combined_fasta = work_dir / 'combined_query.faa'
        counts = write_combined_fasta(records, combined_fasta, force=force)
        if not sum(counts.values()):
            raise ValueError('No sequences found in any of the manifest genomes')

        hits_path = combined_fasta.with_suffix('.hits.tsv')
        if rescore_only:
            if not hits_path.exists():
                raise FileNotFoundError(
                    f'--rescore-only needs existing search results at {hits_path}'
                )
            logger.info('Rescoring existing search results: %s', hits_path)
        else:
            if db_path is None:
                raise ValueError('A database is required unless rescore_only is set')
            hits_path = run_search(
                combined_fasta,
                Path(db_path),
                method=self.params.search_method,
                # Deliberately absent: the clade to exclude differs per genome,
                # so self-hits are removed after loading instead.
                exclude_taxid=None,
                max_hits=self.params.max_hits,
                evalue=self.params.evalue,
                tuning=tuning or SearchTuning(threads=self.params.threads),
                output_file=hits_path,
                force=force,
            )

        if select_engine(engine, hits_path) == 'duckdb':
            return self._run_out_of_core(
                records, records, hits_path, output_dir, per_genome,
                counts, {}, taxonomy_dump, resume,
            )

        hits = self.detector.load_hits(hits_path)
        logger.info('Resolving taxonomy for %d hit(s) across %d genome(s)...',
                    len(hits), len(records))
        taxids = list(self.detector._hit_taxids(hits).unique())  # noqa: SLF001
        taxids += [str(record.taxid) for record in records]
        taxonomy = build_index(self.taxonomy_provider, taxids)

        usable_records, outcomes = self._partition_records(records, taxonomy)
        if not usable_records:
            raise RuntimeError(
                'None of the manifest taxids could be scored: check the taxids, '
                'that the local taxonomy database is current, and that their '
                f'lineages have a {self.params.tax_level}-rank ancestor (the run '
                'summary gives the reason for each genome).'
            )

        hits = self.annotate_hits(hits, taxonomy, usable_records)
        result = self._score_and_write(
            usable_records, records, hits, taxonomy, output_dir, per_genome, outcomes,
            counts, resume,
        )
        result.hits_path = hits_path
        return result

    def _run_out_of_core(
        self,
        records: Sequence[GenomeRecord],
        all_records: Sequence[GenomeRecord],
        hits_path: Path,
        output_dir: Path,
        per_genome: bool,
        counts: Dict[str, int],
        outcomes: Dict[str, GenomeOutcome],
        taxonomy_dump: Optional[Path],
        resume: bool,
    ) -> MultiRunResult:
        """Score a hit table too large to load, one genome at a time.

        The aggregation happens once for the whole table; the per-genome split
        then falls out of the namespaced query ids.
        """
        from . import aggregate

        logger.info('Scoring out of core (duckdb): %s', hits_path)
        taxids = aggregate.scan_taxids(hits_path, threads=self.params.threads)
        taxids += [str(record.taxid) for record in records]
        taxonomy = build_index(self.taxonomy_provider, taxids)

        usable, skipped = self._partition_records(records, taxonomy)
        outcomes.update(skipped)
        if not usable:
            raise RuntimeError(
                'None of the manifest taxids could be scored: check the taxids, '
                'that the local taxonomy database is current, and that their '
                f'lineages have a {self.params.tax_level}-rank ancestor (the run '
                'summary gives the reason for each genome).'
            )
        if taxonomy_dump:
            write_index(taxonomy, taxonomy_dump)

        rank = self.params.tax_level
        lookup = aggregate.taxonomy_table(taxonomy, taxids, rank)
        queries = aggregate.query_table([
            (record.genome_id,
             taxonomy.rank_taxid(record.taxid, rank),
             self._exclusion_species(record, taxonomy))
            for record in usable
        ])

        genome_dir, scratch = self._genome_dir(output_dir, per_genome)
        try:
            with aggregate.HitTable(
                hits_path, lookup, queries=queries, threads=self.params.threads,
            ) as table:
                per_genome_counts = table.gene_counts_by_genome()
                candidates, errors, _ = self.detector.process_genes_out_of_core(
                    table, taxonomy,
                )

            by_genome: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidates:
                genome, _ = split_query_id(candidate['gene'])
                by_genome.setdefault(genome, []).append(candidate)
            errors_by_genome: Dict[str, int] = {}
            for gene, _message in errors:
                genome, _ = split_query_id(gene)
                errors_by_genome[genome] = errors_by_genome.get(genome, 0) + 1

            for record in usable:
                results_path = genome_dir / self._genome_filename(record)
                if resume:
                    reused = self._reused_outcome(record, results_path, counts)
                    if reused is not None:
                        # The aggregation covers the whole table either way, but
                        # a genome the user has already accepted results for must
                        # not be silently rewritten.
                        outcomes[record.genome_id] = reused
                        continue
                found = by_genome.get(record.genome_id, [])
                frame = pd.DataFrame(
                    found, columns=['gene', 'scores', 'taxonomy', 'top_hits']
                )
                analysis = AnalysisResult(
                    candidates=frame,
                    genes_total=counts.get(record.genome_id, 0),
                    genes_with_hits=per_genome_counts.get(record.genome_id, 0),
                )
                self.detector.write_results(analysis, results_path)
                outcomes[record.genome_id] = GenomeOutcome(
                    genome_id=record.genome_id,
                    taxid=record.taxid,
                    genes_total=analysis.genes_total,
                    genes_with_hits=analysis.genes_with_hits,
                    candidates=len(frame),
                    errors=errors_by_genome.get(record.genome_id, 0),
                    results_path=results_path,
                )

            ordered = [
                outcomes[record.genome_id]
                for record in all_records if record.genome_id in outcomes
            ]
            combined_path = output_dir / f'combined_{rank}_HGT.tsv'
            self._write_combined(ordered, combined_path)
            summary_path = output_dir / 'run_summary.tsv'
            self._write_summary(ordered, summary_path)
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)

        return MultiRunResult(
            outcomes=ordered,
            combined_path=combined_path,
            summary_path=summary_path,
            hits_path=hits_path,
            per_genome_dir=genome_dir if per_genome else None,
        )

    def _genome_filename(self, record: GenomeRecord) -> str:
        return f'{record.genome_id}_{self.params.tax_level}_HGT.tsv'

    @staticmethod
    def _gene_ids(
        record: GenomeRecord,
        genome_hits: pd.DataFrame,
        counts: Dict[str, int],
    ) -> List[str]:
        """Query ids for one genome, preferring the hit table over its FASTA.

        Genes with no hits cannot be candidates, so the hit table is enough for
        scoring; ``counts`` (captured while building the combined FASTA) supplies
        the totals.  Falling back to the FASTA keeps the method usable when a
        caller has no counts.
        """
        if not genome_hits.empty:
            ids = genome_hits[QSEQID]
            if isinstance(ids.dtype, pd.CategoricalDtype):
                ids = ids.cat.remove_unused_categories().cat.categories
                return list(ids)
            return list(dict.fromkeys(ids))
        if counts.get(record.genome_id):
            return []                      # has sequences, just no hits
        return list(iter_protein_ids(record))

    @staticmethod
    def _genome_dir(output_dir: Path, per_genome: bool) -> Tuple[Path, Optional[str]]:
        """Where per-genome tables go, and any scratch dir to clean up after.

        Per-genome files are always written, then concatenated into the
        combined table. Reusing HGTDetect.write_results keeps a single
        implementation of the output format; if the caller does not want the
        individual files, they are produced in a scratch directory instead.
        """
        if per_genome:
            genome_dir, scratch = output_dir / 'genomes', None
        else:
            scratch = tempfile.mkdtemp(prefix='flexihgt-multi-', dir=str(output_dir))
            genome_dir = Path(scratch)
        genome_dir.mkdir(parents=True, exist_ok=True)
        return genome_dir, scratch

    @staticmethod
    def _reused_outcome(
        record: GenomeRecord,
        results_path: Path,
        counts: Dict[str, int],
    ) -> Optional[GenomeOutcome]:
        """The outcome for an already-written genome under ``--resume``.

        A long run must be interruptible; a genome already written is a genome
        already scored.  ``None`` means there is nothing to reuse.
        """
        if not results_path.exists():
            return None
        logger.info('Skipping %s: results already present', record.genome_id)
        with results_path.open(encoding='utf-8') as handle:
            written = max(0, sum(1 for _ in handle) - 1)   # minus the header
        return GenomeOutcome(
            genome_id=record.genome_id, taxid=record.taxid, status='ok',
            note='reused existing results',
            genes_total=counts.get(record.genome_id, 0),
            candidates=written,
            results_path=results_path,
        )

    def _partition_records(
        self,
        records: Sequence[GenomeRecord],
        taxonomy: TaxonomyIndex,
    ) -> Tuple[List[GenomeRecord], Dict[str, GenomeOutcome]]:
        """Split off genomes that cannot be scored at the requested rank.

        One bad taxid in a manifest of thousands must not abort the run; it is
        recorded as skipped and reported in the summary.

        A genome whose lineage has no ancestor at ``tax_level`` is skipped for
        the same reason: nothing could ever be in-group for it, so every one of
        its genes would score as if it had no recipient hits and be reported as
        a candidate.  Ranks are patchy in NCBI taxonomy, so across a large
        manifest this is expected rather than exceptional.
        """
        rank = self.params.tax_level
        usable: List[GenomeRecord] = []
        outcomes: Dict[str, GenomeOutcome] = {}
        for record in records:
            if taxonomy.get(record.taxid) is None:
                logger.warning(
                    'Skipping %s: taxid %s could not be resolved',
                    record.genome_id, record.taxid,
                )
                outcomes[record.genome_id] = GenomeOutcome(
                    genome_id=record.genome_id, taxid=record.taxid,
                    status='skipped', note='taxid could not be resolved',
                )
                continue
            if taxonomy.rank_taxid(record.taxid, rank) is None:
                logger.warning(
                    'Skipping %s: taxid %s has no %s-rank ancestor, so no hit '
                    'could be classified as in-group',
                    record.genome_id, record.taxid, rank,
                )
                outcomes[record.genome_id] = GenomeOutcome(
                    genome_id=record.genome_id, taxid=record.taxid,
                    status='skipped', note=f'no {rank}-rank ancestor',
                )
                continue
            usable.append(record)
        return usable, outcomes

    def _score_and_write(
        self,
        records: Sequence[GenomeRecord],
        all_records: Sequence[GenomeRecord],
        hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
        output_dir: Path,
        per_genome: bool,
        outcomes: Dict[str, GenomeOutcome],
        counts: Optional[Dict[str, int]] = None,
        resume: bool = False,
    ) -> MultiRunResult:
        """Score each genome and write per-genome, combined and summary tables."""
        grouped = hits.groupby(_GENOME, observed=True, sort=False)
        empty = hits.iloc[0:0]
        counts = counts or {}
        genome_dir, scratch = self._genome_dir(output_dir, per_genome)

        ordered: List[GenomeOutcome] = []
        try:
            for record in records:
                outcome = self._score_one(
                    record, grouped, empty, taxonomy, genome_dir, counts, resume
                )
                outcomes[record.genome_id] = outcome

            # Report in manifest order, skipped genomes included.
            ordered = [
                outcomes[record.genome_id]
                for record in all_records
                if record.genome_id in outcomes
            ]

            combined_path = output_dir / f'combined_{self.params.tax_level}_HGT.tsv'
            self._write_combined(ordered, combined_path)
            summary_path = output_dir / 'run_summary.tsv'
            self._write_summary(ordered, summary_path)
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)

        logger.info(
            'Multi-genome run complete: %d candidate(s) across %d genome(s)',
            sum(o.candidates for o in ordered), len(ordered),
        )
        return MultiRunResult(
            outcomes=ordered,
            combined_path=combined_path,
            summary_path=summary_path,
            per_genome_dir=genome_dir if per_genome else None,
        )

    def _score_one(
        self,
        record: GenomeRecord,
        grouped,
        empty: pd.DataFrame,
        taxonomy: TaxonomyIndex,
        genome_dir: Path,
        counts: Dict[str, int],
        resume: bool = False,
    ) -> GenomeOutcome:
        """Score a single genome using the single-genome implementation."""
        results_path = genome_dir / self._genome_filename(record)
        if resume:
            reused = self._reused_outcome(record, results_path, counts)
            if reused is not None:
                return reused
        try:
            genome_hits = grouped.get_group(record.genome_id)
        except KeyError:
            genome_hits = empty                     # no hits for this genome at all

        if not genome_hits.empty:
            # A slice keeps every category of the parent frame, so this genome's
            # few hundred query ids would still carry the whole run's hundreds of
            # thousands. Every groupby below scales with that count, which turns
            # the per-genome loop quadratic in the number of genomes.
            genome_hits = genome_hits.copy()
            genome_hits[QSEQID] = genome_hits[QSEQID].cat.remove_unused_categories()

        try:
            # Gene ids come from the hit table where possible: every proteome was
            # already read once to build the combined FASTA, and re-reading
            # thousands of them to recover ids that are recoverable here would
            # double the file I/O for no gain.
            genes = self._gene_ids(record, genome_hits, counts)
            if not genes and not counts.get(record.genome_id):
                # No hits *and* no sequences: the proteome itself is unusable.
                # A genome with sequences but no hits is a normal, reportable
                # outcome, not a skip.
                return GenomeOutcome(
                    genome_id=record.genome_id, taxid=record.taxid,
                    status='skipped', note='no sequences in FASTA',
                )
            candidates, errors, with_hits = self.detector.process_genes(
                genes, genome_hits, taxonomy
            )
        except Exception as exc:  # noqa: BLE001 - one genome must not kill the run
            logger.error('Genome %s failed: %s', record.genome_id, exc, exc_info=True)
            return GenomeOutcome(
                genome_id=record.genome_id, taxid=record.taxid,
                status='failed', note=str(exc),
            )

        frame = pd.DataFrame(candidates, columns=['gene', 'scores', 'taxonomy', 'top_hits'])
        genes_total = counts.get(record.genome_id) or len(genes)
        analysis = AnalysisResult(
            candidates=frame,
            genes_total=genes_total,
            genes_with_hits=with_hits,
            gene_errors=errors,
        )
        self.detector.write_results(analysis, results_path)

        return GenomeOutcome(
            genome_id=record.genome_id,
            taxid=record.taxid,
            genes_total=genes_total,
            genes_with_hits=with_hits,
            candidates=len(frame),
            errors=len(errors),
            results_path=results_path,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _write_combined(self, outcomes: Sequence[GenomeOutcome], path: Path) -> None:
        """Concatenate the per-genome tables, prefixed with a Genome column.

        Built by streaming the files just written rather than re-formatting the
        rows, so the combined table cannot drift from the per-genome ones.
        """
        header_written = False
        with open(path, 'w', encoding='utf-8') as out:
            for outcome in outcomes:
                if outcome.results_path is None or not outcome.results_path.exists():
                    continue
                with open(outcome.results_path, encoding='utf-8') as handle:
                    header = handle.readline().rstrip('\n')
                    if not header_written:
                        out.write('Genome\t' + header + '\n')
                        header_written = True
                    for line in handle:
                        row = line.rstrip('\n')
                        if row.strip():
                            out.write(outcome.genome_id + '\t' + row + '\n')
        if not header_written:
            # Nothing passed anywhere; still emit a well-formed header.
            path.write_text(
                '\t'.join(['Genome', *HGTDetect.RESULT_COLUMNS]) + '\n', encoding='utf-8'
            )
        logger.info('Combined results written to %s', path)

    def _write_summary(self, outcomes: Sequence[GenomeOutcome], path: Path) -> None:
        import csv

        with open(path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(self.SUMMARY_COLUMNS)
            for outcome in outcomes:
                writer.writerow([
                    outcome.genome_id,
                    outcome.taxid,
                    outcome.status,
                    outcome.genes_total,
                    outcome.genes_with_hits,
                    outcome.candidates,
                    outcome.errors,
                    outcome.note,
                ])
        logger.info('Run summary written to %s', path)
