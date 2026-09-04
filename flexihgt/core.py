"""HGT detection: scoring, filtering and the analysis pipeline.
Reworked from the last version, some stuff is in different files now.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .search import (
    BITSCORE,
    EVALUE,
    HIT_COLUMNS,
    PIDENT,
    QSEQID,
    SSEQID,
    STAXIDS,
    SearchTuning,
    run_search,
)
from .taxonomy import (
    TAX_RANKS,
    Ete3Taxonomy,
    TaxonomyIndex,
    TaxonomyProvider,
    build_index,
    write_index,
)

logger = logging.getLogger(__name__)

#: Internal columns added to the hit table during preprocessing.
_TAXID = '_taxid'
_IS_RECIPIENT = '_is_recipient'
_IS_SYNTHETIC = '_is_synthetic'
_SPECIES = '_species'

#: Columns produced by :meth:`HGTDetect.score_all_genes`, in HGTScores order.
_SCORE_COLUMNS = [
    'max_outgroup_bitscore', 'max_recipient_bitscore', 'hgt_index', 'out_pct',
    'alien_index', 'outgroup_count', 'recipient_count', 'min_outgroup_evalue',
    'min_recipient_evalue', 'no_recipient_hits',
]

#: Bumped when the annotated-hit layout changes, so old caches are ignored.
_ANNOTATION_CACHE_VERSION = 2

#: E-value substituted for "no hit at all" when computing the Alien Index for a
#: gene with no in-group hits.  1.0 is the worst score BLAST/DIAMOND report, so
#: this is the conventional cap rather than an arbitrary constant.
NO_HIT_EVALUE = 1.0


#: Hit tables above this size use the out-of-core engine when ``engine='auto'``
#: and DuckDB is installed. Below it, pandas is faster (no query planning).
AUTO_ENGINE_BYTES = 2 * 1024 ** 3

ENGINES = ('auto', 'pandas', 'duckdb')


def select_engine(engine: str, hits_path: Path) -> str:
    """Resolve ``auto`` to a concrete engine for this hit table."""
    if engine not in ENGINES:
        raise ValueError(f'Unknown engine {engine!r}; expected one of {", ".join(ENGINES)}')

    from . import aggregate

    if engine == 'duckdb':
        if not aggregate.available():
            raise RuntimeError(
                "engine='duckdb' needs DuckDB: pip install 'flexihgt[bigdata]'"
            )
        return 'duckdb'
    if engine == 'pandas':
        return 'pandas'

    size = Path(hits_path).stat().st_size
    if size < AUTO_ENGINE_BYTES:
        return 'pandas'
    if not aggregate.available():
        logger.warning(
            'Hit table is %.1f GB but DuckDB is not installed, so it must be loaded '
            "into memory. Install it with: pip install 'flexihgt[bigdata]'",
            size / 1024 ** 3,
        )
        return 'pandas'
    logger.info('Hit table is %.1f GB; using the out-of-core engine', size / 1024 ** 3)
    return 'duckdb'


def compute_scores(
    genes: pd.Index,
    *,
    out_bits: pd.Series,
    rec_bits: pd.Series,
    out_evalue: pd.Series,
    rec_evalue: pd.Series,
    out_species: pd.Series,
    rec_species: pd.Series,
    e_minus: float,
) -> pd.DataFrame:
    """Turn per-gene, per-side aggregates into the four HGT scores.

    The arithmetic lives here, on plain arrays, so the in-memory and the
    out-of-core engines cannot drift apart: both produce the same six
    aggregates and hand them to this function.  ``NaN`` on the recipient side
    means the gene has no in-group hits.
    """
    no_recipient = rec_bits.isna().to_numpy()
    rec_bits_filled = rec_bits.fillna(0.0).to_numpy(dtype=float)
    rec_evalue_filled = rec_evalue.fillna(NO_HIT_EVALUE).to_numpy(dtype=float)
    out_bits_values = out_bits.to_numpy(dtype=float)

    ratio = np.divide(
        out_bits_values, rec_bits_filled,
        out=np.ones(len(genes), dtype=float), where=rec_bits_filled > 0,
    )
    hgt_index = np.where(no_recipient, 1.0, ratio)

    out_species_values = out_species.to_numpy(dtype=float)
    total_species = out_species_values + rec_species.to_numpy(dtype=float)
    out_pct = np.where(
        total_species > 0,
        np.divide(
            out_species_values, total_species,
            out=np.zeros(len(genes), dtype=float), where=total_species > 0,
        ),
        # Neither side has species-rank annotation; fall back to the hit split
        # so a gene is not silently given out_pct == 0.
        np.where(no_recipient, 1.0, 0.0),
    )

    alien_index = (
        np.log(rec_evalue_filled + e_minus)
        - np.log(out_evalue.to_numpy(dtype=float) + e_minus)
    )

    return pd.DataFrame({
        'max_outgroup_bitscore': out_bits_values,
        'max_recipient_bitscore': rec_bits_filled,
        'hgt_index': hgt_index,
        'out_pct': out_pct,
        'alien_index': alien_index,
        'outgroup_count': out_species_values,
        'recipient_count': rec_species.to_numpy(),
        'min_outgroup_evalue': out_evalue.to_numpy(),
        'min_recipient_evalue': rec_evalue_filled,
        'no_recipient_hits': no_recipient,
    }, index=genes)


class HGTScores(NamedTuple):
    """Scores summarising the evidence for HGT at a single gene."""

    max_outgroup_bitscore: float
    max_recipient_bitscore: float
    hgt_index: float
    out_pct: float
    alien_index: float
    outgroup_count: int
    recipient_count: int
    min_outgroup_evalue: float
    min_recipient_evalue: float
    #: True when the gene has no in-group hits at all -- the strongest possible
    #: signal, previously discarded because it produced all-zero scores.
    no_recipient_hits: bool = False


@dataclass
class HGTParameters:
    """Configuration for HGT detection."""

    bitscore_parameter: float = 100
    hgt_index: float = 0.5
    out_pct: float = 0.8
    ai_threshold: float = 45
    tax_level: str = 'family'
    search_method: str = 'diamond'
    e_minus: float = 1e-200
    query_taxid: Optional[int] = None
    max_hits: int = 200
    #: Search-tool E-value cutoff. Tighter than DIAMOND's 0.001 default, which
    #: keeps a proteome-scale hit table substantially smaller.
    evalue: float = 1e-5
    threads: int = 0
    top_hits: int = 5

    SEARCH_METHODS = ('diamond', 'mmseqs')

    def __post_init__(self) -> None:
        if not 0 <= self.hgt_index <= 1:
            raise ValueError('HGT index must be between 0 and 1')
        if not 0 <= self.out_pct <= 1:
            raise ValueError('Out percentage must be between 0 and 1')
        if self.query_taxid is not None and self.query_taxid <= 0:
            raise ValueError('Query taxid must be positive')
        if self.tax_level not in TAX_RANKS:
            raise ValueError(
                f'Unknown taxonomic level {self.tax_level!r}; '
                f'expected one of {", ".join(TAX_RANKS)}'
            )
        if self.search_method not in self.SEARCH_METHODS:
            raise ValueError(
                f'Unknown search method {self.search_method!r}; '
                f'expected one of {", ".join(self.SEARCH_METHODS)}'
            )
        if self.e_minus <= 0:
            raise ValueError('e_minus must be positive')
        if self.max_hits <= 0:
            raise ValueError('max_hits must be positive')
        if not 0 < self.evalue <= 10:
            raise ValueError('evalue must be greater than 0 and at most 10')
        if self.top_hits < 0:
            raise ValueError('top_hits must not be negative')


@dataclass
class AnalysisResult:
    """Outcome of a full run, including what went wrong along the way."""

    candidates: pd.DataFrame
    genes_total: int = 0
    genes_with_hits: int = 0
    #: Genes whose every hit was a synthetic construct, and so had nothing left
    #: to score.  ``None`` means "not counted": the out-of-core engine discards
    #: synthetic rows in SQL, and counting them there would cost a second pass
    #: over a file chosen for that engine precisely because it is enormous.
    genes_all_synthetic: Optional[int] = None
    gene_errors: List[Tuple[str, str]] = field(default_factory=list)
    results_path: Optional[Path] = None
    top_hits_path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.candidates)


class HGTDetect:
    """Detect HGT candidates in a set of protein sequences."""

    TAX_RANKS = TAX_RANKS

    def __init__(
        self,
        params: Optional[HGTParameters] = None,
        taxonomy: Optional[TaxonomyProvider] = None,
    ) -> None:
        self.params = params or HGTParameters()
        # Injectable so the scoring logic can be tested without ete3's 500 MB
        # taxonomy database; constructed lazily otherwise.
        self._taxonomy = taxonomy

    @property
    def taxonomy(self) -> TaxonomyProvider:
        if self._taxonomy is None:
            self._taxonomy = Ete3Taxonomy()
        return self._taxonomy

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    @staticmethod
    def read_fasta_ids(fasta_path: Path) -> Iterator[str]:
        """Stream sequence identifiers from a FASTA file.

        Scans for header lines directly rather than going through
        ``SeqIO.parse``, which would parse and immediately discard every
        sequence; only the identifiers are used downstream.
        """
        with open(fasta_path, encoding='UTF-8') as handle:
            for line in handle:
                if not line.startswith('>'):
                    continue
                # Biopython's record.id is the first whitespace-delimited token.
                identifier = line[1:].split(None, 1)
                if identifier:
                    yield identifier[0]

    def load_hits(self, hits_path: Path) -> pd.DataFrame:
        """Read a search result table with explicit names and dtypes.

        The taxid column is read as text unconditionally.  Left to its own
        devices pandas infers ``int64`` whenever every hit carries a single
        numeric taxid, and the ``.str`` accessor then raises on every row --
        which surfaced as "0 HGT events found" rather than as an error.

        ``length``/``pident`` are narrowed to float32 because they play no part
        in scoring.  ``evalue`` and ``bitscore`` stay float64: float32's
        smallest normal value is ~1e-38, so significant e-values would flush to
        zero and silently corrupt the Alien Index.
        """
        hits = self._read_hit_csv(hits_path)
        if hits.empty:
            raise ValueError(f'Search results are empty: {hits_path}')

        for column in (EVALUE, BITSCORE):
            # Forced to float64: integral bitscores would otherwise land in an
            # int column and turn later ratio arithmetic into integer division.
            hits[column] = pd.to_numeric(hits[column], errors='coerce').astype('float64')
        hits[PIDENT] = pd.to_numeric(hits[PIDENT], errors='coerce').astype('float32')

        unscored = hits[BITSCORE].isna() | hits[EVALUE].isna()
        if unscored.any():
            logger.warning('Dropping %d hit(s) with unparseable e-value/bitscore', int(unscored.sum()))
            hits = hits[~unscored]

        untaxed = hits[STAXIDS].isna()
        if untaxed.any():
            logger.warning(
                'Dropping %d hit(s) with no taxid; is the database taxonomy-aware?',
                int(untaxed.sum()),
            )
            hits = hits[~untaxed]

        if hits.empty:
            raise ValueError(f'No usable hits with taxonomy information in {hits_path}')

        hits = hits.reset_index(drop=True)
        # Query ids repeat max_hits times each, so a categorical cuts memory
        # several-fold and makes the per-gene groupby substantially faster.
        # Subject ids are deliberately left as plain strings: they are close to
        # unique, so a categorical would save nothing and would make every
        # row-wise access convert the whole category array.
        hits[QSEQID] = hits[QSEQID].astype('category')
        return hits

    #: Columns actually read from the search output. ``length`` is requested
    #: from the search tool (it costs nothing there and keeps the format
    #: conventional) but never used by any score or output column, so parsing
    #: it for every one of millions of rows is pure waste.
    USED_COLUMNS = [QSEQID, SSEQID, EVALUE, BITSCORE, PIDENT, STAXIDS]

    @staticmethod
    def _read_hit_csv(hits_path: Path) -> pd.DataFrame:
        """Read the hit table, preferring pyarrow's multithreaded CSV reader.

        pyarrow parses these tables around 2.3x faster than the default engine.
        It does not support ``usecols`` alongside ``names`` (it looks for the
        generated ``f0..fN`` labels and fails), so the unused column is dropped
        after the read instead -- still well ahead of the default engine even
        having parsed one extra column.

        Both readers are given the same NA handling: on pandas' defaults a
        protein or accession literally called ``NA``/``null`` parses as missing,
        so whether such a hit survived depended on whether pyarrow happened to
        be installed.  Only a genuinely empty field counts as missing.

        Any failure falls back rather than breaking the run.
        """
        dtypes = {QSEQID: str, SSEQID: str, STAXIDS: str}
        try:
            hits = pd.read_csv(
                hits_path, sep='\t', header=None, names=HIT_COLUMNS,
                dtype=dtypes, engine='pyarrow',
                keep_default_na=False, na_values=[''],
            )
            return hits.drop(columns=[
                name for name in hits.columns if name not in HGTDetect.USED_COLUMNS
            ])
        except Exception as exc:  # noqa: BLE001 - pyarrow missing or unusable here
            logger.debug('pyarrow CSV reader unavailable (%s); using the default engine', exc)
        return pd.read_csv(
            hits_path, sep='\t', header=None, names=HIT_COLUMNS,
            usecols=HGTDetect.USED_COLUMNS, dtype=dtypes,
            keep_default_na=False, na_values=[''],
        )

    @staticmethod
    def _hit_taxids(hits: pd.DataFrame) -> pd.Series:
        """Last taxid of each (possibly ``;``-separated) staxids field.

        ``rsplit(n=1)`` stops at the first separator from the right instead of
        splitting the whole field, and the work is done once per *distinct*
        staxids value -- taxids repeat across millions of hits.

        Distinct fields routinely collapse onto the same taxid: ``562;1280``
        and ``1280`` both end at 1280.  ``rename_categories`` cannot express
        that (categories must stay unique, and it raises), so the codes are
        remapped instead -- still one pass over the categories rather than over
        the rows.
        """
        column = hits[STAXIDS].astype('category')
        cleaned = pd.Index(
            column.cat.categories.to_series()
            .str.rsplit(';', n=1).str[-1].str.strip()
            .to_list(),
            dtype=object,
        )
        categories = pd.Index(cleaned.unique(), dtype=object)
        old_to_new = categories.get_indexer(cleaned)

        codes = column.cat.codes.to_numpy()
        # Code -1 is "missing" and must survive the remap unchanged; clip keeps
        # it from indexing off the end of the mapping array.
        new_codes = np.where(codes >= 0, old_to_new[codes.clip(min=0)], -1)
        return pd.Series(
            pd.Categorical.from_codes(new_codes, categories=categories),
            index=hits.index,
        )

    def annotate_hits(self, hits: pd.DataFrame, taxonomy: TaxonomyIndex) -> pd.DataFrame:
        """Attach taxid / in-group / synthetic / species columns in one pass.

        Every classification is derived per *unique taxid* and applied with a
        vectorised ``map``, so cost scales with the number of distinct taxa
        rather than the number of hits.  The frame is mutated in place -- the
        caller hands over ownership -- because copying a proteome-scale hit
        table doubles peak memory for no benefit.
        """
        hits[_TAXID] = self._hit_taxids(hits)

        unique_taxids = list(hits[_TAXID].cat.categories)
        recipient_map = {
            taxid: taxonomy.share_rank(self.params.query_taxid, taxid, self.params.tax_level)
            for taxid in unique_taxids
        }
        synthetic_map = {taxid: taxonomy.is_synthetic(taxid) for taxid in unique_taxids}
        species_map = taxonomy.rank_taxid_map(unique_taxids, 'species')

        # Taxids the taxonomy database could not resolve are unusable either
        # way, so treat them like synthetic records and drop them.
        unknown = [taxid for taxid in unique_taxids if taxonomy.get(taxid) is None]
        if unknown:
            logger.warning(
                '%d taxid(s) could not be resolved and their hits will be ignored '
                '(e.g. %s)',
                len(unknown), ', '.join(sorted(unknown)[:5]),
            )
            for taxid in unknown:
                synthetic_map[taxid] = True

        hits[_IS_RECIPIENT] = hits[_TAXID].map(recipient_map).astype(bool)
        hits[_IS_SYNTHETIC] = hits[_TAXID].map(synthetic_map).astype(bool)
        # Species-rank ancestor per hit; NA where the taxon has none. Counting
        # distinct values of this column is what out_pct is built from.
        hits[_SPECIES] = pd.to_numeric(
            hits[_TAXID].map(species_map), errors='coerce'
        ).astype('Int64')

        # Belt and braces: the search tool is asked to exclude the query taxon,
        # but a database without taxonomy filtering support would ignore that.
        # Compare at species rank so sister *strains* of the query organism are
        # excluded too, not just an exact taxid match.
        query_species = self._query_exclusion_taxid(taxonomy)
        if query_species is not None:
            self_hits = (hits[_SPECIES] == query_species).fillna(False)
            if self_hits.any():
                logger.info('Dropping %d self-hit(s) to the query taxon', int(self_hits.sum()))
                hits = hits[~self_hits]

        return hits

    def _search_exclusion_taxid(self) -> Optional[int]:
        """Clade to hand the search tool's taxon-exclude filter.

        Resolved before the search, so it costs one taxonomy lookup. If that
        lookup fails we fall back to the raw taxid rather than blocking the
        run -- ``annotate_hits`` filters self-hits again afterwards.
        """
        if not self.params.query_taxid:
            return None
        try:
            index = build_index(self.taxonomy, [self.params.query_taxid])
        except Exception as exc:  # noqa: BLE001 - non-fatal, we have a fallback
            logger.warning(
                'Could not resolve query taxid %s before the search (%s); '
                'excluding the exact taxid only',
                self.params.query_taxid, exc,
            )
            return self.params.query_taxid
        return self._query_exclusion_taxid(index)

    def _check_query_rank(self, taxonomy: TaxonomyIndex) -> None:
        """Fail when the query taxon has no ancestor at ``tax_level``.

        Without one, *nothing* can be in-group: the pandas engine then scores
        every gene as if it had no recipient hits (a proteome of false
        positives), while the out-of-core engine's SQL comparison against NULL
        excludes every row and reports nothing at all.  Neither answer is
        meaningful, and both used to be produced silently.

        Ranks are patchy in NCBI taxonomy -- plenty of bacteria have no family,
        and ranks like ``tribe`` or ``subgenus`` are absent almost everywhere --
        so this is a routine mistake, not an exotic one.
        """
        if not self.params.query_taxid:
            return
        if taxonomy.rank_taxid(self.params.query_taxid, self.params.tax_level) is not None:
            return
        info = taxonomy.get(self.params.query_taxid)
        available = ', '.join(rank for rank in TAX_RANKS if info and rank in info.alignment)
        raise RuntimeError(
            f'Query taxid {self.params.query_taxid} '
            f'({info.name if info else "unknown"}) has no {self.params.tax_level}-rank '
            'ancestor, so no hit could ever be classified as in-group and every '
            'score would be meaningless. Pick a rank its lineage actually has '
            f'with -t/--tax_level (available: {available or "none"}).'
        )

    def _query_exclusion_taxid(self, taxonomy: TaxonomyIndex) -> Optional[int]:
        """Taxid whose whole clade counts as "the query organism".

        Resolves up to species rank when the user supplies a strain-level
        taxid, so the search tool excludes the strain's siblings as well.
        """
        if not self.params.query_taxid:
            return None
        species = taxonomy.rank_taxid(self.params.query_taxid, 'species')
        if species is not None:
            return species
        info = taxonomy.get(self.params.query_taxid)
        return info.taxid if info else self.params.query_taxid

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _species_set(self, hits: pd.DataFrame, taxonomy: TaxonomyIndex) -> Set[int]:
        """Distinct species-rank taxa represented in ``hits``."""
        if _SPECIES in hits.columns:
            return set(hits[_SPECIES].dropna().unique().tolist())
        species: Set[int] = set()
        for taxid in hits[_TAXID].unique():
            species_taxid = taxonomy.rank_taxid(taxid, 'species')
            if species_taxid is not None:
                species.add(species_taxid)
        return species

    def score_all_genes(self, hits: pd.DataFrame) -> pd.DataFrame:
        """Score every gene at once, returning a frame indexed by gene.

        This is the vectorised equivalent of calling :meth:`calculate_scores`
        per gene.  Six groupby aggregations replace a Python loop over the
        whole proteome, and the expensive per-candidate work (donor lineage,
        top hits) then runs only on the genes that clear the thresholds --
        typically a handful out of tens of thousands.

        ``hits`` must already be annotated and free of synthetic records.
        """
        scored = hits[[QSEQID, _IS_RECIPIENT, _SPECIES, BITSCORE, EVALUE]]
        recipient = scored[scored[_IS_RECIPIENT]]
        outgroup = scored[~scored[_IS_RECIPIENT]]

        # Only genes with an out-group hit can be candidates, so they define
        # the index; everything else is reindexed onto it.
        out_bits = outgroup.groupby(QSEQID, observed=True)[BITSCORE].max()
        genes = out_bits.index
        if genes.empty:
            return pd.DataFrame(columns=_SCORE_COLUMNS)

        out_evalue = outgroup.groupby(QSEQID, observed=True)[EVALUE].min().reindex(genes)
        rec_bits = recipient.groupby(QSEQID, observed=True)[BITSCORE].max().reindex(genes)
        rec_evalue = recipient.groupby(QSEQID, observed=True)[EVALUE].min().reindex(genes)

        def species_counts(side: pd.DataFrame) -> pd.Series:
            distinct = side[[QSEQID, _SPECIES]].dropna(subset=[_SPECIES]).drop_duplicates()
            counts = distinct.groupby(QSEQID, observed=True).size()
            return counts.reindex(genes, fill_value=0).astype('int64')

        out_species = species_counts(outgroup)
        rec_species = species_counts(recipient)

        return compute_scores(
            genes,
            out_bits=out_bits, rec_bits=rec_bits,
            out_evalue=out_evalue, rec_evalue=rec_evalue,
            out_species=out_species, rec_species=rec_species,
            e_minus=self.params.e_minus,
        )

    def candidate_mask(self, scores: pd.DataFrame) -> pd.Series:
        """Vectorised equivalent of :meth:`meets_hgt_criteria`."""
        if scores.empty:
            return pd.Series(dtype=bool)
        return (
            (scores['max_outgroup_bitscore'] >= self.params.bitscore_parameter)
            & (scores['hgt_index'] >= self.params.hgt_index)
            & (scores['out_pct'] >= self.params.out_pct)
            & (scores['alien_index'] >= self.params.ai_threshold)
        )

    @staticmethod
    def _row_to_scores(row: pd.Series) -> HGTScores:
        return HGTScores(
            max_outgroup_bitscore=float(row['max_outgroup_bitscore']),
            max_recipient_bitscore=float(row['max_recipient_bitscore']),
            hgt_index=float(row['hgt_index']),
            out_pct=float(row['out_pct']),
            alien_index=float(row['alien_index']),
            outgroup_count=int(row['outgroup_count']),
            recipient_count=int(row['recipient_count']),
            min_outgroup_evalue=float(row['min_outgroup_evalue']),
            min_recipient_evalue=float(row['min_recipient_evalue']),
            no_recipient_hits=bool(row['no_recipient_hits']),
        )

    def calculate_scores(
        self,
        recipient_hits: pd.DataFrame,
        outgroup_hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
    ) -> Optional[HGTScores]:
        """Score one gene, or return ``None`` when there is no signal to score.

        A gene with no *out-group* hits cannot be an HGT candidate.  A gene with
        no *in-group* hits is the strongest candidate there is, and is scored
        against ``NO_HIT_EVALUE`` instead of being discarded.
        """
        if outgroup_hits.empty:
            logger.debug('No outgroup hits; not an HGT candidate')
            return None

        outgroup_species = self._species_set(outgroup_hits, taxonomy)
        max_outgroup_bitscore = float(outgroup_hits[BITSCORE].max())
        min_outgroup_evalue = float(outgroup_hits[EVALUE].min())

        if recipient_hits.empty:
            recipient_species: Set[int] = set()
            max_recipient_bitscore = 0.0
            min_recipient_evalue = NO_HIT_EVALUE
            hgt_index = 1.0
            no_recipient_hits = True
        else:
            recipient_species = self._species_set(recipient_hits, taxonomy)
            max_recipient_bitscore = float(recipient_hits[BITSCORE].max())
            min_recipient_evalue = float(recipient_hits[EVALUE].min())
            hgt_index = (
                max_outgroup_bitscore / max_recipient_bitscore
                if max_recipient_bitscore > 0 else 1.0
            )
            no_recipient_hits = False

        total_species = len(recipient_species) + len(outgroup_species)
        if total_species:
            out_pct = len(outgroup_species) / total_species
        else:
            # Neither side has species-rank annotation; fall back to the hit
            # split so a gene is not silently given out_pct == 0.
            out_pct = 1.0 if recipient_hits.empty else 0.0

        e_minus = self.params.e_minus
        alien_index = (
            math.log(min_recipient_evalue + e_minus)
            - math.log(min_outgroup_evalue + e_minus)
        )

        return HGTScores(
            max_outgroup_bitscore=max_outgroup_bitscore,
            max_recipient_bitscore=max_recipient_bitscore,
            hgt_index=hgt_index,
            out_pct=out_pct,
            alien_index=alien_index,
            outgroup_count=len(outgroup_species),
            recipient_count=len(recipient_species),
            min_outgroup_evalue=min_outgroup_evalue,
            min_recipient_evalue=min_recipient_evalue,
            no_recipient_hits=no_recipient_hits,
        )

    def meets_hgt_criteria(self, scores: Optional[HGTScores]) -> bool:
        if scores is None:
            return False
        return (
            scores.max_outgroup_bitscore >= self.params.bitscore_parameter
            and scores.hgt_index >= self.params.hgt_index
            and scores.out_pct >= self.params.out_pct
            and scores.alien_index >= self.params.ai_threshold
        )

    # ------------------------------------------------------------------
    # Per-gene processing
    # ------------------------------------------------------------------
    def process_single_gene(
        self,
        gene: str,
        gene_hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
    ) -> Optional[Dict[str, Any]]:
        """Score one gene; ``None`` means "not a candidate"."""
        hits = gene_hits[~gene_hits[_IS_SYNTHETIC]]
        if hits.empty:
            logger.debug('All hits for %s were synthetic constructs', gene)
            return None

        recipient_hits = hits[hits[_IS_RECIPIENT]]
        outgroup_hits = hits[~hits[_IS_RECIPIENT]]

        scores = self.calculate_scores(recipient_hits, outgroup_hits, taxonomy)
        if scores is None or not self.meets_hgt_criteria(scores):
            logger.debug('Gene %s does not meet HGT criteria', gene)
            return None

        logger.info('Gene %s meets HGT criteria (AI=%.1f, HGT index=%.2f)',
                    gene, scores.alien_index, scores.hgt_index)
        return {
            'gene': gene,
            'scores': scores,
            'taxonomy': self._donor_taxonomy(outgroup_hits, taxonomy),
            'top_hits': self._extract_top_hits(
                recipient_hits, outgroup_hits, taxonomy, n=self.params.top_hits
            ),
        }

    def process_genes(
        self,
        genes: List[str],
        hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], int]:
        """Score every gene, returning (candidates, errors, genes_with_hits).

        Two phases: score the whole proteome with vectorised aggregations, then
        build the detailed record (donor lineage, top hits) only for the genes
        that pass.  Runs single-threaded on purpose -- the work is GIL-bound
        pandas, so the old ``ThreadPoolExecutor`` added overhead without
        concurrency, and after vectorisation there is little left to spread.
        """
        gene_order = pd.Index(genes)
        usable = hits[~hits[_IS_SYNTHETIC]]
        # unique() on a categorical returns just the categories, so this costs
        # one pass rather than a full isin() scan of every hit row.
        with_hits = len(gene_order.intersection(pd.Index(hits[QSEQID].unique()), sort=False))

        scores = self.score_all_genes(usable)
        candidates: List[Dict[str, Any]] = []
        errors: List[Tuple[str, str]] = []

        if not scores.empty:
            passing = scores[self.candidate_mask(scores)]
            # Report in input order rather than groupby order.
            ordered = gene_order.intersection(passing.index, sort=False)
            passing = passing.reindex(ordered)

            detail_hits = usable[usable[QSEQID].isin(passing.index)]
            grouped = detail_hits.groupby(QSEQID, observed=True, sort=False)
            for gene, row in passing.iterrows():
                try:
                    gene_hits = grouped.get_group(gene)
                    gene_scores = self._row_to_scores(row)
                    recipient_hits = gene_hits[gene_hits[_IS_RECIPIENT]]
                    outgroup_hits = gene_hits[~gene_hits[_IS_RECIPIENT]]
                    logger.info('Gene %s meets HGT criteria (AI=%.1f, HGT index=%.2f)',
                                gene, gene_scores.alien_index, gene_scores.hgt_index)
                    candidates.append({
                        'gene': gene,
                        'scores': gene_scores,
                        'taxonomy': self._donor_taxonomy(outgroup_hits, taxonomy),
                        'top_hits': self._extract_top_hits(
                            recipient_hits, outgroup_hits, taxonomy, n=self.params.top_hits
                        ),
                    })
                except Exception as exc:  # noqa: BLE001 - one bad gene must not kill the run
                    logger.error('Error processing gene %s: %s', gene, exc, exc_info=True)
                    errors.append((str(gene), str(exc)))

        logger.info(
            'Scored %d gene(s) with hits (%d without); %d candidate(s), %d error(s)',
            with_hits, len(genes) - with_hits, len(candidates), len(errors),
        )
        return candidates, errors, with_hits

    def process_genes_out_of_core(
        self,
        table,
        taxonomy: TaxonomyIndex,
        genes_total: Optional[int] = None,
        gene_order: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], int]:
        """Score from a :class:`~flexihgt.aggregate.HitTable` instead of memory.

        Same two-phase shape as :meth:`process_genes` -- aggregate everything,
        then fetch details only for the genes that passed -- but the aggregation
        streams the file rather than loading it, and the detail pass is a second
        query restricted to the candidates.
        """
        raw = table.aggregate()
        if raw.empty:
            return [], [], table.gene_count()

        scores = compute_scores(
            raw.index,
            out_bits=raw['max_outgroup_bitscore'],
            rec_bits=raw['max_recipient_bitscore'],
            out_evalue=raw['min_outgroup_evalue'],
            rec_evalue=raw['min_recipient_evalue'],
            out_species=raw['out_species'],
            rec_species=raw['recipient_species'],
            e_minus=self.params.e_minus,
        )
        passing = scores[self.candidate_mask(scores)]
        if passing.empty:
            return [], [], table.gene_count()

        if gene_order is not None:
            # Report in input order, as the in-memory engine does. Only the
            # candidates are reordered, so this stays cheap.
            position = {gene: index for index, gene in enumerate(gene_order)}
            passing = passing.reindex(
                sorted(passing.index, key=lambda gene: position.get(gene, len(position)))
            )

        genes = list(passing.index)
        donors = table.best_outgroup_hits(genes)
        details = table.top_hits(genes, n=self.params.top_hits)

        candidates: List[Dict[str, Any]] = []
        errors: List[Tuple[str, str]] = []
        for gene, row in passing.iterrows():
            try:
                donor_taxid, subject_id = donors.get(gene, ('', ''))
                top = details.get(gene, {'recipient': [], 'outgroup': []})
                # The query leaves species unresolved; the taxonomy index does
                # it here rather than joining names into every hit row.
                for side in top.values():
                    for hit in side:
                        hit['species'] = taxonomy.species_name(hit['taxid'])
                candidates.append({
                    'gene': gene,
                    'scores': self._row_to_scores(row),
                    'taxonomy': self._donor_record(donor_taxid, subject_id, taxonomy),
                    'top_hits': top,
                })
            except Exception as exc:  # noqa: BLE001 - one bad gene must not kill the run
                logger.error('Error processing gene %s: %s', gene, exc, exc_info=True)
                errors.append((str(gene), str(exc)))

        with_hits = genes_total if genes_total is not None else table.gene_count()
        logger.info('Scored %d gene(s) with hits; %d candidate(s), %d error(s)',
                    with_hits, len(candidates), len(errors))
        return candidates, errors, with_hits

    def _donor_record(
        self,
        donor_taxid: str,
        subject_id: str,
        taxonomy: TaxonomyIndex,
    ) -> Dict[str, Any]:
        """Donor description shared by both engines."""
        info = taxonomy.get(donor_taxid)
        if info is None:
            return {'donor_taxid': donor_taxid, 'donor_taxonomy': {}, 'donor_name': 'Unknown'}
        return {
            'donor_taxid': str(info.taxid),
            'donor_name': info.name,
            'donor_subject_id': subject_id,
            'donor_taxonomy': taxonomy.lineage_names(donor_taxid),
        }

    def _donor_taxonomy(
        self,
        outgroup_hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
    ) -> Dict[str, Any]:
        """Describe the best out-group hit, i.e. the putative donor."""
        if outgroup_hits.empty:
            return {'donor_taxid': '', 'donor_taxonomy': {}, 'donor_name': ''}

        # idxmax avoids the float equality comparison the previous version used
        # to re-find this row, which could match nothing and raise IndexError.
        # Scalar .at[] lookups rather than extracting the whole row: a row spans
        # mixed dtypes, so pandas would materialise every categorical column's
        # full category array to build it.
        best = outgroup_hits[BITSCORE].idxmax()
        # Resolved from the full lineage, including ancestors -- an index that
        # only held hit taxids left this nearly always empty.
        return self._donor_record(
            str(outgroup_hits.at[best, _TAXID]),
            outgroup_hits.at[best, SSEQID],
            taxonomy,
        )

    def _extract_top_hits(
        self,
        recipient_hits: pd.DataFrame,
        outgroup_hits: pd.DataFrame,
        taxonomy: TaxonomyIndex,
        n: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Best ``n`` hits from each side, for manual inspection.

        Values are pulled out column by column rather than with ``iterrows``.
        A row spans mixed dtypes, so pandas builds it by converting every
        categorical column in full -- once per row, over categories numbering
        in the thousands.
        """
        def format_hits(hits: pd.DataFrame) -> List[Dict[str, Any]]:
            if hits.empty or n <= 0:
                return []
            top = hits.nlargest(n, BITSCORE)
            # Narrow the categories to the handful actually present first.
            taxids = [str(t) for t in top[_TAXID].cat.remove_unused_categories()]
            subjects = top[SSEQID].tolist()
            evalues = top[EVALUE].tolist()
            bitscores = top[BITSCORE].tolist()
            pidents = top[PIDENT].tolist()
            return [
                {
                    'subject_id': subject,
                    'evalue': float(evalue),
                    'bitscore': float(bitscore),
                    'pident': float(pident) if pd.notna(pident) else float('nan'),
                    'taxid': taxid,
                    'species': taxonomy.species_name(taxid),
                }
                for subject, evalue, bitscore, pident, taxid
                in zip(subjects, evalues, bitscores, pidents, taxids)
            ]

        return {
            'recipient': format_hits(recipient_hits),
            'outgroup': format_hits(outgroup_hits),
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _run_out_of_core(
        self,
        hits_path: Path,
        taxonomy_dump: Optional[Path] = None,
        gene_order: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], int]:
        """Score a hit table without loading it, via the DuckDB engine."""
        from . import aggregate

        logger.info('Scoring out of core (duckdb): %s', hits_path)
        taxids = aggregate.scan_taxids(hits_path, threads=self.params.threads)
        if self.params.query_taxid:
            taxids.append(str(self.params.query_taxid))
        taxonomy = build_index(self.taxonomy, taxids)
        if self.params.query_taxid and taxonomy.get(self.params.query_taxid) is None:
            raise RuntimeError(
                f'Query taxid {self.params.query_taxid} could not be resolved; '
                'check the taxid and that the local taxonomy database is current.'
            )
        self._check_query_rank(taxonomy)
        if taxonomy_dump:
            write_index(taxonomy, taxonomy_dump)

        lookup = aggregate.taxonomy_table(taxonomy, taxids, self.params.tax_level)
        with aggregate.HitTable(
            hits_path, lookup,
            query_boundary=taxonomy.rank_taxid(self.params.query_taxid, self.params.tax_level),
            query_species=self._query_exclusion_taxid(taxonomy),
            threads=self.params.threads,
        ) as table:
            return self.process_genes_out_of_core(table, taxonomy, gene_order=gene_order)

    @staticmethod
    def count_all_synthetic(hits: pd.DataFrame) -> int:
        """Genes left with nothing to score once synthetic hits are removed."""
        synthetic = hits[_IS_SYNTHETIC]
        if not synthetic.any():
            return 0
        usable_genes = set(hits.loc[~synthetic, QSEQID].unique())
        blocked = set(hits.loc[synthetic, QSEQID].unique())
        return len(blocked - usable_genes)

    def _finish(
        self,
        candidates: List[Dict[str, Any]],
        errors: List[Tuple[str, str]],
        with_hits: int,
        genes_total: int,
        output_file: Path,
        input_file: Optional[Path] = None,
        db_path: Optional[Path] = None,
        all_synthetic: Optional[int] = None,
    ) -> AnalysisResult:
        """Assemble the result and write every output file."""
        frame = pd.DataFrame(candidates, columns=['gene', 'scores', 'taxonomy', 'top_hits'])
        result = AnalysisResult(
            candidates=frame,
            genes_total=genes_total,
            genes_with_hits=with_hits,
            genes_all_synthetic=all_synthetic,
            gene_errors=errors,
            results_path=output_file,
        )
        self.write_results(result, output_file)
        self.write_provenance(output_file, input_file, db_path)
        return result

    def write_provenance(
        self,
        output_file: Path,
        input_file: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> Path:
        """Record how a results file was produced, next to it.

        Parameters previously existed only in the log, which is routinely lost;
        a results TSV on its own could not be reproduced.  A sidecar is used
        rather than comment lines in the TSV so downstream parsers are
        unaffected.
        """
        from . import __version__

        path = output_file.with_name(output_file.name + '.params.json')
        payload = {
            'flexihgt_version': __version__,
            'written': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'input': str(Path(input_file).resolve()) if input_file else None,
            'database': str(Path(db_path).resolve()) if db_path else None,
            'parameters': asdict(self.params),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
        logger.info('Run parameters written to %s', path)
        return path

    @staticmethod
    def annotation_cache_paths(hits_path: Path) -> Tuple[Path, Path]:
        """(parquet, signature) paths for a hit file's annotation cache.

        Suffixes are appended rather than substituted, matching the search
        sidecar convention and keeping the origin of the file obvious.
        """
        base = hits_path.name + '.annotated'
        return (
            hits_path.with_name(base + '.parquet'),
            hits_path.with_name(base + '.json'),
        )

    def _annotation_signature(self, hits_path: Path) -> str:
        """Fingerprint of everything that determines the annotated hit table."""
        stat = hits_path.stat()
        payload = json.dumps({
            'version': _ANNOTATION_CACHE_VERSION,
            'hits': str(hits_path.resolve()),
            'hits_size': stat.st_size,
            'hits_mtime': int(stat.st_mtime),
            'query_taxid': self.params.query_taxid,
            'tax_level': self.params.tax_level,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _load_annotation_cache(self, hits_path: Path) -> Optional[pd.DataFrame]:
        """Reuse a previously annotated hit table when nothing relevant changed.

        Threshold tuning is the normal workflow, and thresholds affect neither
        parsing nor taxonomy -- so a sweep should not repeat either.
        """
        cache_path, meta_path = self.annotation_cache_paths(hits_path)
        if not (cache_path.exists() and meta_path.exists()):
            return None
        try:
            if meta_path.read_text(encoding='utf-8').strip() != self._annotation_signature(hits_path):
                logger.info('Annotation cache is stale; rebuilding')
                return None
            frame = pd.read_parquet(cache_path)
        except Exception as exc:  # noqa: BLE001 - a bad cache must never be fatal
            logger.info('Could not read annotation cache (%s); rebuilding', exc)
            return None
        logger.info('Reusing annotated hits from %s', cache_path)
        return frame

    def _save_annotation_cache(self, hits_path: Path, hits: pd.DataFrame) -> None:
        cache_path, meta_path = self.annotation_cache_paths(hits_path)
        try:
            hits.to_parquet(cache_path, index=False)
            meta_path.write_text(self._annotation_signature(hits_path), encoding='utf-8')
            logger.info('Cached annotated hits to %s', cache_path)
        except Exception as exc:  # noqa: BLE001 - needs pyarrow/fastparquet
            logger.info(
                'Annotation cache not written (%s); install pyarrow to enable it', exc
            )

    def run_analysis(
        self,
        input_file: Path,
        db_path: Optional[Path] = None,
        output_file: Optional[Path] = None,
        force: bool = False,
        taxonomy_dump: Optional[Path] = None,
        tuning: Optional[SearchTuning] = None,
        rescore_only: bool = False,
        use_cache: bool = True,
        engine: str = 'auto',
    ) -> AnalysisResult:
        """Run the full pipeline.

        Infrastructure failures (missing input, search tool errors, an
        unusable taxonomy database) are raised rather than logged and turned
        into an empty result set.

        With ``rescore_only`` the search is skipped entirely and cached results
        are required -- for re-running with different thresholds.
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f'Input file not found: {input_file}')

        output_file = Path(output_file) if output_file else Path(
            f'{input_file.stem}_{self.params.tax_level}_HGT.tsv'
        )

        if rescore_only:
            hits_path = input_file.with_suffix('.hits.tsv')
            if not hits_path.exists():
                raise FileNotFoundError(
                    f'--rescore-only needs existing search results at {hits_path}'
                )
            logger.info('Rescoring existing search results: %s', hits_path)
        else:
            if db_path is None:
                raise ValueError('A database is required unless rescore_only is set')
            hits_path = run_search(
                input_file,
                Path(db_path),
                method=self.params.search_method,
                exclude_taxid=self._search_exclusion_taxid(),
                max_hits=self.params.max_hits,
                evalue=self.params.evalue,
                tuning=tuning or SearchTuning(threads=self.params.threads),
                force=force,
            )

        genes = list(self.read_fasta_ids(input_file))
        if not genes:
            raise ValueError(f'No sequences found in {input_file}')

        if select_engine(engine, hits_path) == 'duckdb':
            candidates, errors, with_hits = self._run_out_of_core(
                hits_path, taxonomy_dump, gene_order=genes
            )
            return self._finish(
                candidates, errors, with_hits, len(genes), output_file, input_file, db_path
            )

        cached = self._load_annotation_cache(hits_path) if use_cache else None
        if cached is not None:
            hits = cached
            taxids = list(hits[_TAXID].unique())
        else:
            hits = self.load_hits(hits_path)
            logger.info('Resolving taxonomy for %d hit(s)...', len(hits))
            taxids = list(self._hit_taxids(hits).unique())
        if self.params.query_taxid:
            taxids.append(str(self.params.query_taxid))

        taxonomy = build_index(self.taxonomy, taxids)
        if self.params.query_taxid and taxonomy.get(self.params.query_taxid) is None:
            raise RuntimeError(
                f'Query taxid {self.params.query_taxid} could not be resolved; '
                'check the taxid and that the local taxonomy database is current.'
            )
        self._check_query_rank(taxonomy)

        if cached is None:
            hits = self.annotate_hits(hits, taxonomy)
            if use_cache:
                self._save_annotation_cache(hits_path, hits)

        if taxonomy_dump:
            write_index(taxonomy, taxonomy_dump)
            logger.info('Wrote resolved taxonomy to %s', taxonomy_dump)

        candidates, errors, with_hits = self.process_genes(genes, hits, taxonomy)
        return self._finish(
            candidates, errors, with_hits, len(genes), output_file, input_file, db_path,
            all_synthetic=self.count_all_synthetic(hits),
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    RESULT_COLUMNS = [
        'Gene/Protein',
        'Max outgroup bitscore',
        'Max recipient bitscore',
        'Out_pct',
        'HGT index',
        'Alien Index',
        'Min outgroup E-value',
        'Min recipient E-value',
        'Outgroup species',
        'Recipient species',
        'No recipient hits',
        'Donor taxid',
        'Donor name',
        'Donor taxonomy',
    ]

    TOP_HIT_COLUMNS = [
        'Gene', 'Hit type', 'Subject ID', 'E-value', 'Bitscore', 'Pident', 'TaxID', 'Species',
    ]

    def write_results(self, result: AnalysisResult, output_file: Path) -> None:
        """Write the candidate table and the accompanying top-hits table.

        Both files are always written -- an empty run produces headers only, so
        downstream tooling does not have to special-case a missing file.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        top_hits_file = output_file.with_name(f'{output_file.stem}_top_hits.tsv')

        with open(output_file, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(self.RESULT_COLUMNS)
            for _, row in result.candidates.iterrows():
                scores: HGTScores = row['scores']
                taxonomy_info: Dict[str, Any] = row['taxonomy']
                donor_taxonomy = taxonomy_info.get('donor_taxonomy') or {}
                writer.writerow([
                    row['gene'],
                    f'{scores.max_outgroup_bitscore:.2f}',
                    f'{scores.max_recipient_bitscore:.2f}',
                    f'{scores.out_pct:.3f}',
                    f'{scores.hgt_index:.3f}',
                    f'{scores.alien_index:.2f}',
                    f'{scores.min_outgroup_evalue:.2e}',
                    f'{scores.min_recipient_evalue:.2e}',
                    scores.outgroup_count,
                    scores.recipient_count,
                    'yes' if scores.no_recipient_hits else 'no',
                    taxonomy_info.get('donor_taxid', ''),
                    taxonomy_info.get('donor_name', ''),
                    '; '.join(f'{rank}: {name}' for rank, name in donor_taxonomy.items()),
                ])

        with open(top_hits_file, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(self.TOP_HIT_COLUMNS)
            for _, row in result.candidates.iterrows():
                top_hits = row['top_hits'] or {}
                for label, key in (('Recipient', 'recipient'), ('Outgroup', 'outgroup')):
                    for hit in top_hits.get(key, []):
                        writer.writerow([
                            row['gene'],
                            label,
                            hit['subject_id'],
                            f'{hit["evalue"]:.2e}',
                            f'{hit["bitscore"]:.2f}',
                            f'{hit["pident"]:.1f}',
                            hit['taxid'],
                            hit['species'],
                        ])

        result.results_path = output_file
        result.top_hits_path = top_hits_file
        logger.info('Results written to %s', output_file)
        logger.info('Top hits written to %s', top_hits_file)
