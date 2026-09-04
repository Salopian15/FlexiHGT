"""
This module (should ) compute exactly the same per-gene aggregates with SQL, streaming
the file from disk.  Two queries do the work:

1. one grouped aggregation producing the six numbers per gene and side, and
2. a second, restricted to the genes that passed, fetching their top hits.

The taxonomy is passed in as a small lookup table (one row per distinct hit
taxid), so no taxonomy work happens per hit row.

DuckDB is optional.  :func:`available` reports whether this path can be used;
callers fall back to the pandas implementation when it cannot.

Note my testing on this has been limited, not much real world use has been applied
so be warned for unknown issues and silent crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .search import HIT_COLUMNS

logger = logging.getLogger(__name__)

#: Columns of the per-gene aggregate, matching HGTDetect.score_all_genes.
AGGREGATE_COLUMNS = [
    'max_outgroup_bitscore', 'max_recipient_bitscore', 'out_species', 'recipient_species',
    'min_outgroup_evalue', 'min_recipient_evalue',
]


def available() -> bool:
    """True when DuckDB is importable."""
    import importlib.util

    return importlib.util.find_spec('duckdb') is not None


def _sql_literal(path: Path) -> str:
    """A path as a SQL string literal, with quotes escaped.

    Paths come from the user, and a single quote anywhere in one would
    otherwise terminate the literal and break the query.
    """
    return Path(path).as_posix().replace("'", "''")


def _connect(threads: int = 0, memory_limit: Optional[str] = None):
    import duckdb

    con = duckdb.connect()
    if threads:
        con.execute(f'SET threads = {int(threads)}')
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    return con


class HitTable:
    """A hit table on disk, queried without loading it into memory.

    ``taxonomy`` maps each distinct hit taxid onto the three facts scoring
    needs: its species-rank ancestor, its ancestor at the in-group rank, and
    whether it is a synthetic construct.  ``queries`` maps each query id onto
    the in-group boundary of the genome it came from -- one row per genome for
    a multi-genome run, or a single row's worth of constant for one proteome.
    """

    def __init__(
        self,
        path: Path,
        taxonomy: pd.DataFrame,
        queries: Optional[pd.DataFrame] = None,
        query_boundary: Optional[int] = None,
        query_species: Optional[int] = None,
        threads: int = 0,
        memory_limit: Optional[str] = None,
    ) -> None:
        if queries is None and query_boundary is None:
            raise ValueError('Provide either a per-genome query table or a single boundary')
        self.path = Path(path)
        self._con = _connect(threads, memory_limit)
        self._con.register('tax', taxonomy)
        self._single_boundary = query_boundary
        self._single_species = query_species
        self._per_genome = queries is not None
        if queries is not None:
            self._con.register('queries', queries)
        self._con.execute(self._base_view_sql())

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> 'HitTable':
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _base_view_sql(self) -> str:
        """A view of the file with taxonomy joined and the split decided."""
        columns = ', '.join(f"'{name}': '{self._column_type(name)}'" for name in HIT_COLUMNS)
        # Only the last taxid of a ';'-separated field is used, matching
        # HGTDetect._hit_taxids.
        taxid_expr = "regexp_extract(raw.staxids, '([^;]+)$', 1)"

        if self._per_genome:
            genome_expr = "split_part(raw.qseqid, '|', 1)"
            join = (
                'JOIN queries q ON q.genome_id = ' + genome_expr + '\n'
                'LEFT JOIN tax t ON t.taxid = TRY_CAST(' + taxid_expr + ' AS BIGINT)'
            )
            boundary = 'q.boundary'
            own_species = 'q.species'
        else:
            join = 'LEFT JOIN tax t ON t.taxid = TRY_CAST(' + taxid_expr + ' AS BIGINT)'
            boundary = 'NULL' if self._single_boundary is None else str(self._single_boundary)
            own_species = 'NULL' if self._single_species is None else str(self._single_species)

        return f"""
        CREATE OR REPLACE VIEW hits AS
        SELECT
            raw.qseqid                                   AS qseqid,
            raw.sseqid                                   AS sseqid,
            raw.evalue                                   AS evalue,
            raw.bitscore                                 AS bitscore,
            raw.pident                                   AS pident,
            {taxid_expr}                                 AS taxid,
            t.species                                    AS species,
            -- COALESCE, not a bare comparison: a NULL boundary on either side
            -- makes '=' return NULL, and a NULL is_recipient satisfies neither
            -- FILTER (WHERE is_recipient) nor FILTER (WHERE NOT is_recipient),
            -- so every row would drop out of both sides of the aggregate and
            -- the run would report no candidates at all.
            COALESCE(t.boundary = {boundary}, FALSE)     AS is_recipient
        FROM read_csv('{self._quoted_path()}',
                      delim='\\t', header=false,
                      columns={{{columns}}},
                      nullstr='') AS raw
        {join}
        WHERE t.taxid IS NOT NULL           -- unresolved taxids are unusable
          AND NOT t.synthetic
          AND raw.bitscore IS NOT NULL
          AND raw.evalue IS NOT NULL
          -- Self-hits: the genome's own species never counts as evidence.
          AND (t.species IS NULL OR {own_species} IS NULL OR t.species <> {own_species})
        """

    def _quoted_path(self) -> str:
        return _sql_literal(self.path)

    @staticmethod
    def _column_type(name: str) -> str:
        # Everything but the three numeric columns is parsed as text; ``length``
        # in particular is never read, and declaring it VARCHAR avoids a cast.
        if name in ('evalue', 'bitscore', 'pident'):
            return 'DOUBLE'
        return 'VARCHAR'

    # ------------------------------------------------------------------
    def aggregate(self) -> pd.DataFrame:
        """Per-gene aggregates, indexed by query id.

        One pass over the file. ``COUNT(DISTINCT species)`` is what ``out_pct``
        is built from, and is the reason a naive streaming implementation needs
        a set per gene -- DuckDB does it without holding the table.
        """
        frame = self._con.execute("""
            SELECT
                qseqid,
                max(bitscore) FILTER (WHERE NOT is_recipient)      AS max_outgroup_bitscore,
                max(bitscore) FILTER (WHERE is_recipient)          AS max_recipient_bitscore,
                min(evalue)   FILTER (WHERE NOT is_recipient)      AS min_outgroup_evalue,
                min(evalue)   FILTER (WHERE is_recipient)          AS min_recipient_evalue,
                count(DISTINCT species) FILTER (WHERE NOT is_recipient) AS out_species,
                count(DISTINCT species) FILTER (WHERE is_recipient)     AS recipient_species
            FROM hits
            GROUP BY qseqid
            HAVING max(bitscore) FILTER (WHERE NOT is_recipient) IS NOT NULL
        """).df()
        return frame.set_index('qseqid')

    def gene_count(self) -> int:
        """Distinct query ids with at least one usable hit."""
        return int(self._con.execute(
            'SELECT count(DISTINCT qseqid) FROM hits'
        ).fetchone()[0])

    def gene_counts_by_genome(self) -> Dict[str, int]:
        """Distinct query ids per genome, for the multi-genome summary."""
        frame = self._con.execute("""
            SELECT split_part(qseqid, '|', 1) AS genome_id,
                   count(DISTINCT qseqid)     AS genes
            FROM hits GROUP BY 1
        """).df()
        return {row.genome_id: int(row.genes) for row in frame.itertuples(index=False)}

    def top_hits(self, genes: Sequence[str], n: int = 5) -> Dict[str, Dict[str, List[dict]]]:
        """Best ``n`` hits per side, for the given genes only.

        This is the second pass, and it is why the design is affordable: the
        expensive per-row detail is fetched for the handful of candidates
        rather than for the whole table.
        """
        if not genes or n <= 0:
            return {}
        wanted = pd.DataFrame({'qseqid': list(dict.fromkeys(genes))})
        self._con.register('wanted', wanted)
        frame = self._con.execute(f"""
            SELECT qseqid, is_recipient, sseqid, evalue, bitscore, pident, taxid, species
            FROM hits JOIN wanted USING (qseqid)
            QUALIFY row_number() OVER (
                PARTITION BY qseqid, is_recipient ORDER BY bitscore DESC
            ) <= {int(n)}
            ORDER BY qseqid, is_recipient DESC, bitscore DESC
        """).df()
        self._con.unregister('wanted')

        results: Dict[str, Dict[str, List[dict]]] = {
            gene: {'recipient': [], 'outgroup': []} for gene in wanted['qseqid']
        }
        for row in frame.itertuples(index=False):
            side = 'recipient' if row.is_recipient else 'outgroup'
            results[row.qseqid][side].append({
                'subject_id': row.sseqid,
                'evalue': float(row.evalue),
                'bitscore': float(row.bitscore),
                'pident': float(row.pident) if pd.notna(row.pident) else float('nan'),
                'taxid': str(row.taxid),
                'species': None,          # resolved by the caller, which holds the index
            })
        return results

    def best_outgroup_hits(self, genes: Sequence[str]) -> Dict[str, Tuple[str, str]]:
        """``{gene: (donor taxid, subject id)}`` for the best out-group hit."""
        if not genes:
            return {}
        wanted = pd.DataFrame({'qseqid': list(dict.fromkeys(genes))})
        self._con.register('wanted', wanted)
        frame = self._con.execute("""
            SELECT qseqid, taxid, sseqid
            FROM hits JOIN wanted USING (qseqid)
            WHERE NOT is_recipient
            QUALIFY row_number() OVER (PARTITION BY qseqid ORDER BY bitscore DESC) = 1
        """).df()
        self._con.unregister('wanted')
        return {
            row.qseqid: (str(row.taxid), row.sseqid)
            for row in frame.itertuples(index=False)
        }

    def distinct_taxids(self) -> List[str]:
        """Every distinct hit taxid in the file, read without loading it.

        Used to build the taxonomy index before the annotated view exists, so
        it deliberately queries the raw file rather than the ``hits`` view.
        """
        columns = ', '.join(f"'{name}': 'VARCHAR'" for name in HIT_COLUMNS)
        frame = self._con.execute(f"""
            SELECT DISTINCT regexp_extract(staxids, '([^;]+)$', 1) AS taxid
            FROM read_csv('{self._quoted_path()}',
                          delim='\\t', header=false, columns={{{columns}}}, nullstr='')
            WHERE staxids IS NOT NULL
        """).df()
        return [t for t in frame['taxid'].tolist() if t]


def scan_taxids(path: Path, threads: int = 0) -> List[str]:
    """Distinct hit taxids in a table too large to load."""
    columns = ', '.join(f"'{name}': 'VARCHAR'" for name in HIT_COLUMNS)
    con = _connect(threads)
    try:
        frame = con.execute(f"""
            SELECT DISTINCT regexp_extract(staxids, '([^;]+)$', 1) AS taxid
            FROM read_csv('{_sql_literal(path)}',
                          delim='\\t', header=false, columns={{{columns}}}, nullstr='')
            WHERE staxids IS NOT NULL
        """).df()
    finally:
        con.close()
    return [t for t in frame['taxid'].tolist() if t]


def taxonomy_table(index, taxids: Sequence[str], rank: str) -> pd.DataFrame:
    """Flatten a :class:`~flexihgt.taxonomy.TaxonomyIndex` into a lookup table.

    One row per distinct hit taxid, which is at most a few hundred thousand
    even for a whole phylum -- small enough to hand to DuckDB directly.
    """
    rows: List[Dict[str, Any]] = []
    for raw in dict.fromkeys(taxids):
        info = index.get(raw)
        if info is None:
            continue
        rows.append({
            'taxid': int(info.original_taxid),
            'species': index.rank_taxid(raw, 'species'),
            'boundary': index.rank_taxid(raw, rank),
            'synthetic': bool(index.is_synthetic(raw)),
        })
    frame = pd.DataFrame(rows, columns=['taxid', 'species', 'boundary', 'synthetic'])
    return frame.astype({
        'taxid': 'int64',
        'species': 'Int64',
        'boundary': 'Int64',
        'synthetic': 'bool',
    })


def query_table(genomes: Sequence[Tuple[str, Optional[int], Optional[int]]]) -> pd.DataFrame:
    """``(genome_id, boundary, species)`` rows for a multi-genome run."""
    frame = pd.DataFrame(genomes, columns=['genome_id', 'boundary', 'species'])
    return frame.astype({'genome_id': 'object', 'boundary': 'Int64', 'species': 'Int64'})
