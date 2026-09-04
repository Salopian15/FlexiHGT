"""Taxonomy lookups for FlexiHGT.

The whole pipeline talks to NCBI taxonomy through the :class:`TaxonomyProvider`
protocol rather than importing ``ete3`` directly.  That keeps the scoring logic
testable without a 500 MB ``taxa.sqlite`` on disk, and leaves room for swapping
the unmaintained ete3 backend out later.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Ranks we care about, ordered from most to least specific.
TAX_RANKS: Tuple[str, ...] = (
    'species', 'subgenus', 'genus',
    'subtribe', 'tribe', 'subfamily', 'family', 'superfamily',
    'infraorder', 'suborder', 'order', 'superorder',
    'infraclass', 'subclass', 'class', 'superclass',
    'subphylum', 'phylum', 'superphylum',
    'subkingdom', 'kingdom', 'superkingdom',
)

#: Ancestors that mark a sequence as a synthetic construct rather than a real
#: organism.  Matching on lineage is far more reliable than matching on name.
SYNTHETIC_TAXIDS = frozenset({
    32630,  # synthetic construct
    81077,  # artificial sequences
    28384,  # other sequences
})

#: Fallback name matching, for records whose lineage does not reach a synthetic
#: ancestor.  Compared against a lower-cased name, so every entry must be lower
#: case -- the previous capitalised duplicates could never match.
SYNTHETIC_KEYWORDS = frozenset({
    'synthetic', 'vector', 'construct', 'artificial',
    'engineered', 'cloning', 'expression', 'plasmid',
})


class TaxonomyInfo(NamedTuple):
    """Resolved taxonomy for a single taxid."""

    taxid: int                    # taxid after merged-taxid translation
    original_taxid: int           # taxid as it appeared in the search results
    rank: str
    name: str
    lineage: Tuple[int, ...]
    alignment: Dict[str, int]     # rank -> taxid, for the ranks in TAX_RANKS


class TaxonomyProvider:
    """Minimal interface onto an NCBI taxonomy database."""

    def translate_merged(self, taxids: Sequence[int]) -> Tuple[List[int], Dict[int, int]]:
        """Return (live taxids, {retired taxid: live taxid})."""
        raise NotImplementedError

    def get_lineage_translator(self, taxids: Sequence[int]) -> Dict[int, List[int]]:
        raise NotImplementedError

    def get_rank(self, taxids: Sequence[int]) -> Dict[int, str]:
        raise NotImplementedError

    def get_taxid_translator(self, taxids: Sequence[int]) -> Dict[int, str]:
        raise NotImplementedError

    def update_database(self) -> None:
        raise NotImplementedError


class Ete3Taxonomy(TaxonomyProvider):
    """:class:`TaxonomyProvider` backed by ``ete3.NCBITaxa``.

    ``ete3`` is imported lazily so that importing :mod:`flexihgt` (and running
    the test suite) does not require the taxonomy database to be present.
    """

    def __init__(self, ncbi=None) -> None:
        # Construction must stay cheap and import-free: the CLI builds a
        # provider before it knows whether the run will need one (--rescore-only
        # against a warm cache does not), and the tests inject a fake.
        self._ncbi = ncbi

    @property
    def ncbi(self):
        if self._ncbi is None:
            from ete3 import NCBITaxa  # noqa: PLC0415 - deliberately lazy

            self._ncbi = NCBITaxa()
        return self._ncbi

    def translate_merged(self, taxids: Sequence[int]) -> Tuple[List[int], Dict[int, int]]:
        # ``_translate_merged`` is private API but it is the only way to map
        # retired taxids onto their replacements; fall back to a no-op if a
        # future ete3 release removes it.
        translate = getattr(self.ncbi, '_translate_merged', None)
        if translate is None:
            return list(taxids), {}
        live, merged = translate(list(taxids))
        return list(live), dict(merged)

    #: Taxids per ``IN (...)`` clause. SQLite copes with far more, but very
    #: large literal lists slow the query planner down.
    _CHUNK = 5000

    def get_lineage_translator(self, taxids: Sequence[int]) -> Dict[int, List[int]]:
        if not taxids:
            return {}
        bulk = self._bulk_lineages(list(taxids))
        if bulk is not None:
            return bulk
        return self.ncbi.get_lineage_translator(list(taxids))

    def _bulk_lineages(self, taxids: List[int]) -> Optional[Dict[int, List[int]]]:
        """Fetch every lineage with a handful of queries instead of one each.

        ete3 stores each taxon's full path to the root in a ``track`` column, so
        the lineages can be read directly rather than resolved by per-taxid
        recursion -- the difference between a few queries and ~100k of them on a
        proteome-scale search.

        Returns ``None`` if the schema is not what we expect, in which case the
        caller falls back to ete3's own (correct, slower) implementation.
        """
        db = getattr(self.ncbi, 'db', None)
        if db is None:
            return None
        try:
            lineages: Dict[int, List[int]] = {}
            for start in range(0, len(taxids), self._CHUNK):
                chunk = taxids[start:start + self._CHUNK]
                placeholders = ','.join(str(int(taxid)) for taxid in chunk)
                cursor = db.execute(
                    f'SELECT taxid, track FROM species WHERE taxid IN ({placeholders})'
                )
                for taxid, track in cursor.fetchall():
                    if not track:
                        continue
                    # ``track`` runs from the taxon up to the root; callers
                    # expect root-first ordering, as ete3's get_lineage returns.
                    lineages[int(taxid)] = [
                        int(node) for node in reversed(str(track).split(','))
                    ]
            return lineages
        except Exception as exc:  # noqa: BLE001 - any schema surprise -> fall back
            logger.debug('Bulk lineage query unavailable (%s); using ete3 API', exc)
            return None

    def get_rank(self, taxids: Sequence[int]) -> Dict[int, str]:
        if not taxids:
            return {}
        return self.ncbi.get_rank(list(taxids))

    def get_taxid_translator(self, taxids: Sequence[int]) -> Dict[int, str]:
        if not taxids:
            return {}
        return self.ncbi.get_taxid_translator(list(taxids))

    def update_database(self) -> None:
        self.ncbi.update_taxonomy_database()


class TaxopyTaxonomy(TaxonomyProvider):
    """:class:`TaxonomyProvider` backed by `taxopy <https://github.com/apcamargo/taxopy>`_.

    An alternative to ete3, which is unmaintained and is what pins FlexiHGT
    below Python 3.13. taxopy holds ``nodes.dmp``/``names.dmp`` in memory, so
    lineage resolution is a dictionary walk rather than a database query.

    Select it with ``--taxonomy-backend taxopy``; ete3 remains the default.
    """

    def __init__(self, taxdb=None, taxdb_dir: Optional[str] = None) -> None:
        if taxdb is None:
            import taxopy  # noqa: PLC0415 - optional dependency, imported lazily

            taxdb = taxopy.TaxDb(taxdb_dir=taxdb_dir) if taxdb_dir else taxopy.TaxDb()
        self._db = taxdb

    def translate_merged(self, taxids: Sequence[int]) -> Tuple[List[int], Dict[int, int]]:
        old_to_new = getattr(self._db, 'oldtaxid2newtaxid', {}) or {}
        merged = {t: old_to_new[t] for t in taxids if t in old_to_new}
        return [t for t in taxids if t not in merged], merged

    def get_lineage_translator(self, taxids: Sequence[int]) -> Dict[int, List[int]]:
        parents = self._db.taxid2parent
        cache: Dict[int, List[int]] = {}

        def lineage(taxid: int) -> List[int]:
            # Ancestors are shared heavily, so memoising makes the whole set
            # close to linear in the number of distinct nodes touched.
            if taxid in cache:
                return cache[taxid]
            if taxid not in parents:
                return []
            parent = parents[taxid]
            result = [taxid] if parent == taxid else lineage(parent) + [taxid]
            cache[taxid] = result
            return result

        return {t: lineage(t) for t in taxids if lineage(t)}

    def get_rank(self, taxids: Sequence[int]) -> Dict[int, str]:
        ranks = self._db.taxid2rank
        return {t: ranks[t] for t in taxids if t in ranks}

    def get_taxid_translator(self, taxids: Sequence[int]) -> Dict[int, str]:
        names = self._db.taxid2name
        return {t: names[t] for t in taxids if t in names}

    def update_database(self) -> None:
        raise NotImplementedError(
            'taxopy reads nodes.dmp/names.dmp directly; download them from '
            'https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/ rather than via FlexiHGT.'
        )


def get_provider(backend: str = 'ete3', taxdb_dir: Optional[str] = None) -> TaxonomyProvider:
    """Construct the requested taxonomy backend."""
    if backend == 'ete3':
        return Ete3Taxonomy()
    if backend == 'taxopy':
        return TaxopyTaxonomy(taxdb_dir=taxdb_dir)
    raise ValueError(f'Unknown taxonomy backend: {backend!r}')


class TaxonomyIndex:
    """Every taxonomy fact needed for one analysis, resolved up front.

    Crucially this includes the *ancestors* of every hit taxid, not just the
    hit taxids themselves.  Donor lineages and species names are drawn from
    ancestors, so an index that only knows about leaves reports them as empty.
    """

    def __init__(
        self,
        info: Dict[int, TaxonomyInfo],
        names: Dict[int, str],
        ranks: Dict[int, str],
        merged: Optional[Dict[int, int]] = None,
    ) -> None:
        self._info = info
        self._names = names
        self._ranks = ranks
        self._merged = merged or {}

    def __len__(self) -> int:
        return len(self._info)

    def __contains__(self, taxid: object) -> bool:
        return self._as_int(taxid) in self._info

    @staticmethod
    def _as_int(taxid: object) -> Optional[int]:
        """Coerce a taxid of unknown provenance (str/float/int) to ``int``."""
        if taxid is None:
            return None
        try:
            text = str(taxid).strip()
            if not text or text.lower() in {'nan', 'none', 'n/a', '*'}:
                return None
            return int(float(text))
        except (TypeError, ValueError):
            return None

    def get(self, taxid: object) -> Optional[TaxonomyInfo]:
        key = self._as_int(taxid)
        if key is None:
            return None
        return self._info.get(key)

    def name_of(self, taxid: object, default: str = 'Unknown') -> str:
        key = self._as_int(taxid)
        if key is None:
            return default
        return self._names.get(key, default)

    def rank_of(self, taxid: object, default: str = 'no rank') -> str:
        key = self._as_int(taxid)
        if key is None:
            return default
        return self._ranks.get(key, default)

    def rank_taxid(self, taxid: object, rank: str) -> Optional[int]:
        """Taxid of ``taxid``'s ancestor at ``rank``, if any."""
        info = self.get(taxid)
        if info is None:
            return None
        return info.alignment.get(rank)

    def rank_taxid_map(self, taxids: Iterable[object], rank: str) -> Dict[object, Optional[int]]:
        """``{taxid: ancestor at rank}`` for many taxids at once.

        Built once per run and applied to the hit table with a vectorised
        ``Series.map``, instead of a Python call per hit.
        """
        return {taxid: self.rank_taxid(taxid, rank) for taxid in taxids}

    def species_name(self, taxid: object, default: str = 'Unknown') -> str:
        species = self.rank_taxid(taxid, 'species')
        if species is None:
            # No species-rank ancestor (e.g. a hit annotated only to genus).
            info = self.get(taxid)
            return info.name if info else default
        return self.name_of(species, default)

    def lineage_names(self, taxid: object) -> Dict[str, str]:
        """``{rank: name}`` for the hit's lineage, ordered specific -> general."""
        info = self.get(taxid)
        if info is None:
            return {}
        return {
            rank: self.name_of(info.alignment[rank])
            for rank in TAX_RANKS
            if rank in info.alignment
        }

    def is_synthetic(self, taxid: object) -> bool:
        info = self.get(taxid)
        if info is None:
            return False
        if SYNTHETIC_TAXIDS.intersection(info.lineage):
            return True
        name = info.name.lower()
        return any(keyword in name for keyword in SYNTHETIC_KEYWORDS)

    def share_rank(self, left: object, right: object, rank: str) -> bool:
        """True when both taxids resolve to the same ancestor at ``rank``."""
        left_taxid = self.rank_taxid(left, rank)
        if left_taxid is None:
            return False
        return left_taxid == self.rank_taxid(right, rank)


def build_index(
    provider: TaxonomyProvider,
    taxids: Iterable[object],
) -> TaxonomyIndex:
    """Resolve ``taxids`` (and their full lineages) into a :class:`TaxonomyIndex`.

    Raises:
        RuntimeError: if the taxonomy backend resolved nothing at all, which
            almost always means the local database is missing or corrupt.  This
            used to be swallowed and reported as "0 HGT events found".
    """
    wanted: set[int] = set()
    skipped = 0
    for raw in taxids:
        parsed = TaxonomyIndex._as_int(raw)
        if parsed is None or parsed <= 0:
            skipped += 1
            continue
        wanted.add(parsed)

    if skipped:
        logger.warning('Ignored %d hit(s) with an unparseable taxid', skipped)
    if not wanted:
        raise RuntimeError('No usable taxids found in the search results')

    live, merged = provider.translate_merged(sorted(wanted))
    # A retired taxid is not in ``live``; look up its replacement instead.
    lookup_ids = sorted(set(live) | {merged[t] for t in merged})
    lineages = provider.get_lineage_translator(lookup_ids)

    # Resolve names and ranks for every ancestor as well, in one bulk call.
    ancestors: set[int] = set(lookup_ids)
    for lineage in lineages.values():
        ancestors.update(lineage)
    ranks = provider.get_rank(sorted(ancestors))
    names = provider.get_taxid_translator(sorted(ancestors))

    info: Dict[int, TaxonomyInfo] = {}
    unresolved = 0
    for original in sorted(wanted):
        final = merged.get(original, original)
        lineage = lineages.get(final) or lineages.get(original)
        if not lineage:
            unresolved += 1
            continue
        alignment = {
            rank: node
            for node in lineage
            for rank in (ranks.get(node, 'no rank'),)
            if rank in TAX_RANKS
        }
        record = TaxonomyInfo(
            taxid=final,
            original_taxid=original,
            rank=ranks.get(final, 'no rank'),
            name=names.get(final, 'unknown'),
            lineage=tuple(lineage),
            alignment=alignment,
        )
        info[original] = record
        # Index under the live taxid too, so lookups by either id succeed.
        info.setdefault(final, record)

    if unresolved:
        logger.warning('No lineage available for %d taxid(s)', unresolved)
    if not info:
        raise RuntimeError(
            'Taxonomy lookup resolved no lineages at all -- is the ete3 database '
            'installed? Run `flexihgt --update` or `ete3 ncbiquery --update`.'
        )

    logger.info(
        'Resolved taxonomy for %d taxid(s) (%d merged, %d ancestors cached)',
        len(wanted), len(merged), len(names),
    )
    return TaxonomyIndex(info=info, names=names, ranks=ranks, merged=merged)


def write_index(index: TaxonomyIndex, path) -> None:
    """Dump the resolved taxonomy to TSV for debugging/provenance."""
    import csv

    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['original_taxid', 'taxid', 'rank', 'name', 'lineage', 'alignment'])
        seen: set[int] = set()
        for record in index._info.values():  # noqa: SLF001 - same module
            if record.original_taxid in seen:
                continue
            seen.add(record.original_taxid)
            writer.writerow([
                record.original_taxid,
                record.taxid,
                record.rank,
                record.name,
                ','.join(str(node) for node in record.lineage),
                ';'.join(f'{rank}:{taxid}' for rank, taxid in sorted(record.alignment.items())),
            ])
