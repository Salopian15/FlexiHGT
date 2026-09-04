"""Tests for hit loading, scoring and output."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from flexihgt.core import NO_HIT_EVALUE, AnalysisResult, HGTDetect, HGTParameters
from flexihgt.search import HIT_COLUMNS


def make_hits(rows):
    """Build a hit table from (gene, subject, evalue, bitscore, taxid) tuples."""
    return pd.DataFrame(
        [(gene, subject, evalue, bits, 100, 50.0, str(taxid))
         for gene, subject, evalue, bits, taxid in rows],
        columns=HIT_COLUMNS,
    )


@pytest.fixture()
def detector():
    """Default thresholds, in-group defined at family level."""
    return HGTDetect(HGTParameters(query_taxid=7227, tax_level='family'))


@pytest.fixture()
def permissive():
    """As above but with a lower out_pct, so the tiny fixture tree can pass it.

    The default 0.8 needs at least four out-group species per in-group species,
    which the miniature taxonomy in conftest cannot supply.
    """
    return HGTDetect(HGTParameters(query_taxid=7227, tax_level='family', out_pct=0.5))


# ---------------------------------------------------------------- parameters
def test_rejects_unknown_tax_level():
    with pytest.raises(ValueError, match='taxonomic level'):
        HGTParameters(tax_level='clade')


def test_rejects_unknown_search_method():
    with pytest.raises(ValueError, match='search method'):
        HGTParameters(search_method='blast')


def test_rejects_out_of_range_thresholds():
    with pytest.raises(ValueError):
        HGTParameters(hgt_index=1.5)
    with pytest.raises(ValueError):
        HGTParameters(out_pct=-0.1)
    with pytest.raises(ValueError):
        HGTParameters(query_taxid=0)


# ------------------------------------------------------------------ loading
def test_numeric_taxid_column_is_read_as_text(tmp_path, detector):
    """Regression: an all-numeric staxids column was inferred as int64.

    Every downstream ``.str.split(';')`` then raised, each gene was caught and
    skipped, and the run reported "0 HGT events" with exit code 0.
    """
    path = tmp_path / 'hits.tsv'
    path.write_text(
        'g1\ts1\t1e-90\t300\t100\t50.0\t1390\n'
        'g1\ts2\t1e-10\t120\t100\t45.0\t7240\n',
        encoding='utf-8',
    )
    hits = detector.load_hits(path)
    # The point is that the .str accessor works; the exact string dtype pandas
    # picks varies by version.
    assert all(isinstance(value, str) for value in hits['staxids'])
    assert list(detector._hit_taxids(hits)) == ['1390', '7240']


def test_multi_taxid_field_uses_last_entry(tmp_path, detector):
    path = tmp_path / 'hits.tsv'
    path.write_text('g1\ts1\t1e-90\t300\t100\t50.0\t1423;1390\n', encoding='utf-8')
    hits = detector.load_hits(path)
    assert list(detector._hit_taxids(hits)) == ['1390']


def test_hits_without_taxids_are_dropped(tmp_path, detector):
    path = tmp_path / 'hits.tsv'
    path.write_text(
        'g1\ts1\t1e-90\t300\t100\t50.0\t\n'
        'g1\ts2\t1e-80\t280\t100\t50.0\t1390\n',
        encoding='utf-8',
    )
    hits = detector.load_hits(path)
    assert len(hits) == 1
    assert hits.iloc[0]['sseqid'] == 's2'


def test_empty_hit_table_raises(tmp_path, detector):
    path = tmp_path / 'hits.tsv'
    path.write_text('', encoding='utf-8')
    with pytest.raises(ValueError):
        detector.load_hits(path)


# --------------------------------------------------------------- annotation
def test_annotate_splits_ingroup_and_outgroup(detector, index):
    hits = make_hits([
        ('g1', 's1', 1e-90, 300, 1390),   # Bacillaceae -> outgroup
        ('g1', 's2', 1e-20, 150, 7240),   # Drosophilidae -> in-group
        ('g1', 's3', 1e-15, 140, 7165),   # Culicidae -> outgroup at family level
        ('g1', 's4', 1e-05, 90, 32630),   # synthetic
    ])
    annotated = detector.annotate_hits(hits, index)
    recipient = dict(zip(annotated['sseqid'], annotated['_is_recipient']))
    assert recipient == {'s1': False, 's2': True, 's3': False, 's4': False}
    assert dict(zip(annotated['sseqid'], annotated['_is_synthetic']))['s4'] is True


def test_annotate_drops_self_hits(detector, index):
    hits = make_hits([
        ('g1', 'self', 0.0, 900, 7227),
        ('g1', 's1', 1e-90, 300, 1390),
    ])
    annotated = detector.annotate_hits(hits, index)
    assert list(annotated['sseqid']) == ['s1']


def test_tax_level_changes_the_split(index):
    """The in-group boundary must follow --tax_level."""
    hits = make_hits([('g1', 's1', 1e-15, 140, 7165)])  # Anopheles
    at_family = HGTDetect(HGTParameters(query_taxid=7227, tax_level='family'))
    at_order = HGTDetect(HGTParameters(query_taxid=7227, tax_level='order'))
    assert not at_family.annotate_hits(hits, index)['_is_recipient'].iloc[0]
    assert at_order.annotate_hits(hits, index)['_is_recipient'].iloc[0]


# ------------------------------------------------------------------ scoring
def score(detector, index, rows):
    annotated = detector.annotate_hits(make_hits(rows), index)
    annotated = annotated[~annotated['_is_synthetic']]
    return detector.calculate_scores(
        annotated[annotated['_is_recipient']],
        annotated[~annotated['_is_recipient']],
        index,
    )


def test_score_arithmetic(detector, index):
    scores = score(detector, index, [
        ('g1', 's1', 1e-100, 400, 1390),
        ('g1', 's2', 1e-10, 200, 7240),
    ])
    assert scores.max_outgroup_bitscore == 400
    assert scores.max_recipient_bitscore == 200
    assert scores.hgt_index == pytest.approx(2.0)
    assert scores.out_pct == pytest.approx(0.5)          # 1 outgroup / 2 species
    assert scores.alien_index == pytest.approx(
        math.log(1e-10 + 1e-200) - math.log(1e-100 + 1e-200)
    )
    assert scores.alien_index > 0                        # outgroup hit is better
    assert not scores.no_recipient_hits


def test_e_minus_parameter_is_used(index):
    """Regression: e_minus was a parameter but the code hardcoded 1e-200."""
    detector = HGTDetect(HGTParameters(query_taxid=7227, e_minus=1e-50))
    scores = score(detector, index, [
        ('g1', 's1', 0.0, 400, 1390),
        ('g1', 's2', 0.0, 200, 7240),
    ])
    # Both e-values are 0, so the AI is log(e_minus) - log(e_minus) == 0 and the
    # only observable effect is that it did not raise on log(0).
    assert scores.alien_index == pytest.approx(0.0)


def test_no_outgroup_hits_is_not_a_candidate(detector, index):
    assert score(detector, index, [('g1', 's2', 1e-10, 200, 7240)]) is None


def test_no_recipient_hits_is_the_strongest_candidate(detector, index):
    """Regression: these genes used to score all-zero and be discarded."""
    scores = score(detector, index, [
        ('g1', 's1', 1e-100, 400, 1390),
        ('g1', 's3', 1e-90, 380, 1423),
    ])
    assert scores is not None
    assert scores.no_recipient_hits
    assert scores.hgt_index == 1.0
    assert scores.out_pct == 1.0
    assert scores.min_recipient_evalue == NO_HIT_EVALUE
    assert scores.alien_index == pytest.approx(
        math.log(NO_HIT_EVALUE + 1e-200) - math.log(1e-100 + 1e-200)
    )
    assert detector.meets_hgt_criteria(scores)


def test_species_counts_are_distinct_species(detector, index):
    scores = score(detector, index, [
        ('g1', 's1', 1e-100, 400, 1390),
        ('g1', 's2', 1e-99, 390, 1390),   # same species, counted once
        ('g1', 's3', 1e-98, 380, 1423),
        ('g1', 's4', 1e-10, 200, 7240),
    ])
    assert scores.outgroup_count == 2
    assert scores.recipient_count == 1
    assert scores.out_pct == pytest.approx(2 / 3)


def test_thresholds_gate_candidates(index):
    detector = HGTDetect(HGTParameters(
        query_taxid=7227, bitscore_parameter=100, hgt_index=0.5, out_pct=0.6, ai_threshold=45,
    ))
    strong = score(detector, index, [
        ('g1', 's1', 1e-100, 400, 1390),
        ('g1', 's2', 1e-99, 390, 1423),
        ('g1', 's3', 1e-1, 120, 7240),
    ])
    assert detector.meets_hgt_criteria(strong)

    # Same hits, but the in-group match is just as good -> not HGT.
    weak = score(detector, index, [
        ('g1', 's1', 1e-100, 400, 1390),
        ('g1', 's2', 1e-99, 390, 1423),
        ('g1', 's3', 1e-100, 400, 7240),
    ])
    assert not detector.meets_hgt_criteria(weak)
    assert not detector.meets_hgt_criteria(None)


# ------------------------------------------------------------- gene results
def test_process_single_gene_reports_donor_lineage(permissive, index):
    hits = permissive.annotate_hits(make_hits([
        ('g1', 'bact1', 1e-100, 400, 1390),
        ('g1', 'fly1', 1e-2, 120, 7240),
    ]), index)
    result = permissive.process_single_gene('g1', hits, index)
    assert result is not None
    assert result['taxonomy']['donor_taxid'] == '1390'
    assert result['taxonomy']['donor_subject_id'] == 'bact1'
    # Regression: this dict used to be empty because ancestors were not indexed.
    assert result['taxonomy']['donor_taxonomy']['family'] == 'Bacillaceae'
    assert result['top_hits']['outgroup'][0]['species'] == 'Bacillus amyloliquefaciens'
    assert result['top_hits']['recipient'][0]['species'] == 'Drosophila simulans'


def test_all_synthetic_hits_are_not_candidates(detector, index):
    hits = detector.annotate_hits(make_hits([
        ('g1', 's1', 1e-100, 400, 32630),
    ]), index)
    assert detector.process_single_gene('g1', hits, index) is None


def test_process_genes_groups_without_scanning_per_gene(permissive, index):
    hits = permissive.annotate_hits(make_hits([
        ('g1', 'b1', 1e-100, 400, 1390),
        ('g1', 'f1', 1e-2, 120, 7240),
        ('g2', 'f2', 1e-90, 380, 7240),   # in-group only -> no outgroup signal
    ]), index)
    candidates, errors, with_hits = permissive.process_genes(['g1', 'g2', 'g3'], hits, index)
    assert [c['gene'] for c in candidates] == ['g1']
    assert with_hits == 2          # g3 had no hits at all
    assert errors == []


def test_gene_errors_are_collected_not_swallowed(detector, index, monkeypatch):
    hits = detector.annotate_hits(make_hits([('g1', 'b1', 1e-100, 400, 1390)]), index)

    def boom(*_args, **_kwargs):
        raise RuntimeError('donor lookup exploded')

    # Fails while building the detail record for a gene that passed scoring.
    monkeypatch.setattr(detector, '_donor_taxonomy', boom)
    candidates, errors, _ = detector.process_genes(['g1'], hits, index)
    assert candidates == []
    assert errors == [('g1', 'donor lookup exploded')]


# ------------------------------------------------------------------- output
def test_write_results_emits_both_files(permissive, index, tmp_path):
    hits = permissive.annotate_hits(make_hits([
        ('g1', 'bact1', 1e-100, 400, 1390),
        ('g1', 'fly1', 1e-2, 120, 7240),
    ]), index)
    candidates = [permissive.process_single_gene('g1', hits, index)]
    result = AnalysisResult(candidates=pd.DataFrame(candidates))

    out = tmp_path / 'out.tsv'
    permissive.write_results(result, out)

    body = out.read_text(encoding='utf-8').splitlines()
    assert body[0].split('\t')[0] == 'Gene/Protein'
    assert 'Bacillaceae' in body[1]

    top = result.top_hits_path.read_text(encoding='utf-8').splitlines()
    assert top[0].startswith('Gene\tHit type')
    assert any('Bacillus amyloliquefaciens' in line for line in top)


def test_write_results_on_empty_run_writes_headers(detector, tmp_path):
    result = AnalysisResult(
        candidates=pd.DataFrame(columns=['gene', 'scores', 'taxonomy', 'top_hits'])
    )
    out = tmp_path / 'empty.tsv'
    detector.write_results(result, out)
    assert out.read_text(encoding='utf-8').strip() == '\t'.join(detector.RESULT_COLUMNS)
    assert result.top_hits_path.exists()


def test_read_fasta_ids_streams_identifiers(tmp_path):
    fasta = tmp_path / 'p.faa'
    fasta.write_text('>g1 some description\nMKV\n>g2\nMKVA\n', encoding='utf-8')
    assert list(HGTDetect.read_fasta_ids(fasta)) == ['g1', 'g2']


# ------------------------------------------------------- vectorised scoring
def test_vectorised_scores_match_per_gene_scores(permissive, index):
    """The fast path must agree with calculate_scores gene for gene.

    score_all_genes replaced a Python loop with groupby aggregations; this is
    the guard that the two stay equivalent.
    """
    rows = [
        # Mixed: strong outgroup, weak ingroup.
        ('g1', 'b1', 1e-100, 400, 1390),
        ('g1', 'b2', 1e-95, 390, 1423),
        ('g1', 'f1', 1e-3, 120, 7240),
        # Outgroup only.
        ('g2', 'b3', 1e-130, 440, 1390),
        # Ingroup better than outgroup.
        ('g3', 'f2', 1e-150, 500, 7240),
        ('g3', 'b4', 1e-4, 100, 1390),
        # Repeated species on both sides.
        ('g4', 'b5', 1e-60, 250, 1390),
        ('g4', 'b6', 1e-59, 240, 1390),
        ('g4', 'f3', 1e-58, 230, 7240),
        # Duplicate e-values / bitscores.
        ('g5', 'b7', 1e-20, 150, 1390),
        ('g5', 'f4', 1e-20, 150, 7240),
        # Hit whose taxon has no species-rank ancestor.
        ('g6', 'b8', 1e-70, 300, 1386),
        ('g6', 'f5', 1e-2, 110, 7240),
    ]
    annotated = permissive.annotate_hits(make_hits(rows), index)
    usable = annotated[~annotated['_is_synthetic']]
    vectorised = permissive.score_all_genes(usable)

    for gene, gene_hits in usable.groupby('qseqid', observed=True):
        expected = permissive.calculate_scores(
            gene_hits[gene_hits['_is_recipient']],
            gene_hits[~gene_hits['_is_recipient']],
            index,
        )
        if expected is None:            # no outgroup hits -> not scored at all
            assert gene not in vectorised.index
            continue
        actual = permissive._row_to_scores(vectorised.loc[gene])
        for field_name in expected._fields:
            got, want = getattr(actual, field_name), getattr(expected, field_name)
            if isinstance(want, float):
                assert got == pytest.approx(want), f'{gene}.{field_name}'
            else:
                assert got == want, f'{gene}.{field_name}'


def test_candidate_mask_matches_meets_hgt_criteria(permissive, index):
    rows = [
        ('g1', 'b1', 1e-100, 400, 1390),
        ('g1', 'b2', 1e-95, 390, 1423),
        ('g1', 'f1', 1e-3, 120, 7240),
        ('g2', 'f2', 1e-150, 500, 7240),
        ('g2', 'b3', 1e-4, 100, 1390),
    ]
    annotated = permissive.annotate_hits(make_hits(rows), index)
    usable = annotated[~annotated['_is_synthetic']]
    scores = permissive.score_all_genes(usable)
    mask = permissive.candidate_mask(scores)

    for gene in scores.index:
        scalar = permissive.meets_hgt_criteria(permissive._row_to_scores(scores.loc[gene]))
        assert bool(mask.loc[gene]) is scalar


def test_score_all_genes_on_empty_input(permissive, index):
    hits = permissive.annotate_hits(make_hits([('g1', 'f1', 1e-10, 200, 7240)]), index)
    # In-group only, so there is no out-group frame to aggregate over.
    assert permissive.score_all_genes(hits).empty


# ----------------------------------------------------------------- taxid IO
def test_species_column_is_populated(detector, index):
    hits = detector.annotate_hits(make_hits([
        ('g1', 'b1', 1e-90, 300, 1390),
        ('g1', 'f1', 1e-20, 150, 7240),
    ]), index)
    assert list(hits['_species']) == [1390, 7240]


def test_self_hits_are_dropped_at_species_rank(index):
    """A strain-level query must not treat its own species as an in-group hit."""
    detector = HGTDetect(HGTParameters(query_taxid=7227, tax_level='family'))
    hits = make_hits([
        ('g1', 'self', 0.0, 900, 7227),
        ('g1', 'b1', 1e-90, 300, 1390),
    ])
    assert list(detector.annotate_hits(hits, index)['sseqid']) == ['b1']


def test_bitscore_is_float_even_when_integral(tmp_path, detector):
    """Integral bitscores must not land in an int column and break ratios."""
    path = tmp_path / 'hits.tsv'
    path.write_text('g1\ts1\t1e-90\t300\t100\t50\t1390\n', encoding='utf-8')
    hits = detector.load_hits(path)
    assert hits['bitscore'].dtype == 'float64'
    assert hits['evalue'].dtype == 'float64'
    assert hits['pident'].dtype == 'float32'


def test_evalue_keeps_full_dynamic_range(tmp_path, detector):
    """float32 would flush anything below ~1e-38 to zero and wreck the AI."""
    path = tmp_path / 'hits.tsv'
    path.write_text('g1\ts1\t1e-180\t300\t100\t50.0\t1390\n', encoding='utf-8')
    hits = detector.load_hits(path)
    assert hits['evalue'].iloc[0] == pytest.approx(1e-180)
    assert hits['evalue'].iloc[0] > 0


def test_qseqid_is_categorical_and_sseqid_is_not(tmp_path, detector):
    """qseqid repeats max_hits times, so a categorical pays off there.

    sseqid is close to unique, where a categorical saves no memory and makes
    every row-wise access convert the whole category array -- which is what
    made the top-hits extraction pathologically slow.
    """
    path = tmp_path / 'hits.tsv'
    path.write_text(
        ''.join(f'g1\ts{i}\t1e-90\t300\t100\t50.0\t1390\n' for i in range(50)),
        encoding='utf-8',
    )
    hits = detector.load_hits(path)
    assert str(hits['qseqid'].dtype) == 'category'
    assert str(hits['sseqid'].dtype) != 'category'
    assert str(detector._hit_taxids(hits).dtype) == 'category'


def test_pyarrow_reader_is_actually_used(tmp_path, detector, caplog):
    """Regression: usecols+names silently broke the fast path on every read.

    pyarrow looks for its generated f0..fN labels when usecols is combined with
    names, so the reader raised and every run quietly fell back to the slower
    engine.
    """
    pytest.importorskip('pyarrow')
    path = tmp_path / 'hits.tsv'
    path.write_text('g1\ts1\t1e-90\t300\t100\t50.0\t1390\n', encoding='utf-8')
    with caplog.at_level('DEBUG', logger='flexihgt.core'):
        hits = detector.load_hits(path)
    assert 'pyarrow CSV reader unavailable' not in caplog.text
    assert set(hits.columns) == set(HGTDetect.USED_COLUMNS)


def test_unused_length_column_is_dropped(tmp_path, detector):
    path = tmp_path / 'hits.tsv'
    path.write_text('g1\ts1\t1e-90\t300\t100\t50.0\t1390\n', encoding='utf-8')
    hits = detector.load_hits(path)
    # `length` is requested from the search tool but never scored or reported.
    assert 'length' not in hits.columns
    assert set(hits.columns) == set(HGTDetect.USED_COLUMNS)


def test_both_readers_agree(tmp_path, detector, monkeypatch):
    """The fallback must produce the same frame as the fast path."""
    pytest.importorskip('pyarrow')
    path = tmp_path / 'hits.tsv'
    path.write_text(
        'g1\ts1\t1e-90\t300\t100\t50.0\t1390\n'
        'g1\ts2\t0.0\t420\t210\t62.5\t1423;7240\n',
        encoding='utf-8',
    )
    fast = detector.load_hits(path)

    import pandas as _pd
    real_read_csv = _pd.read_csv

    def no_pyarrow(*args, **kwargs):
        if kwargs.get('engine') == 'pyarrow':
            raise ImportError('pyarrow disabled for this test')
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(_pd, 'read_csv', no_pyarrow)
    slow = detector.load_hits(path)

    assert list(fast.columns) == list(slow.columns)
    assert fast.to_dict('records') == slow.to_dict('records')


def test_colliding_staxids_do_not_crash(detector, index):
    """Regression: distinct staxids fields can end at the same taxid.

    '7240;1390' and '1390' both resolve to 1390, and renaming a categorical's
    categories onto a non-unique list raises. NCBI's nr routinely annotates a
    subject with several taxids, so this aborted real runs after the search.
    """
    hits = make_hits([
        ('g1', 's1', 1e-90, 300.0, '7240;1390'),
        ('g1', 's2', 1e-80, 280.0, '1390'),
        ('g1', 's3', 1e-70, 260.0, '1423;7240;1390'),
        ('g2', 's4', 1e-60, 240.0, '7240'),
    ])
    taxids = HGTDetect._hit_taxids(hits)
    assert list(taxids) == ['1390', '1390', '1390', '7240']
    assert list(taxids.cat.categories) == ['1390', '7240']

    annotated = detector.annotate_hits(hits, index)
    assert list(annotated['_taxid']) == ['1390', '1390', '1390', '7240']


def test_hit_taxids_keeps_missing_values_missing(detector):
    hits = make_hits([('g1', 's1', 1e-9, 200.0, '1390')])
    hits.loc[len(hits)] = ('g2', 's2', 1e-9, 200.0, 100, 50.0, None)
    taxids = HGTDetect._hit_taxids(hits)
    assert taxids.tolist()[0] == '1390'
    assert pd.isna(taxids.tolist()[1])


def test_query_taxid_without_the_requested_rank_is_fatal(index):
    """A rank the query lineage lacks makes every score meaningless.

    Nothing can be in-group, so the pandas engine scores the whole proteome as
    if it had no recipient hits while the out-of-core engine reports nothing.
    Both used to happen silently; D. melanogaster has no tribe.
    """
    detector = HGTDetect(HGTParameters(query_taxid=7227, tax_level='tribe'))
    with pytest.raises(RuntimeError, match='no tribe-rank ancestor'):
        detector._check_query_rank(index)


def test_query_taxid_with_the_requested_rank_is_accepted(index):
    detector = HGTDetect(HGTParameters(query_taxid=7227, tax_level='family'))
    detector._check_query_rank(index)          # must not raise


def test_genes_hitting_only_synthetic_constructs_are_counted(detector, index):
    hits = detector.annotate_hits(make_hits([
        ('g1', 's1', 1e-90, 300.0, 32630),
        ('g1', 's2', 1e-80, 280.0, 32630),
        ('g2', 's3', 1e-70, 260.0, 1390),
        ('g3', 's4', 1e-60, 240.0, 32630),
        ('g3', 's5', 1e-50, 220.0, 1390),
    ]), index)
    # g1 only; g3 still has a real hit and g2 has no synthetic hit at all.
    assert HGTDetect.count_all_synthetic(hits) == 1
