# FlexiHGT

A horizontal gene transfer (HGT) detection suite derived from the HGTphyloDetect
software suite. It allows flexible selection of the taxonomic level of interest,
and processes data quickly enough to run at proteome scale.

See: Yuan, Le, et al. *HGTphyloDetect: facilitating the identification and
phylogenetic analysis of horizontal gene transfer.* Briefings in Bioinformatics
(2023). <https://doi.org/10.1093/bib/bbad035> —
<https://github.com/SysBioChalmers/HGTphyloDetect>

## Installation

Create a conda environment (Python 3.9–3.12; `ete3` does not install cleanly on
3.13) and pull the search tools from bioconda:

```bash
conda create -n flexihgt -c bioconda -c conda-forge python=3.11 diamond mmseqs2
conda activate flexihgt
pip install .
```

Then download the NCBI taxonomy database once (~100 MB, cached in
`~/.etetoolkit/`):

```bash
flexihgt --update-only
```

See [INSTALL.md](INSTALL.md) for installing DIAMOND/MMseqs2 without conda.

## Usage

You need three things: a proteome FASTA file, the NCBI taxid of the organism
those proteins come from, and a search database.

```bash
flexihgt input.faa -q 12344 -db path/to/db.dmnd -s diamond
```

With the defaults, hits that do **not** share the query's *family* are treated
as the out-group, and a gene is reported when it clears all four thresholds:
out-group bitscore ≥ 100, HGT index ≥ 0.5, out-group species fraction ≥ 0.8 and
Alien Index ≥ 45.

`--tax_level` moves the in-group/out-group boundary to any NCBI rank, from
`species` up to `superkingdom`:

```bash
flexihgt input.faa -q 12344 -db path/to/db.dmnd -t phylum
```

The rank has to be one the query organism's own lineage has. NCBI assigns ranks
patchily — plenty of bacteria have no family, and `tribe` or `subgenus` are
absent almost everywhere — and without an ancestor at the chosen rank nothing
could ever count as in-group, so the run stops and tells you which ranks are
available instead of scoring the whole proteome as foreign.

### Databases

| `-s` | Database | Built with |
| --- | --- | --- |
| `diamond` (default) | DIAMOND `.dmnd` file | `diamond makedb --taxonmap … --taxonnodes …` |
| `mmseqs` | MMseqs2 database prefix | `mmseqs createdb` then `mmseqs createtaxdb` |

Both must carry taxonomy information — FlexiHGT needs a taxid per hit and drops
hits that do not have one.

### Output

Two TSV files are written per run:

- `<input>_<tax_level>_HGT.tsv` — one row per candidate, with each score, the
  donor taxid and the donor's full resolved lineage. Override with `-o`.
- `<name>_top_hits.tsv` — the best hits from each side of the split, for
  eyeballing individual calls.

### Caching and threshold sweeps

Two intermediates are cached next to the input, each with a sidecar recording
what produced it:

| File | Holds | Reused when |
| --- | --- | --- |
| `<input>.hits.tsv` | raw search results | database, query taxon, `--max_hits` and `--evalue` all unchanged |
| `<input>.hits.tsv.annotated.parquet` | hits with taxonomy resolved | the above, plus `--tax_level` unchanged |

Thresholds affect neither, so re-running with different cutoffs skips both the
search *and* the taxonomy step:

```bash
flexihgt input.faa -q 12344 --rescore-only --AI 30 --out_pct 0.6
```

`--force` re-runs the search regardless; `--no-cache` disables the annotation
cache. The parquet cache needs `pyarrow` (`pip install '.[cache]'`); without it
the pipeline works normally and just re-annotates each time.

### Reproducibility

Every results file gets a `<name>.params.json` sidecar recording the FlexiHGT
version, the input, the database and every parameter, so a TSV found six months
later can be traced back to the run that made it. It is a sidecar rather than
comment lines in the TSV so downstream parsers are unaffected.

Options can also come from a config file, which command line flags override:

```bash
flexihgt input.faa -q 12344 -db db.dmnd --config analysis.toml
```

```toml
# analysis.toml
tax_level = "phylum"
AI = 30
out_pct = 0.6
max_hits = 500
```

### Performance

The homology search dominates runtime — everything downstream is vectorised and
runs in seconds even at proteome scale. The levers that matter:

| Flag | Effect |
| --- | --- |
| `--threads` | Cores for the search tool (default: all) |
| `--block_size` (DIAMOND `-b`) | Main throughput lever; needs roughly 6× its value in GB of RAM |
| `--index_chunks` (DIAMOND `-c`) | Fewer chunks, fewer passes over the database. `-b6 -c1` suits a large-RAM node |
| `--evalue` | Default `1e-5`, tighter than DIAMOND's own `1e-3`; keeps the hit table much smaller |
| `--tmpdir` | Point at local NVMe when the working directory is on a network filesystem |
| `--memory_limit` | MMseqs2 `--split-memory-limit`, e.g. `64G` |
| `--engine` | `pandas` loads the hit table into memory; `duckdb` streams it from disk. `auto` (default) picks `duckdb` above 2 GB |

Install `pyarrow` (`pip install '.[cache]'`) for ~2.3× faster hit-table parsing,
and `duckdb` (`pip install '.[bigdata]'`) to score tables larger than memory.
Both engines are checked against each other by the test suite and produce
byte-identical output.

Before a long job, `--dry-run` reports what would happen — sequence counts,
projected hit-table size, which engine, and what is already cached — without
running anything.

`--taxonomy-backend taxopy` swaps ete3 for [taxopy](https://github.com/apcamargo/taxopy),
which keeps `nodes.dmp`/`names.dmp` in memory instead of querying SQLite. Install
it with `pip install '.[taxopy]'` and point `--taxonomy-dir` at the dump files.

## Many genomes at once

`flexihgt-multi` analyses a set of proteomes from a **single** combined search.
It is a separate command — `flexihgt` and its behaviour are unchanged.

Describe the genomes in a TSV or CSV manifest. `taxid` and `fasta` are required;
`genome_id` defaults to the FASTA's stem, and relative paths resolve against the
manifest's own directory:

```
genome_id	taxid	fasta
GCA_000001215	7227	proteomes/dmel.faa
GCA_000005575	7165	proteomes/agam.faa.gz
```

```bash
flexihgt-multi manifest.tsv -db path/to/db.dmnd -o results/ -t family
```

Every genome gets its own in-group boundary from its own taxid, so `-t family`
means "outside *this* genome's family", evaluated per gene. Gzipped proteomes are
read directly, and protein ids containing `|` (`sp|P12345|NAME`) are preserved.

A genome whose taxid cannot be resolved, or whose lineage has no ancestor at the
chosen rank, is skipped with its reason recorded in `run_summary.tsv` rather than
scored against a boundary that does not exist. One bad row does not stop the run;
the command exits 4 so a wrapper can notice.

Output:

| Path | Contents |
| --- | --- |
| `results/combined_<tax_level>_HGT.tsv` | every candidate, with a `Genome` column |
| `results/genomes/<genome_id>_<tax_level>_HGT.tsv` | per genome, identical in format to a single-genome run |
| `results/run_summary.tsv` | one row per genome: gene counts, candidates, errors, skip reasons |
| `results/work/combined_query.faa` | the concatenated query set, rebuilt only when the genome set changes |

Building a manifest by hand is fine for ten genomes and miserable for thousands,
so `--make-manifest` scans a directory and works the taxids out for you:

```bash
flexihgt-multi proteomes/ --make-manifest manifest.tsv
```

Taxids are taken, in order of preference, from a `--taxid-map` override file, an
NCBI datasets `assembly_data_report.jsonl`, an NCBI `*_assembly_report.txt`
header, or a UniProt-style `UP000000803_7227.fasta` filename. Proteomes whose
taxid cannot be determined are listed rather than silently dropped, and the
command exits 4 so a script notices.

Useful flags:

- `--genomes` — restrict to a comma-separated list of ids or a file of ids, for
  splitting one manifest across a job array.
- `--resume` — skip genomes whose results already exist, so an interrupted run
  can be restarted.
- `--rescore-only` — reuse the combined hit table and just re-apply thresholds.
- `--no-per-genome` — write only the combined table.
- `--dry-run` — report the plan and stop.
- `--log-level WARNING` — the default logs a line per candidate, which is too
  much for very large runs.

A genome whose taxid cannot be resolved is skipped and recorded in the summary
rather than aborting the run; the command exits 4 if anything was skipped or
failed.

### Scale

The search is issued once for all genomes, which is what makes this affordable —
one database load and index build instead of one per genome. Because the taxon to
exclude differs per genome, `--taxon-exclude` is not used; self-hits are removed
after loading by comparing at species rank, which is exact.

With `duckdb` installed the hit table is scored from disk, so memory stops being
the limit; without it, the table is loaded and a warning is logged past 2 GB.
Per-genome cost is dominated by the number of *candidates*, not the number of
genomes — roughly 10 ms per genome at a realistic candidate rate.

What is **not** yet addressed for phylum-scale work: query dereplication before
the search, and clustering candidates across species so that one ancient
transfer shared by 400 relatives is counted once rather than 400 times.

### Scores

| Score | Meaning |
| --- | --- |
| HGT index | best out-group bitscore ÷ best in-group bitscore |
| Out_pct | fraction of the distinct hit species that fall outside the in-group |
| Alien Index | `ln(best in-group E-value) − ln(best out-group E-value)`; positive means the out-group match is the better one |

A gene with **no in-group hits at all** is reported with `No recipient hits =
yes`, an HGT index of 1.0, and an Alien Index computed against `E = 1.0`.

Hits to the query organism are excluded twice over: the search tool is asked to
skip the query's whole *species* clade (so sister strains go too), and any that
slip through are dropped after loading.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Completed; results written |
| 2 | Bad arguments, missing input file/database, or a failed environment check |
| 3 | The analysis failed (search error, unusable taxonomy database) |
| 4 | Completed, but some genes could not be scored — treat the results as partial |

## Development

```bash
pip install -e '.[test]'
pytest
```

The test suite substitutes an in-memory taxonomy for ete3 and stubs the search
tools, so it runs without downloading `taxa.sqlite` and without DIAMOND or
MMseqs2 installed.
