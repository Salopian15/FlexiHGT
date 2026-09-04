# Installation Instructions for FlexiHGT

## Prerequisites

FlexiHGT requires:

- Python 3.9–3.12 (`ete3` does not install cleanly on 3.13)
- [DIAMOND](https://github.com/bbuchfink/diamond) — required for the default
  `-s diamond` search
- [MMseqs2](https://github.com/soedinglab/MMseqs2) — only needed for `-s mmseqs`

## Recommended: conda

This installs the search tools and an isolated Python in one step:

```bash
conda create -n flexihgt -c bioconda -c conda-forge python=3.11 diamond mmseqs2
conda activate flexihgt
pip install .
```

## Without conda

The helper script downloads current Linux or macOS builds into a directory of
your choice (`~/.local/bin` by default) — no `sudo` required:

```bash
bash scripts/install_dependencies.sh              # installs to ~/.local/bin
bash scripts/install_dependencies.sh /opt/bin     # or a directory you pick
```

Make sure the target directory is on your `PATH`, then install FlexiHGT itself:

```bash
pip install .
```

## Taxonomy database

FlexiHGT resolves lineages from a local NCBI taxonomy database (~100 MB, cached
in `~/.etetoolkit/taxa.sqlite`). Download or refresh it with:

```bash
flexihgt --update-only
```

## Verifying the installation

```bash
flexihgt --version
diamond --version
```

Every prerequisite is re-checked automatically at the start of each run, along
with a one-second probe confirming the search database actually carries
taxonomy — a database built without `--taxonmap` returns hits with no taxids,
all of which would be discarded after the search had already run. Use
`--skip-checks` to bypass both.

## Optional extras

```bash
pip install '.[cache]'    # pyarrow: faster parsing + annotated-hits cache
pip install '.[bigdata]'  # duckdb: score hit tables larger than memory
pip install '.[taxopy]'   # taxopy: faster alternative to the ete3 taxonomy backend
```

None is required. Without them FlexiHGT parses with pandas, re-annotates on each
run, loads the hit table into memory, and uses ete3 respectively.

## Development install

```bash
pip install -e '.[test]'
pytest
```

The tests need neither the taxonomy database nor the search tools.
