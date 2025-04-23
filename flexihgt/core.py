import logging
import sys
import os
import warnings
import math
import csv
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any, Optional, Iterator, NamedTuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import pandas as pd
import numpy as np
from Bio import SeqIO, BiopythonWarning
from ete3 import NCBITaxa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaxonomyInfo(NamedTuple):
    """Structure for taxonomy information"""
    taxid: str
    rank: str
    name: str
    lineage: Tuple[int, ...]
    alignment: Dict[str, int]

class HGTScores(NamedTuple):
    """Structure for HGT-related scores"""
    max_outgroup_bitscore: float
    max_recipient_bitscore: float
    hgt_index: float
    out_pct: float
    alien_index: float
    outgroup_count: int
    recipient_count: int
    min_outgroup_evalue: float
    min_recipient_evalue: float

@dataclass
class HGTParameters:
    """Configuration parameters for HGT detection"""
    bitscore_parameter: float = 100
    hgt_index: float = 0.5
    out_pct: float = 0.8
    ai_threshold: float = 45
    tax_level: str = "family"
    search_method: str = "diamond"
    e_minus: float = 1e-200
    query_taxid: Optional[int] = None
    max_hits: int = 200

    def __post_init__(self):
        """Validate parameters after initialization"""
        if not 0 <= self.hgt_index <= 1:
            raise ValueError("HGT index must be between 0 and 1")
        if not 0 <= self.out_pct <= 1:
            raise ValueError("Out percentage must be between 0 and 1")
        if self.query_taxid is not None and self.query_taxid <= 0:
            raise ValueError("Query taxid must be positive")

class HGTDetect:
    """Class to detect HGT events in protein sequences with improved performance and error handling"""

    SYNTHETIC_KEYWORDS = {
                    'synthetic', 'vector', 'construct', 'artificial', 
                    'engineered', 'cloning', 'expression', 'plasmid',
                    'synthetic construct', 'vector:', 'plasmid:', 'cloning:',
                    'expression:', 'artificial construct', 'engineered construct',
                    'Synthetic', 'Vector', 'Construct', 'Artificial',
                    'Engineered', 'Cloning', 'Expression', 'Plasmid',
                    'Synthetic Construct', 'Vector:', 'Plasmid:', 'Cloning:',
                    'Expression:', 'Artificial Construct', 'Engineered Construct'
    }

    TAX_RANKS = [
        'species', 'subgenus', 'genus', 
        'subtribe', 'tribe', 'subfamily', 'family', 'superfamily', 
        'infraorder', 'suborder', 'order', 'superorder', 
        'infraclass', 'subclass', 'class', 'superclass', 
        'subphylum', 'phylum', 'superphylum', 
        'subkingdom', 'kingdom', 'superkingdom'
    ]

    def __init__(self, params: Optional[HGTParameters] = None) -> None:
        """Initialize HGT detection with optional custom parameters"""
        self.params = params or HGTParameters()
        self.ncbi = NCBITaxa()
        self._setup_caches()

    def _setup_caches(self, cache_size: int = 10000) -> None:
        """Initialize LRU caches with reasonable size limits"""
        self.get_lineage = lru_cache(maxsize=cache_size)(self._get_lineage)
        self.get_rank = lru_cache(maxsize=cache_size)(self._get_rank)
        self.get_name = lru_cache(maxsize=cache_size)(self._get_name)

    def precompute_taxonomy(self, diamond_results: pd.DataFrame) -> Dict[str, TaxonomyInfo]:
        """Precompute all taxonomy information using ete3, including handling merged TaxIDs."""
        try:
            # Initialize NCBITaxa
            ncbi = NCBITaxa()

            # Extract unique taxids from input and clean them
            unique_taxids = set()
            for taxid in pd.concat([
                diamond_results[6].dropna().str.split(';').str[-1],
                pd.Series([str(self.params.query_taxid)])
            ]):
                try:
                    if pd.notna(taxid) and str(taxid).strip():
                        unique_taxids.add(int(float(taxid)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid taxid found: {taxid}")
                    continue

            # Handle merged TaxIDs
            final_taxids, merged_map = ncbi._translate_merged(list(unique_taxids))
            final_taxids = list(final_taxids)

            # Fetch taxonomy data in bulk
            lineages = ncbi.get_lineage_translator(final_taxids)
            ranks = ncbi.get_rank(final_taxids)
            names = ncbi.get_taxid_translator(final_taxids)

            # Build taxonomy dictionary with validation and alignments
            taxonomy_info = {}
            for orig_id in unique_taxids:
                try:
                    final_id = merged_map.get(orig_id, orig_id)
                    lineage = lineages.get(final_id, [])
                    lineageorig = lineages.get(orig_id, [])
                    if not lineage or not lineageorig:
                        logger.warning(f"No lineage found for taxid {orig_id}")
                        continue
                    # Check if lineage is empty and use lineageorig if available
                    if not lineage and lineageorig:
                        lineage = lineageorig
                    # Create taxonomy alignment as in reference code
                    gene_lineage = lineage
                    gene_lineage2ranks = ncbi.get_rank(gene_lineage)
                    gene_ranks2lineage = dict((rank, taxid) for (taxid, rank) in gene_lineage2ranks.items())
                    taxonomy_alignment = gene_ranks2lineage

                    # Create TaxonomyInfo with alignment included
                    taxonomy_info[str(orig_id)] = TaxonomyInfo(
                        taxid=str(final_id),
                        rank=ranks.get(final_id, "unknown"),
                        name=names.get(final_id, "unknown"),
                        lineage=tuple(lineage),
                        alignment=taxonomy_alignment
                    )

                except Exception as e:
                    logger.warning(f"Error processing taxid {orig_id}: {str(e)}")
                    continue

            logger.info(f"Successfully precomputed taxonomy for {len(taxonomy_info)} taxids")

            # Save to file for debugging/reference
            with open("taxonomy_info.csv", "w") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["taxid", "rank", "name", "lineage", "alignment"])
                for taxid, info in taxonomy_info.items():
                    writer.writerow([
                        info.taxid,
                        info.rank,
                        info.name,
                        info.lineage,
                        str(info.alignment)
                    ])

            return taxonomy_info

        except Exception as e:
            logger.error(f"Failed to precompute taxonomy: {str(e)}")
            return {}



    def _get_taxonomy_alignment(self, taxid: str, taxonomy_info: Dict[str, TaxonomyInfo]) -> Dict[str, int]:
        """Get taxonomy alignment at different ranks"""
        try:
            tax_info = taxonomy_info.get(str(taxid))
            if not tax_info or not tax_info.lineage:
                return {}

            # Get ranks for all taxids in lineage
            ranks = {str(tid): self.ncbi.get_rank([tid])[tid]
                    for tid in tax_info.lineage}

            # Create alignment dictionary
            alignment = {}
            for tid, rank in ranks.items():
                if rank in self.TAX_RANKS:
                    alignment[rank] = tid

            return alignment
        except Exception as e:
            logger.warning(f"Error getting taxonomy alignment for {taxid}: {e}")
            return {}

    def _is_recipient_taxid(self, taxid: str, taxonomy_info: Dict[str, TaxonomyInfo]) -> bool:
        """Check if taxid belongs to recipient group based on tax level"""
        if not self.params.query_taxid:
            return False

        try:
            # Get taxonomy info directly
            query_info = taxonomy_info.get(str(self.params.query_taxid))
            target_info = taxonomy_info.get(str(taxid))
            
            if not query_info or not target_info:
                return False
                
            # Compare at specified tax level using alignments
            query_taxlevel = query_info.alignment.get(self.params.tax_level)
            target_taxlevel = target_info.alignment.get(self.params.tax_level)
            
            if query_taxlevel and target_taxlevel:
                return query_taxlevel == target_taxlevel
                
            return False
            
        except Exception as e:
            logger.warning(f"Error comparing taxids {taxid}: {str(e)}")
            return False

    def _calculate_scores(self, results: pd.DataFrame, taxonomy_info: Dict[str, TaxonomyInfo]) -> HGTScores:
        """Calculate HGT-related scores using taxonomy alignments from TaxonomyInfo"""
        try:
            # Split hits based on taxonomy alignment
            is_recipient = results[6].str.split(';').str[-1].apply(
                lambda x: self._is_recipient_taxid(str(x).strip(), taxonomy_info)
            )
            recipient_hits = results[is_recipient]
            outgroup_hits = results[~is_recipient]

            if recipient_hits.empty or outgroup_hits.empty:
                logger.warning("No recipient or outgroup hits found, skipping scores.")
                logger.info(
                    "Recipient hits: %s, "
                    "Outgroup hits: %s",
                    len(
                        recipient_hits
                    ),
                    len(
                        outgroup_hits
                    )
                )
                return HGTScores(
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                )
                
            # Get unique species counts using taxonomy alignments directly from TaxonomyInfo
            recipient_species: Set[int] = set()
            outgroup_species: Set[int] = set()
            
            # Process species for both recipient and outgroup hits
            for hits, species_set in [(recipient_hits, recipient_species), 
                                    (outgroup_hits, outgroup_species)]:
                for taxid in hits[6].str.split(';').str[-1].unique():
                    tax_info = taxonomy_info.get(str(taxid))
                    if tax_info and 'species' in tax_info.alignment:
                        species_set.add(tax_info.alignment['species'])

            # Calculate scores
            max_recipient_bitscore = recipient_hits[3].max()
            max_outgroup_bitscore = outgroup_hits[3].max()

            # Calculate HGT index with safety checks
            hgt_index = (max_outgroup_bitscore / max_recipient_bitscore 
                        if max_recipient_bitscore > 0 else 0)

            # Calculate species counts
            total_species = len(recipient_species) + len(outgroup_species)
            out_pct = len(outgroup_species) / total_species if total_species > 0 else 0

            # Calculate alien index
            min_recipient_evalue = recipient_hits[2].min()
            min_outgroup_evalue = outgroup_hits[2].min()
            e_minus = 1e-200

            alien_index = (
                math.log(min_recipient_evalue + e_minus) -
                math.log(min_outgroup_evalue + e_minus)
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
                min_recipient_evalue=min_recipient_evalue
            )

        except Exception as e:
            logger.error("Error calculating scores: %s", e)
            return HGTScores(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def process_fasta(self, fasta_path: Path, batch_size: int = 1000) -> Iterator[Tuple[str, str]]:
        """Process FASTA file in batches to conserve memory"""
        with open(fasta_path, encoding='UTF-8') as handle:
            batch = []
            for record in SeqIO.parse(handle, "fasta"):
                batch.append((str(record.id), str(record.seq)))
                if len(batch) >= batch_size:
                    yield from batch
                    batch = []
            if batch:
                yield from batch
        logger.info("Finished processing FASTA file: %s", fasta_path)

    def run_diamond_search(self, input_file: Path, db_path: Path) -> Path:
        """Run DIAMOND search with better error handling and validation"""
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        output_file = input_file.with_suffix('.tsv')
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f'Using existing Diamond results: {output_file}')
            return output_file

        cmd = (
            f'diamond blastp -d {db_path} -q {input_file} '
            '--max-target-seqs 250 --outfmt 6 qseqid sseqid evalue bitscore length pident staxids '
            f'-o {output_file} '
            f'--taxon-exclude {self.params.query_taxid}'  # Now part of the same string
        )

        logger.info(f"Running Diamond search: {cmd}")
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError(f"Diamond search failed with exit code {result}")

        logger.info("Finished Diamond search. Results saved to %s", output_file)
        return output_file

    def process_single_gene(self, gene: str, gene_results: pd.DataFrame,
                          taxonomy_info: Dict[str, TaxonomyInfo]) -> Optional[Dict[str, Any]]:
        """Process a single gene with precomputed taxonomy information"""
        try:
            if gene_results.empty:
                logger.warning("No results found for gene %s", gene)
                return None

            # Filter synthetic results using precomputed taxonomy
            gene_results = self._filter_synthetic_results(gene_results, taxonomy_info)
            if gene_results is None or gene_results.empty:
                logger.warning("No non-synthetic results for gene %s", gene)
                return None

            # Calculate scores using precomputed taxonomy
            scores = self._calculate_scores(gene_results, taxonomy_info)
            
            # Extract top hits (both recipient and outgroup)
            top_hits = self._extract_top_hits(gene_results, taxonomy_info, n=5)
            
            if self._meets_hgt_criteria(scores):
                logger.info("Gene %s meets HGT criteria. Scores: %s", gene, scores)
            if not self._meets_hgt_criteria(scores):
                logger.info("Gene %s does not meet HGT criteria.", gene)
                return None

            return {
                'gene': gene,
                'scores': scores,
                'taxonomy': self._get_taxonomy_info(gene_results, scores, taxonomy_info),
                'top_hits': top_hits  # Add top hits to the results
            }

        except Exception as e:
            logger.error(f"Error processing gene {gene}: {e}", exc_info=True)
            return None

    def process_genes_parallel(self, genes: List[str], diamond_results: pd.DataFrame,
                             taxonomy_info: Dict[str, TaxonomyInfo],
                             num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process genes in parallel using ProcessPoolExecutor"""
        num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for gene in genes:
                future = executor.submit(
                    self.process_single_gene,
                    gene,
                    diamond_results[diamond_results[0] == gene],
                    taxonomy_info
                )
                futures.append(future)

            results = []
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error("Error processing gene: %s", e, exc_info=True)
                    continue
        logger.info("Finished processing %d genes", len(genes))
        return results

    def run_analysis(self, input_file: Path, db_path: Path) -> pd.DataFrame:
        """Main method to run the complete HGT analysis pipeline"""
        try:
            # Validate input files
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")

            # Run DIAMOND search
            diamond_results_path = self.run_diamond_search(input_file, db_path)
            diamond_results = pd.read_csv(diamond_results_path, sep='\t', header=None)

            # Precompute taxonomy information
            logger.info("Precomputing taxonomy information...")
            taxonomy_info = self.precompute_taxonomy(diamond_results)

            # Load and preprocess sequences
            sequences = list(self.process_fasta(input_file))
            genes = [seq[0] for seq in sequences]

            # Process genes in parallel with precomputed taxonomy
            results = self.process_genes_parallel(genes, diamond_results, taxonomy_info)

            # Convert results to DataFrame
            df = pd.DataFrame(results)

            # Write results
            self._write_results(df, input_file.stem)
            logger.info("Analysis complete")
            logger.info("Found %d potential HGT events", len(df))
            logger.info("Results saved to %s", f"{input_file.stem}_hgt_results.tsv")
            return df

        except Exception as e:
            logger.error("Analysis failed: %s", e, exc_info=True)
            raise

    def _meets_hgt_criteria(self, scores: HGTScores) -> bool:
        """
        Determine if a gene meets the criteria for HGT
        """
        return (
            scores.max_outgroup_bitscore >= self.params.bitscore_parameter and
            scores.hgt_index >= self.params.hgt_index and
            scores.out_pct >= self.params.out_pct and
            scores.alien_index >= self.params.ai_threshold
        )

    def _filter_synthetic_results(self, gene_results: pd.DataFrame, 
                                taxonomy_info: Dict[str, TaxonomyInfo]) -> Optional[pd.DataFrame]:
        """Filter synthetic results using precomputed taxonomy"""
        def check_taxid(taxid: str) -> bool:
            info = taxonomy_info.get(taxid)
            if info is None:
                return False
            return not any(keyword in info.name.lower() for keyword in self.SYNTHETIC_KEYWORDS)

        mask = gene_results[6].str.split(';').str[-1].apply(check_taxid)
        filtered_df = gene_results[mask]

        if filtered_df.empty:
            logger.warning("All hits were synthetic constructs")
            return None

        return filtered_df


    def _get_taxonomy_info(
        self,
        results: pd.DataFrame,
        scores: HGTScores,
        taxonomy_info: Dict[str, TaxonomyInfo]
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get detailed taxonomy information for potential HGT donors"""
        if not scores.max_outgroup_bitscore:
            return {"donor_taxonomy": "No potential donors found"}

        # Get the hit with maximum bitscore from outgroup
        max_hit = results[
            (results[3] == scores.max_outgroup_bitscore) &
            (results[6].str.split(';').str[-1].map(
                lambda x: not self._is_recipient_taxid(x, taxonomy_info)
            ))
        ].iloc[0]

        donor_taxid = max_hit[6].split(';')[-1]
        donor_info = taxonomy_info.get(donor_taxid)

        if donor_info is None:
            return {"donor_taxonomy": "Taxonomy lookup failed"}

        try:
            lineage_info = {
                str(taxid): taxonomy_info.get(str(taxid))
                for taxid in donor_info.lineage
                if str(taxid) in taxonomy_info
            }

            return {
                "donor_taxid": donor_taxid,
                "donor_taxonomy": {
                    info.rank: info.name
                    for info in lineage_info.values()
                    if info and info.rank
                },
                "donor_alignment": donor_info.alignment
            }

        except Exception as e:
            logger.error("Error getting donor taxonomy: %s", str(e))
            return {"donor_taxonomy": "Taxonomy lookup failed"}

    def _write_results(self, results: pd.DataFrame, prefix: str) -> None:
        """Write analysis results to files in a structured TSV format
        
        Args:
            results: DataFrame containing HGT analysis results
            prefix: Prefix for output filenames
        """
        # Write main results
        output_file = f"{prefix}_hgt_results.tsv"
        if results.empty:
            logger.warning("No results to write")
            return

        try:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                tsv_writer = csv.writer(outfile, delimiter='\t')
                
                # Write header
                columns = [
                    'Gene/Protein',
                    'Bitscore',
                    'Out_pct', 
                    'HGT index',
                    'Alien Index',
                    'Min Outgroup E-value',
                    'Donor taxonomy'
                ]
                tsv_writer.writerow(columns)

                # Process each result row
                for _, row in results.iterrows():
                    scores = row['scores']
                    taxonomy = row['taxonomy']

                    # Format donor taxonomy string
                    donor_tax = taxonomy.get('donor_taxonomy', {})
                    if isinstance(donor_tax, dict):
                        donor_tax_str = '; '.join(f"{rank}: {name}" 
                                                  for rank, name in donor_tax.items())
                    else:
                        donor_tax_str = str(donor_tax)

                    # Prepare row data
                    row_data = [
                        row['gene'],
                        f"{scores.max_outgroup_bitscore:.2f}",
                        f"{scores.out_pct:.2f}",
                        f"{scores.hgt_index:.2f}",
                        f"{scores.alien_index:.2f}",
                        f"{scores.min_outgroup_evalue:.2e}",
                        donor_tax_str
                    ]

                    tsv_writer.writerow(row_data)

        except Exception as e:
            logger.error(f"Error writing results: {str(e)}")
            raise

        # Add a new file for top hits
        if not results.empty:
            hits_file = f"{prefix}_top_hits.tsv"
            with open(hits_file, 'w', encoding='utf-8') as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                
                # Write header
                header = ['Gene', 'Hit Type', 'Subject ID', 'E-value', 'Bitscore', 'TaxID', 'Species']
                tsv_writer.writerow(header)
                
                # Write hits for each gene
                for _, row in results.iterrows():
                    gene = row['gene']
                    top_hits = row.get('top_hits', {'recipient': [], 'outgroup': []})
                    
                    # Write recipient hits
                    for hit in top_hits.get('recipient', []):
                        tsv_writer.writerow([
                            gene,
                            'Recipient',
                            hit['subject_id'],
                            f"{hit['evalue']:.2e}",
                            f"{hit['bitscore']:.2f}",
                            hit['taxid'],
                            hit['species']
                        ])
                    
                    # Write outgroup hits
                    for hit in top_hits.get('outgroup', []):
                        tsv_writer.writerow([
                            gene,
                            'Outgroup',
                            hit['subject_id'],
                            f"{hit['evalue']:.2e}",
                            f"{hit['bitscore']:.2f}",
                            hit['taxid'],
                            hit['species']
                        ])
                
                logger.info(f"Top hits written to {hits_file}")

    # Base methods for caching
    def _get_lineage(self, taxid: int) -> Tuple[int, ...]:
        """Get taxonomy lineage for a taxid"""
        try:
            return tuple(self.ncbi.get_lineage(taxid))
        except Exception as e:
            logger.error(f"Error getting lineage for taxid {taxid}: {e}")
            return tuple()

    def _get_rank(self, taxid: int) -> str:
        """Get taxonomic rank for a taxid"""
        try:
            return self.ncbi.get_rank([taxid])[taxid]
        except Exception as e:
            logger.error(f"Error getting rank for taxid {taxid}: {e}")
            return "unknown"

    def _get_name(self, taxid: int) -> str:
        """Get scientific name for a taxid"""
        try:
            return self.ncbi.get_taxid_translator([taxid])[taxid]
        except Exception as e:
            logger.error(f"Error getting name for taxid {taxid}: {e}")
            return "unknown"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'HGTDetect':
        """Create HGTDetect instance from command line arguments"""
        params = HGTParameters(
            bitscore_parameter=args.bitscore_parameter,
            hgt_index=args.HGTIndex,
            out_pct=args.out_pct,
            ai_threshold=args.AI,
            tax_level=args.tax_level,
            search_method=args.search,
            query_taxid=args.query_tax
        )
        return cls(params)

    def _extract_top_hits(self, gene_results: pd.DataFrame, 
                        taxonomy_info: Dict[str, TaxonomyInfo], 
                        n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Extract top hits from both recipient and outgroup taxa
        
        Args:
            gene_results: DataFrame with DIAMOND results
            taxonomy_info: Dictionary with taxonomy information
            n: Number of top hits to extract
            
        Returns:
            Dictionary with recipient and outgroup top hits
        """
        try:
            # Split hits based on taxonomy
            is_recipient = gene_results[6].str.split(';').str[-1].apply(
                lambda x: self._is_recipient_taxid(str(x).strip(), taxonomy_info)
            )
            recipient_hits = gene_results[is_recipient]
            outgroup_hits = gene_results[~is_recipient]
            
            # Sort by bitscore (column 3) in descending order
            recipient_hits = recipient_hits.sort_values(by=3, ascending=False).head(n)
            outgroup_hits = outgroup_hits.sort_values(by=3, ascending=False).head(n)
            
            # Format hits with more readable information
            formatted_recipient = []
            formatted_outgroup = []
            
            for _, hit in recipient_hits.iterrows():
                taxid = hit[6].split(';')[-1]
                tax_info = taxonomy_info.get(taxid)
                species = "Unknown"
                if tax_info:
                    if 'species' in tax_info.alignment:
                        species_taxid = tax_info.alignment['species']
                        species_info = taxonomy_info.get(str(species_taxid))
                        if species_info:
                            species = species_info.name
                
                formatted_recipient.append({
                    'subject_id': hit[1],
                    'evalue': hit[2],
                    'bitscore': hit[3],
                    'taxid': taxid,
                    'species': species
                })
            
            for _, hit in outgroup_hits.iterrows():
                taxid = hit[6].split(';')[-1]
                tax_info = taxonomy_info.get(taxid)
                species = "Unknown"
                if tax_info:
                    if 'species' in tax_info.alignment:
                        species_taxid = tax_info.alignment['species']
                        species_info = taxonomy_info.get(str(species_taxid))
                        if species_info:
                            species = species_info.name
                
                formatted_outgroup.append({
                    'subject_id': hit[1],
                    'evalue': hit[2],
                    'bitscore': hit[3],
                    'taxid': taxid,
                    'species': species
                })
            
            return {
                'recipient': formatted_recipient,
                'outgroup': formatted_outgroup
            }
            
        except Exception as e:
            logger.error(f"Error extracting top hits: {e}")
            return {'recipient': [], 'outgroup': []}
