"""HGT Detection CLI Tool."""

import argparse
import sys
import logging
import time
from pathlib import Path
from .core import HGTDetect, HGTParameters
import ete3

def validate_input_file(input_path: Path) -> bool:
    """Validate that input file exists and is a file."""
    if not input_path.exists() or not input_path.is_file():
        logging.error("Invalid input file: %s", input_path)
        return False
    return True

def validate_database(db_path: Path) -> bool:
    """Validate that database exists."""
    if not db_path.exists():
        logging.error("Database not found: %s", db_path)
        return False
    return True

def main():
    """
    Runs main HGT detection pipeline.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        start_time = time.time()
        logger.info("Starting HGT detection pipeline...")

        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="HGT Detection Tool",
            epilog="Author: Jack A. Crosby, Aberystwyth University/Queens University Belfast"
        )

        # Required arguments
        parser.add_argument("input_file", help="Input FASTA file of protein sequences")
        parser.add_argument(
            "-q", "--query_tax",
            type=int,
            required=True,
            help="Taxid associated with the query sequence"
        )
        parser.add_argument(
            "-db", "--database",
            required=True,
            help="Path to the search database (e.g., Diamond or MMseqs database)"
        )

        # Optional arguments
        parser.add_argument(
            "--bitscore_parameter",
            type=float,
            default=100,
            help="Bitscore parameter"
        )
        parser.add_argument(
            "--HGTIndex",
            type=float,
            default=0.5,
            help="HGT Index threshold"
        )
        parser.add_argument(
            "--out_pct",
            type=float,
            default=0.8,
            help="Outgroup percentage"
        )
        parser.add_argument(
            "--AI",
            type=float,
            default=45,
            help="Alien Index"
        )
        parser.add_argument(
            "-t", "--tax_level",
            type=str,
            default="family",
            choices=["superkingdom", "kingdom", "phylum", "subphylum", "class",
                    "order", "family", "genus", "species"],
            help="Taxonomic level"
        )
        parser.add_argument(
            "-s", "--search",
            type=str,
            default="diamond",
            choices=["diamond", "mmseqs"],
            help="Search method"
        )
        parser.add_argument(
            "-u", "--update",
            action="store_true",
            help="Update the NCBI taxonomy database"
        )
        parser.add_argument(
            "-o", "--outfile",
            help="Output file name, default is output_taxlevel_HGT.tsv"
        )

        args = parser.parse_args()

        # Validate input paths
        input_path = Path(args.input_file)
        db_path = Path(args.database)

        if not validate_input_file(input_path) or not validate_database(db_path):
            sys.exit(1)

        # Create HGTParameters instance
        params = HGTParameters(
            bitscore_parameter=args.bitscore_parameter,
            hgt_index=args.HGTIndex,
            out_pct=args.out_pct,
            ai_threshold=args.AI,
            tax_level=args.tax_level,
            search_method=args.search,
            query_taxid=args.query_tax
        )

        # Create HGTDetect instance
        hgt = HGTDetect(params)

        # Handle taxonomy database update if requested
        if args.update:
            try:
                ete3.ncbi.update_taxonomy_database()
                if not args.input_file:  # If only updating taxonomy
                    logger.info("Taxonomy database update complete. Exiting...")
                    sys.exit(0)
            except Exception as e:
                logger.error("Failed to update taxonomy database: %s", e)
                sys.exit(1)
        logger.info(f"Quert taxid: {args.query_tax}")
        logger.info(f"Parameters used: {params}")
        try:
            # Run analysis
            results_df = hgt.run_analysis(input_path, db_path)

            # Calculate execution time
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)

            # Log results
            logger.info("Analysis complete. Found %d potential HGT events.", len(results_df))
            logger.info("Total execution time: %dh %dm %ds", hours, minutes, seconds)

            # Save results
            output_file = args.outfile or f"output_{args.tax_level}_HGT.tsv"
            results_df.to_csv(output_file, sep='\t', index=False)
            logger.info("Results saved to %s", output_file)

        except Exception as e:
            logger.error("Error during analysis", exc_info=True)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
