import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import os
import time
from .core import HGTDetect

def noargs(args):
    """
    Checks to see if arguments are provided
    """
    if len(args) == 0:
        print("FlexiHGT: No arguments given, please run with -h or --help for help")
        sys.exit()
    else:
        pass

def main():
    """
    Runs main pipeline
    """
    hgt = HGTDetect()
    args = hgt.parse_args()
    #hgt.set_params(args)
    host_taxlevel = hgt.get_refTax(args.query_tax, args.tax_level)
    genes = hgt.load_fasta(args.input_file, hgt.genes, hgt.geneSeq)
    hgt.run_search(args.input_file)
    combined_file = f"{os.path.splitext(args.input_file)[0]}.tsv"
    taxonomy_alignments, ranks, names = hgt.fetch_all_taxonomy_data(combined_file, args.query_tax)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda gene: hgt.process_gene(gene,
                                    combined_file, args, taxonomy_alignments,
                                    ranks, names, host_taxlevel), genes))
    results = [r for r in results if r is not None]
    hgt.write_output(results, args.tax_level)

if __name__ == "__main__":
    noargs(sys.argv)
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    print(f"Elapsed time: {elapsed_time}: hours", flush = True)
# End of Core run.
