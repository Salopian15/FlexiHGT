#!/usr/bin/env python3
import sys, os, warnings, math, csv, argparse, time
from concurrent.futures import ThreadPoolExecutor
from ete3 import NCBITaxa
from typing import List, Dict, Tuple, Set, Any
from functools import lru_cache
import pandas as pd
from Bio import SeqIO, BiopythonWarning

class HGTDetect:
    """
    Class to detect HGT events in protein sequences
    """

    def __init__(self):
        # Initialize the class
        # Set up the NCBI Taxonomy database
        self.ncbi = NCBITaxa()
        self.bitscore_parameter = 100
        self.HGTIndex = 0.5
        self.out_pct = 0.8
        self.tax_level = "family"
        self.search = "diamond"
        self.query_tax = None
        self.genes = list()
        self.geneSeq = dict()
        self.HGT = list()
        self.set_params(self.parse_args())
        #self.ncbi.update_taxonomy_database()
        #self.taxdb = "~/.etetoolkit/taxa.sqlite"
        self.taxdb = "~/.etetoolkit/taxa.sqlite"
        self.dmnd_dbpath = None

    def parse_args(self):
        """
        Parses command line arguments
        """
        parser = argparse.ArgumentParser(description="Modified version of HGTPhyloDetect close workflow for HGT events, takes protein fasta file and iterates through each sequence outputting a likelihood of HGT origin for each", epilog="Author: Jack A. Crosby, Aberystwyth University/Queens University Belfast")
        parser.add_argument("input_file", help="Input file path, should be a fasta file of protein sequences")
        parser.add_argument("--bitscore_parameter", type=float, default=100, help="Bitscore parameter, default is 100")
        parser.add_argument("--HGTIndex", type=float, default=0.5, help="HGT Index, default is 0.5")
        parser.add_argument("--out_pct", type=float, default=0.8, help="Out Pct, default is 0.8")
        parser.add_argument("-t", "--tax_level", type=str, default="family", choices=["superkingdom", "kingdom", "phylum", "subphylum", "class", "order", "family", "genus", "species"], help="Taxonomic level, organisms outisde of this level will be classified as 'outgroup', default is family.")
        parser.add_argument("-s", "--search", type=str, default="diamond", choices=["diamond", "mmseqs"], help="Search methods, diamond & mmseqs use local database for search, default is diamond.")
        parser.add_argument("-u", "--update", action="store_true", help="Update the NCBI taxonomy database")
        parser.add_argument("-q", "--query_tax", type=int, help="Taxid associated with the query sequence")
        parser.add_argument("-db", "--database", help="Path to the search database (e.g., Diamond or MMseqs database)")
        return parser.parse_args()        

    def set_params(self, args):
        """
        Set the parameters
        """
        self.bitscore_parameter = args.bitscore_parameter
        self.HGTIndex = args.HGTIndex
        self.out_pct = args.out_pct
        self.tax_level = args.tax_level.lower()
        self.search = args.search.lower()
        self.query_tax = args.query_tax
        self.dmnd_dbpath = args.database
        name = args.input_file
        bitscore_parameter = args.bitscore_parameter
        HGTIndex = args.HGTIndex
        out_pct = args.out_pct
        tax_level = args.tax_level.lower()
        search = args.search.lower()
        update = args.update
        qtaxid = args.query_tax

        if update:
            self.ncbi.update_taxonomy_database()
        
        warnings.simplefilter('ignore', BiopythonWarning)
        if not os.path.exists(self.dmnd_dbpath):
            print(f'Error: database not found at {self.dmnd_dbpath}')
            sys.exit()
        # Print the table header
        print("Input Parameters:")
        print("-----------------")
        print(f"{'Input File':<20} | {name}")
        print(f"{'Bitscore Parameter':<20} | {bitscore_parameter}")
        print(f"{'HGT Index':<20} | {HGTIndex}")
        print(f"{'Outgroup Percentage':<20} | {out_pct}")
        print(f"{'Taxonomic Level':<20} | {tax_level}")
        print(f"{'Search Method':<20} | {search}")
        print("-----------------")

    def load_fasta(self, name: str , genes: List[str], geneSeq: Dict[str, str]) -> List[str]:
        """
        Loads the fasta file into a dictionary
        """
        with open(name, 'r') as handleGene:
            for record in SeqIO.parse(handleGene, "fasta"):
                gene = str(record.id)
                sequence = str(record.seq)
                geneSeq[gene] = sequence
                genes.append(gene)
        return genes

    def run_search(self, name: str) -> None:
        """
        Runs the homology search
        """
        outf = str(name.split(".")[0] + ".tsv")
        if os.path.exists(f"{os.path.splitext(name)[0]}.tsv") and os.path.getsize(f"{os.path.splitext(name)[0]}.tsv") > 0:
        #accession_number, accession_bitscore = parse_NCBI(gene)
            print(f'Diamond file found for {os.path.splitext(name)[0]}')
            return
        elif self.search == "diamond":
            myCmd =f'diamond blastp -d {self.dmnd_dbpath} -q {name} --max-target-seqs 250 --outfmt 6 qseqid sseqid evalue bitscore length pident staxids -o {outf}'
            myCmd = str(myCmd)
            os.system(myCmd)
        elif self.search == "mmseqs":
            myCmd = f'mmseqs easy-search {name} {self.mmseqs_dbpath} {outf} --max-seqs 250 --format-output "query,target,evalue,bits,alnlen,pident,taxid'
            myCmd = str(myCmd)
            os.system(myCmd)
        else:
            print("Error: Search method not recognized")
            sys.exit()  

    def load_diamond_results(self, combined_file: str , gene: str) -> pd.DataFrame:
        """
        Load the diamond results file into a dataframe and filter for the gene of interest
        """
        try:
            results=pd.read_csv(combined_file, sep='\t', header=None)
            gene_results = results[results[0] == gene]
        except pd.errors.EmptyDataError:
            print(f"Error: No results found for {gene}")
            sys.exit()
        return gene_results

    def get_refTax(self, qtaxid, tax_level):
        """
        Get the taxonomy of the host organism (the organism of the input sequences)
        """
        try:
            gene_lineage = self.ncbi.get_lineage(qtaxid)
            gene_lineage2ranks = self.ncbi.get_rank(gene_lineage)
            gene_ranks2lineage = dict((rank, taxid) for (taxid, rank) in gene_lineage2ranks.items())
            gene_taxonomy_alignment = gene_ranks2lineage
            gene_taxlevel = gene_taxonomy_alignment.get(tax_level)
            
            if gene_taxlevel is None:
                raise ValueError(f"Specified tax_level '{tax_level}' not found in the lineage")
            #print("Gene Taxonomy Information:")
            #print("--------------------------")
            #for rank, taxid in gene_taxonomy_alignment.items():
            #    print(f"{rank.capitalize():<20} | {taxid}")
            #print("--------------------------")
        except Exception as e:
            print(f"Error type: {e.__class__.__name__}, Message: {e}")
            print("Exiting...")
            sys.exit()
        return gene_taxlevel

    def get_query_taxids(self, result_file, accession_number):
        """
        Get the taxids of the query sequences
        """
        taxids = []
        accession_to_taxid = {}  # To map each accession to its taxid for later use
        for accession in accession_number[:200]:
            try:
                taxid = self.get_taxid(result_file, accession)
                taxids.append(taxid)
                accession_to_taxid[accession] = taxid
            except Exception as e:
                print(f"Error fetching taxid for {accession}: {e}")
                continue
        return taxids, accession_to_taxid

    def get_taxid(self, gene_results, accession_number):
        """
        Gets taxids of results from diamond search result file
        """
        df = pd.read_csv(gene_results, sep='\t', header=None)
        filtered_results = df[df[1] == accession_number]
        taxid = filtered_results[6].str.split(';').str[-1].values[0]
        return taxid

    def hgt_calc(self, gene, max_outgroup_bitscore, max_recipient_organism_bitscore, outgroup_species_number, recipient_species_number, HGT, HGTIndex, out_pct, tax_level, names, taxonomy_alignments, bitscore_parameter, donor_taxonomy):
        """
        Calculates the likelihood of a HGT event
        """
        HGT_index = format(max_outgroup_bitscore / max_recipient_organism_bitscore, '.4f')
        Outg_pct = format(outgroup_species_number / (outgroup_species_number + recipient_species_number), '.4f')
        print(f'HGT index: {HGT_index}', flush=True)
        print(f'Out_pct: {Outg_pct}', flush=True)
        is_hgt_event = (
            max_outgroup_bitscore >= bitscore_parameter and
            float(HGT_index) >= HGTIndex and
            float(Outg_pct) >= out_pct
        )
        if is_hgt_event:
            print('This is a HGT event', flush=True)
            taxonomy = donor_taxonomy
            # check if donor_taxonomy is not empty
            if donor_taxonomy:
                taxonomy = donor_taxonomy
            else:
                taxonomy = 'Not available'
            #for taxid, alignment in taxonomy_alignments.items():
            #    if tax_level in alignment:
            #        taxonomy = names.get(alignment[tax_level], 'Not available')
            #        break
        else:
            print('This is not a HGT event', flush=True)
            taxonomy = 'No'
        item = [gene, max_outgroup_bitscore, Outg_pct, HGT_index, taxonomy]
        HGT.append(item)
        return HGT

    def write_output(self, HGT, tax_level):
        """
        Writes results of the HGT detection to a file
        """
        outfile = open(f"./output_{tax_level}_HGT.tsv", "wt")
        tsv_writer = csv.writer(outfile, delimiter="\t")
        column = ['Gene/Protein', 'Bitscore', 'Out_pct',
                  'HGT index', 'Donor taxonomy']
        tsv_writer.writerow(column)
        for HGT_info in HGT :
            tsv_writer.writerow(HGT_info)
        outfile.close()

    def process_gene(self, gene, combined_file, args, taxonomy_alignments, ranks, names, hosttax):
        """
        Runs the main analysis for each gene, slices the results 
        for the first 200 hits and returns the HGT results
        """
        try:
            # Slices first 200 hits and pulls out the accession number, bitscore and taxids
            gene_results = self.load_diamond_results(combined_file, gene)
            gene_results = gene_results[:200]
            gene_results = gene_results.dropna(subset=[6])
            accession_number = gene_results[1].values
            accession_bitscore = gene_results[3].values
            taxids = gene_results[6].str.split(';').str[-1].values
            accession_to_taxid = dict(zip(accession_number, taxids))
            #print(f"Debug: Query taxid {args.query_tax}, Taxonomy alignments keys: {list(taxonomy_alignments.keys())[:10]}...")
            gene_taxlevel = taxonomy_alignments[str(args.query_tax)].get(args.tax_level)
            #gene_taxlevel = hosttax
            #if str(args.query_tax) not in taxonomy_alignments:
            #    print(f"Warning: Query taxid {args.query_tax} not found in taxonomy alignments. Skipping gene {gene}.")
            #    return None
            
            if gene_taxlevel is None:
                print(f"Warning: Tax level {args.tax_level} not found for query taxid {args.query_tax}. Skipping gene {gene}.")
                return None
            recipient_accession = set()
            recipient_species = set()
            outgroup_accession = set()
            outgroup_species = set()
            #evalue_dict = {}
            for accession, taxid in accession_to_taxid.items():
                if taxid not in taxonomy_alignments:
                    print(f"Warning: Taxid {taxid} not found in taxonomy alignments. Skipping this accession.")
                    continue
                taxonomy_alignment = taxonomy_alignments[taxid]
                if args.tax_level in taxonomy_alignment and taxonomy_alignment[args.tax_level] == gene_taxlevel:
                    recipient_accession.add(accession)
                    recipient_species.add(names.get(taxonomy_alignment.get('species'), 'Unknown'))
                else:
                    outgroup_accession.add(accession)
                    outgroup_species.add(names.get(taxonomy_alignment.get('species'), 'Unknown'))
                #evalue = gene_results[gene_results[1] == accession].iloc[0][2]
                #evalue_dict[accession] = evalue
            recipient_accession_bitscore = {acc: bs for acc, bs in zip(accession_number, accession_bitscore) if acc in recipient_accession}
            outgroup_accession_bitscore = {acc: bs for acc, bs in zip(accession_number, accession_bitscore) if acc in outgroup_accession}
            max_recipient_organism_bitscore = max(recipient_accession_bitscore.values()) if recipient_accession_bitscore else 0
            max_outgroup_bitscore = max(outgroup_accession_bitscore.values()) if outgroup_accession_bitscore else 0
            recipient_species_number = len(recipient_species)
            outgroup_species_number = len(outgroup_species)
            #se_minus = 1e-200
            if max_outgroup_bitscore and max_recipient_organism_bitscore:
                #min_outgroup_key = min(outgroup_accession_bitscore,
                #                       key=outgroup_accession_bitscore.get)
                #min_outgroup_evalue = evalue_dict.get(min_outgroup_key, e_minus)
                #min_ingroup_key = min(recipient_accession_bitscore,
                #                      key=recipient_accession_bitscore.get)
                #min_ingroup_evalue = evalue_dict.get(min_ingroup_key, e_minus)
                #alienindex = format(math.log(min_ingroup_evalue + e_minus, math.e) - math.log(min_outgroup_evalue + e_minus, math.e), '.2f')
                donor_taxid = None
                donor_taxonomy = 'Not available'
                if outgroup_accession_bitscore:
                    max_outgroup_acc = max(outgroup_accession_bitscore, key=outgroup_accession_bitscore.get)
                    donor_taxid = accession_to_taxid.get(max_outgroup_acc)
                    if donor_taxid in taxonomy_alignments:
                        donor_alignment = taxonomy_alignments[donor_taxid]
                        if args.tax_level in donor_alignment:
                            donor_taxonomy = names.get(donor_alignment[args.tax_level], 'Not available')

                hgt_result = self.hgt_calc(
                    gene, max_outgroup_bitscore, max_recipient_organism_bitscore,
                    outgroup_species_number, recipient_species_number, [],
                    args.HGTIndex, args.out_pct, args.tax_level,
                    names, taxonomy_alignments, args.bitscore_parameter, donor_taxonomy
                )
                print("Result for ", gene, "processed", flush= True)
                return hgt_result[0] if hgt_result else None
            else:
                print(f"Skipping HGT calculation for gene {gene} due to missing bitscore data")
                return None
        except Exception as e:
            print(f'Error in process_gene for gene {gene}: {e.__class__.__name__}, Message: {e}')
            return None

    @lru_cache(maxsize=None)
    def get_lineage(self, taxid: int) -> Tuple[int, ...]:
        """Cached method to get lineage for a taxid."""
        return tuple(self.ncbi.get_lineage(taxid))

    @lru_cache(maxsize=None)
    def get_rank(self, taxid: int) -> str:
        """Cached method to get rank for a taxid."""
        return self.ncbi.get_rank([taxid])[taxid]

    @lru_cache(maxsize=None)
    def get_name(self, taxid: int) -> str:
        """Cached method to get name for a taxid."""
        return self.ncbi.get_taxid_translator([taxid])[taxid]

    def fetch_all_taxonomy_data(self, combined_file: str, query_taxid: int) -> Tuple[Dict[str, Dict[str, int]], Dict[int, str], Dict[int, str]]:
        """
        Fetches all the taxonomy data from the diamond results file
        """
        df = pd.read_csv(combined_file, sep='\t', header=None, usecols=[6])
        df[6] = df[6].fillna('').astype(str)
        unique_taxids: Set[int] = set()
        for tid in df[6].str.split(';').explode().unique():
            try:
                if tid and not pd.isna(tid):
                    unique_taxids.add(int(float(tid)))
            except ValueError:
                print(f"Warning: Invalid taxid '{tid}'. Skipping.")
        
        unique_taxids.add(query_taxid)
        lineages: Dict[int, Tuple[int, ...]] = {}
        for tid in unique_taxids:
            try:
                lineages[tid] = self.get_lineage(tid)
            except Exception as e:
                print(f"Error fetching lineage for taxid {tid}: {e}")
        all_taxids: Set[int] = set(tid for lineage in lineages.values() for tid in lineage) | unique_taxids
        ranks: Dict[int, str] = {}
        names: Dict[int, str] = {}
        for tid in all_taxids:
            try:
                ranks[tid] = self.get_rank(tid)
                names[tid] = self.get_name(tid)
            except Exception as e:
                print(f"Error fetching rank or name for taxid {tid}: {e}")
        taxonomy_alignments: Dict[str, Dict[str, int]] = {}
        for taxid, lineage in lineages.items():
            taxonomy_alignments[str(taxid)] = {ranks[tid]: tid for tid in lineage if tid in ranks}
            taxonomy_alignments[str(taxid)][ranks.get(taxid, 'no rank')] = taxid
        return taxonomy_alignments, ranks, names


#if __name__ == "__main__":
#    start_time = time.time()
#    main()
#    end_time = time.time()
#    elapsed_time = (end_time - start_time) / 3600
#    print(f"Elapsed time: {elapsed_time}: hours")
# End of Phylotest.py
