#!/bin/bash
# Set variables for paths and versions
OUTPUT_DIR="1000G_ICGC_data"
TG_PREFIX="20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr"
TG_SUFFIX=".recalibrated_variants.annotated.coding.txt"
CODING_VARIANTS="transcript_ablation|splice_acceptor_variant|splice_donor_variant|stop_gained|frameshift_variant|stop_lost|start_lost|inframe_insertion|inframe_deletion|missense_variant|protein_altering_variant|coding_sequence_variant|coding_transcript_variant"
AF_THRESHOLD=0.05
AF_COLUMN=29

# Create output directory
mkdir -p ${OUTPUT_DIR}

############################
# 1000 Genomes Processing #
############################
echo "Processing 1000 Genomes data..."

# Download and process all chromosomes in parallel
process_chromosome() {
    local chr=$1
    local file="${TG_PREFIX}${chr}${TG_SUFFIX}"
    
    # Download if file doesn't exist
    if [ ! -f "$file" ]; then
        wget -q "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/${file}"
    fi
    
    # Apply all filters in a single pass to avoid multiple file reads/writes
    awk -F"\t" -v af_threshold=${AF_THRESHOLD} -v af_col=${AF_COLUMN} -v coding="${CODING_VARIANTS}" '
        # Check if coding variant
        $8 ~ coding && 
        # Check if on autosomes (not X or Y)
        $1 != "chrX" && $1 != "chrY" && 
        # Check if SNV (single nucleotide variant)
        length($4) == 1 && length($5) == 1 && 
        # Check allele frequency
        $af_col > af_threshold && $af_col != "NA" 
        { print }
    ' "$file" > "${OUTPUT_DIR}/filtered_chr${chr}.txt"
    
    echo "Processed chromosome ${chr}"
}

# Process all chromosomes in parallel (adjust -P based on your system)
for chr in {1..22}; do
    process_chromosome $chr &
done
wait

# Combine all filtered chromosome files
cat ${OUTPUT_DIR}/filtered_chr*.txt > ${OUTPUT_DIR}/1000G_great_${AF_THRESHOLD}_coding.txt
rm ${OUTPUT_DIR}/filtered_chr*.txt

echo "Completed 1000 Genomes processing"

####################
# ICGC Processing #
####################
echo "Processing ICGC data..."

# Download ICGC data
wget -O ${OUTPUT_DIR}/simple_somatic_mutation.aggregated.vcf.gz "https://dcc.icgc.org/api/v1/download?fn=/current/Summary/simple_somatic_mutation.aggregated.vcf.gz"
gunzip -c ${OUTPUT_DIR}/simple_somatic_mutation.aggregated.vcf.gz | grep -v "^#" > ${OUTPUT_DIR}/simple_somatic_mutation.aggregated.bed

# Apply all filters in a single pass
awk -F"\t" -v coding="${CODING_VARIANTS}" '
    # Check if SNV
    length($4) == 1 && length($5) == 1 && 
    # Check if autosome
    $1 != "X" && $1 != "Y" { 
        # Check if coding variant (field 7 in pipe-delimited section)
        split($8, fields, "|");
        if (fields[7] ~ coding) {
            print;
        }
    }
' ${OUTPUT_DIR}/simple_somatic_mutation.aggregated.bed > ${OUTPUT_DIR}/filtered_somatic_mutation.bed

echo "Note: ICGC data is in GRCh37 format and needs to be converted to GRCh38"
echo "Processing complete. Final datasets are in ${OUTPUT_DIR}/"
