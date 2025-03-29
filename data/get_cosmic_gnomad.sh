#!/bin/bash
# Set variables for paths and versions
OUTPUT_DIR="cosmicGnomadControlsAndNonCancer"
PHYLO_DIR="${OUTPUT_DIR}/queryPhyloPhastCons"
COSMIC_DIR="/user/home/uw20204/data/encode/public/cosmic"
GNOMAD_VERSION="r3.1.2"
COSMIC_VERSION="v99"

# Create required directories
mkdir -p ${OUTPUT_DIR} ${PHYLO_DIR}

# Load required modules
module load apps/bedops/2.4.38 apps/bedtools/2.30.0
module load apps/bcftools apps/samtools/1.9 apps/tabix/0.2.5 lib/htslib/1.10.2-gcc

##########################
# COSMIC Data Processing #
##########################
# Get and process COSMIC dataset
wget -O CosmicMutantExport.tsv.gz https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/${COSMIC_VERSION}/CosmicMutantExport.tsv.gz

# Process COSMIC data in a single pipeline
gunzip -c CosmicMutantExport.tsv.gz | 
  tr '\t' ',' | 
  sed '1s/ /_/g' | 
  awk -F, '$22 ~ /Substitution/' > CosmicMutantExportSubs.csv

#########################
# GNOMAD Data Processing #
#########################
# Get and process GNOMAD exome dataset
EXOME_FILE="gnomad.exomes.${GNOMAD_VERSION}.sites.liftover_grch38"
wget -O ${EXOME_FILE}.vcf.bgz https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/liftover_grch38/vcf/exomes/${EXOME_FILE}.vcf.bgz

# Extract and filter GNOMAD data 
bcftools view ${EXOME_FILE}.vcf.bgz | 
  sed '1d' | 
  awk '($8 ~ "/non_cancer_AF=/") || ($8 ~ "/controls_AF=/") {print $0}' > ${OUTPUT_DIR}/gnomadAlleleFreqIncl.txt

# Get variants with allele frequency > 0.05
python getAlleleFrequencies.py

#############################
# Compare COSMIC and GNOMAD #
#############################
# Extract and format GNOMAD variants in a single operation
cat ${OUTPUT_DIR}/gnomadAlleleFreqIncl.txt | 
  awk 'length($4) == 1 && length($5) == 1 {print $1 "\t" $2 "\t" $2 "\t" $4 "\t" $5}' | 
  sort > ${OUTPUT_DIR}/gnomadMatch.sorted.bed

# Extract and format COSMIC variants
cat ${COSMIC_DIR}/CosmicMutantExportFilteredSubst.bed | 
  awk '{print $1 "\t" $2 "\t" $2 "\t" $3 "\t" $4 "\t" $5}' | 
  sort > ${COSMIC_DIR}/cosmicMatch.sorted.bed

# Remove likely benign variants from COSMIC
comm -23 ${COSMIC_DIR}/cosmicMatch.sorted.bed ${OUTPUT_DIR}/gnomadMatch.sorted.bed > ${OUTPUT_DIR}/cosmicGnomadExcl.bed

# Create 1000bp window COSMIC dataset for GNOMAD variant detection
cat ${OUTPUT_DIR}/cosmicGnomadExcl.bed | 
  awk '{print $1 "\t" $2 - 1000 "\t" $2 + 1000 "\t" $4 "\t" $5 "\t" $2 "\t" $6}' > ${OUTPUT_DIR}/cosmicQuery.bed

# Find GNOMAD variants within 1000bp of COSMIC variants
bedtools intersect -wa -wb \
  -a ${OUTPUT_DIR}/cosmicQuery.bed \
  -b ${OUTPUT_DIR}/gnomadMatch.sorted.bed > ${OUTPUT_DIR}/overlaps.bed

# Extract unique GNOMAD and COSMIC variants from intersect results
cat ${OUTPUT_DIR}/overlaps.bed | 
  awk '{print $8 "\t" $9 "\t" $9 "\t" $11 "\t" $12 "\t" $7}' | 
  awk '!seen[$1,$2,$3,$4,$5]++' | 
  sort > ${OUTPUT_DIR}/gnomad.bed

cat ${OUTPUT_DIR}/overlaps.bed | 
  awk '{print $1 "\t" $6 "\t" $6 "\t" $4 "\t" $5 "\t" $7}' | 
  awk '!seen[$0]++' | 
  sort > ${OUTPUT_DIR}/cosmic.bed


echo "Processing complete. Final datasets are in ${OUTPUT_DIR}/cosmic.bedand ${OUTPUT_DIR}/gnomad.bed
