#!/bin/bash
# Convert FrameDiPT Predictions from backbone to full atom.

# Set a timer
SECONDS=0

# The first argument to the bash script is the path to the
# root directory that contains the FrameDiPT predictions.
root_dir=$1
standard_name=$2

# Iterate over subdirectories
for dir in $(find $root_dir -type d -name '*length*'); do
   for sample_dir in $dir/sample_*; do
     # Iterate over each .pdb file that does not contain 'all_atom'
     for in_pdb in $(find $sample_dir -name "sample_*.pdb" ! -name "*all_atom.pdb"); do
       out_pdb="${in_pdb%.pdb}_all_atom.pdb"
       if [ ! -f "$out_pdb" ]; then
         echo "$out_pdb";
         # Run full atom model conversion
         if [ "$standard_name" = "--standard-name" ]; then
           echo "Using IUPAC standard atom names"
           convert_cg2all -p $in_pdb -o $out_pdb --cg MainchainModel --device cpu --standard-name
         else
           convert_cg2all -p $in_pdb -o $out_pdb --cg MainchainModel --device cpu
         fi
       fi
     done
  done
done

echo "The script execution took $SECONDS seconds"
