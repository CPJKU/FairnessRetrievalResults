# 
# bash script to split a file in n-similiar sized chunks (to be fed into the optimized multiprocess batch generators)
# also returns the correct batch count (that the multiprocess generators will produce)
#

# arg 1: base file
# arg 2: n chunks (should correspond to the number of processes you want to use)
# arg 3: output folder

if [ $# -ne 3 ]
  then
    echo "Usage (bash): ./generate_file_split.sh <base_file> <n file chunks> <output_folder>"
    exit 1
fi

base_file=$1
n_chunks=$2
output_folder=$3

# make sure the directory exists
mkdir -p $output_folder
output_folder_prefix="${output_folder}/part"

#
# script start
#

echo "Output directory: "$output_folder

split --number=l/$n_chunks -d --additional-suffix ".tsv" $base_file $output_folder_prefix

echo "done!"