# squiggle-simulator

Train machine learning models against whole human genome dataset NA12878. (https://github.com/nanopore-wgs-consortium/NA12878/blob/master/Genome.md)

Attempting to train simulator that can output ONT signal data given reference DNA sequence

Note: This takes non-trivial amount of time to set up.

Steps:
1. Obtain a pair of tarballs from hosted AWS buckets, one FASTQ and one FAST5. Configure AWS client and use aws-cli to obtain. (This project has been tested against FAB42804)
2. Unzip and untar as required.
3. Move all FASTQ files into a folder named FASTQ. Move all FAST5 files into a folder named FAST5. Keep `sequencing_summary.txt` at root along with python files in this repo.
4. Use `python data-aggregate.py` to combine both datasets into a single dataset. On successful run you will see a `merged` folder with 3 files.
5. Use `python squiggle-simulator.py` to load this data into pytorch dataloader as tensors and train your own pytorch model. DNA sequences have been one-hot encoded.
