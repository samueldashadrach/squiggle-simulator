# File structure required for this program to work:
# "sequencing_summary.txt" file in base directory
# All fastq files in "fastq" subdirectory
# All fast5 files in "fast5" subdirectory

import time
import h5py
import os
import numpy as np
import sys

fastq_filenames = os.listdir("fastq")
fast5_filenames = os.listdir("fast5")



with open("sequencing_summary.txt", "r") as f_summary:
	with open("merged/read_id.txt", "w") as f_read_id, open("merged/dna.txt", "w") as f_dna, open("merged/signal.txt", "w") as f_signal:

		for line_no, line in enumerate(f_summary):
			if line_no == 0:
				assert line.split()[1] == "read_id"
				continue

			read_id = line.split()[1]

			# Find fastq match

			fastq_matches = 0
			for fastq_filename in fastq_filenames:
				with open("fastq/" + fastq_filename, "r") as f_fastq:
					for line_no_fastq, line_fastq in enumerate(f_fastq):

						if line_no_fastq % 4 == 0:
							grab_next_lines = False
							if line_fastq.split()[0] == "@" + read_id:
								grab_next_lines = True
						elif line_no_fastq % 4 == 1:
							if grab_next_lines:
								DNA_seq = line_fastq[:-1] # -1 used to ignore newline chars

								fastq_matches += 1
			assert fastq_matches == 1

			# Find fast5 match

			fast5_matches = 0
			for fast5_filename in fast5_filenames:
				h5 = h5py.File("fast5/" + fast5_filename, "r")
				key = "read_" + read_id
				try:
					signal = h5[key]["Raw"]["Signal"][:]
					fast5_matches += 1
				except KeyError:
					pass
			assert fast5_matches == 1


			# Now all matches found, write to respective files
			f_read_id.write(str(read_id))
			f_read_id.write("\n")

			f_dna.write(str(DNA_seq))
			f_dna.write("\n")

			for s in signal:
				f_signal.write(str(s))
				f_signal.write(" ")
			f_signal.write("\n")


			if line_no % 100 == 0:
				print(line_no)


