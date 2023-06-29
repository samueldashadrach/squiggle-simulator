# import itertools

# ratio = 450 / 4000 = 4.5 / 40 = 9 / 80
# 90 bases needs 80 numbers each of which is multiple chars


with open("merged/dna.txt", "r") as f_dna, open("merged/signal.txt", "r") as f_signal, open("gpt3/gpt3.jsonl", "w") as f_gpt3:
	for line_dna, line_signal in zip(f_dna, f_signal):

		# delete terminating newline char
		dna_str = line_dna[:-1]
		dna_str = dna_str[:90].lower()

		signal_str = line_signal[:-1]
		signal_str = signal_str[:3600]

		str_to_write = "{\"prompt\":\"" + signal_str + "->\", \"completion\":\" " + dna_str + "\\n\"}\n"

		print(str_to_write)
		f_gpt3.write(str_to_write)

