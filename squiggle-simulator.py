
import os
import sys
import torch
import numpy as np

import matplotlib.pyplot as plt

# 450 basepairs / second, 4000 signal reads / second

# Uses first 90 bases and first 800 signal ints

class ONT_DNA_dataset(torch.utils.data.Dataset):
	def __init__(self):
		self.read_ids = []
		self.dnas = []
		self.signals = []

		with open("merged/read_id.txt", "r") as f_read_id:
			for line_no, line in enumerate(f_read_id):
				self.read_ids.append(line)

		with open("merged/dna.txt", "r") as f_dna, open("merged/signal.txt", "r") as f_signal:
			for line_no, line_dna in enumerate(f_dna):

				dna_str = line_dna[:-1]
				dna_str = dna_str[:90].lower()
				
				# def base_to_onehot(base):
				# 	if base == "a":
				# 		return torch.tensor([[1,0,0,0]])
				# 	elif base == "t":
				# 		return torch.tensor([[0,1,0,0]])
				# 	elif base == "g":
				# 		return torch.tensor([[0,0,1,0]])
				# 	elif base == "c":
				# 		return torch.tensor([[0,0,0,1]])
				# 	else:
				# 		raise Exception("WTF")

				def base_to_serial(base):
					if base == "a":
						return torch.tensor([[0]], dtype = torch.float32)
					elif base == "t":
						return torch.tensor([[1]], dtype = torch.float32)
					elif base == "g":
						return torch.tensor([[2]], dtype = torch.float32)
					elif base == "c":
						return torch.tensor([[3]], dtype = torch.float32)
					else:
						raise Exception("WTF")

				dna = torch.zeros((90))

				for i, base in enumerate(dna_str):
					dna[i] = base_to_serial(base).squeeze()
					if i >= 90:
						break

				self.dnas.append(dna)

			for line_no, line_signal in enumerate(f_signal):

				signal_arr = [int(num) for num in line_signal.split(None, 801)[:800]]
				signal = torch.tensor(signal_arr, dtype = torch.float32)
				# print(signal)

				self.signals.append(signal)


	def __len__(self):
		return(len(self.read_ids))

	def __getitem__(self, idx):

		read_id = self.read_ids[idx]
		dna = self.dnas[idx]
		signal = self.signals[idx]

		return {"read_id": read_id, "signal" : signal, "dna" : dna}

class FC_test(torch.nn.Module):
	def __init__(self):
		# call constructor from superclass
		super().__init__()

		# define network layers
		self.fc1 = torch.nn.Linear(800, 2000)
		self.fc2 = torch.nn.Linear(2000, 5000)
		self.fc3 = torch.nn.Linear(5000, 500)
		self.fc4 = torch.nn.Linear(500, 90)

	def forward(self, x):
		# define forward pass
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		x = torch.nn.functional.relu(self.fc3(x))
		x = torch.sigmoid(self.fc4(x))
		return x

def train_FC():

	# Dataset

	dataset = ONT_DNA_dataset()
	train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
	
	# Model

	model = FC_test()

	if torch.backends.mps.is_available():
	    mps_device = torch.device("mps")
	    using_mps = True
	    model.to(mps_device)
	print(model)

	# Loss function, Optimmiser

	loss_fn = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)

	# Training

	print("training started")
	model.train(True)

	num_epochs = 100

	for epoch in range(num_epochs):
		for i_batch, train_batch in enumerate(train_dataloader):

			model_input = train_batch["signal"]
			model_output_label = train_batch["dna"]
			if using_mps:
				model_input = model_input.to(mps_device)
				model_output_label = model_output_label.to(mps_device)

			model_output = model(model_input)

			loss = loss_fn(model_output, model_output_label)
			
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

		print(epoch, loss)



# train_FC()



####################################################

# Uses first 16 bases and 142 signals

class ONT_DNA_dataset_onehot(torch.utils.data.Dataset):
	def __init__(self):
		self.read_ids = []
		self.dnas = []
		self.signals = []

		self.num_bases = 16
		self.num_signals = 142

		with open("merged/read_id.txt", "r") as f_read_id:
			for line_no, line in enumerate(f_read_id):
				self.read_ids.append(line)

		with open("merged/dna.txt", "r") as f_dna, open("merged/signal.txt", "r") as f_signal:
			for line_no, line_dna in enumerate(f_dna):

				dna_str = line_dna[:-1]
				dna_str = dna_str[:self.num_bases].lower()
				
				def base_to_onehot(base):
					if base == "a":
						return torch.tensor([[1,0,0,0]], dtype = torch.float32)
					elif base == "t":
						return torch.tensor([[0,1,0,0]], dtype = torch.float32)
					elif base == "g":
						return torch.tensor([[0,0,1,0]], dtype = torch.float32)
					elif base == "c":
						return torch.tensor([[0,0,0,1]], dtype = torch.float32)
					else:
						raise Exception("WTF")


				dna = torch.zeros((4, self.num_bases))

				for i, base in enumerate(dna_str):
					dna[:,i] = base_to_onehot(base).squeeze()
					if i >= self.num_bases:
						break

				self.dnas.append(dna)

			for line_no, line_signal in enumerate(f_signal):

				signal_arr = [int(num) for num in line_signal.split(None, self.num_signals + 1)[:self.num_signals]]
				signal = torch.tensor(signal_arr, dtype = torch.float32)
				# print(signal)

				self.signals.append(signal)


	def __len__(self):
		return(len(self.read_ids))

	def __getitem__(self, idx):

		read_id = self.read_ids[idx]
		dna = self.dnas[idx]
		signal = self.signals[idx]

		return {"read_id": read_id, "signal" : signal, "dna" : dna}


class CNN_scrappie_test(torch.nn.Module):
	def __init__(self):

		super().__init__()

		self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=32, kernel_size=9, stride=1, padding=4, bias=True)
		self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4, bias=True)
		self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4, bias=True)

		self.conv4 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
		self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
		self.conv6 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)

		self.fc7 = torch.nn.Linear(64, 142)

	def forward(self, x):

		# (N, 4, 16)
		# N sequences in mini-batch, each with 4 channels (4 bases) and 16 bases
		# print(x.shape)

		# To (N, 32, 16)
		x = self.conv1(x)
		x = torch.nn.functional.relu(x)
		# To (N, 32, 16) twice
		x = self.conv2(x)
		x = torch.nn.functional.relu(x)
		x = self.conv3(x)
		# To (N, 32, 8)
		x = torch.nn.MaxPool1d(kernel_size = 2, stride = 2, padding = 0)(x)

		# To (N, 64, 8)
		x = self.conv4(x)
		x = torch.nn.functional.relu(x)
		# To (N, 64, 8) twice
		x = self.conv5(x)
		x = torch.nn.functional.relu(x)
		x = self.conv6(x)
		# To (N, 64, 1)
		x = torch.nn.MaxPool1d(kernel_size = 8, stride = 8, padding = 0)(x)

		x = x.view(x.size(0), x.size(1))

		x = self.fc7(x)
		# print(x.shape)

		return x


def train_CNN():
	dataset = ONT_DNA_dataset_onehot()
	train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
	
	model = CNN_scrappie_test()
	print(model)

	# for i_batch, train_batch in enumerate(train_dataloader):
	# 	print(train_batch["signal"].shape)
	# 	print(train_batch["dna"].shape)

	# 	print(train_batch["signal"])
	# 	print(train_batch["dna"])

	# 	break

	if torch.backends.mps.is_available():
	    mps_device = torch.device("mps")
	    using_mps = True
	    model.to(mps_device)
	print(model)

	# Loss function, Optimmiser

	loss_fn = torch.nn.MSELoss()
	optimiser = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-4)

	# Training

	print("training started")
	model.train(True)

	num_epochs = 100

	for epoch in range(num_epochs):
		for i_batch, train_batch in enumerate(train_dataloader):

			model_input = train_batch["dna"]
			model_output_label = train_batch["signal"]
			if using_mps:
				model_input = model_input.to(mps_device)
				model_output_label = model_output_label.to(mps_device)

			model_output = model(model_input)

			loss = loss_fn(model_output, model_output_label)


			if i_batch % 1 == 0:
				print("aaaa", model_output)
				print("bbb", model_output_label)
				print("loss", loss)
				temp = input()

	
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

		print(epoch, loss)





train_CNN()

