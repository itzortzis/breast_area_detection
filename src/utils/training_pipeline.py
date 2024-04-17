
import time
import torch
import calendar
import numpy as np
from tqdm import tqdm
from torchmetrics import F1Score
from matplotlib import pyplot as plt


from utils.data_preprocessing import DPP
from utils.initializer import INIT
from utils.print_info import Printer


class Training():

	def __init__(self, comps, params, paths):
		self.comps = INIT(comps, params, paths)
		self.comps.run()
		self.init()


	def init(self):
		self.pr = Printer(self.comps)
		dpp = DPP(self.comps.dataset)
		dpp.run()
		self.train_ldr = dpp.train_ldr
		self.valid_ldr = dpp.valid_ldr
		self.test_ldr = dpp.valid_ldr
		self.losses = np.zeros((self.comps.epochs, 2))
		self.scores = np.zeros((self.comps.epochs, 2))
		self.max_score = 0
		self.log = open("logs.txt", "a")  # append mode


		if self.comps.device == 'cuda':
			print("Cuda available")
			self.comps.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.comps.model = self.comps.model.to(self.comps.device)




	# Main_training:
	# --------------
	# The supervisor of the training procedure.
	def main_training(self):
		if not (self.pr.print_train_details()):
			return
		self.get_current_timestamp()
		print("Training is starting...")
		start_time = time.time()
		self.metric = F1Score(task="binary", num_classes=2)
		self.metric.to(self.comps.device)
		for epoch in tqdm(range(self.comps.epochs)):
			
			tr_score, tr_loss = self.epoch_training()
			vl_score, vl_loss = self.epoch_validation()

			self.losses[epoch, 0] = tr_loss
			self.losses[epoch, 1] = vl_loss
			self.scores[epoch, 0] = tr_score
			self.scores[epoch, 1] = vl_score

			print()
			print("\t Training - Score: ", tr_score, " Loss: ", tr_loss)
			print("\t Validation: - Score: ", vl_score, " Loss: ", vl_loss)
			print()
			self.save_model_weights(epoch, vl_score, vl_loss)
		self.exec_time = time.time() - start_time
		print("Total execution time: ", self.exec_time, " seconds")
		self.test_set_score = self.inference()
		self.log_line = str(self.test_set_score) + " " + self.log_line
		self.save_metrics()
		self.update_log()


	def update_log(self):
		self.log.write(self.log_line)
		self.log.close()


	# Get_current_timestamp:
	# ----------------------
	# This function calculates the current timestamp that is
	# used as unique id for saving the experimental details
	def get_current_timestamp(self):
		current_GMT = time.gmtime()
		self.timestamp = calendar.timegm(current_GMT)


	# Save_model_weights:
	# -------------------
	# This funtion saves the model weights during training
	# procedure, if some requirements are satisfied.
	#
	# --> epoch: current epoch of the training
	# --> score: current epoch score value
	# --> loss: current epoch loss value
	def save_model_weights(self, epoch, score, loss):

		if score > self.max_score and epoch > self.comps.epoch_thr:
			path_to_model = self.comps.trained_models + self.comps.dtst_name
			path_to_model += "_" + str(self.timestamp) + ".pth"
			torch.save(self.comps.model.state_dict(), path_to_model)
			self.comps.model_dict = self.comps.model.state_dict()
			log = str(epoch) + " " + str(score) + " " + path_to_model + "\n"
			self.log_line = log
			self.max_score = score


	# Prepare_data:
	# -------------
	# Given x and y tensors, this function applies some basic
	# transformations/changes related to dimensions, data types,
	# and device.
	#
	# --> x: tensor containing a batch of input images
	# --> y: tensor containing a batch of annotation masks
	# <-- x, y: the updated tensors
	def prepare_data(self, x, y):
		# print('This is x size: ', x.size())
		# print('This is x_ size: ', x_.size())
		if len(x.size()) < 4:
			x = torch.unsqueeze(x, 1)
		else:
			x = x.movedim(2, -1)
			x = x.movedim(1, 2)

		x = x.to(torch.float32)
		y = y.to(torch.int64)

		x = x.to(self.comps.device)
		y = y.to(self.comps.device)

		return x, y
		

	# Epoch_training:
	# ---------------
	# This function is used for implementing the training
	# procedure during a single epoch.
	#
	# <-- epoch_score: performance score achieved during
	#                  the training
	# <-- epoch_loss: the loss function score achieved during
	#                 the training
	def epoch_training(self):
		self.comps.model.train(True)
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		# print("Simple epoch training...")

		step = 0
		for x, y in self.train_ldr:
			x, y = self.prepare_data(x, y)
			step += 1
			self.comps.opt.zero_grad()
			outputs = self.comps.model(x)
			# print(y, outputs)
			loss = self.comps.loss_fn(outputs, y)
			# print("Simple training loss: ", loss)
			loss.backward()
			self.comps.opt.step()
			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = self.metric.compute()
		self.metric.reset()
		epoch_loss  = current_loss / len(self.train_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()

	
 
	# Epoch_validation:
	# ---------------
	# This function is used for implementing the validation
	# procedure during a single epoch.
	#
	# <-- epoch_score: performance score achieved during
	#                  the validation
	# <-- epoch_loss: the loss function score achieved during
	#                 the validation
	def epoch_validation(self):
		self.comps.model.train(False)
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()

		for x, y in self.valid_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.comps.model(x)
				loss = self.comps.loss_fn(outputs, y)

			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = self.metric.compute()
		epoch_loss  = current_loss / len(self.valid_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()


	# Inference:
	# ----------
	# Applies inference to the testing set extracted from
	# the input dataset during the initialization phase
	#
	# <-- test_set_score: the score achieved by the trained model
	def inference(self):
		self.comps.model.load_state_dict(self.comps.model_dict)
		self.comps.model.eval()
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		for x, y in self.test_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.comps.model(x)

			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)

		test_set_score = self.metric.compute()
		self.metric.reset()
		return test_set_score.item()


	def ext_inference(self, set_ldr):
		path_to_model = self.comps.trained_models + self.comps.inf_model
		self.comps.model.load_state_dict(torch.load(path_to_model))
		self.comps.model.eval()
		current_score = 0.0
		self.metric = F1Score(task="binary", num_classes=2)
		self.metric.to(self.comps.device)
		# self.metric.reset()
		for x, y in set_ldr:
			x,  y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.comps.model(x)

			preds = torch.argmax(outputs, dim=1)
			self.metric.update(preds, y)

		test_set_score = self.metric.compute()
		self.metric.reset()
		return test_set_score.item()



	def save_metrics(self):
		postfix = self.comps.dtst_name + "_" + str(self.timestamp)
		np.save(self.comps.metrics + "scores_" + postfix, self.scores)
		np.save(self.comps.metrics + "losses_" + postfix, self.losses)
		self.save_figures()


	def save_figures(self):
		postfix = self.comps.dtst_name + "_" + str(self.timestamp) + ".png"
		plt.figure()
		plt.plot(self.scores[:, 0])
		plt.savefig(self.comps.figures + "train_s_" + postfix)
		plt.figure()
		plt.plot(self.scores[:, 1])
		plt.savefig(self.comps.figures + "valid_s_" + postfix)

		plt.figure()
		plt.plot(self.losses[:, 0])
		plt.savefig(self.comps.figures + "train_l_" + postfix)
		plt.figure()
		plt.plot(self.losses[:, 1])
		plt.savefig(self.comps.figures + "valid_l_" + postfix)