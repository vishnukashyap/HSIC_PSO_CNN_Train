import torch
import torchvision
import os
import tqdm

import ResNet18
import PSO_funcs
import validation
import utils

'''
	The current training code written here is specifically to attempt to train the network for the task of classification on ResNet-18 architecture.
'''

def initialize_global(model,num_layers):

	ResNet18.global_best_particle = ResNet18.Global_Best(num_layers)
	for i in range(10):
		ResNet18.global_best_particle.update_global_best(i,model.get_particles(i)[0],-1000)

	return


def train(dataset_dir,dataset_name,epochs,batch_size,save_checkpoint_dir,train_val_split,num_workers,device,checkpoint_save_interval,validation_interval,load_checkpoint=None):

	# Dataset Loader
	if dataset_name == 'CIFAR10':
		normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR10(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 10

	elif dataset_name == 'CIFAR100':
		normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],std=[0.2009, 0.1984, 0.2023])
		train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
		dataset = torchvision.datasets.CIFAR100(dataset_dir,train=True,download=True,transform=train_transform)
		indices = len(dataset)
		num_classes = 100

	train_indices = range(int(indices*train_val_split))
	train_data_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
	train_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=train_data_sampler)

	val_indices = range(int(indices*train_val_split),indices)
	val_data_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
	val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=val_data_sampler)

	# Model Initialization
	num_particles_per_layer = [2,2,2,2,2,2,2,2,2,5]
	num_layers = len(num_particles_per_layer)
	model = ResNet18.ResNet18(num_particles_per_layer,device)
	initialize_global(model,num_layers)
	PSO_funcs.init_PSO_params()
	
	start_epoch = 0

	# Load Checkpoint

	if load_checkpoint is not None:
		last_checkpoint = torch.load(load_checkpoint)
		model = last_checkpoint["Model"]
		ResNet18.global_best_particle = last_checkpoint["Global_Best"]
		start_epoch = last_checkpoint["Epoch"]+1

	# Loss Criterion
	# The loss criterion here is present only to check how well the model is learning
	loss_criterion = torch.nn.CrossEntropyLoss()

	with torch.no_grad():
		# Begin Training
		for epoch in range(start_epoch,epochs):
			
			progress_bar = tqdm.tqdm(enumerate(train_dataloader))
			progress_bar.set_description("Epoch: "+str(epoch))
			print("\n")

			for batch_idx,(data,target) in progress_bar:

				data = data.to(device)
				one_hot_target = utils.to_categorical(target,num_classes).to(device)
				target = target.to(device)

				for i in range(num_layers):

					outputs = model(data,i)
					if outputs == None:
						return
					costs = PSO_funcs.PSO_Update_Particles(model,outputs,data,one_hot_target,i)
					if costs == None:
						return
					print("The costs for layer "+str(i)+" calculated using nHSIC are: "+str(costs))

				output = model(data,num_layers)
				loss = loss_criterion(output,one_hot_target)
				accuracy = utils.get_classification_accuracy(output,target)
				print("\nTraining Loss: "+str(loss)+" and Accuracy: "+str(accuracy))

			if epoch%validation_interval==0:
				accuracy,total_loss = validation.validation(val_dataloader,model)
				print("\nValidation Loss: "+str(total_loss)+" and Accuracy: "+str(accuracy))

			if epoch%checkpoint_save_interval==0:
				checkpoint_path = os.path.join(save_checkpoint_dir,"PSO_Model_"+str(epoch))
				checkpoint = {}
				checkpoint["Epoch"] = epoch
				checkpoint["Model"] = model
				checkpoint["Global_Best"] = ResNet18.global_best_particle
				torch.save(checkpoint,checkpoint_path)

	return