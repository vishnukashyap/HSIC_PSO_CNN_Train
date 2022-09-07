import torch
from ResNet18 import global_best_particle
import utils

def validation(validation_dataset,model):

	with torch.no_grad():

		correct = 0.
		total = 0.
		total_loss = 0.
		count = 0
		layers = len(global_best_particle.Particles)
		loss_criterion = torch.nn.CrossEntropyLoss()

		for batch_idx,(data,target) in enumerate(validation_dataset):

			output = model(data,layers)

			loss = loss_criterion(output,target)
			batch_correct,batch_total = utils.get_classification_accuracy(output,target)

			correct += batch_correct
			total += batch_total
			
			total_loss = (count*total_loss + loss.item())/(count+1)
			count += 1

		accuracy = (correct/total)*100

		return accuracy,total_loss

def eval(dataset_path,dataset_name,batch_size,num_workers,device,last_checkpoint):

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

	val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers)

	# Load Model and Global Best
	if last_checkpoint is not None:
		last_checkpoint = torch.load(last_checkpoint)
	else:
		print("Unable to load checkpoint, checkpoint path is None")
		return
	model = last_checkpoint["Model"]
	ResNet18.global_best_particle = last_checkpoint["Global_Best"]

	accuracy,total_loss = validation.validation(val_dataloader,model)

	print("The trained model evaluated on "+str(dataset_name))
	print("\n Loss of model on dataset: "+str(total_loss))
	print("\n Accuracy of model on dataset: "+str(accuracy))

	return