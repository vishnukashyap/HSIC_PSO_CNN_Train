import torch

def get_classification_accuracy(output,target):
	'''
		This function returns the number of correct predictions and the total number of predictions
	'''
	_ , output_idx = torch.topk(output,k=1,dim=1)
	output_idx = output_idx.squeeze(1)

	correct = (output_idx==target).sum().item()

	total = int(target.shape[0])

	return int(correct),total

def to_categorical(y,num_classes):
	'''
		1 hot encoding of the targets
	'''
	return torch.squeeze(torch.eye(num_classes)[y])