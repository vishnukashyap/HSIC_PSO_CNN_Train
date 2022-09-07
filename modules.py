import torch
import random

'''
	In these modules, the weights and biases are the particles
	the velocity for the particles are part for the Linear and Conv2D classes
'''

class Linear(torch.nn.Module):
	def __init__(self,in_channels,out_channels,device,bias=False):
		super(Linear,self).__init__()

		self.weights = torch.empty(out_channels,in_channels)
		torch.nn.init.xavier_uniform_(self.weights)
		self.bias = None
		self.device = device

		if bias == True:
			self.bias = torch.empty(out_channels)
			torch.nn.init.ones_(self.bias)

		# PSO Parameters
		self.Velocity = random.random()*10 # Select a random number between 0 and 10
		self.P_Best_w = self.weights
		self.P_Best_b = self.bias
		self.P_Best_cost = -1000000


	def forward(self,input_tensor):
		if self.bias is not None:
			return torch.nn.functional.linear(input_tensor,self.weights.to(self.device),self.bias.to(self.device))
		return torch.nn.functional.linear(input_tensor,self.weights.to(self.device),self.bias)

class Conv2D(torch.nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,device,stride=1,padding=1,bias=False):
		super(Conv2D,self).__init__()
		
		self.stride = stride
		self.padding = padding
		self.device = device

		if type(kernel_size)==tuple:
			self.weights = torch.empty(out_channels,in_channels,kernel_size[0],kernel_size[1])
		elif type(kernel_size)==int:
			self.weights = torch.empty(out_channels,in_channels,kernel_size,kernel_size)
		torch.nn.init.xavier_uniform_(self.weights)

		self.bias = None
		if bias == True:
			self.bias = torch.empty(out_channels)
			torch.nn.init.ones_(self.bias)

		# PSO Parameters
		self.Velocity = random.random()*10# Select a random number between 0 and 10
		self.P_Best_w = self.weights
		self.P_Best_b = self.bias
		self.P_Best_cost = -1000

	def forward(self,input_tensor):
		if self.bias is not None:
			return torch.nn.functional.conv2d(input_tensor, self.weights.to(self.device), bias=self.bias.to(self.device), stride=self.stride, padding=self.padding)
		return torch.nn.functional.conv2d(input_tensor, self.weights.to(self.device), bias=self.bias, stride=self.stride, padding=self.padding)

class ResidualBlock(torch.nn.Module):
	def __init__(self,in_channels,out_channels,device,stride=1,bais=False):
		super(ResidualBlock,self).__init__()
		self.Conv1 = Conv2D(in_channels,out_channels,kernel_size=3,device=device,stride=stride,padding=1,bias=False)
		self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
		self.Act1 = torch.nn.ReLU()

		self.Conv2 = Conv2D(out_channels,out_channels,kernel_size=3,device=device,stride=1,padding=1,bias=False)
		self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
		self.Act2 = torch.nn.ReLU()

		self.Identity = torch.nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.Identity = torch.nn.Sequential(*[Conv2D(in_channels,out_channels,device=device,kernel_size=1,stride=stride,padding=0,bias=False),torch.nn.BatchNorm2d(out_channels)])

	def forward(self,input_tensor):
		output = self.Act1(self.batch_norm1(self.Conv1(input_tensor)))
		output = self.batch_norm2(self.Conv2(output)) + self.Identity(input_tensor)
		output = self.Act2(output)
		return output