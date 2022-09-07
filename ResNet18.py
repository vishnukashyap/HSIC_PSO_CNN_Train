import torch
import modules

'''
	Training for this network will be done based on the HSIC BottleNeck.
	During the updation of the weights, for layers which depend on the output of the previous layer, the global best weights and biases will be used to calculate the output upto that point
	Since each layer is a swarm, trying to train all layers in one pass would require increase the memory requirement since the number of outputs
	would be equal to the number of product of the number of particle in each layer. Hence as a work around to this, the layers are trained iteratively
	i.e. the batch is forward propagated, the output of one layer is taken, updated, then the dataset is forward propagated again to update the next layer.
	The Network class has been updated so as to be able to train the network as per the method described above.
'''

global_best_particle = None

class ResNet18(torch.nn.Module):
	def __init__(self,particles,device):
		super(ResNet18,self).__init__()
		
		self.device = device
		if len(particles)<10:
			particles = particles + [particles[-1]]*(10-len(particles))

		self.Conv1 = [modules.Conv2D(3,64,kernel_size=3,device=device,stride=1,padding=1) for _ in range(particles[0])]
		self.batch_norm1 = torch.nn.BatchNorm2d(64).to(self.device)

		self.ResBlock1_1 = [modules.ResidualBlock(64,64,device=device) for _ in range(particles[1])]
		self.ResBlock1_2 = [modules.ResidualBlock(64,64,device=device) for _ in range(particles[2])]

		self.ResBlock2_1 = [modules.ResidualBlock(64,128,stride=2,device=device) for _ in range(particles[3])]
		self.ResBlock2_2 = [modules.ResidualBlock(128,128,device=device) for _ in range(particles[4])]

		self.ResBlock3_1 =[ modules.ResidualBlock(128,256,stride=2,device=device) for _ in range(particles[5])]
		self.ResBlock3_2 = [modules.ResidualBlock(256,256,device=device) for _ in range(particles[6])]

		self.ResBlock4_1 = [modules.ResidualBlock(256,512,stride=2,device=device) for _ in range(particles[7])]
		self.ResBlock4_2 = [modules.ResidualBlock(512,512,device=device) for _ in range(particles[8])]

		self.FC = [modules.Linear(512,10,bias=True,device=device) for _ in range(particles[9])]	

		self.SoftMax = torch.nn.Softmax().to(self.device)

	def forward(self,input_tensor,layer_idx):
		global global_best_particle
		outputs = []
		output = None

		if layer_idx > 10:
			print("Error: layer index cannot be greater than 10 given "+str(layer_idx))
			return None

		if layer_idx < -10:
			print("Error: layer index cannot be lesser than -9 given "+str(layer_idx))
			return None

		if layer_idx < 0:
			layer_idx = 10 - layer_idx

		################## Layer 0 ##################
		if layer_idx == 0:
			for i in range(len(self.Conv1)):
				module = self.Conv1[i].to(self.device)
				outputs.append(self.batch_norm1(module(input_tensor)))
				del module
			return outputs
		
		module = global_best_particle.Particles[0].to(self.device)
		output = module(input_tensor)
		output = self.batch_norm1(output)
		del module

		################## Layer 1 ##################
		if layer_idx == 1:
			for i in range(len(self.ResBlock1_1)):
				module = self.ResBlock1_1[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[1].to(self.device)
		output = module(output)
		del module

		################## Layer 2 ##################
		if layer_idx == 2:
			for i in range(len(self.ResBlock1_2)):
				module = self.ResBlock1_2[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[2].to(self.device)
		output = module(output)
		del module

		################## Layer 3 ##################
		if layer_idx == 3:
			for i in range(len(self.ResBlock2_1)):
				module = self.ResBlock2_1[i].to(self.device)
				outputs.append(module(output))
			return outputs
		
		module = global_best_particle.Particles[3].to(self.device)
		output = module(output)
		del module

		################## Layer 4 ##################
		if layer_idx == 4:
			for i in range(len(self.ResBlock2_2)):
				module = self.ResBlock2_2[i].to(self.device)
				outputs.append(module(output))
			return outputs
		
		module = global_best_particle.Particles[4].to(self.device)
		output = module(output)
		del module

		################## Layer 5 ##################
		if layer_idx == 5:
			for i in range(len(self.ResBlock3_1)):
				module = self.ResBlock3_1[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[5].to(self.device)
		output = module(output)
		del module

		################## Layer 6 ##################
		if layer_idx == 6:
			for i in range(len(self.ResBlock3_2)):
				module = self.ResBlock3_2[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[6].to(self.device)
		output = module(output)
		del module

		################## Layer 7 ##################
		if layer_idx == 7:
			for i in range(len(self.ResBlock4_1)):
				module = self.ResBlock4_1[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[7].to(self.device)
		output = module(output)
		del module

		################## Layer 8 ##################
		if layer_idx == 8:
			for i in range(len(self.ResBlock4_2)):
				module = self.ResBlock4_2[i].to(self.device)
				outputs.append(module(output))
				del module
			return outputs
		
		module = global_best_particle.Particles[8].to(self.device)
		output = module(output)
		output = torch.nn.functional.avg_pool2d(output,4)
		output = output.view(output.size(0),-1)
		del module

		################## Layer 9 ##################
		if layer_idx == 9:
			for i in range(len(self.FC)):
				module = self.FC[i].to(self.device)
				outputs.append(torch.sigmoid(module(output)))
				del module
			return outputs

		module = global_best_particle.Particles[9].to(self.device)
		output = module(output)
		output = torch.sigmoid(output)
		del module

		return output

	def get_particles(self,layer_idx):
		if layer_idx == 0:
			return self.Conv1

		if layer_idx == 1:
			return self.ResBlock1_1

		if layer_idx == 2:
			return self.ResBlock1_2

		if layer_idx == 3:
			return self.ResBlock2_1

		if layer_idx == 4:
			return self.ResBlock2_2

		if layer_idx == 5:
			return self.ResBlock3_1

		if layer_idx == 6:
			return self.ResBlock3_2

		if layer_idx == 7:
			return self.ResBlock4_1

		if layer_idx == 8:
			return self.ResBlock4_2

		if layer_idx == 9:
			return self.FC

		return None

	def set_particles(self,layer_idx,particles):
		if layer_idx == 0:
			self.Conv1 = particles

		elif layer_idx == 1:
			self.ResBlock1_1 = particles

		elif layer_idx == 2:
			self.ResBlock1_2 = particles

		elif layer_idx == 3:
			self.ResBlock2_1 = particles

		elif layer_idx == 4:
			self.ResBlock2_2 = particles

		elif layer_idx == 5:
			self.ResBlock3_1 = particles

		elif layer_idx == 6:
			self.ResBlock3_2 = particles

		elif layer_idx == 7:
			self.ResBlock4_1 = particles

		elif layer_idx == 8:
			self.ResBlock4_2 = particles

		elif layer_idx == 9:
			self.FC = particles

		else:
			print("Invalid layer_idx:"+str(layer_idx))

class Global_Best:
	def __init__(self,num_layers):
		self.Particles = [None]*num_layers
		self.Cost = [None]*num_layers

	def update_global_best(self,layer_idx,particle,cost):
		self.Particles[layer_idx] = particle.to("cpu")
		self.Cost[layer_idx] = cost