import random
import torch

import hsic_funcs
import ResNet18

'''
	Implementation of Particle Swarm Optimization
	- The code for auto-hyperparameters to be implemented later
'''

weight = None
C1 = None
C2 = None
lambdaY = None

def init_PSO_params():
	global weight,C1,C2,lambdaY
	weight = 0.0001
	C1 = 1
	C2 = 1
	lambdaY = 0.25

def Calculate_Cost(outputs,inputs,targets):
	'''
		This function calculates the cost value of the particles of the given layer using normalized HSIC formula
	'''
	costs = []
	batch_size = inputs.shape[0]
	for i in range(len(outputs)):
		cost_x,cost_y = hsic_funcs.HSIC_objective(outputs[i].view(batch_size,-1),inputs.view(batch_size,-1),targets.view(batch_size,-1),None)
		print("cost_x: ",cost_x)
		print("cost_y: ",cost_y)
		cost = cost_x*10. - lambdaY*cost_y
		costs.append(cost)
	return costs

def Update(particle_weight,particle_bias,particle_p_best_weight,particle_p_best_bias,velocity,g_best_weight,g_best_bias):
	'''
		This function calculates the velocity at time t+1 as well as the updated particle as well as particle's best position
	'''
	diff_p_best_p = (particle_p_best_weight - particle_weight).norm(p=1)
	# if particle_bias is not None and particle_p_best_bias is not None:
	# 	diff_p_best_p = diff_p_best_p + (particle_p_best_bias - particle_bias).norm(p=1)

	diff_g_best_p = (g_best_weight - particle_weight).norm(p=1)
	# if particle_bias is not None and g_best_bias is not None:
	# 	diff_g_best_p = diff_g_best_p + (g_best_bias - particle_bias).norm(p=1)

	Velocity = weight*velocity + C1*random.uniform(0,2)*diff_p_best_p + C2*random.uniform(0,2)*diff_g_best_p
	weights = particle_weight + velocity
	if particle_bias is not None:
		bias = particle_bias + velocity
	else:
		bias = None

	return Velocity, weights, bias

def Update_Particles(particle,cost,global_best_particle):
	'''
		This function updates all the particles of a given layer in the model
	'''
	particle_weight = particle.weights
	particle_p_best_weight = particle.P_Best_w
	g_best_weight = global_best_particle.weights
	
	particle_bias = particle.bias
	particle_p_best_bias = particle.P_Best_b
	g_best_bias = global_best_particle.bias

	particle_velocity = particle.Velocity

	Velocity,weights,bias = Update(particle_weight,particle_bias,particle_p_best_weight,particle_p_best_bias,particle_velocity,g_best_weight,g_best_bias)

	particle.Velocity = Velocity
	particle.weights = weights
	particle.bias = bias

	if cost < particle.P_Best_cost:
		particle.P_Best_w = weights
		particle.P_Best_b = bias
		particle.P_Best_cost = cost

	return particle

def PSO_Update_Particles(model,outputs,inputs,targets,layer_idx):
	'''
		This function will update the particles (weights and biases of layer at layer_idx) using the particle updation method of PSO algorithm.
		The cost function to be minimized will be the normalized HSIC value combination for inputs, layer_wise outputs and targets.
	'''
	particle_costs = Calculate_Cost(outputs,inputs,targets)
	particles = model.get_particles(layer_idx)
	global_best_particle = ResNet18.global_best_particle.Particles[layer_idx]

	if particles == None:
		return None

	length = len(particles)
	
	for i in range(length):

		module = particles[i]

		if layer_idx==0 or layer_idx==9:
			module = Update_Particles(module,particle_costs[i],global_best_particle)
		elif layer_idx>0 or layer_idx<9:
			module.Conv1 = Update_Particles(module.Conv1,particle_costs[i],global_best_particle.Conv1)
			module.Conv2 = Update_Particles(module.Conv2,particle_costs[i],global_best_particle.Conv2)
			if len(module.Identity) != 0:
				module.Identity[0] = Update_Particles(module.Identity[0],particle_costs[i],global_best_particle.Identity[0])

		if particle_costs[i] < ResNet18.global_best_particle.Cost[layer_idx]:
			ResNet18.global_best_particle.Cost[layer_idx] = particle_costs[i]
			ResNet18.global_best_particle.Particles[layer_idx] = global_best_particle

		particles[i] = module

	model.set_particles(layer_idx,particles)

	return particle_costs