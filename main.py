import argparse
import yaml
import torch

import ResNet18
import train
import validation

parser = argparse.ArgumentParser()
parser.add_argument("--run_type",type=str,help="Network to be trained or evaluated")

def main():

	args = parser.parse_args()
	run_type = args.run_type

	config_file_path = "config.yaml"
	config_file = open(config_file_path,'r')
	config_dict = yaml.load(config_file,Loader=yaml.FullLoader)

	if run_type=="train":
		train.train(config_dict["dataset_dir"],
			config_dict["dataset_name"],
			config_dict["epochs"],
			config_dict["batch_size"],
			config_dict["checkpoint_path"],
			config_dict["train_val_split"],
			config_dict["num_workers"],
			config_dict["device"],
			config_dict["checkpoint_save_interval"],
			config_dict["validation_interval"],
			config_dict["load_checkpoint_path"])
	elif run_type=="eval":
		eval(config_dict["dataset_dir"],
			config_dict["dataset_name"],
			config_dict["batch_size"],
			config_dict["num_workers"],
			config_dict["device"],
			config_dict["load_checkpoint_path"])

if __name__ == "__main__":
	main()