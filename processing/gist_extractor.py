from processing import gist
from tqdm import tqdm
import sys
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import re
import os
import multiprocessing as mp

param = {
	"orientationsPerScale": np.array([8, 8]),
	"numberBlocks": [10, 10],
	"fc_prefilt": 10,
	"boundaryExtension": 32
}


class Dataloader():
	def __init__(self, input_path, output_path):
		self.input_path = input_path
		self.output_path = output_path
		self.is_dir = 0 if re.search("\.", input_path) != None else 1

	def get_inputfile(self) -> list:
		if self.is_dir:
			# dirctory in images
			# path = f"./{self.input_path}/"
			path = self.input_path
			a = sorted(os.listdir(path))
			file_list = list(map(lambda x: path + x, a))

			return file_list
		else:
			# image file such png, jpg etc..
			# path = f"./{self.input_path}"
			path = self.input_path
			return [path]

	def save_feature(self, x: np.array):
		if self.is_dir:
			gist_df = pd.DataFrame(x, columns=[f"gist_{i}" for i in range(x.shape[1])])
		else:
			gist_df = pd.DataFrame(x.reshape(1, -1), columns=[f"gist_{i}" for i in range(x.shape[1])])

		gist_df.to_feather(f"{self.output_path}")


def _get_gist_single(param: dict, file_path):
	img = Image.open(file_path).convert("L")
	img = img.resize((256, 256), Image.BILINEAR)
	#img = img.resize((int(img.size[0] / 4), int(img.size[1] / 4)), Image.BILINEAR)

	gist_runner = gist.GIST(param)

	file_list = [file_path]
	img_list = list(map(lambda f: np.array(img), file_list))

	with mp.Pool(mp.cpu_count()) as pool:
		p = pool.imap(gist_runner._gist_extract, img_list[:])
		gist_feature = list(tqdm(p, total=len(img_list)))
	return np.array(gist_feature)

def _get_gist(param: dict, file_list: list) -> np.array:
	img_list = list(map(lambda f: np.array(Image.open(f).convert("L")), file_list))
	gist_runner = gist.GIST(param)

	with mp.Pool(mp.cpu_count()) as pool:
		p = pool.imap(gist_runner._gist_extract, img_list[:])
		gist_feature = list(tqdm(p, total=len(img_list)))
	return np.array(gist_feature)

def store_summaries(file_list, base_path):
	BENCHMARK_PATH = base_path + "metrics.txt"

	average = 0.0
	lowest = 99999.0
	highest = 0.0
	with open(BENCHMARK_PATH, "w") as f:
		for file_path in file_list:
			file_name = file_path.split("/")[7]
			gist_feature = _get_gist_single(param, file_path)
			norm_rounded = np.round(np.linalg.norm(gist_feature), 5)
			average = average + norm_rounded
			print("Image: " + file_name + " : " + str(norm_rounded), file = f)

			if(norm_rounded < lowest):
				lowest = norm_rounded
			if(norm_rounded > highest):
				highest = norm_rounded

		average = np.round(average / len(file_list), 5)
		print("Average GIST norm: " +str(average), file = f)
		print("Lowest GIST norm: " + str(lowest), file=f)
		print("Highest GIST norm: " + str(highest), file=f)

def main():
	BASE_PATH = "G:/My Drive/CONFERENCES/WSCG 2021/Experiments/"

	arg = argparse.ArgumentParser()
	arg.add_argument("--input_path", default=BASE_PATH + "Impressionism sorted/post-1886/")
	arg.add_argument("--output_path", default=BASE_PATH + "gist.feather")
	arg.add_argument("--save", default=True)
	args = arg.parse_args()
	print(args)
	data = Dataloader(args.input_path, args.output_path)
	file_list = data.get_inputfile()
	store_summaries(file_list, BASE_PATH)

	# if args.save == True:
	# 	data.save_feature(gist_feature)


if __name__ == "__main__":
	main()