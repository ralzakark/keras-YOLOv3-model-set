import json
import re
import base64
import os
from PIL import Image
from io import BytesIO
from absl import app, flags, logging
from absl.flags import FLAGS
import tqdm
import random

flags.DEFINE_string('json_dir', './data/json/', 'json dir')
flags.DEFINE_string('output_dir', './data/imageset/', 'jgp')
flags.DEFINE_string('anno_train', './data/annotation.train', 'Trainset annotation file')
flags.DEFINE_string('anno_val', './data/annotation.val', 'Validation annotation file')
flags.DEFINE_string('anno_test', './data/annotation.test', 'Test annotation file')
flags.DEFINE_string('class_map', './data/class_map', 'Class map')
flags.DEFINE_integer('random_seed', 1, 'Random seed used for dataset splitting', lower_bound=1)
flags.DEFINE_integer('validation_holdout', 20, '% of initial dataset to save for validation purpose')
flags.DEFINE_integer('test_holdout', 0, '% of initial dataset to save for test purpose')

def main(_argv):
	# Load class map as a "class to index" dictionary
	class_map = {name: idx for idx, name in enumerate(
		open(FLAGS.class_map).read().splitlines())}
	logging.info("Class mapp loaded: %s", class_map)

	# JPGs output directory
	out_dir = os.path.join(FLAGS.output_dir)
	os.makedirs(out_dir, exist_ok=True)

	# Input JSONs path
	json_dir = os.path.join(FLAGS.json_dir)

	# Annotation list
	# Three files will be written: for training, validation and test sets
	# Each file has one row (an only one for image)
	# Row format: image_file_path box1 box2...boxN
	# Box format: x_min,y_min,x_max,y_max,class_id (no space)
	annotations = set()

	for json_file in os.listdir(os.path.join(json_dir)):
		with open(os.path.join(json_dir, json_file)) as f:
			logging.info("Processing: %s", json_file)
			data = json.load(f)
			filename = os.path.splitext(os.path.basename(json_file))[0]

			for i in tqdm.tqdm(range(len(data))):
				d = data[i]
				s = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", d['image'], re.DOTALL).groupdict().get(
					"data")
				b = base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))
				b = BytesIO(b)
				im = Image.open(b)
				im = im.convert('RGB')
				jpg_file_name = os.path.join(out_dir, filename + '_' + str(i).zfill(5) + '.jpg')
				im.save(jpg_file_name)
				annotation_entry = str(jpg_file_name)
				for bb in d['bbox']:
					coord = []
					coord.append(bb['xmin'])
					coord.append(bb['ymin'])
					coord.append(bb['xmax'])
					coord.append(bb['ymax'])
					for i in range(len(coord)):
						if coord[i] < 0: coord[i] = 0
						if coord[i] > 1: coord[i] = 1
					annotation_entry = annotation_entry + \
									  (" %f,%f,%f,%f,%d" % (
									  int(coord[0]*416), int(coord[1]*416),
									  int(coord[2]*416), int(coord[3]*416),
									  class_map[bb['class']]))
				annotations.add(annotation_entry)
			f.close()

	# A random seed is generated, default to 1
	random.seed(FLAGS.random_seed)

	# Total number of annotation (i.e. jpg images)
	anno_num = len(annotations)
	val_num = int(anno_num / 100 * FLAGS.validation_holdout)
	test_num = int(anno_num / 100 * FLAGS.test_holdout)
	train_num = anno_num - val_num - test_num
	train_set = set(random.sample(annotations, train_num))
	val_set = set(random.sample(annotations - train_set , val_num))
	test_set = set(annotations - train_set - val_set)

	with open(FLAGS.anno_train, 'w') as f:
		for item in train_set:
			f.write("%s\n" % item)
		logging.info("%d annotations written in training set", len(train_set))
	with open(FLAGS.anno_val, 'w') as f:
		for item in val_set:
			f.write("%s\n" % item)
		logging.info("%d annotations written in validation set", len(val_set))
	with open(FLAGS.anno_test, 'w') as f:
		for item in test_set:
			f.write("%s\n" % item)
		logging.info("%d annotations written in test set", len(test_set))

if __name__ == '__main__':
	app.run(main)
