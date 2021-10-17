# -*- coding: utf-8 -*-
"""
TAGGING
"""


def _load_data(file_or_files: 'list files|txt file|folder', tagging_type: 'str ["img", "text", "email"]', **kwargs):
	"""
	Loading data file
	"""
	import os
	from PIL import Image
	import base64
	constants = {
		'img_formats': ['png', 'jpg', 'jpeg'],
		'txt_formats': ['txt', 'csv', 'tsv']
	}
	if 'metadata' in kwargs.keys():
		metadata = kwargs['metadata']
	else:
		metadata = {}
	metadata['tagging_type'] = tagging_type
	if tagging_type == "text" \
		and (type(file_or_files) == str or (type(file_or_files) == list and len(file_or_files) == 1)) \
		and True in [i in list(file_or_files)[0] for i in constants['txt_formats']]:
		if type(file_or_files) == list:
			file_or_files = file_or_files[0]
		# single file txt
		if 'sep' not in kwargs.keys():
			print('If you want load txt file you have to insert in function sep="separator_value"')
			return 'Error in loading file', metadata
		else:
			with open(file_or_files, 'r') as f:
				to_tag = {str(k): v for k, v in enumerate('\n'.join(f.readlines()).split(kwargs['sep']))}
	elif tagging_type == "text" and type(file_or_files) == list \
		and False not in [i.split('.')[1] in constants['txt_formats'] for i in file_or_files]:
		# multiple files
		to_tag = {}
		for file in file_or_files:
			with open(file, 'r') as f:
				to_tag[file] = '\n'.join(f.readlines())
	elif tagging_type == "text" and type(file_or_files) == str and '.' not in file_or_files:
		# single folder
		to_tag = {}
		for file in [i for i in os.listdir(str(file_or_files)) if i.split('.')[1] in constants['txt_formats']]:
			with open(f'{file_or_files}/{file}', 'r') as f:
				to_tag[f'{file_or_files}/{file}'] = '\n'.join(f.readlines())
	elif tagging_type == "img" and type(file_or_files) == list \
		and False not in [i.split('.')[1] in constants['img_formats'] for i in file_or_files]:
		# multiples files
		to_tag = {}
		for file in file_or_files:
			if 'base64' in kwargs.keys() and kwargs['base64']:
				with open(file, 'rb') as f:
					to_tag[file] = 'data:image/' + str(file.split(".")[-1]) + ';base64,'\
									+ str(base64.b64encode(f.read()))\
									.replace("b'", "").replace("'", "").replace(" ", "+")
			else:
				to_tag[file] = Image.open(file)
	elif tagging_type == "img" and type(file_or_files) == str and '.' not in file_or_files:
		# single folder
		to_tag = {}
		for file in os.listdir(str(file_or_files)):
			if 'base64' in kwargs.keys() and kwargs['base64']:
				with open(f'{file_or_files}/{file}', 'rb') as f:
					to_tag[f'{file_or_files}/{file}'] = 'data:image/' + str(file.split(".")[-1]) + ';base64,'\
												+ str(base64.b64encode(f.read()))\
												.replace("b'", "").replace("'", "").replace(" ", "+")
			else:
				to_tag[file] = Image.open(f'{file_or_files}/{file}')
	else:
		print('Generic error in load data read documentation')
		return 'Error in loading file', metadata
	metadata['tot_to_tag'] = len(to_tag.keys())
	metadata['tot_tagged'] = 0
	return to_tag, metadata


def _tagging_dynamic(tagged, file, category):
	"""
	Tagging fun
	"""
	try:
		tagged[file] = category
		return tagged
	except Exception as e:
		print(e)
		return 'Error in tagging dynamic'


def _extract_to_tag(to_tag, tagged, select_random=False):
	"""
	Extract next to tag fun
	"""
	import random
	if not select_random:
		for k, _ in to_tag.items():
			try:
				# search with dict keys is the fastest
				tagged[str(k)]
			except KeyError as err:
				_ = err
				return k
			except Exception as e:
				print(e)
				return 'Error in tagging dynamic'
	else:
		ls = [k for k, _ in to_tag.items() if k not in tagged.keys()]
		return ls[random.randint(0, len(ls)-1)]
	return 'All tagged'


def _load_tag_files(analysis_token: 'analysis token'):
	import pickle
	import json
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	with open(f'{dir_path}/{analysis_token}/to_tag.pkl', 'rb') as f:
		to_tag = pickle.load(f)
	with open(f'{dir_path}/{analysis_token}/tagged.pkl', 'rb') as f:
		tagged = pickle.load(f)
	with open(f'{dir_path}/{analysis_token}/metadata.json', 'r', encoding='utf8') as f:
		metadata = json.load(f)
	return to_tag, tagged, metadata['tagging_type'], metadata['categories']


def _add_tagged(analysis_token: 'analysis token'):
	import json
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	with open(f'{dir_path}/{analysis_token}/metadata.json', 'r', encoding='utf8') as f:
		metadata = json.load(f)
	metadata['tot_tagged'] = int(metadata['tot_tagged']) + 1
	with open(f'{dir_path}/{analysis_token}/metadata.json', 'w', encoding='utf8') as f:
		json.dump(metadata, f, indent=6, allow_nan=False, ensure_ascii=True)


def _save_file_pkl(input_obj: 'obj binary', name: 'file name', analysis_token: 'analysis token'):
	import pickle
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	try:
		os.mkdir(dir_path)
	except FileExistsError:
		pass
	try:
		os.mkdir(f'{dir_path}/{analysis_token}')
	except FileExistsError:
		pass
	try:
		with open(f'{dir_path}/{analysis_token}/{name}.pkl', 'wb') as f:
			pickle.dump(input_obj, f)
	except Exception as e:
		print(e)
		pass


def _save_file_json(input_obj: dict, name: 'file name', analysis_token: 'analysis token'):
	import json
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	try:
		os.mkdir(dir_path)
	except FileExistsError:
		pass
	try:
		os.mkdir(f'{dir_path}/{analysis_token}')
	except FileExistsError:
		pass
	try:
		with open(f'{dir_path}/{analysis_token}/{name}.json', 'w', encoding='utf8') as f:
			json.dump(input_obj, f, indent=6, allow_nan=False, ensure_ascii=True)
	except Exception as e:
		print(e)
		pass


def _all_analyzes(**kwargs):
	import json
	import shutil
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	if 'delete' in kwargs.keys():
		shutil.rmtree(f'{dir_path}\\{kwargs["delete"]}')
	if 'deletemodel' in kwargs.keys():
		shutil.rmtree(f'{dir_path}\\{kwargs["deletemodel"]}\\{kwargs["tipo"]}')
	if 'stop' not in kwargs.keys() or kwargs["stop"] != 'y':
		try:
			analyzes_folder = os.listdir(dir_path)
			num_analyzes = len(analyzes_folder)
			token_analysis = []
			title_analysis = []
			tot_tagged = []
			tot_to_tag = []
			percentage_analysis = []
			typetag = []
			for analysis_folder in analyzes_folder:
				with open(f'{dir_path}/{analysis_folder}/metadata.json', 'r', encoding='utf8') as f:
					metadata = json.load(f)
				token_analysis.append(metadata['analysis_token'])
				title_analysis.append(metadata['analysis_title'])
				tot_tagged.append(metadata['tot_tagged'])
				tot_to_tag.append(metadata['tot_to_tag'])
				typetag.append(metadata['tagging_type'])
				percentage_analysis.append(int(metadata['tot_tagged']/metadata['tot_to_tag']*100))

			return num_analyzes, token_analysis, title_analysis, tot_tagged, tot_to_tag, percentage_analysis, typetag
		except Exception as e:
			print(e)
			return 0, ['no analysis found'], ['no analysis found'], \
				['no analysis found'], ['no analysis found'], ['no analysis found'], ['no analysis found']


def _export_analysis(analysis_token: 'analysis token', **kwargs):
	import tarfile
	import pickle
	import base64
	import shutil
	import json
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'

	if 'tipo' in kwargs.keys():
		folder_path = f'{dir_path}\\{analysis_token}\\{kwargs["tipo"]}'
		with tarfile.open(folder_path + ".tgz", "w:gz") as tar:
			for subfolder in os.listdir(folder_path):
				tar.add(f'{folder_path}/{subfolder}', arcname=subfolder)
		return folder_path + ".tgz"
	else:
		with open(f'{dir_path}/{analysis_token}/to_tag.pkl', 'rb') as f:
			to_tag = pickle.load(f)
		with open(f'{dir_path}/{analysis_token}/tagged.pkl', 'rb') as f:
			tagged = pickle.load(f)
		with open(f'{dir_path}/{analysis_token}/metadata.json', 'r', encoding='utf8') as f:
			metadata = json.load(f)
		folder_path = f'{os.path.dirname(path)}/Static/{analysis_token}'
		try:
			os.mkdir(folder_path)
		except FileExistsError:
			pass
		for categ in set(tagged.values()):
			try:
				os.mkdir(f'{folder_path}/{categ}')
			except FileExistsError as e:
				print(e)
				pass
		if metadata['tagging_type'] == 'img':
			for k, v in tagged.items():
				imgdata = base64.b64decode(to_tag[k].split(';base64,')[-1])
				filename = f'{folder_path}/{v}/{k.replace("upload_tmp/", "")}'
				with open(filename, 'wb') as f:
					f.write(imgdata)
		else:
			for k, v in tagged.items():
				textdata = to_tag[k]
				filename = f'{folder_path}/{v}/{k.replace("upload_tmp/", "").split(".")[0]}.txt'
				with open(filename, 'w') as f:
					f.write(textdata)
		with tarfile.open(folder_path + ".tgz", "w:gz") as tar:
			for subfolder in os.listdir(folder_path):
				tar.add(f'{folder_path}/{subfolder}', arcname=subfolder)
		shutil.rmtree(folder_path)
		return folder_path + ".tgz"


def _remove_analysis():
	import os
	path = os.path.abspath(__file__)
	files = [i for i in os.listdir(f'{os.path.dirname(path)}/Static') if '.tgz' in i]
	if len(files) > 0:
		for file in files:
			os.remove(f'{os.path.dirname(path)}/Static/{file}')


def _exist_in_analysis(analysis_token: 'analysis_token', folder: str):
	import os
	path = os.path.abspath(__file__)
	dir_path = os.path.dirname(path) + '\\Uploads'
	return folder in os.listdir(f'{dir_path}/{analysis_token}')