import argparse
import json
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description='This script will generate a list of all the filtered wbc and pbmc cells and write it to the output directory. Consecutively it will create config json files for all the cells and write them to the generated config directory')

parser.add_argument("--meta_file", help="File containing the names of the cells, the filters and the percentages")
parser.add_argument("--percent_minimum", help="The minimum percentage of a certain class to be present")
parser.add_argument("--generated_config_dir",
                    help="Path where to store the generated train and predicition config json files")
parser.add_argument("--model_dir", help="Path where the best model is stored (needed for prediction config file)")
parser.add_argument("--output_dir", help="Output directory for storing list of wbc and pbms cells after filtering")
args = parser.parse_args()


def generate_json_file_for_hpc(type,generated_config_dir,model_dir):
  with open(generated_config_dir+'train-2-deepflow_example.json', 'r') as json_file:
    train_json = json.load(json_file)
  with open(generated_config_dir+'pred-2-deepflow_example.json', 'r') as json_file:
    pred_json = json.load(json_file)
  path = generated_config_dir+type+'/'
  train_path = path + 'train-2-deepflow_hpc_combined.json'
  pred_path = path + 'pred-2-deepflow_hpc_combined.json'
  # Create directories if they don't exist
  os.makedirs(os.path.dirname(train_path), exist_ok=True)
  os.makedirs(os.path.dirname(pred_path), exist_ok=True)
  train_json['h5_data'] = 'combined.h5'
  train_json['meta']='combined.csv'
  train_json['name_experiment']= 'combined_' + type
  train_json['split_dir'] = 'Split/combined/'
  pred_json['h5_data'] = 'combined.h5'
  pred_json['model_hdf5'] = model_dir + type +'/combined/best-model.hdf5'
  pred_json['split_dir'] = 'Split/combined/'
  with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_json, f, ensure_ascii=False, indent=4)
  with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(pred_json, f, ensure_ascii=False, indent=4)


#def cell_name(plate, well):
    #if len(well) == 2:
        #well = well[0] + '0' + well[1]
    #name = 'Plate' + str(plate) + '_' + well
    #return name

def cell_name(meta_plate,Well):
    if len(Well)==2:
      Well=Well[0]+'0'+Well[1]
    name= 'Plate'+str(meta_plate)+'_'+Well
    return name

meta_file="/kyukon/home/gent/455/vsc45530/Metadata2.csv"
metadata = pd.read_csv(meta_file)
metadata1 = metadata[(~ metadata['Marker name'].isin(['BLANK', 'Isotype']) & (metadata['Valid'] == 1) & (metadata['meta_cells'] == 'WBC')&
                     (metadata['Grans'] == 'Y')& (metadata['Monos'] == 'N')&(metadata['Super grans'] == 'N'))]
#metadata1['cellname']= metadata1.apply(lambda row : cell_name(row['meta_plate'],row['Well']), axis = 1)
metadata1.loc[:, 'cellname'] = metadata1.apply(lambda row: cell_name(row['meta_plate'], row['Well']), axis=1)


for i in list(metadata1['cellname'][:3]):
    generate_json_file_for_hpc('wbc', args.generated_config_dir, args.model_dir)
# Create parent directories of model_dir if they don't exist
os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
print('Script Completed')

wbc_cellnames=metadata1['cellname']
first_three = wbc_cellnames[:3]

output_file = os.path.join(args.output_dir, "wbc_cellnames.txt")
with open(output_file, "w") as f:
    f.write(first_three.to_string(index=False))

