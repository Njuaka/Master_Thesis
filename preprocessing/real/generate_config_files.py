import argparse
import json 
import pandas as pd

parser = argparse.ArgumentParser(description='This script will generate a list of all the filtered wbc and pbmc cells and write it to the output directory. Consecutively it will create config json files for all the cells and write them to the generated config directory') 

parser.add_argument("--meta_file", help="File containing the names of the cells, the filters and the percentages")
parser.add_argument("--percent_minimum", help="The minimum percentage of a certain class to be present")
parser.add_argument("--generated_config_dir", help="Path where to store the generated train and predicition config json files")
parser.add_argument("--model_dir", help="Path where the best model is stored (needed for predicition config file)")
parser.add_argument("--output_dir", help="Output directory for storing list of wbc and pbms cells after filtering")  
args = parser.parse_args()

def generate_combined_json_file_for_hpc(type, generated_config_dir, model_dir):
    with open(generated_config_dir + 'train-2-deepflow_example.json', 'r') as json_file:
        train_json = json.load(json_file)
    with open(generated_config_dir + 'pred-2-deepflow_example.json', 'r') as json_file:
        pred_json = json.load(json_file)
        
    train_json['h5_data'] = 'combined.h5'
    train_json['meta'] = 'combined.csv'
    train_json['name_experiment'] = 'combined_' + type
    train_json['split_dir'] = 'Split/combined/'
        
    pred_json['h5_data'] = 'combined.h5'
    pred_json['model_hdf5'] = model_dir + type + '/combined/best-model.hdf5'
    pred_json['split_dir'] = 'Split/combined/'
        
    path = generated_config_dir + type + '/'
    train_path = path + 'train-2-deepflow_hpc_combined.json'
    pred_path = path + 'pred-2-deepflow_hpc_combined.json'
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=4)
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=4)


def cell_name(plate,well):
    if len(well)==2:
      well=well[0]+'0'+well[1]
    name= 'Plate'+str(plate)+'_'+well
    return name

def select_cells(meta_file,percent_minimum,output_dir):
    cells = pd.read_csv(meta_file,sep=';')
    cells.rename(columns = {'Metadata_Plate':'Plate'}, inplace = True)
    cells['cellname']= cells.apply(lambda row : cell_name(row['Plate'],row['Well']), axis = 1)
    # First selection based on flags
    wbc_cells = cells[cells['WBC']==1]
    pbmc_cells = cells[cells['PBMC']==1]
    wbc_cells.PosWBCIsx = wbc_cells.PosWBCIsx.str.replace(',','.')
    wbc_cells.PosWBCIsx = wbc_cells.PosWBCIsx.astype(float)
    pbmc_cells.PosPBMCIsx = pbmc_cells.PosPBMCIsx.str.replace(',','.')
    pbmc_cells.PosPBMCIsx = pbmc_cells.PosPBMCIsx.astype(float)
    
    # Selectiom based on minimum percentage
    percent_minimum= int(percent_minimum)
    wbc_cells = wbc_cells[wbc_cells['PosWBCIsx']>percent_minimum]
    pbmc_cells = pbmc_cells[pbmc_cells['PosPBMCIsx']>percent_minimum]
    wbc_cells = wbc_cells[wbc_cells['PosWBCIsx']<(100-percent_minimum)]
    pbmc_cells = pbmc_cells[pbmc_cells['PosPBMCIsx']<(100-percent_minimum)]
    with open(output_dir+'wbc_cellnames.txt', 'w') as f:
        file_content = "\n".join(list(wbc_cells['cellname']))
        f.write(file_content)
    with open(output_dir+'pbmc_cellnames.txt', 'w') as f:
        file_content = "\n".join(list(pbmc_cells['cellname']))
        f.write(file_content)
    return wbc_cells,pbmc_cells

wbc_cells, pbmc_cells = select_cells(args.meta_file, args.percent_minimum, args.output_dir)

generate_combined_json_file_for_hpc('wbc', args.generated_config_dir, args.model_dir)
generate_combined_json_file_for_hpc('pbmc', args.generated_config_dir, args.model_dir)

print('Script Completed')
