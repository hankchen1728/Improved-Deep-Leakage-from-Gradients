import pandas as pd

csv_path = '/data2/sinai/CheXpert/CheXpert-v1.0-small/train.csv'
ent_file = pd.read_csv(csv_path).fillna(0)
ent_file = ent_file[ent_file['Frontal/Lateral'] == 'Frontal']
with_device = ent_file[ent_file['Support Devices'] == 1].sample(n=915)
no_device = ent_file[ent_file['Support Devices'] == -1]
no_device *= 0
ent_file = pd.concat([with_device, no_device])
ent_file = ent_file[['Path', 'Support Devices']]
ent_file['Path'] = '/data2/sinai/CheXpert/' + ent_file['Path']
ent_file.to_csv('idlg_data_entry.csv', index=False)
