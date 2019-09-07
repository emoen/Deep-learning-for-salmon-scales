cd = '/gpfs/gpfs0/deep/data/salmon-scales/from_RB_2017/'
#cd = '/gpfs/gpfs0/deep/data/salmon-scales/from_RB/'
  
count = 0
rb_path_total = pd.DataFrame(columns=['ID nr.', 'Totalt'])
for i in range(0, len(df)):
    file_name = df['ID nr.'].values[i]
    total = df['Totalt'].values[i]
    #fylke = df['Fylke'].values[i]
    from_RB_root = cd  # + fylke
    found = False
    if not pd.isnull(total):
        for root, dirs, files in os.walk(from_RB_root):
            if found == True:
                break
            for name in files:
                if name == file_name+'.jpg':
                    #print(file_name)
                    #print(root + '/'+name + " - "+str(total))
                    #print(fylke)
                    path_to_jpg = root + '/'+name
                    dict = {'ID nr.': path_to_jpg, 'Totalt': total}
                    rb_path_total = rb_path_total.append(dict, ignore_index=True)
                    count += 1
                    found = True  