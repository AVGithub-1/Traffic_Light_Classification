def fix_filename(filename):
    """
    change the filename/path to file of each image in the dataset

    (string) filename: the name and path of the image as a file
    """
    fil1, fil2 = filename.split('/')
    if 'daySequence1' in filename:
        return 'daySequence1/daySequence1/frames/' + fil2
    elif 'daySequence2' in filename:
        return 'daySequence2/daySequence2/frames/' + fil2
    elif 'dayTraining' in filename:
        clip, c = fil2.split("-", maxsplit=1)
        return 'dayTrain/dayTrain/' + clip + "/frames/" + fil2
    elif 'nightSequence1' in filename:
        return 'nightSequence1/nightSequence1/frames/' + fil2
    elif 'nightSequence2' in filename:
        return 'nightSequence2/nightSequence2/frames/' + fil2
    elif 'nightTraining' in filename:
        clip, c = fil2.split("-", maxsplit=1)
        return 'nightTrain/nightTrain/' + clip + "/frames/" + fil2

#import the box annotation data
df_box = pd.DataFrame()
for dirname, _, filenames in os.walk('C:/Users/akhpv/OneDrive/Documents/Machine_Learning_Practice/LISA_Dataset'):
    for filename in filenames:
        if "BOX.csv" in filename:
            add_to_df = pd.read_csv(os.path.join(dirname,filename), sep=';')
            add_to_df = add_to_df.drop(add_to_df.columns[-4:], axis=1)
            for i, row in add_to_df.iterrows():
                filename = fix_filename(add_to_df.loc[i,'Filename'])
                add_to_df.loc[i,'Filename'] = filename

                
            df_box = pd.concat([df_box, add_to_df], ignore_index=True)



df_box.to_csv('df_img.csv')

#if you're using 
df_box = pd.read_csv('df_img.csv')
df_box.columns

df_box = df_box.drop('Unnamed: 0', axis=1)
