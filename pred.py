
from head import *
from prep import *
from test import *


# Loads the model, predicts the labels, and generates the requested .csv file.
def main():
    
    while True:
        
        h5_name = input('Please enter .h5 file name (e.g., "SynthText_test.h5")\n')
        try:
            model = pickle.load(zp.ZipFile('model.zip', 'r').open('model', 'r'))
        except:
            print('Failed to read model.zip.')
            break

        try:
            db = h5py.File(h5_name, 'r')
        except:
            print('Failed to read '+h5_name+'.')
            break
        names = list(db['data'].keys())
        setDB(db, names) # Set db globals.
        setInter(cv2.INTER_CUBIC) # Set interpolation method.
        setImgSz(64) # Set character image size.
        setFilSz(5) # Set morphology filter size.

        print(f'Processing images...')
        start = time.time()
        X, y = prepData()
        X = np.concatenate(X) 
        y = np.concatenate(y)
        prepTime = time.time() - start
        print(f'Preparation time: {round(prepTime / len(y) * 1000, 3)} milliseconds per character.')
        
        print(f'Predicting labels...')
        start = time.time()
        pred = majVot(distScrs(model, X), y)
        inferTime = time.time() - start
        print(f'Inference time: {round(inferTime / len(y) * 1000, 3)} milliseconds per character.')

        del X

        header = [' ', 'image', 'char'] + [l.decode('utf-8') for l in labels]

        I = np.identity(len(labels), int) # One-hot encoding is requested.
        csvDat = [[i, names[t[0]], chr(t[2]), *I[p]] 
                  for i, (t, p) in enumerate(zip(y, pred))]

        csvName = 'labels_test.csv'

        # Write the CSV file.
        with open(csvName, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(csvDat)

        print('Verifying...')
        try:
            if filecmp.cmp(csvName, 'labels_verify.csv'):
                print('Done!')
            else:
                print('Something went wrong: '+csvName+' does not match labels_verify.csv!')
        except:
            print('labels_verify.csv is missing.')
        break
    
    input('Press \'Enter\' to exit.')
        

if __name__ == '__main__':
    main()
