
# Data preparation methods.

from head import *
    


S = None # The size of the character images.
A = None # The inverse of the destination coordinates.
# Sets S and A.
def setA():
    global S
    global A
    A = np.linalg.inv(np.array([[0, 0, 1],
                                [S, 0, 1],
                                [S, S, 1]]).T)



# Sets character image size and resets A accordingly.
def setImgSz(sz):
    global S
    global A
    S = sz
    setA()
    


inter = cv2.INTER_CUBIC # Default interpolation method.
# Sets interpolation method.
def setInter(INTER):
    global inter
    inter = INTER

    
    
# This method extracts a given character from the image
# by transforming it to the SxS upper-left corner of the image.
def extChr(img, BB):
    global inter
    B = np.array([[*BB[:,0], 1],
                  [*BB[:,1] ,1],
                  [*BB[:,2], 1]]).T # The source coordinates.
    M = np.matmul(B, A) # Compute the matrix that transforms the character to the desired location.
    charImg = cv2.warpAffine(
        img, M[:2], (S,S), flags=cv2.WARP_INVERSE_MAP + inter)
    return charImg




fil = None
# Sets the morphology filter.
def setFilSz(sz):
    global fil
    fil = cv2.getStructuringElement(cv2.MORPH_RECT, 2*[sz])


    
# Cleans the received character image.
def clnChr(img):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, fil)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, fil)
    return img



# Computes HOG features using b bins per cell (c^2 cells in total).
def hog(img, b=8, c=16, k=7): # k is the Sobel filter size.
    # Compute the derivatives.
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
    mags, angs = cv2.cartToPolar(gx, gy) # Compute the magnitudes and angles.
    # Get the maximum magnitudes and their angles.
    angs = np.take_along_axis(angs, mags.argmax(-1, keepdims=True), -1).squeeze()
    mags = mags.max(-1)
    bins = (b * angs / (2 * np.pi)).astype(int)
    n = S//c # Cell size.
    # Fast binning using one-hot encoding:
    I = np.identity(b)
    feat = np.multiply(*[np.stack([t[i::n,j::n]
                                   for i in range(n)
                                   for j in range(n)]).reshape(n**2,c**2,-1)
                         for t in [mags, I[bins]]]).sum(0).flatten()
    return feat, mags, angs



db = None
names = None
# Sets data globals for easy access.
def setDB(d, n):
    global db
    global names
    db = d
    names = n



# This method receives an image name,
# and returns the corresponding data point (image, font, etc.).
def getDatPt(name):
    img = db['data'][name][:].astype('uint8')
    try:
        fnt = db['data'][name].attrs['font']
    except:
        fnt = None
    txt = db['data'][name].attrs['txt'] 
    cBB = db['data'][name].attrs['charBB'] 
    wBB = db['data'][name].attrs['wordBB']
    return img, fnt, txt, cBB, wBB



# The data preparation method.
# fExt is a character extraction function, 
# fCln is a data cleaning function (e.g., morphologies), and 
# fFeat is a feature extraction function (e.g., HOG).
def prepData(fExt=extChr, fCln=clnChr, fFeat=lambda x: hog(x)[0]):
    X_data = []
    y_data = []
    # For each image:
    for i, name in enumerate(names):  
        img, fnt, txt, cBB, wBB = getDatPt(name)
        X = []
        y = []
        chrIdx = 0
        # For each word:
        for j in range(wBB.shape[-1]):
            # For each character:
            for k in range(len(txt[j])):
                # Extract, clean, and transform the character. 
                X.append(fFeat(fCln(fExt(img, cBB[...,chrIdx+k]))))
                y.append([i, j, txt[j][k]]) # Create metadata.
                try: # Try to add the font label.
                    y[-1].append(labDict[fnt[chrIdx+k]])
                except: # The test set has no fonts.
                    pass 
            chrIdx += len(txt[j]) # Update the character index.
        X_data.append(np.stack(X))
        y_data.append(np.stack(y))
    return X_data, y_data
