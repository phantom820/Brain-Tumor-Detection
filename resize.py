import matplotlib.pyplot as plt
import skimage.exposure as exposure
from skimage.filters import unsharp_mask
from skimage import color
from skimage.transform import resize
import glob

def resize_images(path):
    no_files = glob.glob('raw_data/'+path+'/no/*.jpg')
    yes_files = glob.glob('raw_data/'+path+'/yes/*.jpg')
    
    # saving no files 0
    for i in range(len(no_files)):
        image=plt.imread(no_files[i])
        image=color.rgb2grey(image)
        image = unsharp_mask(image,amount=1,radius=1)
        image = exposure.equalize_hist(image,nbins=256)
        image_resized = resize(image,(128,128),
                               anti_aliasing=True)
        
        plt.imsave("data/"+path+"/no/"+str(i)+".jpg",image_resized,cmap="gray")
    
    # saving yes files
    for i in range(len(yes_files)):
        image=plt.imread(yes_files[i])
        image=color.rgb2grey(image)
        image = unsharp_mask(image,amount=1,radius=1)
        image = exposure.equalize_hist(image,nbins=256)
        image_resized = resize(image,(128,128),
                               anti_aliasing=True)
        
        plt.imsave("data/"+path+"/yes/"+str(i)+".jpg",image_resized,cmap="gray")

print('Preprocessing training set')
resize_images('training')
print('Preprocessing testing set')
resize_images('testing')
