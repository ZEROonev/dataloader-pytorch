'''
data
-train
--indoor
---1.1
-----1.1.mp3
-----1.101.png
-----1.101.jpg
-----1.102.jpg
-----1.103.jpg
-----1.103.png
--outdoor
---1.1
-----1.1.mp3
-----1.101.png
-----1.101.jpg
-----1.102.jpg
-----1.103.jpg
-----1.103.png
'''
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ImagesDataset(Dataset):
    def __init__(self, root, typel,mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.root=root
        
        self.indoor_path=os.path.join(self.root,self.typel,indoor)
        self.o=os.path.join(self.root,self.typel,outdoor)
        self.filenames=sorted(os.listdir(self.indoor_path))
        self.fil=sorted(os.listdir(self.o))
        self.indor_name=[]
        self.outdoor_name=[]
        for item,data in enumerate(self.filenames):
          f=sorted(os.listdir(os.path.join(self.indoor_path,data)))
          f=f.split('.')[0]
          if f!=data and f not in self.indor_name :
            self.indor_name+=f
        for item,data in enumerate(self.fil):
          f=sorted(os.listdir(os.path.join(self.indoor_path,data)))
          f=f.split('.')[0]
          if f!=data and f not in self.indor_name :
            self.outdoor_name+=f
        self.to=[]
        self.to=self.indor_name+self.outdoor_name
        self.num=len(self.to)
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        name=self.to[idx]
        b=name.split('_')
        fold_name=''.join(b[0],b[1],b[2])
        ids=name.split('_')[-1]
        image_path=os.path.join(self.root,self.typel,fold_name,ids,name+'.jpg')
        label_path=os.path.join(self.root,self.typel,fold_name,ids,name+'.png')
        image=Image.open(image_path).convert('RGB')
        label=Image.open(label_path).convert('L')
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
