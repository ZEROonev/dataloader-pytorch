'''
cvpr
--train
---image
-----00001
-----00002
---seg
-----00001
-----00002
'''
class data(Dataset): 

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='../cvpr',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None,typel='train'):
       
        self.typel=typel
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train'
        else:
            fname = 'val'

        if self.seq_name is None:
                seqs=os.listdir(os.path.join(db_root_dir,typel,'image'))
                print(seqs)
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, self.typel,'image', seq.strip())))
                    images_path = list(map(lambda x: os.path.join(db_root_dir,typel,'image',seq.strip(), x), images))
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir,self.typel, 'seg', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join(db_root_dir,typel, 'seg', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, self.typel, 'image',  str(seq_name))))
            img_list = list(map(lambda x: os.path.join(db_root_dir, self.typel, 'image',str(seq_name), x), names_img))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir,self.typel, 'seg', str(seq_name))))
            labels = list(map(lambda x: os.path.join(db_root_dir, self.typel, 'seg',str(seq_name), x), name_label))
            
          

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        #print(self.img_list,self.labels)

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
        
   def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        path=os.path.join(self.img_list[idx])
        #print(path)
        img = cv2.imread(path)
        #print(img.shape)
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.labels[idx]))
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt
db_train = data(train=True, db_root_dir='../cvpr', transform=composed_transforms, seq_name='0001',typel='train')


# code modified from https://github.com/kmaninis/OSVOS-PyTorch.git
