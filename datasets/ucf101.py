import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UCF101(Dataset):
  def __init__(self, root, train=True, clip_len=16, output_clip=True, transform=None, target_transform=None):
    super(UCF101, self).__init__()
    self.train = train
    self.clip_len = clip_len
    self.output_clip = output_clip

    if transform is not None:
      self.transform = transform
    else:
      self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    self.target_transform = target_transform

    videos, self.data = [], []
    self.targets = []

    classInd = {}
    with open(os.path.join(root, 'classInd.txt')) as f:
      for line in f:
        class_id, class_name = [item.strip() for item in line.split()]
        class_id = int(class_id) - 1
        classInd[class_name] = class_id

    if self.train:
      txt = os.path.join(root, 'trainlist01.txt')
    else:
      txt = os.path.join(root, 'testlist01.txt')

    with open(txt, 'r') as f:
      for line in f:
        line = line.split()[0].strip()
        video_class, video_name = [item.strip() for item in line.split('/')]

        videos.append(video_name)
        self.targets.append(classInd[video_class])

    videos_dir = os.path.join(root, 'videos')

    for video_name in videos:
      records = []
      video_dir = os.path.join(videos_dir, video_name)
      for frame_name in sorted(os.listdir(video_dir)):
        frame_path = os.path.join(video_dir, frame_name)
        records.append(frame_path)
      length = len(records)
      if length < clip_len:
        for i in range(clip_len-length):
          records.append(records[i])
      self.data.append(records)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    if self.output_clip:
      video, label = self.data[index], self.targets[index]
      length = len(video)
      start = np.random.randint(0, length-self.clip_len+1)
      clip = []
      for frame_path in video[start:start+self.clip_len]:
        img = Image.open(frame_path)
        img = self.transform(img)
        '''
        img = img.resize((64, 64), Image.ANTIALIAS)
        img = np.asarray(img)
        img = np.swapaxes(img, 2, 1)
        img = np.swapaxes(img, 1, 0)
        img = img.astype(np.float32)
        '''
        clip.append(img)
      clip = torch.stack(clip,1)
      if self.target_transform is not None:
        label = self.target_transform(label)
      return clip, label, index
    else:
      video, label = self.data[index], self.targets[index]
      length = len(video)
      data = []
      for start in range(0, length-self.clip_len+1, self.clip_len):
        clip = []
        for frame_path in video[start:start+self.clip_len]:
          img = Image.open(frame_path)
          img = self.transform(img)
          clip.append(img)
        clip = torch.stack(clip, 1)
        data.append(clip)
      data = torch.stack(data, 0)
      if self.target_transform is not None:
        label = self.target_transform(label)
      return data, label, index
