# INBreast Breast Annotation Mask Generator
import csv
import numpy as np
from tqdm import tqdm
from MAMG import MAMG
from matplotlib import pyplot as plt

paths = {
    'xml': '../../../../datasets/INBreast/XML/',
    'dcm': '../../../../datasets/INBreast/DICOM/',
    'csv': '../../../../datasets/INBreast/INbreast_compact.csv'
}



def build_dcm_dict(path_to_csv):
  dcm_list = []

  with open(path_to_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
      if line_count == 0:
        line_count += 1
        continue
      img_name = row[5]
      dcm_list.append(img_name)
      line_count += 1
    print(f'Processed {line_count} lines.')

  return dcm_list


def generate_masks(dcm_list):
  
  dataset = np.zeros((len(dcm_list)*4, 512, 512, 2))
  
  print(dataset.shape)
  idx = 0
  for i in tqdm(range(len(dcm_list))):
    dcm_name = dcm_list[i]
    
    try:
      m = MAMG(paths, dcm_name)
      m.run()
      for w in range(m.pas.shape[0]):
        dataset[idx, :, :, 0] = m.pas[w, :, :]
        dataset[idx, :, :, 1] = m.clustered_img
        idx += 1
    except:
      continue
    
  print(idx)
  return dataset[:idx, :, :, :]


def main():
  dcm_list = build_dcm_dict(paths['csv'])
  dataset = generate_masks(dcm_list)
  np.save("../../generated/dataset", dataset)


def check_dataset():
  
  
  w = 0
  dataset = np.load("../../generated/dataset.npy")
  print("Dataset shape: ", dataset.shape)
  ans = input("Next? (y/n)")
  while w < dataset.shape[0]:
    # w = 36
    img = dataset[w, :, :, 0]
    # img[1,1] = 4095
    # print(np.min(img), np.max(img))
    print("Index: ", w)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.savefig("xaxa.png")
    plt.close()
    plt.figure()
    plt.imshow(dataset[w, :, :, 1], cmap='gray')
    plt.savefig("xaxa_mask.png")
    plt.close()
    w += 1
    ans = input("Next? (y/n)")
  
  
if __name__ == "__main__":
  # main()
  check_dataset()