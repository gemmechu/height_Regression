import numpy as np

label_path = '../DOH/output/openpose/gt_2.txt'
content = []
with open(label_path, 'r') as f:
    content = f.readlines()
    content = [x.strip('\n') for x in content]
    
def get_height(centers, height_file_path):
    height = np.load(height_file_path)
    result = []
    for i in ([10,13]):
        try:
            x,y = centers[i]
            temp = height[y][x]
            result.append(temp)
        except:
            print('no key')
    return result
def write_to_file(name, height):
    with open ('img_height.txt', 'a') as f:
        f.write(name+ ',' + str(height))
        f.write('\n')
for line in content:
    height_file_path = '../Ruler_Person/height/'+'_'.join(line.split(',')[0].split('/')[:-1]) +'.npy'
    centers = eval(','.join(line.split(',')[1:]))
    
    height = get_height(centers, height_file_path)
    if(height == []):
        continue
    height = np.nanmean(height)
    #write to file
    name = line.split(',')[0]
    write_to_file(name, height)
    