import os

def make_list(path):
    f = open(os.path.join(path, 'images.txt'), 'w')
    ff = open(os.path.join(path, 'image_class_labels.txt'), 'w')
    i = 0
    for dirlist in os.listdir(path):
        if dirlist != '.DS_Store' and dirlist != 'images_AD_CN.txt' and dirlist != 'image_class_labels_AD_CN.txt' \
                and dirlist != 'images.txt' and dirlist != 'image_class_labels.txt' and dirlist != 'images_MCI_CN.txt' \
                and dirlist != 'image_class_labels_MCI_CN.txt' and dirlist != 'images_AD_MCI.txt' and dirlist != 'image_class_labels_AD_MCI.txt':
            for imagelist in os.listdir(os.path.join(path, dirlist)):
                if dirlist == 'AD':
                    name = str(i) + ' ' + os.path.join(dirlist, imagelist) + '\n'
                    name2 = str(i) + ' 0\n'
                    f.write(name)
                    ff.write(name2)
                    i += 1
                elif dirlist == 'MCI':
                    name = str(i) + ' ' + os.path.join(dirlist, imagelist) + '\n'
                    name2 = str(i) + ' 1\n'
                    f.write(name)
                    ff.write(name2)
                    i += 1
                else:
                    name = str(i) + ' ' + os.path.join(dirlist, imagelist) + '\n'
                    name2 = str(i) + ' 2\n'
                    f.write(name)
                    ff.write(name2)
                    i += 1
                # else:
                #     name = str(i) + ' ' + os.path.join(dirlist, imagelist) + '\n'
                #     name2 = str(i) + ' 2\n'


    f.close()



if __name__ == '__main__':
    #make_list('/home/trojan/Desktop/dimentia/dataset_large/data_3categ/train')
    make_list('/home/trojan/Desktop/dimentia/dataset_large/data_3categ/validation')