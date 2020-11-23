import os
import random
import argparse

import shutil

random.seed(0)

def prepare_dataset(root_dir):

    back_dir = os.path.join(root_dir,'images_background')
    eval_dir = os.path.join(root_dir, 'images_evaluation')
    
    # Moving 10 alphabets to background as per original paper
    for i in os.listdir(eval_dir)[:10]:
        shutil.move(os.path.join(eval_dir, i), os.path.join(back_dir, i))

    back_alpha =[os.path.join(back_dir, x) for x in os.listdir(back_dir)]

    train_alpha = random.sample(back_alpha, 30) # Randomly picking 30 alphabets for training 
    print('number of training alphabets {}'.format(len(train_alpha)))

    val_alpha = [ x for x in back_alpha if x not in train_alpha ]
    print('number of val alphabets {}'.format(len(val_alpha)))

    test_alpha = [os.path.join(eval_dir, x) for x in os.listdir(eval_dir)]
    print('number of testing alphabets {}'.format(len(test_alpha)))

    train_root = os.path.join(root_dir,'train')
    os.makedirs(train_root, exist_ok=True)
    
    #extracting characterwise folders
    for i in train_alpha:
        train_alpha_dir = os.path.join(train_root, i.split('/')[-1] + '_')
        for j in os.listdir(i):
            char_dir = train_alpha_dir + j
            os.makedirs(char_dir, exist_ok=True)
            for k in os.listdir(os.path.join(i,j)):
                drawer_img = os.path.join(i,j,k)
                shutil.copyfile(drawer_img, os.path.join(char_dir, k))

    val_root = os.path.join(root_dir, 'val')
    os.makedirs(val_root, exist_ok=True)

    for i in val_alpha:
        val_alpha_dir = os.path.join(val_root, i.split('/')[-1] + '_')
        for j in os.listdir(i):
            char_dir = val_alpha_dir + j
            os.makedirs(char_dir, exist_ok=True)
            for k in os.listdir(os.path.join(i,j)):
                drawer_img = os.path.join(i,j,k)
                shutil.copyfile(drawer_img, os.path.join(char_dir, k))
    
    test_root = os.path.join(root_dir, 'test')
    os.makedirs(test_root, exist_ok=True)

    for i in test_alpha:
        test_alpha_dir = os.path.join(test_root, i.split('/')[-1] + '_')
        for j in os.listdir(i):
            char_dir = test_alpha_dir + j
            os.makedirs(char_dir, exist_ok=True)
            for k in os.listdir(os.path.join(i,j)):
                drawer_img = os.path.join(i,j,k)
                shutil.copyfile(drawer_img, os.path.join(char_dir, k))

if __name__== '__main__':

    my_parser = argparse.ArgumentParser(description='Prepare and split Omniglot dataset into train-val-test')

    
    my_parser.add_argument('Path',
                        metavar='path',
                        type=str,
                        help='the root directory containing omniglot-background and omniglot-eval')

    
    args = my_parser.parse_args()

    input_path = args.Path

    prepare_dataset(input_path)