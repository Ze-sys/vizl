'''
A script for downloading team crest images from visl.org
OUTPUT: 1. team_crests/teamcrest_<team_id>.jpg into a folder called visl_team_crests under the main directory
        2. team_names.csv into a folder named csv under the main directory
        3. Creates thumbnails of the team crest images and saves them in the same folder as the team crest images
        4. Adds alpha channel to the thumbnails
        5. Creates icons from the team crest images
'''

import re
import os
import cv2
import bs4 as bs
import requests
import subprocess
import pandas as pd
from PIL import Image


def download_team_crests(d):
    '''
    Downloads team crest images from visl.org
    INPUT: dictionary of team names and ids
    '''

    for i in d.team_id.values:
        bash_command = f'wget -O visl_team_crests/teamcrest_{i}.jpg https://visl.org/upload/img/teamcrest_{i}.jpg'
                        
        process = subprocess.Popen(bash_command.split(), shell=False, stdout=subprocess.PIPE)

        output, error = process.communicate()

        str_output = str(output.decode("utf-8"))

        print('Any Error?: {}'.format(error)) 

        print('Status: {}'.format(str_output)) 

def resize_image(image_path = 'visl_team_crests/', size = [75,75], suffix='_thumb'):
    '''
    Resizes an image to a given size
    INPUT: image path, size of the image to be resized
    OUTPUT: resized image
    '''
    for file in os.listdir(image_path):
        if file.endswith(".jpg"):
            im = Image.open(image_path + file)
            filename, extension = file.split('.')
            # remove alpha channel ( not needed ) if found. Add it back later for all images
            save_in = image_path + filename + suffix + '.' + extension
            im.thumbnail(size, Image.ANTIALIAS)
            try:
                 im.save(save_in)
            except OSError:  #cannot write mode RGBA as JPEG
                im = im.convert("RGB")
                im.save(save_in)

            im.show()
            print('Saved image to: {}'.format(save_in))


def add_alpha_channel(image_path = 'visl_team_crests/'):
    '''
    Adds alpha channel to the images
    INPUT: image path
    OUTPUT: alpha channel added images
    '''
    for file in os.listdir(image_path):
        if file.endswith("_thumb.jpg"):
            img = cv2.imread(image_path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            cv2.imwrite(image_path + file, img)
            print('Added alpha channel to: {}'.format(file))

def image_to_icon(image_path = 'visl_team_crests/', icon_sizes = [(16,16), (32, 32), (48, 48), (64,64)]):
    '''
    Creates icons from the images
    '''
    for file in os.listdir(image_path):
        if file.endswith(".jpg"):
            im = Image.open(image_path + file)
            filename, extension = file.split('.')[0],'ico'
            # remove alpha channel ( not needed ) if found. Add it back later for all images
            save_in = image_path + filename + '.' + extension
            im.convert('RGB').save(save_in, icon_sizes=icon_sizes)
            # im.show()
            print('Saved image to: {}'.format(save_in))



import imageio
def  create_gif( path = 'visl_team_crests/', suffix='_thumb_alpha', extension='png'):
    '''
    Creates a gif from the images
    INPUT: image path
    OUTPUT: gif
    '''
    images = [img for img in os.listdir(path) if img.endswith(suffix + '.' + extension)]
    
    for i in range(len(images)):
        images[i] = imageio.imread(path + images[i])
    imageio.mimsave(path + 'visl_team_crests.gif',images, format='GIF', duration=0.75) 


# Use BeautifulSoup to extract team names and ids    

url = 'https://visl.org/'
r = requests.get(url)
soup = bs.BeautifulSoup(r.text, 'lxml')
team_names = soup.find_all('a', {"href": lambda x: x and x.startswith("/webapps/spappz_live/team_info?id=")})
team_crests = soup.find_all('img', {"src": lambda x: x and x.startswith("/upload/img/teamcrest_")})

def extract_team_names_and_ids(team_names):
    team_dict = {team.text:re.search(r'\d+', team.attrs['href']).group() for team in team_names}
    return team_dict

if team_names:
    team_dict = extract_team_names_and_ids(team_names)
    d= pd.DataFrame([team_dict], index=['team_id']).T.astype(int)
    d.index.name = 'team_name'
    # save teams and ids to csv for later use
    with  open('csv/visl_team_names_and_ids.csv', 'a') as f:
        d.to_csv(f, header=False)


if __name__ == '__main__':

    download_team_crests(d)
    resize_image(size=[75,75])
    add_alpha_channel()
    create_gif( path = 'visl_team_crests/', suffix='_thumb', extension='jpg')
    image_to_icon()