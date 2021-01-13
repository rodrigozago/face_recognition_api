import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import glob
import re

regex = r"(./.+)/(.+)\.(.+)"

# Read all the images from the tests dataset
images_directories = glob.glob("./yalefaces/subject*")

# Create dict of tests and known people
images = dict()

# Dict with known people encodings
known_encodings = dict()

# list with known people encodings
known_face_encodings_array = []
known_face_names_array = []

total_of_faces_cont = 0
recognized_faces_cont = 0


for image in images_directories:
    #get the subject from the filename
    matches = re.match(regex, image)
    key = matches.group(2)
    if key in images:
        images[key].append(image)
    else:
        images[key] = []
        images[key].append(image)

# encoding one face for every people
for key, value in images.items():
    aux = dict()
    images_array = value
    link = images_array[0]
    del images_array[0]
    print("Aprendi a reconhecer: {}".format(key))
    aux["image"] = face_recognition.load_image_file(link)
    aux["encoding"] = face_recognition.face_encodings(aux["image"])[0]
    known_encodings[key] = aux

# create array of face encodings and names
for key, value in known_encodings.items():
    known_face_encodings_array.append(value["encoding"])
    known_face_names_array.append(key)

print(known_face_encodings_array)
print(known_face_names_array)

# try to recognize faces
for key, value in images.items():
    print("\n--------------")
    print("Tentando reconhecer {} fotos de: {}".format(len(value), key))
    total_images = len(value)
    total_of_faces_cont += total_images

    recognized = 0
    for img_path in value:
        print("Reconhecendo: {}".format(img_path))
        unknown = face_recognition.load_image_file(img_path)

        face_locations = face_recognition.face_locations(unknown)
        face_encodings = face_recognition.face_encodings(unknown, face_locations)
        # Percorre cada face encontrada na imagem
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compara faces com conhecidas
            matches = face_recognition.compare_faces(known_face_encodings_array, face_encoding)
            name = "Unknown"

            # Encontra a face mais parecida
            face_distances = face_recognition.face_distance(known_face_encodings_array, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names_array[best_match_index]
                print("Matches with: {}".format(name))
                recognized+=1
                recognized_faces_cont+=1
    print("Reconhecidas: {}/{} imagens.".format(recognized, total_images))


print('\n-------------')
print('Resultados: ')
print("Total de rostos encontrados: {} \nTotal de rostos reconhecidos: {} \n".format(total_of_faces_cont, recognized_faces_cont))
print("Percentual de acerto: {}%".format((recognized_faces_cont*100)/total_of_faces_cont))