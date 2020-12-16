import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display


# Cria os encodings das faces
jon_snow_image = face_recognition.load_image_file("known_people/jon_snow.jpg")
jon_snow_face_encoding = face_recognition.face_encodings(jon_snow_image)[0]

hermione_image = face_recognition.load_image_file("known_people/hermione.jpg")
hermione_face_encoding = face_recognition.face_encodings(hermione_image)[0]

# Cria um array com os encodings e com os nomes
known_face_encodings = [
    jon_snow_face_encoding,
    hermione_face_encoding
]
known_face_names = [
    "Jon Snow",
    "Hermione Granger"
]
print('Aprendi a reconhecer ', len(known_face_encodings), ' faces.')


# Carrega uma imagem com face desconhecida 
snow_unknown = face_recognition.load_image_file("unknown_people/snow.jpg")

# Encontra todas as faces na foto
face_locations = face_recognition.face_locations(snow_unknown)
face_encodings = face_recognition.face_encodings(snow_unknown, face_locations)

# Converte a imagem para o formato PIL para poder desenhar sobre ela
pil_image = Image.fromarray(snow_unknown)
# Cria um ImageDraw instance para poder desenhar
draw = ImageDraw.Draw(pil_image)

# Percorre cada face encontrada na imagem
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compara faces com conhecidas
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Encontra a face mais parecida
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Desenha um retângulo aonde a face foi encontrada
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Desenha uma label com o nome abaixo do rosto
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove o desenho da memória
del draw

# Mostra a imagem
# display(pil_image)
pil_image.show()