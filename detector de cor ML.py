import os
from collections import Counter
from typing import Optional, Any

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def rgb2hex(rgb):
    hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    # hex = f"#{rgb[0]}{}{}"
    return hex


rgb = [0, 1, 2]
hex = ["#"]
[hex.append(f"{c:}") for c in rgb]
''.join(hex)

PATH = "./SOLO1.jpg"
WIDTH = 128
HEIGHT = 128
CLUSTERS = 6

image: Optional[Any] = Image.open(PATH)

var = image.size

print("Loaded {f} image. Size: {s:.2f} KB. Dimensions: ({d})".format(
    f=image.format, s=os.path.getsize(PATH) / 1024, d=image.size))

image


# DADOS E DIMENSÕES DA IMAGEM FOI APRESENTADA NA TELA
# EM SEGUIDA A FUNÇÃO VAI CALCULAR OS DADOS - MINERAR

def calculate_new_size(image):
    if image.width >= image.height:
        wpercent = (WIDTH / float(image.width))
        hsize = int((float(image.height) * float(wpercent)))
        new_width, new_height = WIDTH, hsize
    else:
        hpercent = (HEIGHT / float(image.height))
        wsize = int((float(image.width) * float(hpercent)))
        new_width, new_height = wsize, HEIGHT

    image.resize((new_width, new_height), Image.ANTIALIAS)
    return image, new_width, new_height


# AGORA A APRENDIZAGEM DE MAQUINA (MACHINE LEARNING) A FUNÇÃO DEVE AGRUPAR SIMILARES

new_image, new_width, new_height = calculate_new_size(image)
print(f"New dimensions: {new_width}x{new_height}")
img_array = np.array(new_image)
img_vector = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
new_image

model = KMeans(n_clusters=CLUSTERS)
labels = model.fit_predict(img_vector)
label_counts = Counter(labels)
model.cluster_centers_

total_count = sum(label_counts.values())
total_count

hex_colors = [rgb2hex(center) for center in model.cluster_centers_]
hex_colors

hex_colors

# AGRUPOU NAS CORES ACIMA EM FORMATO HEX - '#a89679', '#968468', '#c7c4c7', '#645c53', '#827158', '#afabab' 
# PRECISO QUE DÊ EM MUNSELL TIPO - colour.xyY_to_munsell_colour([0.38736945, 0.35751656, 0.59362000]) DANDO ISSO '4.2YR 8.1/5.3'

list(zip(hex_colors, list(label_counts.values())))

# FEZ UMA LISTA ASSOCIANDO A QUANTIDADE DE PIXEL, ACREDITO QUE SEJA ISSO. / DE SIMILARES PREDOMINANTES

plt.figure(figsize=(14, 8))

plt.subplot(221)

plt.imshow(image)

plt.axis('off')

plt.subplot(222)

plt.pie(label_counts.values(), labels=hex_colors, colors=[color / 255 for color in model.cluster_centers_],
        autopct='%1.1f%%',
        shadow=True, startangle=90)

plt.axis('equal')

plt.title('CORES DO SOLO')

plt.show()