import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFile




img = Image.open('euler.jpg')

#img = img.resize()
#print(img.size)

sq_size = 100

sq_pos = np.array([250, 100])

move_pos = 200 + sq_pos

img_np = np.asarray(img)

#print(img_np.shape)

sq_copy = img_np[sq_pos[0]:sq_pos[0]+sq_size, sq_pos[1]:sq_pos[1]+sq_size]

sq_col = sq_copy.mean(axis=(0,1))

print(sq_copy.shape)

img_np.setflags(write=1)
print(img_np.flags)

img_np[move_pos[0]:move_pos[0]+sq_size, move_pos[1]:move_pos[1]+sq_size, :] = sq_copy


img = Image.fromarray(img_np)

img.show()
exit()

















#
