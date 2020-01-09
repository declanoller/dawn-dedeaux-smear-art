import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFile

from collections import namedtuple

# pixels is a 3D numpy array, the actual chunk of the original image
# avg_color is the average color of the block
# block_corner is the corner of the block, relative to the coords of the original image
# block_center_rel is the center of the block, relative to (0, 0) being at the center of the image
# dist_from_img_center is the distance from the image center, where (0,0) is the image center
PixelBlock = namedtuple('PixelBlock', ['pixels', 'avg_color', 'block_corner', 'block_center_rel', 'dist_from_img_center'])


class SmearImage:

    def __init__(self, image_fname):

        self.image_fname = image_fname

        self.img = Image.open(image_fname)

        self.img_dims = self.img.size


        self.img_center = np.array([self.img_dims[0]/2, self.img_dims[1]/2])


        # self.img_smear will be the canvas that's being produced.
        self.img_smear = self.img

        # Number of pixel blocks across
        self.N_blocks_across = 10

        self.pixel_blocks = None


    def get_pixel_blocks(self):

        # Chop self.img up into pixel blocks. For each one, create a namedtuple
        # from above with all the relevant info. Block width should be an int.
        # Because it probably doesn't divide the image evenly, the blocks at the edge
        # won't be perfect squares.

        block_width = int(self.img_dims[0]/self.N_blocks_across)

        '''
        for i in range(self.N_blocks_across):
            for j in range(...)
                pb = self.img[image coords]
                # get other info about this pixel block
                self.pixel_blocks.append(PixelBlock(pb, avg_color, ...))

        '''

        pass


    def smear_all(self):

        if self.pixel_blocks is None:
            self.get_pixel_blocks()

        outside_sorted_blocks = sorted(self.pixel_blocks, key=lambda x: -x.dist_from_img_center)

        for pb in outside_sorted_blocks:

            self.smear_pixel_block(pb)

        # show/output image
        self.img_smear.save(self.image_fname.replace('.jpg', '_smeared.jpg'))
        self.img_smear.show()


    def smear_pixel_block(self, pb):

        # Smears a single pixel block. Uses its info, calls self.vector_field(),
        # and updates self.img_smear.

        # pos_final = self.vector_field(pb.block_center_rel)
        # trail_coords = self.get_trail_coords(pb.block_center_rel, pos_final)
        # new_pos_bds = self.pixel_block_bds(pos_final)
        # self.img_smear[new_pos_bds] = pb.pixels
        # self.img_smear[trail_coords] = pb.avg_color

        pass


    def get_trail_coords(self, pos_init, pos_final):

        # Given the pixel block initial and final positions, get the coordinates
        # of the pixels of its trail, i.e., what it sweeps out.
        pass


    def pixel_block_bds(self, pb_center):

        # Returns the bounds of the image indices given the pb_center and a block's
        # width.
        pass


    def dist_from_center(self, pos):

        # Get the distance from the center of the image. Pos is the absolute coords.

        return np.linalg.norm(pos - self.img_center)


    def avg_color(self, pixel_block):

        # Gets avg color from a pixel block
        pass


    def vector_field(self, pos):

        # Returns 2-tuple representing vector field at that position (x, y)

        x, y = pos

        # Will likely need to be tuned
        scale_const = 5

        return [scale_const*x, scale_const*y]





















#
