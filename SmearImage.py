import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import time
from PIL import Image, ImageDraw, ImageFile
from collections import namedtuple

'''

When you turn an image into an array using PIL, it's a matrix where the first
coord is y, and the second coord is x. So you have to address pixels like
img[y_coord, x_coord]. I'll probably fix this at some point to reverse it.


'''

# pixels is a 3D numpy array, the actual chunk of the original image
# avg_color is the average color of the block
# block_corner is the corner of the block, relative to (0,0) of the original image
# block_center_rel is the center of the block, relative to (0, 0) being at the center of the image
# x, y are coords of the corner of the block. w, h are the width and height of the block.
PixelBlock = namedtuple('PixelBlock', ['pixels', 'avg_color', 'block_corner', 'x', 'y', 'w', 'h', 'block_center_rel'])


class SmearImage:

    def __init__(self, image_fname, **kwargs):

        self.image_fname = image_fname

        self.img = np.array(Image.open(image_fname))
        #self.img = self.img.swapaxes(0, 1)

        self.img_dims = self.img.shape
        print('Img dims: ', self.img_dims)

        self.img_center = 0.5*np.array(self.img_dims[:2])

        self.img_r_max = self.img_center[0]

        # self.img_smear will be the canvas that's being produced.
        # You have to use copy(), otherwise it just copies by ref, and changes
        # to the copy apply to the original.
        self.img_smear = copy(self.img)

        # Number of pixel blocks across
        self.N_blocks_x = kwargs.get('N_blocks', 30)
        self.block_width = np.ceil(self.img_dims[0]/self.N_blocks_x).astype(int)
        self.N_blocks_y = np.ceil(self.img_dims[1]/self.block_width).astype(int)

        print('N_blocks_x = {}, N_blocks_y = {}, block_width = {}, img_center = {}'.format(self.N_blocks_x, self.N_blocks_y, self.block_width, self.img_center))

        self.pixel_blocks = None


    def get_pixel_blocks(self):

        # Chop self.img up into pixel blocks. For each one, create a namedtuple
        # from above with all the relevant info. Block width should be an int.
        # Because it probably doesn't divide the image evenly, the blocks at the edge
        # won't be perfect squares.

        self.pixel_blocks = []
        for i in range(self.N_blocks_x):
            for j in range(self.N_blocks_y):
                x = i*self.block_width
                y = j*self.block_width
                # To account for blocks that aren't the full width at the edges
                w = min(self.img_dims[0], x + self.block_width) - x
                h = min(self.img_dims[1], y + self.block_width) - y
                pixels = self.img[x:x+w, y:y+h]
                avg_color = self.avg_color(pixels)

                block_center = np.array([x + w/2, y + h/2])
                block_center_rel = block_center - self.img_center

                self.pixel_blocks.append(PixelBlock(pixels, avg_color, np.array([x, y]), x, y, w, h, block_center_rel))

        print('\nTotal pixel blocks: {}\n'.format(len(self.pixel_blocks)))


    def center_rel_coord_to_corner_coord(self, pb, center_rel_coord):
        center_coord = self.img_center + center_rel_coord
        corner_coord = center_coord - np.array([pb.w/2, pb.h/2])
        return corner_coord


    def print_all_pixel_blocks(self):

        # Just a diagnostic tool
        assert self.pixel_blocks is not None, 'self.pixel_blocks must be defined to print!'
        print('\nTotal pixel blocks: {}\n'.format(len(self.pixel_blocks)))

        for pb in self.pixel_blocks:
            print('avg color = ({:.2f}, {:.2f}, {:.2f})\t corner = {}\t x = {:.1f}\t y = {:.1f}\t w = {:.1f}\t h = {:.1f}\t block_center = {}'.format(*pb.avg_color,
                                                                                                                                pb.block_corner,
                                                                                                                                pb.x,
                                                                                                                                pb.y,
                                                                                                                                pb.w,
                                                                                                                                pb.h,
                                                                                                                                pb.block_center_rel,))


    def smear_all(self):

        '''
        - Form the pixel blocks if they don't exist yet
        - Sort them by distance from center, so we transform farther out blocks first
        - Smear each block
        - Save image, create image with points and arrows overlaid
        '''

        start_time = time.time()

        # Get blocks
        if self.pixel_blocks is None:
            self.get_pixel_blocks()

        # Sort blocks
        outside_sorted_blocks = sorted(self.pixel_blocks, key=lambda x: -np.linalg.norm(x.block_center_rel))

        # Loop over all blocks
        for i,pb in enumerate(outside_sorted_blocks):
            if i % max(1, int(len(self.pixel_blocks))/10)==0:
                print(f'Smearing pixel block {i}')
            #print(pb.x, pb.y)
            #if pb.x == 200 and pb.y == 320:
            if True:
                self.smear_pixel_block(pb)


        print('\nTook {:.2f} seconds'.format(time.time() - start_time))

        # show/output image
        img = Image.fromarray(self.img_smear)
        img.show()
        img.save(self.image_fname.replace('.jpg', '_smeared.jpg'))
        # Draw original and new positions with line connecting, save also
        #self.draw_arrows_on_img(img)
        #img.save(self.image_fname.replace('.jpg', '_vecfield_smeared.jpg'))
        pass


    def smear_pixel_block(self, pb):

        '''
        Smears a single pixel block. Uses its info, calls self.vector_field(),
        and updates self.img_smear.

        Works by taking the total translation vector, move_vec, and chopping it
        up into steps, move_step. Then it translates the block one step at a time,
        setting all the pixels at the step's new location to the average color.

        It has to do be a bit careful with ones that are translated past the edges.

        Then it copies the whole original block to the new location. Again it has
        to be careful around the edges.


        '''

        '''move_vec = np.trunc(self.vector_field(pb.block_center_rel)).astype(int)

        N_steps = int(max(1, np.max(np.abs(move_vec))))
        move_step = move_vec/N_steps
        all_move_steps = [np.trunc(i*move_step) for i in range(N_steps)]

        all_moved_block_corners = [(pb.block_corner + ms).astype(int) for ms in all_move_steps]
        '''

        n_steps = int(0.8*np.linalg.norm(pb.block_center_rel))
        #print(n_steps)
        #n_steps = 150
        all_move_steps_rel = self.get_move_steps_path(pb.block_center_rel, n_steps)
        all_moved_block_corners = [self.center_rel_coord_to_corner_coord(pb, ms).astype(int) for ms in all_move_steps_rel]
        move_diff = all_move_steps_rel[-1] - all_move_steps_rel[0]
        move_vec = (np.trunc(move_diff) + np.sign(move_diff)).astype(int)

        if move_diff[0] > 0:
            if move_diff[1] > 0:
                move_dir = 'RD'
            else:
                move_dir = 'RU'
        else:
            if move_diff[1] > 0:
                move_dir = 'LD'
            else:
                move_dir = 'LU'

        col = pb.avg_color
        norm_col = col/np.linalg.norm(col)
        lighter_col = np.clip(col + 50*norm_col, 0, 255).astype(np.uint8)

        # Loop over all steps, copy average color to those coords.
        for mb_c in all_moved_block_corners:
            bds_y = np.clip(mb_c[1], 0, self.img_dims[1] - 1)
            bds_yy = np.clip(mb_c[1] + pb.h, 0, self.img_dims[1] - 1)
            bds_x = np.clip(mb_c[0], 0, self.img_dims[0] - 1)
            bds_xx = np.clip(mb_c[0] + pb.w, 0, self.img_dims[0] - 1)

            if 'L' in move_dir:
                self.img_smear[bds_xx, bds_y:bds_yy+1] = lighter_col
            else:
                self.img_smear[bds_x, bds_y:bds_yy+1] = lighter_col
            if 'D' in move_dir:
                self.img_smear[bds_x:bds_xx, bds_y] = col
            else:
                self.img_smear[bds_x:bds_xx, bds_yy] = col


        # new_x, new_y are final corner coords for translated block. new_xx, new_yy
        # are the coords of the other side of the block.
        new_x, new_y = pb.x + move_vec[0], pb.y + move_vec[1]
        new_xx = new_x + pb.w
        new_yy = new_y + pb.h

        new_x_in_bds = new_x.clip(0, self.img_dims[0])
        new_xx_in_bds = new_xx.clip(0, self.img_dims[0])
        new_y_in_bds = new_y.clip(0, self.img_dims[1])
        new_yy_in_bds = new_yy.clip(0, self.img_dims[1])


        # If it's off the edge on either side, the pixels from the block that
        # are copied to the new location start from the "front" or "back" of
        # the block's pixels.
        new_w = new_xx_in_bds - new_x_in_bds
        if new_x >= 0:
            starting_px_ind_x = 0
        else:
            starting_px_ind_x = -new_x

        new_h = new_yy_in_bds - new_y_in_bds
        if new_y >= 0:
            starting_px_ind_y = 0
        else:
            starting_px_ind_y = -new_y

        # Set the values of the img at the block's translated location
        self.img_smear[new_x_in_bds:new_xx_in_bds, new_y_in_bds:new_yy_in_bds] = pb.pixels[starting_px_ind_x:starting_px_ind_x+new_w, starting_px_ind_y:starting_px_ind_y+new_h]


    def avg_color(self, pb):

        # Gets avg color from a pixel block
        col_mean = np.mean(pb, axis=(0,1))
        return col_mean


    def vector_field(self, pos):
        # Returns 2-tuple representing vector field at that position (x, y)

        x, y = pos
        r = max(1, (x**2 + y**2)**0.5)

        theta_rot = 0.2*np.cos(2*np.pi*(r/self.img_r_max))
        #theta_rot = 0

        u_field = x*np.cos(theta_rot)/r + y*np.sin(theta_rot)/r
        v_field = -x*np.sin(theta_rot)/r + y*np.cos(theta_rot)/r

        #x_whorl = 100
        #y_whorl = -100

        x_whorl, y_whorl = self.img_center*2*(np.random.rand(2) - 0.5)

        whorl_rel_x = x - x_whorl
        whorl_rel_y = y - y_whorl

        whorl_rel_r = max(2, (whorl_rel_x**2 + whorl_rel_y**2)**0.5)
        theta_whorl = 0.5*np.pi

        if whorl_rel_r < self.img_r_max/3:
            whorl_scale_div = whorl_rel_r**1.4
        else:
            whorl_scale_div = whorl_rel_r**1.8

        u_whorl = whorl_rel_x*np.cos(theta_whorl)/whorl_scale_div + whorl_rel_y*np.sin(theta_whorl)/whorl_scale_div
        v_whorl = -whorl_rel_x*np.sin(theta_whorl)/whorl_scale_div + whorl_rel_y*np.cos(theta_whorl)/whorl_scale_div

        dot_prod = whorl_rel_x*x + whorl_rel_y*y
        if dot_prod < 0:
            u_whorl *= -1
            v_whorl *= -1

        w = 4.0

        u = u_field + w*u_whorl
        v = v_field + w*v_whorl

        m = np.linalg.norm([u, v])

        return np.array([u, v])/m






    def get_move_steps_path(self, init_pos, n_steps):

        pos = init_pos
        steps_list = [init_pos]
        for i in range(n_steps):

            vf = self.vector_field(pos)
            pos = pos + vf
            steps_list.append(pos)

        theta = 0.3*np.random.randn()
        #theta = 0
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        steps_list_rot = [np.matmul(rot_mat, s - steps_list[0]) for s in steps_list]
        steps_list = [s + steps_list[0] for s in steps_list_rot]
        return steps_list


    def draw_arrows_on_img(self, img):

        # Just a diagnostic tool. Note that you have to use np.flip() here because
        # even though the img array coords are (y, x), you draw on the image in
        # PIL with (x, y).

        draw = ImageDraw.Draw(img)
        c_w = self.block_width/10
        center_xy = np.flip(self.img_center, axis=0)

        # Center dot
        draw.ellipse([*(center_xy - 3*c_w), *(center_xy + 3*c_w)], fill = 'green', outline ='green')

        # Loop over pixel_blocks, draw orig and dest block centers, with line connecting.
        for pb in self.pixel_blocks:
            block_center = np.flip(pb.block_corner + np.array([pb.h, pb.w])/2.0, axis=0)
            block_center_dest = block_center + np.flip(self.vector_field(pb.block_center_rel), axis=0)
            draw.ellipse([*(block_center - c_w), *(block_center + c_w)], fill = 'red', outline ='red')
            draw.ellipse([*(block_center_dest - c_w), *(block_center_dest + c_w)], fill = 'blue', outline ='blue')
            draw.line([tuple(block_center), tuple(block_center_dest)], fill='yellow')












#
