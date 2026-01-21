SOLIDITY = 0.768
MIN_AREA = 0.15

REF_PATH = "../data/images/perfect_nobg.png"

cropped_image_ref_shape = (1014, 903, 3)
MIN_PTS_REF = [(848, 780), #bottrom right cote logo
 (537, 909),#bottom right 
 (291, 780),#bottom left
 (130, 505),#mid left
 (289, 234),#top left
 (611, 234),#top right
 (772, 507)]#mid right

YOLO_KEY_PTS_ID_MIN = [5,7,9,11,13,1,3] #among the 14 key points of my trained yolo models which id correspond to the min points of the star
#YOLO_KEY_PTS_ID_MAX = [] #among the 14 key points of my trained yolo models which id correspond to the max points of the star

MAX_PTS_REF =[(900, 754), (837, 897), (442, 1013), (4, 760), (7, 246), (457, 0), (897, 253)]

def get_centroid_ref(padding=0):
    height_triangle = 116
    side_triangle = 134
    start_y =507-4*height_triangle #top of our first triangle
    start_x= 450-3*side_triangle
    
    # Star layout: (visual_row, start_x, num_cells)
    star_layout = [
        (0, 5, 1),   # Top tip
        (1, 4, 3),
        (2, 0, 11),  # Wide row with left/right points
        (3, 1, 9),
        (4, 1, 9),
        (5, 0, 11),  # Wide row with left/right points
        (6, 4, 3),
        (7, 5, 1),   # Bottom tip
    ]
    triangles_coord={}
    triangles_orientation={}
    id=1
    for row, start_col, num_cells in star_layout:
        y = int(start_y+(row+0.5)*height_triangle)
        for i in range(num_cells):
            col = start_col+i
            if (row+col)%2==1:
                triangles_orientation[id] = "up"
            else:
                triangles_orientation[id] = "down"
            py = -padding if (row+col)%2==0 else padding
            x = int(start_x + (start_col+i+1)*side_triangle/2)
            triangles_coord[id] = (x,y+py)
            id+=1
    return triangles_coord,triangles_orientation    
CENTROID_REF = get_centroid_ref()[0]
TRIANGLES_ORIENTATION = get_centroid_ref()[1]   
cropped_image_ref_SHAPE = (1014, 903, 3) 

COORD_TRIANGLE_UP_REF =[(int(134/2),0),(0,116),(134,116)]
COORD_TRIANGLE_DOWN_REF =[(0,0),(134,0),(int(134/2),116)]

import numpy as np
import cv2
def create_mask_at(top_left,bottom_right,img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.rectangle(mask,top_left, bottom_right, (255), thickness=cv2.FILLED)
    return mask

top_left = 590,760
bottom_right = 860,950
mask_logo = create_mask_at(top_left,bottom_right,(1014, 903))

CELLS_GRAPH={1: [3],
 2: [3, 9],
 3: [2, 4, 1],
 4: [3, 11],
 5: [6],
 6: [5, 7, 16],
 7: [6, 8],
 8: [7, 9, 18],
 9: [8, 10, 2],
 10: [9, 11, 20],
 11: [10, 12, 4],
 12: [11, 13, 22],
 13: [12, 14],
 14: [13, 15, 24],
 15: [14],
 16: [17, 6],
 17: [16, 18, 26],
 18: [17, 19, 8],
 19: [18, 20, 28],
 20: [19, 21, 10],
 21: [20, 22, 30],
 22: [21, 23, 12],
 23: [22, 24, 32],
 24: [23, 14],
 25: [26, 35],
 26: [25, 27, 17],
 27: [26, 28, 37],
 28: [27, 29, 19],
 29: [28, 30, 39],
 30: [29, 31, 21],
 31: [30, 32, 41],
 32: [31, 33, 23],
 33: [32, 43],
 34: [35],
 35: [34, 36, 25],
 36: [35, 37],
 37: [36, 38, 27],
 38: [37, 39, 45],
 39: [38, 40, 29],
 40: [39, 41, 47],
 41: [40, 42, 31],
 42: [41, 43],
 43: [42, 44, 33],
 44: [43],
 45: [46, 38],
 46: [45, 47, 48],
 47: [46, 40],
 48: [46]}