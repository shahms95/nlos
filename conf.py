EPS = 1e-6

# Wall Resolution
START_X = 0
END_X = 25
DELTA_X = 0.1
TOTAL_X = int((END_X - START_X) / DELTA_X)
FIXED_Y = 10

x = START_X
# N = 10
T_DISCRETISATION_FACTOR = 20
TOTAL_T = 20 * T_DISCRETISATION_FACTOR  # time axis is divided into this many units

TOTAL_TIME = 4  # TOTAL_TIME is number of times the entire wall is scanned
BOX_PER_UNIT = 25
MY_MODE = 'seq'

MYDF1 = 0*1e-3
MYDF2 = 1*1e-3

MY_SPEED = 'slow'
MY_N = 5
MY_NUM_POINTS = 8

# MY_MAG_X, MY_MAG_Y = -0.5, -0.5

bb0x, bb0y = 10, 0
bb1x, bb1y = 20, 0
bb2x, bb2y = 20, 10
bb3x, bb3y = 10, 10

MY_DATA_FILE = "data"
MY_IMG_FILE = "image"

ROOT_DIR = '/test/'
IMAGE_DIR = '/images/'
DATA_DIR = '/data/'
INTERPOLATED_DATA_DIR_G = '/interpolated-data-g/'
INTERPOLATED_DATA_DIR_F = '/interpolated-data-f/'
INTERPOLATED_IMAGES_DIR_G = '/interpolated-images-g/'
INTERPOLATED_IMAGES_DIR_F = '/interpolated-images-f/'

MY_ENDPOINTS = [[13, 5], [14, 5.5], [15, 5], [16, 5.5], [17, 5]]
MY_MAG_X, MY_MAG_Y = 0, 2.5