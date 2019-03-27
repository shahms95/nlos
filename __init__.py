import time
from shutil import copy

# import matlab.engine
from nlos.components import *
from nlos.conf import *
from nlos.wallgeometry import *

if __name__ == '__main__':
    # cxn = "cx{}.txt".format(BOX_PER_UNIT)
    # cyn = "cy{}.txt".format(BOX_PER_UNIT)
    # with open(cxn,'w') as f:
    #     for i in cx:
    #         f.write(str(i)+' ')
    # with open(cyn,'w') as f:
    #     for i in cy:
    #         f.write(str(i)+' ')
    # exit()
    configs = [[0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0], [0, 5], [0, -5], [5, 0], [-5, 0], [0.5, 0.5], [-0.5, 0.5],
               [-0.5, -0.5], [0.5, -0.5]]

    for config in configs:
        MY_MAG_X, MY_MAG_Y = config[0], config[1]
        expt_root = os.getcwd() + ROOT_DIR + time.strftime('%m_%d_%H_%M') + "/"
        # expt_root = os.getcwd() + ROOT_DIR + "03_04_00_59/"

        data_path = expt_root + DATA_DIR
        images_path = expt_root + IMAGE_DIR
        int_data_path_g = expt_root + INTERPOLATED_DATA_DIR_G
        int_data_path_f = expt_root + INTERPOLATED_DATA_DIR_F

        int_images_path_g = expt_root + INTERPOLATED_IMAGES_DIR_G
        int_images_path_f = expt_root + INTERPOLATED_IMAGES_DIR_F

        if not os.path.exists(expt_root):
            os.makedirs(expt_root, mode=0o777)
        if not os.path.exists(data_path):
            os.makedirs(data_path, mode=0o777)
        if not os.path.exists(images_path):
            os.makedirs(images_path, mode=0o777)
        if not os.path.exists(int_data_path_g):
            os.makedirs(int_data_path_g, mode=0o777)
        if not os.path.exists(int_data_path_f):
            os.makedirs(int_data_path_f, mode=0o777)
        if not os.path.exists(int_images_path_g):
            os.makedirs(int_images_path_g, mode=0o777)
        if not os.path.exists(int_images_path_f):
            os.makedirs(int_images_path_f, mode=0o777)

        copy("./conf.py", expt_root + "./conf.py")
        with open(expt_root + "./conf.py", 'a') as f:
            s = "MY_MAG_X, MY_MAG_Y = {}, {}".format(MY_MAG_X, MY_MAG_Y)
            f.write(s)

        print("Generating data for ", config)
        # saveData(root=expt_root, osx=MY_OSX, osy=MY_OSY, oex=MY_OEX, oey=MY_OEY, mode=MY_MODE, df1=MYDF1, df2=MYDF2,
        #          mag_x=MY_MAG_X, mag_y=MY_MAG_Y,
        #          filename=MY_DATA_FILE, N=MY_N)

        try:
            saveData(root=expt_root, endpoints=MY_ENDPOINTS, mode=MY_MODE, df1=MYDF1, df2=MYDF2,
                     mag_x=MY_MAG_X, mag_y=MY_MAG_Y,
                     filename=MY_DATA_FILE, N=MY_N)
            print("Data generation done.")
        except:
            print(config, "failed")

        # print("Running backprojection on interpolated data")

        # for testing purposes, small values of s1 and s2 are to be used
        # run2(root=expt_root   , loadfrom=int_data_path_g, s1=2, s2=5, mode=MY_MODE, filename=MY_IMG_FILE, N=MY_N)

        # run2(root=expt_root, loadfrom=int_data_path_g, s1=4, s2=TOTAL_X, mode=MY_MODE, filename=MY_IMG_FILE, N=MY_N)
