import time
from shutil import copy

# import matlab.engine
from nlos.components import *
from nlos.conf import *
from nlos.wallgeometry import *

if __name__ == '__main__':
    # Savedata  :  scan the wall and collect data
    # Getdata   : load stream image data from folders and plot them
    # GetComponents : plot a target interpolated point and its sources
    modes = ["getdata", "savedata", " getcomponents"]
    mode = modes[2]
    # cxn = "cx{}.txt".format(BOX_PER_UNIT)
    # cyn = "cy{}.txt".format(BOX_PER_UNIT)
    # with open(cxn,'w') as f:
    #     for i in cx:
    #         f.write(str(i)+' ')
    # with open(cyn,'w') as f:
    #     for i in cy:
    #         f.write(str(i)+' ')
    # exit()

    if mode == modes[0]:
        # folders = [("03_27_17_49", "04_01_18_14"), ("03_27_17_55", "04_01_18_25"), ("03_27_18_00", "04_01_18_36"),
        #            ("03_27_18_07", "04_01_18_49"), ("03_27_18_13", "04_01_19_02"), ("03_27_18_36", "04_02_09_47"),
        #            ("03_27_18_37", "04_02_09_50"), ("03_27_18_43", "04_02_10_03"), ("03_27_18_50", "04_02_10_16"),
        #            ("03_27_18_57", "04_02_10_29"), ("03_27_19_02", "04_02_10_40")]

        folders = [("04_07_20_08", "04_07_19_48")]
        folders = [folders[0]]
        for folder in folders:
            loadfromorig = os.getcwd() + ROOT_DIR + folder[1] + "/"
            loadfromint = os.getcwd() + ROOT_DIR + folder[0] + "/"
            getData(loadfromorig, loadfromint, num_points=MY_NUM_POINTS)
        exit()

    elif mode == modes[1]:
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
            #     with open(expt_root + "./conf.py", 'a') as f:
            #         s = "MY_MAG_X, MY_MAG_Y = {}, {}".format(MY_MAG_X, MY_MAG_Y)
            #         f.write(s)
            #
            #     print("Generating data for ", config)
            #     # saveData(root=expt_root, osx=MY_OSX, osy=MY_OSY, oex=MY_OEX, oey=MY_OEY, mode=MY_MODE, df1=MYDF1, df2=MYDF2,
            #     #          mag_x=MY_MAG_X, mag_y=MY_MAG_Y,
            #     #          filename=MY_DATA_FILE, N=MY_N)
            #
            try:
                saveData(root=expt_root, endpoints=MY_ENDPOINTS, mode=MY_MODE, df1=MYDF1, df2=MYDF2,
                         mag_x=MY_MAG_X, mag_y=MY_MAG_Y,
                         filename=MY_DATA_FILE, N=MY_N)
                print("Data generation done.")
            except:
                print("failed")
        exit()

    elif mode == modes[2]:
        folders = [("04_07_20_08", "04_07_19_48")]
        folders = [folders[0]]
        for folder in folders:
            loadfromorig = os.getcwd() + ROOT_DIR + folder[1] + "/"
            loadfromint = os.getcwd() + ROOT_DIR + folder[0] + "/"
            getComponents(loadfromorig, loadfromint, location = [2,125,125])
        exit()
