import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nlos.wallgeometry import *


def getStreakImageData(endpoints, mode, df1, df2, mag_x, mag_y, N):
    #     mode = '' #'stag' for staggered, 'rand' for randomised, anything else for sequential
    # N is the skip length while taking staggered points for measurements (0,N,2N,3N....1,2N+1,3N+1....)

    s = Surface()
    s.createSurface(endpoints)
    print("Object started at ")
    s.printSurface()

    objlist = s.getComponents()

    distortion_factor1 = df1
    distortion_factor2 = df2

    delta = 0.1

    nx_wall = len(wallPoints)
    ny_wall = 1
    n_time = TOTAL_T  # discretization along time axis

    deltaTime = 1

    streak_img_data = [np.zeros((nx_wall, n_time)) for _ in range(TOTAL_TIME)]
    speed1 = Point()
    speed1.set(mag_x * distortion_factor1, mag_y * distortion_factor1)

    speed2 = Point()
    speed2.set(mag_x * distortion_factor2, mag_y * distortion_factor2)

    sequential_index = list(range(len(wallPoints)))

    randomised_index = sequential_index.copy()
    random.shuffle(randomised_index)

    staggered_index = []
    for n in range(N):
        staggered_index.extend(sequential_index[n::N])

    if mode == 'stag':  # staggered (0,5,10,1,6,11,2,7,12...)
        index_to_use = staggered_index
    elif mode == 'rand':  # random (4,14,3,17,9,...)
        index_to_use = randomised_index
    else:  # normal (0,1,2,3,4...)
        index_to_use = sequential_index

    wp_delta = distance(wallPoints[0], wallPoints[1])

    # outer loop (time) is only useful for moving objects
    for time in range(0, TOTAL_TIME, deltaTime):
        for wp_index in index_to_use:
            wp = wallPoints[wp_index]
            pLit = wp
            pSeen = wp
            x = wp.getCoords()[0]

            for obj in objlist:
                diff = obj.getEnd() - obj.getStart()
                p = obj.getStart()
                p_new = movePoint(p, delta, diff)
                obj_delta = distance(p, p_new)

                while distance(p, obj.getEnd()) >= EPS:
                    r = distance(pSeen, p)
                    x_index = int((x - START_X) / DELTA_X)
                    t_index = int(T_DISCRETISATION_FACTOR * r)

                    p2 = pSeen

                    l2 = distance(p2, p)
                    c1 = wp_delta / (l2)
                    p_new = movePoint(p, delta, diff)
                    l3 = distance(p2, p_new)
                    c2 = obj_delta * 2 / (l2 + l3)

                    scaling_factor = 1
                    intensity_recd = (scaling_factor / (r ** 4)) * c1 * c2
                    streak_img_data[time][x_index][t_index] = streak_img_data[time][x_index][t_index] + intensity_recd
                    p = p_new

            moveSurface(s, speed2)
        moveSurface(s, speed1)

    print("Object ended at ")
    s.printSurface()
    print("speed1 ", speed1.getCoords())
    print("speed2 ", speed2.getCoords())

    return streak_img_data


def backprojection(streak_img_data):
    #     xs,ys = data[0].shape
    totalTime = len(streak_img_data)
    voxels = [np.zeros((bb_xnum, bb_ynum)) for _ in range(totalTime)]
    print("Bounding box resolution ", voxels[0].shape)
    for time in range(totalTime):
        xmax, tmax = streak_img_data[time].shape
        for x in range(xmax):
            print("t/total : ", time, "/", totalTime, "x : ", x, "/", xmax)
            for t in range(tmax):
                valid_dist = t / T_DISCRETISATION_FACTOR
                #                 add noise instead of doing this
                if (streak_img_data[time][x][t] >= EPS):
                    #                 if(True):
                    for vxi in range(voxels[0].shape[0]):
                        for vyi in range(voxels[0].shape[1]):
                            vx, vy = cx[vxi], cy[vyi]
                            wx = START_X + DELTA_X * x
                            wy = FIXED_Y
                            #                         print('1-')
                            #                         print('wx',wx,'vx',vx,'wy',wy,'vy',vy)
                            l = math.sqrt((wx - vx) ** 2 + (wy - vy) ** 2)  # distance between wallpoint and voxel
                            if abs(l - valid_dist) < (0.707) / BOX_PER_UNIT:
                                voxels[time][vxi][vyi] = voxels[time][vxi][vyi] + streak_img_data[time][x][t] * (l * l)
    return voxels


def hmap(voxels, filename="image"):
    graph = plt.figure(figsize=(20, 10))
    t = 0
    subplots = []
    totalTime = len(voxels)
    for data in voxels:
        df = pd.DataFrame(np.transpose(data))
        xs, ys = data.shape
        subplots.append(graph.add_subplot(totalTime, 1, t + 1))
        subplots[t].figsize = ((10 * xs / (xs + ys)), (10 * ys / (xs + ys)))
        subplots[t].set(title='t=' + str(t))
        sns.heatmap(df, cbar=False, cmap="YlGnBu", xticklabels=10, yticklabels=10).invert_yaxis()
        plt.gca().set_aspect('equal', anchor='C')
        t = t + 1

    plt.tight_layout()
    plt.savefig(filename + ".png")


def hmap_s(voxels, s1, s2, filename="image"):
    # t = int(s1)
    #     subplots=[]
    #     TOTAL_TIME = len(voxels)
    i, j = s1, s2
    for data in voxels:
        print(np.amax(data))
        graph = plt.figure(figsize=(30, 15))
        df = pd.DataFrame(np.transpose(data))
        xs, ys = data.shape
        #         subplots.append(graph.add_subplot(TOTAL_TIME,1,t+1))
        #         subplots[t].figsize=((10*xs/(xs+ys)),(10*ys/(xs+ys)))
        #         subplots[t].set(title='t='+str(t))
        sns.heatmap(df, cbar=True, cmap="YlGnBu", xticklabels=10, yticklabels=10).invert_yaxis()
        plt.gca().set_aspect('equal', anchor='C')
        plt.tight_layout()
        print('Saving...', filename + str(i) + str(i + 1) + "-" + str(j) + ".png")
        plt.savefig(filename + str(i) + str(i + 1) + "-" + str(j) + ".png")
        # plt.cla()
        # plt.clf()
        plt.close()
        j = j + 1
        # if j>2*s2:
        #     i = i+1
        #     j = 1


# collect the data by running the simulation and backproject
def run(root, endpoints, mode, df1, df2, mag_x, mag_y, filename, N):
    print("Scanning the wall and collecting data...")
    streak_img_data = getStreakImageData(endpoints, mode, df1, df2, mag_x, mag_y, N)
    print("Done. Backprojection started...")
    if mode == 'stag':
        totalTime = len(streak_img_data)
        l = streak_img_data[0].shape[0]  # number of points on the wall that were scanned
        ll = int(l / N)
        #         print("Full : ", np.count_nonzero(np.count_nonzero(streak_img_data,axis=1),axis=1))
        for n in range(N):
            new_streak_img_data = []
            #             new_streak_img_data = [streak_img_data[t][n*ll:(n+1)*ll] for t in range(TOTAL_TIME)]
            new_streak_img_data = [streak_img_data[t][n::N] for t in range(totalTime)]
            voxels = backprojection(new_streak_img_data)
            imagename = root + IMAGE_DIR + filename + "-" + mode + "-" + str(n)
            hmap(voxels, imagename)
    #             print("N: ", n)
    #             print(np.count_nonzero(np.count_nonzero(new_streak_img_data,axis=1),axis=1))
    if mode == 'seq' or mode == 'rand':
        voxels = backprojection(streak_img_data)
        imagename = root + IMAGE_DIR + filename + "-" + mode
        hmap(voxels, imagename)

    print("Done.")


# load the interpolated data from the saved files and backproject
def run2(root, loadfrom, s1, s2, mode='seq', filename="image", N=10):
    print("Reading files and collecting data...")
    data = []
    stepsize = 10
    if (os.path.exists(loadfrom)):
        for i in range(1, s1):
            for t in range(0, 2 * s2, stepsize):
                for j in range(t, t + stepsize):
                    if (j % 50 == 0):
                        print(i, j)
                    data.append(np.genfromtxt(loadfrom + 'int' + str(i) + str(i + 1) + '-' + str(j + 1) + '.csv',
                                              delimiter=','))
                streak_img_data = np.asarray(data)
                print("Done. Backprojection started...")
                print('Shape', streak_img_data.shape)
                if mode == 'stag':
                    totalTime = len(streak_img_data)
                    l = streak_img_data[0].shape[0]  # number of points on the wall that were scanned
                    ll = int(l / N)
                    #         print("Full : ", np.count_nonzero(np.count_nonzero(streak_img_data,axis=1),axis=1))
                    for n in range(N):
                        new_streak_img_data = []
                        #             new_streak_img_data = [streak_img_data[t][n*ll:(n+1)*ll] for t in range(TOTAL_TIME)]
                        new_streak_img_data = [streak_img_data[t][n::N] for t in range(totalTime)]
                        voxels = backprojection(new_streak_img_data)
                        imagename = root + INTERPOLATED_IMAGES_DIR_G + filename + "-" + mode + "-" + str(n)
                        hmap_s(voxels, i, t, imagename)
                #             print("N: ", n)
                #             print(np.count_nonzero(np.count_nonzero(new_streak_img_data,axis=1),axis=1))
                if mode == 'seq' or mode == 'rand':
                    voxels = backprojection(streak_img_data)
                    imagename = root + INTERPOLATED_IMAGES_DIR_G + filename + "-" + mode
                    hmap_s(voxels, i, t, imagename)
                #         for l in range(len(voxels)):
                #             print('Voxel Max',np.amax(voxels[l]))
                #             print('SID Max',np.amax(streak_img_data[l]))

                print("Done.")
                data = []

    else:
        print("Interpolated Data not found. Exiting...")
        return


# collects data of streak images formed on the wall from file and displays as a graph of intensity
# at a fixed point varying as a function of time
def getData(loadfromorig, loadfromint, num_points=3):
    print("Reading from file and collecting data...")
    jrange = int((END_X - START_X) / DELTA_X)
    # stepsize = 10
    data = []
    loadfromint = loadfromint + "/interpolated-data-f/"
    if (os.path.exists(loadfromint)):
        for i in range(1, 4):
            # for t in range(0, 2 * s2, stepsize):
            for j in range(0, jrange, 125):
                datapath = "{}int{}-{}.csv".format(loadfromint, i, j + 1)
                # data.append(np.genfromtxt(loadfrom + 'int' + str(i) + str(i + 1) + '-' + str(j + 1) + '.csv',
                #                           delimiter=','))
                data.append(np.genfromtxt(datapath, delimiter=','))
        streak_img_data_int = np.asarray(data)

    data = []
    loadfromorig = loadfromorig + "/data/"
    if (os.path.exists(loadfromorig)):
        for i in range(8):
            datapath = "{}data-{}.txt".format(loadfromorig, i)
            # data.append(np.genfromtxt(loadfrom + 'int' + str(i) + str(i + 1) + '-' + str(j + 1) + '.csv',
            #                           delimiter=','))
            data.append(np.genfromtxt(datapath))
        streak_img_data_orig = np.asarray(data)

    si = 0
    for streak_img_data in [streak_img_data_orig, streak_img_data_int]:
        lt = len(streak_img_data)
        lx = streak_img_data[0].shape[0]

        print("lx ", lx, " lt ", lt, " numpoints ", num_points)
        # lx, ly = lx * 10 / (lx + ly), ly * 10 / (lx + ly)
        xs = streak_img_data[0].shape[0]
        r = range(xs)
        if num_points != 0:
            point_indexes = list(r)[::int(len(r) / num_points)]
            if ((xs - 1) not in point_indexes):
                point_indexes.append(xs - 1)
        else:
            point_indexes = list(r)

        point_indexes = list(r)[-5:]

        print(xs)
        spatial_resolution = len(point_indexes)

        graph = plt.figure(figsize=((spatial_resolution * 8), 3 * lt))
        # graph = plt.figure()
        subplots = []
        # num_points = 3   # depending on number of wallpoints, the actual number of points might be 1 more than specified
        for t in range(lt):
            for i in range(spatial_resolution):
                idx = (t) * spatial_resolution + i
                # print("-------------")
                # print(t,i)
                # print(TOTAL_TIME, spatial_resolution, idx + 1)
                subplots.append(graph.add_subplot(lt, spatial_resolution, idx + 1))
                #             subplots[t].figsize=((10*xs/(xs+ys)),(10*ys/(xs+ys)))
                x = point_indexes[i]
                subplots[idx].set(title='T=' + str(t) + ' x=' + str(x))
                print("x = ", x)
                plt.plot(streak_img_data[t][x])

        plt.tight_layout()
        streak_img_dir = "{}/streak_images/".format(loadfromint)
        if not os.path.exists(streak_img_dir):
            os.makedirs(streak_img_dir, mode=0o777)
        # if streak_img_data == streak_img_data_orig:
        if si == 0:
            filename = "{}streakorig{}-{}.png".format(streak_img_dir, point_indexes[0], point_indexes[-1])
        else:
            filename = "{}streakint{}-{}.png".format(streak_img_dir, point_indexes[0], point_indexes[-1])
        plt.savefig(filename)

        print("Done. Output saved to ", streak_img_dir)
        si += 1


def saveData(root, endpoints, mode, df1, df2, mag_x, mag_y, filename, N):
    print("Scanning the wall and collecting data...")
    print("Scanning the wall")
    streak_img_data = np.asarray(getStreakImageData(endpoints, mode, df1, df2, mag_x, mag_y, N))
    lt = len(streak_img_data)
    print("Wall scanned")
    print("Writing data to file")
    for t in range(lt):
        np.savetxt(root + DATA_DIR + filename + '-' + str(t) + '.txt', streak_img_data[t])
    print("Data written")
    print("Backprojecting on original data...")
    voxels = backprojection(streak_img_data)
    print("Backprojecting on original data finished.")
    imagename = root + IMAGE_DIR + filename + '-' + 'orig_image' + "-" + mode
    print("Saving images ...")
    hmap_s(voxels, 0, 0, imagename)
    print("Saving images finished.")
