import time
import socket
import keyboard
import random
import numpy as np
import sys
from threading import Thread
from scipy.spatial.transform import Rotation as R
from natnet import MotionListener, MotionClient

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# host = '0.0.0.0'
# port = 8009
# s.bind((host, port))
# s.listen(1)
# c, addr = s.accept()
# print('Connected', addr)


class Caliberator:
    def __init__(self):
        self.nums = 200 # sample number
        self.steps = 1  # take 1 sample in steps
        self.coor = {}  # key is body_id, value is a list containing nums samples
        self.quat = {}  # key is body_id, value is a list containing nums quats according to samples
        self.center = {}
        self.docal = True
        self.offset = {}# key is body_id, value is a [x, y, z] offset in probe's coordinte system
        print("Caliberator built")

    def checkNotEnough(self):
        if len(self.coor) == 0:
            return True
        for k in self.coor:
            if len(self.coor[k]) < self.nums:
                return True
        return False


    def takeSamples(self, bodies, markers):
        if self.checkNotEnough():
            for i in range(len(bodies)):
                if not str(bodies[i].body_id) in self.coor:
                    self.coor[str(bodies[i].body_id)] = []
                    self.quat[str(bodies[i].body_id)] = []
                
                self.coor[str(bodies[i].body_id)].append([
                    round(bodies[i].position.x, 4), 
                    round(bodies[i].position.y, 4), 
                    round(bodies[i].position.z, 4)])
                # check if this sequence is correct
                self.quat[str(bodies[i].body_id)].append([
                    round(bodies[i].rotation.x, 4), 
                    round(bodies[i].rotation.y, 4), 
                    round(bodies[i].rotation.z, 4),
                    round(bodies[i].rotation.w, 4)])
        else:
            # print(self.coor)
            self.calSphere()
            self.caliberate()
            self.docal = False
            # sys.exit()
            

    def calSphere(self):
        for k in self.coor:
            points = np.array(self.coor[k])
            points = points.astype(np.float64)  # 防止溢出
            num_points = points.shape[0]
            print(num_points)
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            x_avr = sum(x) / num_points
            y_avr = sum(y) / num_points
            z_avr = sum(z) / num_points
            xx_avr = sum(x * x) / num_points
            yy_avr = sum(y * y) / num_points
            zz_avr = sum(z * z) / num_points
            xy_avr = sum(x * y) / num_points
            xz_avr = sum(x * z) / num_points
            yz_avr = sum(y * z) / num_points
            xxx_avr = sum(x * x * x) / num_points
            xxy_avr = sum(x * x * y) / num_points
            xxz_avr = sum(x * x * z) / num_points
            xyy_avr = sum(x * y * y) / num_points
            xzz_avr = sum(x * z * z) / num_points
            yyy_avr = sum(y * y * y) / num_points
            yyz_avr = sum(y * y * z) / num_points
            yzz_avr = sum(y * z * z) / num_points
            zzz_avr = sum(z * z * z) / num_points

            A = np.array([[xx_avr - x_avr * x_avr, xy_avr - x_avr * y_avr, xz_avr - x_avr * z_avr],
                        [xy_avr - x_avr * y_avr, yy_avr - y_avr * y_avr, yz_avr - y_avr * z_avr],
                        [xz_avr - x_avr * z_avr, yz_avr - y_avr * z_avr, zz_avr - z_avr * z_avr]])
            b = np.array([xxx_avr - x_avr * xx_avr + xyy_avr - x_avr * yy_avr + xzz_avr - x_avr * zz_avr,
                        xxy_avr - y_avr * xx_avr + yyy_avr - y_avr * yy_avr + yzz_avr - y_avr * zz_avr,
                        xxz_avr - z_avr * xx_avr + yyz_avr - z_avr * yy_avr + zzz_avr - z_avr * zz_avr])
            # print(A, b)
            b = b / 2
            center = np.linalg.solve(A, b)
            x0 = center[0]
            y0 = center[1]
            z0 = center[2]
            r2 = xx_avr - 2 * x0 * x_avr + x0 * x0 + yy_avr - 2 * y0 * y_avr + y0 * y0 + zz_avr - 2 * z0 * z_avr + z0 * z0
            r = r2 ** 0.5
            print(k, center, r)
            self.center[k] = center
            

    def caliberate(self):
        for k in self.center:
            n = random.randint(0, self.nums)
            Rm = np.matrix(R.from_quat(self.quat[k][n]))
            v = np.matrix(np.matrix(self.center[k]) - np.matrix(self.coor[k][n])).T
            vp = Rm * v # vp 应该是我们需要记录下来的一个向量，表示在probe坐标系下的笔尖位置
            self.offset[k] = vp
            print(k, vp)





class Listener(MotionListener):
    """
    A class of callback functions that are invoked with information from NatNet server.
    """
    def __init__(self, c: Caliberator):
        super(Listener, self).__init__()
        self.cali = c
        print("Listener successfully built")

    def on_version(self, version):
        print(version)
        print('Version {}'.format(version))
        strr='Version {}'.format(version)
        #c.send(bytes(str(version),encoding="UTF8"))
        #c.send(bytes(str(strr),encoding="UTF8"))

    def on_rigid_body(self, bodies, markers, time_info):
        if self.cali.docal:
            self.cali.takeSamples(bodies, markers)
            return
        strr=''
        # rigidbody pos + rotation
        # marker
        assert len(bodies) == len(markers)
        for i in range(len(bodies)):
            strr += str(bodies[i].body_id) + " pos: "
            strr += str(round(bodies[i].position.x, 4)) + " "
            strr += str(round(bodies[i].position.y, 4)) + " "
            strr += str(round(bodies[i].position.z, 4)) + ";"
            p = [round(bodies[i].position.x, 4),
                round(bodies[i].position.y, 4),
                round(bodies[i].position.z, 4)]
            p = np.matrix(p)
            strr += " quat: "
            strr += str(round(bodies[i].rotation.w, 4)) + " "
            strr += str(round(bodies[i].rotation.x, 4)) + " "
            strr += str(round(bodies[i].rotation.y, 4)) + " "
            strr += str(round(bodies[i].rotation.z, 4)) + ";"
            q = [round(bodies[i].rotation.x, 4), 
                round(bodies[i].rotation.y, 4), 
                round(bodies[i].rotation.z, 4),
                round(bodies[i].rotation.w, 4)]
            q = np.matrix(q)
            off = self.cali.offset[str(bodies[i].body_id)]
            tm = np.linalg.inv(np.matrix(R.from_quat(q)))
            v = tm * off
            tip = v + p
            strr += str(tip)

            # print(type(strr))
            # print(i, strr)
        print(1)
        file.write(strr + "\n")
        # c.send(bytes(str(strr),encoding="UTF8"))
        # i=i+1
        # if i<5000:
        #     fs.write(strr)
        #     fs.write("\n")
        # elif i==5000:
        #     fs.close()

        #print('RigidBodies {}'.format(bodies))
        #strr='RigidBodies {}'.format(bodies)
        #c.send(bytes(str(strr),encoding="UTF8"))

    def on_skeletons(self, skeletons, time_info):
        # print('Skeletons {}'.format(skeletons))
        strr='Skeletons {}'.format(skeletons)
        #c.send(bytes(str(strr),encoding="UTF8"))

    def on_labeled_markers(self, markers, time_info):
        # print('Labeled marker {}'.format(markers))
        strr='Labeled marker {}'.format(markers)
        # c.send(bytes(str(strr),encoding="UTF8"))
        # fs.write(strr)
        # fs.write("\n")

    def on_unlabeled_markers(self, markers, time_info):
        # print('Unlabeled marker {}'.format(markers))
        strr='Unlabeled marker {}'.format(markers)
        #c.send(bytes(str(strr),encoding="UTF8"))


if __name__ == '__main__':
    # set up a server
    # wait for quest
    # Create listener
    
    file = open("positionData.txt", "w")

    # Create a NatNet client
    c = Caliberator()
    listener = Listener(c)
    client = MotionClient(listener)

    time.sleep(10)

    # Data of rigid bodies and markers delivered via listener on a separate thread
    print("start getting data")
    client.get_data()
    

    keyboard.wait('q')

    # send to quest


    # Read version (optional)
    # client.get_version()

    # The client continuously reads data until client.disconnect() is called
    time.sleep(1)

    # Stops data stream and disconnects the client
    client.disconnect()
    # s.close()
    file.close()