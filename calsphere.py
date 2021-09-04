import numpy as np

coor = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]
points = np.array(coor) # coor为点云坐标的列表
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
print(center, r)