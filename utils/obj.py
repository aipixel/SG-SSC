import os
import numpy as np
import math

PI = 3.14159265358979323846

class_colors = [
    (0.1, 0.1, 0.1),
    (0.0649613, 0.467197, 0.0667303),
    (0.1, 0.847035, 0.1),
    (0.0644802, 0.646941, 0.774265),
    (0.131518, 0.273524, 0.548847),
    (1, 0.813553, 0.0392201),
    (1, 0.490452, 0.0624932),
    (0.657877, 0.0505005, 1),
    (0.0363214, 0.0959549, 0.548847),
    (0.316852, 0.548847, 0.186899),
    (0.548847, 0.143381, 0.0045568),
    (1, 0.241096, 0.718126),
    (0.9, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.3, 0.3, 0.3)
    ]

class_names = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa",
               "table", "tvs", "furniture", "objects", "error1", "error2", "error3"]


def write_header(obj_file, mtl_file, name):
    obj_file.write("# EdgeNet360 Wavefront obj exporter v1.0\n")
    obj_file.write("mtllib %s.mtl\n" % os.path.basename(name))
    obj_file.write("o Cube\n")

    mtl_file.write("# EdgeNet360 Wavefront obj exporter v1.0\n")
    # Blender MTL File: 'DWRC1.blend'
    # Material Count: 11


def write_vertice(obj_file, x, y, z, cx, cy, cz, v_unit):
    vu = v_unit * 1
    obj_file.write("v %8.6f %8.6f %8.6f\n" %((x-cx)*vu, (y-cy)*vu, (z-cz)*vu))

def write_vertice_normals(obj_file):
    obj_file.write("vn -1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000 -1.000000\n")
    obj_file.write("vn  1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000  1.000000\n")
    obj_file.write("vn  0.000000 -1.000000  0.000000\n")
    obj_file.write("vn  0.000000  1.000000  0.000000\n")


def write_mtl_faces(obj_file, mtl_file, mtl_faces_list, cl, triangular):
    obj_file.write("g %s_%s\n" % (class_names[cl], str(cl)))
    obj_file.write("usemtl %s_%s\n" % (class_names[cl], str(cl)))
    #obj_file.write("s  off\n")
    mtl_file.write("newmtl %s_%s\n" % (class_names[cl], str(cl)))

    mtl_file.write("Ns 96.078431\n")
    mtl_file.write("Ka 1.000000 1.000000 1.000000\n")
    mtl_file.write("Kd %8.6f %8.6f %8.6f\n" % (class_colors[cl][0], class_colors[cl][1], class_colors[cl][2] ) )
    mtl_file.write("Ks 0.500000 0.500000 0.500000\n")
    mtl_file.write("Ke 0.000000 0.000000 0.000000\n")
    mtl_file.write("Ni 1.000000\n")
    mtl_file.write("d 1.000000\n")
    mtl_file.write("illum 2\n")


    if not triangular:
        for face_vertices in mtl_faces_list:

            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d %d//%d" % (
                                                                face_vertices[1]+1, face_vertices[0],
                                                                face_vertices[2]+1, face_vertices[0],
                                                                face_vertices[3]+1, face_vertices[0],
                                                                face_vertices[4]+1, face_vertices[0],
                                                            ))
            obj_file.write("\n")
    else:
        for face_vertices in mtl_faces_list:

            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d" % (
                                                            face_vertices[1]+1, face_vertices[0],
                                                            face_vertices[2]+1, face_vertices[0],
                                                            face_vertices[3]+1, face_vertices[0],
                                                        ))
            obj_file.write("\n")


def vox2obj(name, vox, shape, v_unit, include_top=True, triangular=False, inner_faces=True, complete_walls=False, colors=None, gap=0.1):
    global class_colors
    if colors is not None:
        class_colors = colors

    vox = np.flip(vox.reshape(shape),axis=0)


    num_classes=len(class_names)

    sx, sy, sz = vox.shape

    _vox = np.ones((sx+2,sy+2,sz+2), dtype=np.uint8)*255
    _vox[1:-1,1:-1,1:-1] = vox

    cx, cy, cz = int(sx//2), int(sy//2), int(sz//2)

    vox_ctrl = np.ones((sx+1, sy+1, sz+1), dtype=np.int32) * -1

    mtl_faces_list =[None] * num_classes

    num_vertices = 0

    with open(name+".obj", 'w') as obj_file, open(name+".mtl", 'w') as mtl_file:

        write_header(obj_file, mtl_file, name)

        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    mtl = int(vox[x,y,z])
                    if mtl == 0 or mtl==255:
                        continue
                    if not include_top and mtl==1:
                        continue
                    delta = [gap/2, 1-gap/2]
                    #delta = [0, 1]
                    for vx in range(2):
                        for vy in range(2):
                            for vz in range(2):
                                #if vox_ctrl[x+vx, y+vy, z+vz] == -1:
                                vox_ctrl[x + vx, y + vy, z + vz] = num_vertices
                                num_vertices += 1
                                write_vertice(obj_file, x+delta[vx], y+delta[vy], z+delta[vz], cx, cy, cz, v_unit)
                    if mtl_faces_list[mtl] is None:
                        mtl_faces_list[mtl] = []

                    if inner_faces:

                        if triangular:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+1]], #OK
                                [1, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+0, y+0, z+1]], #OK

                                [2, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+0, y+0, z+0]], #OK
                                [2, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+0, y+0, z+0]], #OK

                                [3, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+0]], #OK
                                [3, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+1], vox_ctrl[x+1, y+0, z+0]], #OK

                                [4, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+0, z+1]], #OK
                                [4, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+1, y+0, z+1]], #OK

                                [5, vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK
                                [5, vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK

                                [6, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+0]], #OK
                                [6, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+1, z+0]]  #OK
                            ])

                        else:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                 vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]  # OK
                            ])
                    else:
                        _x, _y, _z = x+1, y+1, z+1
                        if triangular:

                            if _vox[_x - 1,_y,_z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x-1,_y,_z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                    [1, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y, _z-1] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y, _z-1]!=255):
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                    [2, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]]])  # OK

                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x + 1, _y, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                    [3, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]]])  # OK

                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y, _z + 1]!=255):
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [4, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y-1, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y-1, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [5, vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y + 1, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]],  # OK
                                    [6, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

                        else:

                            if _vox[_x - 1, _y, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x - 1, _y, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y, _z - 1] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y, _z - 1]!=255):
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]]])  # OK
                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x + 1, _y, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]]])  # OK
                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y, _z + 1]!=255):
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y - 1, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y - 1, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and (complete_walls or _vox[_x, _y + 1, _z]!=255):
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

        write_vertice_normals(obj_file)

        for mtl in range(num_classes):
            if not  mtl_faces_list[mtl] is None:
                write_mtl_faces(obj_file, mtl_file, mtl_faces_list[mtl], mtl, triangular)

    return


def vol2obj(name, vol, shape, v_unit, gap=0.1):
    global class_colors

    vox = np.zeros(vol.shape, np.uint8)

    #vox[:,0:5,:][vol[:,0:5,:] < -1.] = 7
    #vox[:,30:,:][vol[:,30:,:] < -1.] = 7
    #vox[vol < -1.] = 11 #Outside room or fov
    vox[vol == -1.] = 2 #Occluded Empty
    vox[(vol > -1.) & (vol < 0.)] = 3
    vox[vol==0] = 7 #?
    vox[(vol > 0.) & (vol < 0.5)] = 4
    vox[(vol >= 0.5) & (vol < 1)] = 8
    #vox[(vol == 1.)] = 9 #Empty Visible
    vox[(vol > 1.)] = 10 #?

    #vox[:,4:,:] = 0


    vox2obj(name, vox, shape, v_unit, gap=gap)

def weights2obj(name, vol, shape, v_unit, gap=0.1):
    global class_colors

    vox = np.zeros(vol.shape, np.uint8)

    vox[vol < -1] = 7
    vox[vol == -1] = 8 #surface
    vox[(vol > -1.) & (vol < 0.)] = 6
    vox[(vol > 0.) & (vol < 1.)] = 2
    vox[(vol == 1.)] = 3 #occupied
    vox[(vol > 1.)] = 4 #Occluded empty

    vox2obj(name, vox, shape, v_unit, gap=gap)



class WFObject:
    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.normals = []
        self.faces = []

    def get_dim(self):
        x, y, z = [], [], []
        for face in self.faces:
            x.append(self.vertices[face[0][0]][0])
            y.append(self.vertices[face[1][0]][1])
            z.append(self.vertices[face[2][0]][2])
        return min(x), max(x), min(y), max(y), min(z), max(z)

    @staticmethod
    def __add_edge(edges, candidate):
        oposite = (candidate[1],candidate[0])
        if oposite in edges:
            edges.remove(oposite)
        else:
            edges.add(candidate)

    def get_face_polygons(self, h):
        polys = []
        for face in self.faces:
            if self.normals[face[0][1]][1] > 0.9 and \
               self.normals[face[1][1]][1] > 0.9 and \
               self.normals[face[2][1]][1] > 0.9 and \
               self.vertices[face[0][0]][1] > h-.1 and \
               self.vertices[face[1][0]][1] > h-.1 and \
               self.vertices[face[2][0]][1] > h-.1:
                    polys.append([[self.vertices[face[0][0]][0], self.vertices[face[0][0]][2]],
                                  [self.vertices[face[1][0]][0], self.vertices[face[1][0]][2]],
                                  [self.vertices[face[2][0]][0], self.vertices[face[2][0]][2]]])
        return np.array(polys, dtype=np.float32)

    def get_edges(self, h):

        edges = set()

        for face in self.faces:
            if self.normals[face[0][1]][1] > 0.9 and \
               self.normals[face[1][1]][1] > 0.9 and \
               self.normals[face[2][1]][1] > 0.9 and \
               self.vertices[face[0][0]][1] > h-.1 and \
               self.vertices[face[1][0]][1] > h-.1 and \
               self.vertices[face[2][0]][1] > h-.1:
                if (self.vertices[face[0][0]][0], self.vertices[face[0][0]][2]) != \
                   (self.vertices[face[1][0]][0], self.vertices[face[1][0]][2]):
                    self.__add_edge(edges,((self.vertices[face[0][0]][0], self.vertices[face[0][0]][2]),
                                           (self.vertices[face[1][0]][0], self.vertices[face[1][0]][2])))
                if (self.vertices[face[1][0]][0], self.vertices[face[1][0]][2]) != \
                   (self.vertices[face[2][0]][0], self.vertices[face[2][0]][2]):
                    self.__add_edge(edges,((self.vertices[face[1][0]][0], self.vertices[face[1][0]][2]),
                                           (self.vertices[face[2][0]][0], self.vertices[face[2][0]][2])))
                if (self.vertices[face[2][0]][0], self.vertices[face[2][0]][2]) != \
                   (self.vertices[face[0][0]][0], self.vertices[face[0][0]][2]):
                    self.__add_edge(edges,((self.vertices[face[2][0]][0], self.vertices[face[2][0]][2]),
                                           (self.vertices[face[0][0]][0], self.vertices[face[0][0]][2])))

        #print("Edges", len(edges), edges)
        return edges


def join(objects):

        the_obj = WFObject(objects[0].name)

        for obj in objects:
            the_obj.vertices.extend(obj.vertices)
            the_obj.normals.extend(obj.normals)
            the_obj.faces.extend(obj.faces)

        return the_obj


def read(fname, colapse=True):
    objects = []
    obj = None
    with open(fname, "r") as f:
        lines = f.readlines()

    v_count = 1
    vn_count = 1
    for line in lines:
        splits = line.split(sep=" ")
        if splits[0] == "o":
            o_v_count = v_count
            o_vn_count = vn_count

            obj = WFObject(splits[1][:-1])
            objects.append(obj)
        if splits[0] == "v":
            v_count += 1
            #obj.vertices.append([round(float(splits[1]),2), round(float(splits[2]),2), round(float(splits[3]),2)])
            obj.vertices.append([float(splits[1]), float(splits[2]), float(splits[3])])
        if splits[0] == "vn":
            vn_count += 1
            #obj.normals.append([round(float(splits[1]),2), round(float(splits[2]),2), round(float(splits[3]),2)])
            obj.normals.append([float(splits[1]), float(splits[2]), float(splits[3])])
        if splits[0] == "f":
            face = []
            for split in splits[1:]:
                face_data = split.split(sep="/")
                face.append([int(face_data[0]) - o_v_count, int(face_data[2]) - o_vn_count])

            obj.faces.append(face)

    if colapse:
        return join(objects)
    else:
        return objects

def get_edge_angle(e1,e2):
    x1 = e1[1][0]-e1[0][0]
    x2 = e2[1][0]-e2[0][0]
    y1 = e1[1][1]-e1[0][1]
    y2 = e2[1][1]-e2[0][1]

    if e1 == e2:
        angle = 0
    else:
        cos_angle = (x1*x2 + y1*y2)/(math.sqrt(x1*x1 + y1*y1)*math.sqrt(x2*x2+y2*y2))
        if cos_angle>1.0:
            cos_angle = 1.0
        if cos_angle<-1.0:
            cos_angle = -1.0

        angle = math.acos(cos_angle)

    if x2<x1:
        angle = 2*PI - angle

    return angle


def get_polygon(edges):

    n_edges = len(edges)

    if n_edges<3:
        return np.array([])

    max_z = -999999

    for edge in edges:
        if edge[0][1] > max_z:
            max_z = edge[0][1]
            v0 = edge[0]

    v_start = v0
    e0 = (v0,(v0[0],v0[1]+1))
    poly=[list(v0)]
    count=0

    while(True):
        print("edges", edges)
        print("v0", v0)

        count+=1
        #print("v0", v0)
        candidates = []
        for edge in edges:
            if edge[0] == v0:
                candidates.append(edge)
        print("candidates", candidates)
        min_angle = 361 #2*PI + 1
        for edge in candidates:
            angle = get_edge_angle(e0, edge) *180/PI
            #print("v0", v0, "v1", edge[1], angle)
            if angle<min_angle:
                min_angle = angle
                e0 = edge
                print("e0", e0)

        poly.append(list(e0[1]))
        edges.remove(e0)
        try:
            edges.remove((e0[1],e0[0]))
        except:
            pass


        v0 = e0[1]
        if v0 == v_start:
            break
        if count > n_edges:
            raise("Error finding boundary")

    #print("poly", poly)
    #return (Polygon(poly))
    return (np.array(poly))

