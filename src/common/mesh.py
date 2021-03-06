import numpy as np
#https://github.com/dranjan/python-plyfile
from skimage.measure import marching_cubes

class Camera(object):
    pass

class Mesh(object):
    '''
    class for storing mesh data eg as read from a ply file
    '''
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.norms = []

    def load_from_ply(self, filename):
        '''
        loads faces and vertices from a ply file
        '''
        from plyfile import PlyData, PlyElement
        f = open(filename, 'r')
        plydata = PlyData.read(f)
        f.close()
        self.vertices, self.faces = self._extract_plydata(plydata)

    def load_from_obj(self, filename):
        '''
        loads faces and vertices from a ply file
        '''
        with open(filename, 'r') as f:
            for l in f:
                split_line = l.strip().split(' ')
                if split_line[0] == '#':
                    continue
                elif split_line[0] == 'f':
                    self.faces.append([int(split_line[1]) - 1,
                                       int(split_line[2]) - 1,
                                       int(split_line[3]) - 1])
                elif split_line[0] == 'v':
                    self.vertices.append([float(split_line[1]),
                                          float(split_line[2]),
                                          float(split_line[3])])

        self.faces = np.array(self.faces)
        self.vertices = np.array(self.vertices)

    def write_to_obj(self, filename, labels=None):

        with open(filename, 'w') as f:

            f.write("# OBJ file \n")
            f.write("# generated by mfirman in a fit of rage\n")

            if labels != None:  # seems unpythonic but necessary when checking existance of numpy array...
                for vertex, label in zip(self.vertices, labels):
                    if label == 0:
                        f.write("usemtl grey\n")
                        f.write("v %.4f %.4f %.4f\n" %(vertex[0], vertex[1], vertex[2]))
                    elif label==1:
                        f.write("usemtl red\n")
                        f.write("v %.4f %.4f %.4f 0.7 0.1 0.1\n" %(vertex[0], vertex[1], vertex[2]))
            else:
                for vertex in self.vertices:
                    f.write("v %.4f %.4f %.4f\n" %(vertex[0], vertex[1], vertex[2]))
                    #f.write("v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")

            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0]+1, face[1]+1, face[2]+1))

    def write_to_ply(self, filename, labels=None, colours=None):

        with open(filename, 'w') as f:

            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('comment author: Michael Firman\n')
            f.write('element vertex %d\n' % self.vertices.shape[0])
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if labels is not None or colours is not None:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            f.write('element face %d\n' % self.faces.shape[0])
            f.write('property list uchar int vertex_indices\n')
            f.write('end_header\n')

            if labels is not None:
                print "Labels: ", labels.shape, labels.sum()
                for label, v in zip(labels, self.vertices):
                    if label == 0:
                        f.write("%f %f %f 128 128 128\n" % (v[0], v[1], v[2]))
                    elif label == 1:
                        f.write("%f %f %f 200 0 0\n" % (v[0], v[1], v[2]))
            elif colours is not None:
                assert colours.shape == self.vertices.shape
                print "Colours: ", colours.shape, colours.sum()
                for col, v in zip(colours, self.vertices):
                    f.write("%f %f %f %d %d %d\n" % (
                        v[0], v[1], v[2], col[0], col[1], col[2]))
            else:
                for v in self.vertices:
                    f.write("%f %f %f\n" % (v[0], v[1], v[2]))

            if self.faces.shape[1] == 3:
                for face in self.faces:
                    f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))
            else:
                for face in self.faces:
                    f.write("4 %d %d %d %d\n" % (
                        face[0], face[1], face[2], face[3]))

            f.write('element vertex %d\n' % self.vertices.shape[0])

    def read_from_obj(self, filename):
        '''
        not very good or robust - designed to work on simple files!
        '''
        self.faces = []
        self.vertices = []

        with open(filename, 'r') as f:

            for line in f:
                split_line = line.strip().split(' ')

                if split_line[0] == 'v':
                    self.vertices.append([float(v) for v in split_line[1:]])

                elif split_line[0] == 'f':
                    self.faces.append([int(f)-1 for f in split_line[1:]])

                elif split_line[0] == '#':
                    pass

    def scale_mesh(self, scale):
        '''
        applys a scaling factor to the mesh vertices
        '''
        self.vertices *= scale

    def centre_mesh(self):
        '''
        centers the mesh such that the mean of the vertices is at (0, 0, 0)
        '''
        self.vertices = np.array(self.vertices)
        self.vertices -= np.mean(self.vertices, axis=0)

    def _extract_plydata(self, plydata):
        '''
        unpacks the structured array into standard np arrays
        '''
        vertices = plydata['vertex'].data
        np_vertex_data = vertices.view(np.float32).reshape(vertices.shape + (-1,))

        faces = np.zeros((plydata['face'].data.shape[0], 3), dtype=np.int32)
        for idx, t in enumerate(plydata['face'].data):
            faces[idx, :] = t[0]

        return np_vertex_data, faces

    def apply_transformation(self, trans):
        '''
        apply a 4x4 transformation matrix to the vertices
        '''
        n = self.vertices.shape[0]
        temp = np.concatenate((self.vertices, np.ones((n, 1))), axis=1).T
        temp_transformed = trans.dot(temp).T
        for idx in [0, 1, 2]:
            temp_transformed[:, idx] /= temp_transformed[:, 3]
        self.vertices = temp_transformed[:, :3]

    def _normalize_v3(self, arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def compute_vertex_normals(self):
        '''
        https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
        '''
        norms = np.zeros(self.vertices.shape, dtype=self.vertices.dtype)
        #Create an indexed view into the vertex array using the array of three indices for triangles
        tris = self.vertices[self.faces]
        #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        # we need to normalize these, so that our next step weights each normal equally.
        n = self._normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norms[self.faces[:, 0]] += n
        norms[self.faces[:, 1]] += n
        norms[self.faces[:, 2]] += n
        norms = self._normalize_v3(norms)

        self.norms = norms

    def range(self):
        '''
        returns a 1 x 3 vector giving the size along each dimension
        '''
        return np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)

    def from_volume(self, voxel_grid, level=0):
        '''
        generates a mesh from a volume voxel_grid, using marching cubes
        voxel_grid should be a voxel grid object.
        This allows for the coordinates of the found mesh to be in world space.
        '''
        temp_verts, temp_faces, _, _ = marching_cubes(voxel_grid.V, level)

        self.vertices = voxel_grid.idx_to_world(temp_verts)
        self.faces = temp_faces

    def remove_nan_vertices(self):
        '''
        Removes the nan vertices.
        Quite hard as must preserve the face indexing...

        Currently we just remove the faces which are attached to nan vertices
        We do not renumber face indices, so we cannot remove the nan vertices.
        '''

        verts_to_remove = np.any(np.isnan(self.vertices), axis=1)

        # generate a dictionary of verts to remove
        to_remove_dict = {vert: 1 for vert in np.where(verts_to_remove)[0]}

        faces_to_remove = np.zeros((self.faces.shape[0], ), dtype=np.bool)
        for idx, face in enumerate(self.faces):
            faces_to_remove[idx] = (face[0] in to_remove_dict or
                                    face[1] in to_remove_dict or
                                    face[2] in to_remove_dict)

        self.faces = self.faces[~faces_to_remove, :]
