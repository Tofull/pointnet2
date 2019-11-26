'''
    Dataset for Jakarto segmentation
'''

import os
import os.path
import json
import numpy as np
import numpy.lib.recfunctions
import pathlib
import sys

import loader
import np_util


from part_dataset_all_normal import pc_normalize, retrieve_original_pointcloud


class PartNormalJakartoDataset():
    def __init__(self, root, npoints=2500, classification=False, selector='train.txt', normalize=True, return_cls_label=False):
        # store parameters as instance variables
        self.npoints = npoints
        self.root = root
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        self.selector = selector

        # Get available categories
        self.categories = list(pathlib.Path(self.root).glob('*'))
        self.categories = list(filter(pathlib.Path.is_dir, self.categories))
        self.categories = list(map(lambda x: x.name, self.categories))

        self.datapath = []
        with open(pathlib.Path(self.root) / self.selector) as f:
            for line in f:
                file_path = line.strip()
                file_path = pathlib.Path(self.root) / file_path
                file_path = file_path.as_posix()
                category = pathlib.Path(file_path).parent.name

                self.datapath.append((category, file_path))

        self.datapath = list(set(self.datapath))

        self.classes = dict(zip(self.categories, range(len(self.categories))))
        self.seg_classes = {'negatif': [0], 'positif': [1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls, centroid, m = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)

            pointcloud = loader.Pointcloud(fn[1])

            point_set = pointcloud.get_coordinates()
            if self.normalize:
                point_set, centroid, m = pc_normalize(point_set)

            data_except_xyz = np_util.StructuredArray.drop_columns(pointcloud.data, ['X', 'Y', 'Z'])
            try:
                seg, feature_data = np_util.StructuredArray.drop_a_column(data_except_xyz, 'puisard')
                seg = numpy.lib.recfunctions.structured_to_unstructured(seg)
                seg = seg.ravel()
            except:
                feature_data = data_except_xyz
                seg = np.zeros(len(feature_data))

            n_x, feature_data = np_util.StructuredArray.drop_a_column(data_except_xyz, 'nx__0_05')
            n_y, feature_data = np_util.StructuredArray.drop_a_column(data_except_xyz, 'ny__0_05')
            n_z, feature_data = np_util.StructuredArray.drop_a_column(data_except_xyz, 'nz__0_05')

            n_x = numpy.lib.recfunctions.structured_to_unstructured(n_x)
            n_x = n_x.ravel()
            n_x = np.abs(n_x)

            n_y = numpy.lib.recfunctions.structured_to_unstructured(n_y)
            n_y = n_y.ravel()
            n_y = np.abs(n_y)

            n_z = numpy.lib.recfunctions.structured_to_unstructured(n_z)
            n_z = n_z.ravel()
            n_z = np.abs(n_z)

            normal = np.vstack([n_x, n_y, n_z]).transpose()
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls, centroid, m)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice].astype(np.int32)
        normal = normal[choice, :]
        if self.classification:
            return point_set, normal, cls, centroid, m
        else:
            if self.return_cls_label:
                return point_set, normal, seg, cls, centroid, m
            else:
                return point_set, normal, seg, centroid, m

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = PartNormalJakartoDataset(root='../data/jakarto/puisard', selector='train.txt', npoints=3000)
    print(len(dataset))

    i = 500
    ps, normal, seg, centroid, m = dataset[i]
    print(dataset.datapath[i])
    print(np.max(seg), np.min(seg))
    print(ps.shape, seg.shape, normal.shape)
    print(ps)
    print(normal)

    sys.path.append('../utils')
    import show3d_balls
    show3d_balls.showpoints(ps, normal + 1, ballradius=8)

    dataset = PartNormalJakartoDataset(root='../data/jakarto/puisard', classification=True, selector='train.txt', npoints=3000)
    print(len(dataset))
    ps, normal, cls, centroid, m = dataset[0]
    print(ps.shape, type(ps), cls.shape, type(cls))
