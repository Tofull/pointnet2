#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np
import laspy

import np_util


class Pointcloud(object):
    def __init__(self, path: str):
        self.path = path
        self.load()

    def load(self):
        las_file = laspy.file.File(self.path, mode="r")
        self.columns = [spec.name for spec in las_file.reader.point_format]

        self.data = np.copy(las_file.points["point"])

        self.data = np_util.StructuredArray.sort_columns(self.data)
        las_file.close()

    def get_coordinates(self):
        las_file = laspy.file.File(self.path, mode="r")
        x = np.copy(las_file.x)
        y = np.copy(las_file.y)
        z = np.copy(las_file.z)

        self.coordinates = np.vstack([x, y, z]).transpose()
        las_file.close()

        return self.coordinates


class ExtendedPointcloud(Pointcloud):
    def __init__(self, template_path: str, destination_path, *args, **kwargs):
        super().__init__(template_path, *args, **kwargs)
        self.initial_file = laspy.file.File(self.path, mode="r")
        self.output_file = laspy.file.File(destination_path, mode="w", header=self.initial_file.header)

    def add_dimension(self, dimension_name, dimension_type=10, description='No description'):
        self.output_file.define_new_dimension(name=dimension_name, data_type=dimension_type, description=description)

    def copy_existing_dimensions(self):
        for field in self.initial_file.point_format:
            original_data = self.initial_file.reader.get_dimension(field.name)
            self.output_file.writer.set_dimension(field.name, original_data)

    def set_dimension(self, dimension_name, values):
        self.output_file._writer.set_dimension(dimension_name, values)

    def close(self):
        self.output_file.close()
        self.initial_file.close()


if __name__ == "__main__":
    import os.path

    pointcloud_path = os.path.join(os.path.dirname(__file__), '../data/jakarto/puisard/positif/5135246.las')
    pointcloud = Pointcloud(pointcloud_path)
    pointcloud.load()
    print(pointcloud.data)
    print(pointcloud.data.dtype)

    data_except_xyz = np_util.StructuredArray.drop_columns(pointcloud.data, ['X', 'Y', 'Z'])
    print(data_except_xyz, data_except_xyz.dtype)

    data_only_xyz = np_util.StructuredArray.keep_columns(pointcloud.data, ['X', 'Y', 'Z'])
    print(data_only_xyz, data_only_xyz.dtype)

    label_data, feature_data = np_util.StructuredArray.drop_a_column(data_except_xyz, 'puisard')
    print(feature_data, feature_data.dtype)
    print(label_data, label_data.dtype)

    print(label_data['puisard'])
