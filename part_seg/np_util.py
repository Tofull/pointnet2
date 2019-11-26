#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np


class StructuredArray(object):

    @staticmethod
    def drop_a_column(np_structured_array, column_to_remove):
        column_names = list(np_structured_array.dtype.names)
        index = column_names.index(column_to_remove)

        columns_to_keep = column_names[:index] + column_names[index + 1:]

        dropped_column = np_structured_array[[column_to_remove]]
        kept_columns = np_structured_array[columns_to_keep]
        return dropped_column, kept_columns

    @staticmethod
    def drop_columns(np_structured_array, columns_out):
        column_names = list(np_structured_array.dtype.names)
        return np.copy(np_structured_array[[column for column in column_names if column not in columns_out]])

    @staticmethod
    def keep_columns(np_structured_array, columns_in):
        column_names = list(np_structured_array.dtype.names)
        return np.copy(np_structured_array[[column for column in column_names if column in columns_in]])

    @staticmethod
    def sort_columns(np_structured_array):
        column_names = list(np_structured_array.dtype.names)
        return np.copy(np_structured_array[[column for column in sorted(column_names)]])
