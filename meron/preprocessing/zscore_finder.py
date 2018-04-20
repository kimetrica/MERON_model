import numpy as np
from numpy import interp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
import xlrd


class ZScores(object):

    def __init__(self, standards_file):

        self.data = {}
        self.interpolated = {}
        self.fname = standards_file

    def _get_sheet_names(self):
        '''
        Find all sheet names in the excel file.
        '''
        wb = xlrd.open_workbook(self.fname)
        sheet_list = wb.sheet_names()

        return sheet_list

    def _import_standards(self, sheet_list):
        '''
        Use sheet list from excel file to import all standards into self.data
        '''
        for sheet in sheet_list:
            self.data[sheet] = pd.read_excel(self.fname, sheet_name=sheet)

        return True

    def interpolate_zscores(self, measure, xnew, kind='linear'):
        '''
        Stores an interpolated curve for each z-score at each age/length.
        Use this if you just want to plot the WAZ/HAZ/WHZ for the UN standard.
        '''
        x = np.linspace(-4, 4, num=9, endpoint=True)
        self.interpolated[measure] = np.ones((self.data[measure].shape[0], xnew.shape[0]))*np.nan

        for i in self.data[measure].index:
            y = self.data[measure].loc[i][-9:self.data[measure].shape[1]]
            f = interp1d(x, y, kind=kind)
            self.interpolated[measure][i] = f(xnew)

    def _calculate_zscore(self, measure, ind, y_to_find):
        '''
        Calculate the zcore for a specific child.

        measure = column corresponding to WAZ, HAZ, or WHZ
        ind = the index of the age/length of the child
        y_to_find = the height/weight of the child to be converted to a zscore

        '''
        x = np.linspace(-4, 4, num=9, endpoint=True)

        y = self.data[measure].loc[ind][-9:self.data[measure].shape[1]]

        value = interp(y_to_find, y, x, left=-4, right=4)

        return value

    def _find_nearest(self, array, value):
        '''
        Helper function. Used to find the nearest length because it isn't an
        integer like age.
        '''
        idx = (np.abs(array - float(value))).argmin()
        val = array[idx]
        return val, idx

    def z_score(self, gender, measure='wfa', **kwargs):

        snames = self._get_sheet_names()
        self._import_standards(snames)

        if gender == 1:
            measure += '_boys'
        else:
            measure += '_girls'

        if ('wfa' in measure) or ('hfa' in measure):
            if kwargs['age'] > 60:
                return np.nan

            ind = np.where(self.data[measure]['Month'] == kwargs['age'])[0][0]

        else:
            if kwargs['Height'] > 110:
                return np.nan

            ind = self._find_nearest(self.data[measure]['Height'], kwargs['height'])[1]

        if ('wfa' in measure) or ('wfh' in measure):
            y = kwargs['weight']
        else:
            y = kwargs['height']

        return self._calculate_zscore(measure, ind, y)
