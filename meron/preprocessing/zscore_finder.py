import os

import numpy as np
import pandas as pd


class Zscores(object):
    '''A calculator to determine the zscore of the following child growth indicators:

    Weight-for-height (WFH)
    Weight-for-length (WFL)
    Weight-for-age (WFA)
    Height-for-age (HFA)
    Length-for-age (LFA)

    Zscore tables have been compiled by the World Health Organization (WHO). Length should be used
    in place of height for children <= 2 years old.

    Parameters
    ----------
    data_dir : string
               Directory path to the WHO zscore tables.

    file_names : dictionary
                 Dictionary to over-ride default file names for WHO zscore tables. Keys should be
                 'lfa_boys' -- Length for age table, boys
                 'lfa_girls' -- Length for age table, girls
                 'hfa_girls' -- Height for age table, girls
                 'hfa_boys' -- Height for age table, boys
                 'wfa_girls' -- Weight for age table, girls
                 'wfa_boys' -- Weight for age table, boys
                 'wfh_girls' -- Weight for height table, girls
                 'wfh_boys' -- Weight for height table, boys
                 'wfl_girls' -- Weight for length table, girls
                 'wfl_boys' -- Weight for length table, boys
                 Values should be file names
    '''

    def __init__(self, data_dir, file_names={}):

        self.data = {}
        self.data_dir = data_dir
        self.fnames = {}
        self.age_cats = ['lfa', 'hfa', 'wfa']
        self.weight_cats = ['wfa', 'wfh', 'wfl']
        self.all_cats = ['lfa', 'hfa', 'wfa', 'wfh', 'wfl']

        # Default file names for growth indicators
        self.fnames['lfa_boys'] = 'lfa_boys.csv'
        self.fnames['lfa_girls'] = 'lfa_girls.csv'
        self.fnames['hfa_boys'] = 'hfa_boys.csv'
        self.fnames['hfa_girls'] = 'hfa_girls.csv'
        self.fnames['wfa_boys'] = 'wfa_boys.csv'
        self.fnames['wfa_girls'] = 'wfa_girls.csv'
        self.fnames['wfh_boys'] = 'wfh_boys.csv'
        self.fnames['wfh_girls'] = 'wfh_girls.csv'
        self.fnames['wfl_boys'] = 'wfl_boys.csv'
        self.fnames['wfl_girls'] = 'wfl_girls.csv'

        # Update with any user supplied file names
        self.fnames.update(file_names)

        # Read in child growth standard tables
        for k, val in self.fnames.items():
            self.data[k] = pd.read_csv(os.path.join(self.data_dir, val))

    def _calc_zscore(self, measure, ind, x):
        '''Calculate the zscore from the L, M, S parameters assuming a skewed normal distribution.

        Based on:
        Cole, T.J. and Green, P.J., 1992. Smoothing reference centile curves: the LMS method and
        penalized likelihood. Statistics in medicine, 11(10), pp.1305-1319.
        '''

        l, m, s = self.data[measure].iloc[ind][['L', 'M', 'S']].values

        if l == 0:
            zscore = np.log(x / m) / s
        else:
            zscore = (np.power(x / m, l) - 1) / (l * s)

        return np.round(zscore, 2)

    def _find_nearest(self, array, value):
        '''
        Helper function. Used to find the nearest length because it isn't an
        integer like age.
        '''
        idx = (np.abs(array.values - float(value))).argmin()
        val = array[idx]

        return val, idx

    def z_score(self, gender=1, measure='wfa', length=None, height=None, weight=None, age=None):
        '''Calculate the zscore for child growth indicator.

        Calculate the zscore for the child growth indicator based on the World Health Organization
        (WHO) zscore tables. Length should be used in place of height for children <= 2 years old.
        Only specify the specific measurements for the type of indicator to determine zscore. For
        example if indicator is 'wfa' one must specify weight and age, for indcator 'wfl' one
        must specify weight and length.

        Parameters
        ----------
        gender : integer
                 1 = boy (male)
                 0 = girl (female)
        measure : string
                  Type of growth indicator. Options are:
                  'lfa' -- Length for age (age <= 2)
                  'hfa' -- Height for age table (2 < age <= 5)
                  'wfa' -- Weight for age table (0 < age <= 5)
                  'wfh' -- Weight for height table (2 < age <= 5)
                  'wfl' -- Weight for length table (age <= 2)
        length : float
                 Length of child in cm. Measurement of individuals <= 2 years old. Only specify if
                 using measures: 'lfa', 'wfl'
        height : float
                 Height of child in cm. Measurement of individual 2 < age <= 5. Only specify if
                 using measures: 'hfa', 'wfh'
        weight : float
                 Weight of child in kg. Only specify if using measures: 'wfa', 'wfh', 'wfl'
        age : integer
              Age of child in months. Only specify if using measures: 'lfa', 'hfa', 'wfa'


        Returns
        -------
        zscore : float
                 Calculated zscore for growth indicator.
        '''

        measure = measure.lower()

        # Check inputs
        if not (measure in self.all_cats):
            raise ValueError("You must specify one of the following for measures: lfa, hfa, wfa, wfh, wfl")

        if measure == 'lfa':
            if (length is None) or (age is None):
                raise ValueError("For measure lfa, you must specify length and age")

        if measure == 'hfa':
            if (height is None) or (age is None):
                raise ValueError("For measure hfa, you must specify height and age")

        if measure == 'wfa':
            if (weight is None) or (age is None):
                raise ValueError("For measure wfa, you must specify weight and age")

        if measure == 'wfh':
            if (weight is None) or (height is None):
                raise ValueError("For measure wfh, you must specify weight and height")

        if measure == 'wfl':
            if (weight is None) or (length is None):
                raise ValueError("For measure wfl, you must specify weight and length")

        # Add gender
        if gender == 1:
            measure += '_boys'
        else:
            measure += '_girls'

        # Determine index for L,M,S values
        if any(t in measure for t in self.age_cats):
            ind = np.where(self.data[measure]['Month'] == age)[0][0]

        elif 'wfh' in measure:
            ind = self._find_nearest(self.data[measure]['Height'], height)[1]

        elif 'wfl' in measure:
            ind = self._find_nearest(self.data[measure]['Length'], length)[1]

        # Determine type of measure for zscorce calculation
        if any(t in measure for t in self.weight_cats):
            x = weight

        elif 'lfa' in measure:
            x = length

        elif 'hfa':
            x = height

        return self._calc_zscore(measure, ind, x)
