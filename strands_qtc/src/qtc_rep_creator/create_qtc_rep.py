#!/usr/bin/env python

import numpy as np


class QTCCreator():
    """Creating qtc states from positions"""
    def __init__(self, accuracy, no_collaps=True):
        self.accuracy = accuracy
        self.no_collaps = no_collaps
        self.pos1_array = []
        self.pos2_array = []
        self.qtc_rep = []

    def createQTCRep(self, pos1, pos2, accuracy):
        if not self.pos1_array:
            self.pos1_array.append(pos1)
        if not self.pos2_array:
            self.pos2_array.append(pos2)

        if len(self.pos1_array) <= 1 or len(self.pos2_array) <= 1:
            return self.qtc_rep

        RL = np.array([self.pos1_array[-2], self.pos2_array[-2]])
        RL_ext = np.array(
            [
                self.translate(pos1[-2], (pos1[-2]-pos2[-2])/2),
                self.translate(pos2[-2], (pos2[-2]-pos1[-2])/2)
            ]
        )
        rot_RL = self.orthogonalLine(
            np.array(
                [pos1[-2], pos2[-2]-pos1[-2]],
                pos1[-2]
            )
        )
        trans_RL_1 = self.translate(
            [rot_RL[0, 0:2], rot_RL[0, 2:4]],
            (rot_RL[0, 0:2]-rot_RL[0, 2:4])/2
        )
        trans_RL_2 = self.translate(
            trans_RL_1,
            (pos2[-2]-pos1[-2])
        )

    def translate(self, pose, trans_vec):
        """Translating pose by trans_vec.
        pose can be [x,y] or list of lists [[x,y],[x,y]]."""
        pose_array = np.array(pose)
        for ele in pose_array:
            ele = ele + np.array(trans_vec)
        return pose_array

    def orthogonalLine(self, point, line):
        """PERP = orthogonalLine(LINE, POINT);
        Returns the line orthogonal to the line LINE and going through the
        point given by POINT. Directed angle from LINE to PERP is pi/2.
        LINE is given as [x0 y0 dx dy] and POINT is [xp yp]."""
        point = np.array(point)
        line = np.array(line)
        n = max(point.shape[0], line.shape[0])

        if point.shape[0] > 1:
            res = point
        else:
            res = np.ones((n, 1)) * point

        if line.shape[0] > 1:
            res[:, 3] = -line[:, 4]
            res[:, 4] = line[:, 3]
        else:
            res[:, 3] = -np.ones((n, 1))*line[4]
            res[:, 4] = np.ones((n, 1))*line[3]

        res[:, 3] = res[:, 1]+res[:, 3]
        res[:, 4] = res[:, 2]+res[:, 4]

        return res

