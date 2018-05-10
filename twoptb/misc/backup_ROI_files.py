#!/home/yves/anaconda2/bin/python
import h5py
import sys
import os
import shutil
from fnmatch import fnmatch, filter
from os.path import isdir, join
from shutil import copytree


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns

if __name__ == "__main__":
    print "starting copy"

    source = "/media/yves/imaging_yves/"
    shutil.copytree(source,
        "/home/yves/Documents/yves_ROI_backup/backup1/",
        ignore=include_patterns("*_ROIs.p"))
