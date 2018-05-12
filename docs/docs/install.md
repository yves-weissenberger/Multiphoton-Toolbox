# Installation

## Main package
The package has many dependencies so by far the easiest way to install the package is to download <a href="https://www.anaconda.com/download/"> Anaconda 2.7</a>. Thereafter only two additional packages need to be installed. These are:

<li> tifffile </li>
<li> pyqtgraph </li>

The latter from <a href="http://www.pyqtgraph.org/"> here </a> cross plattform. For unix (mac+linux) users, the former is easily obtained by simply running 

    pip install tifffile

while for windows, it can be downloaded from <a href="https://www.lfd.uci.edu/~gohlke/pythonlibs/#tifffile"> here </a> and installed by running:

    pip install /path/to/tifffile‑2018.5.10‑cp27‑cp27m‑win_amd64.whl

assuming you are running a 64 bit version of windows.

Thereafter simply download the repository from <a href="https://github.com/yves-weissenberger/twoptb"> here </a> and install it by running

    pip install . 

in the first twoptb directory (i.e. the one containing setup.py).


## Spike Extraction

The spike extraction algorithm currently used is <a href="https://github.com/lucastheis/c2s"> c2s </a>. For windows users hopes for installing this package are slim at best while for unix users the installation process seems to work fine.