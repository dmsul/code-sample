import os
import socket
import datetime


# Check which machine we're on
HOST = socket.gethostname()
# Dan's
if HOST in ('sullivan-7d', 'sullivan-10d'):
    data_root = "D:\\"
    dropbox_root = os.path.join('c:\\users', 'sullivan', 'dropbox')
# Mark's
elif HOST == 'nepf-7d':
    data_root = "M:\\EPA_AirPollution\\"
    dropbox_root = os.path.join('C:\\Users', 'nepf', 'Dropbox')
# Fall back to Dan's computer over the network
else:
    data_root = r'\\Sullivan-10d\d'


GIS_SRC_PATH = os.path.join(data_root, 'Data', 'gis')

DATA_PATH = os.path.join(data_root, 'Data', 'mon-coverage')


now = datetime.datetime.now()
out_month = str(now.year)[-2:] + str(now.month).zfill(2)
OUT_PATH_ROOT = os.path.join(dropbox_root, 'research', 'mon-coverage', 'out')
OUT_PATH_MONTH = os.path.join(OUT_PATH_ROOT, out_month)


def data_path(*args):
    return os.path.join(DATA_PATH, *args)


def src_path(*args):
    return data_path('src', *args)


def gis_src_path(*args):
    return os.path.join(GIS_SRC_PATH, *args)


def out_path(*args):
    if not os.path.isdir(OUT_PATH_ROOT) and os.path.isdir(OUT_PATH_MONTH):
        os.makedirs(OUT_PATH_MONTH)
    return os.path.join(OUT_PATH_MONTH, *args)
