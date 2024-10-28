
import tarfile
import shutil
import os

final_out_file = "timestamped_gps_coordinate.txt"

output_file = "datasets/twitter.tar.gz"
file = "datasets/twitter.tar.gz.parta"
with open(output_file, 'wb') as outfile:
    for i in ['a','b','c','d','e']:
        part_file = file + i
        with open(part_file, 'rb') as infile:
            shutil.copyfileobj(infile, outfile)

with tarfile.open(output_file, 'r:gz') as tar:
    tar.extractall()

shutil.move(final_out_file, "datasets/"+final_out_file)
os.remove(output_file)