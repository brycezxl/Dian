import os
path = "/home/rlee/cross_line_detection/test_output_10_12/valid/INVALID/violation_20190201bef071b69f8adbdd5ac79ef6311d0b03_cropped_3.jpg"
folder = "/home/rlee/cross_line_detection/data/valid"

output = os.path.join(folder, path.split("/")[-2], path.split("/")[-1])
print(output)