# scale_data.py
# @author: Austin Shin

# convert to txt file
filename = "heart-scan3d.txt"
file = open(filename, "r")

# writeTo = "heart_scan_processed.txt"
# F = open(writeTo, "w")

# write scaled data to new txt file
scaled = "heart_scan_scaled.txt"
F = open(scaled, "w")
for line in file:
    if line.startswith("v "):
        newline = line[2:]
        coords = map(float, newline.split())
        newcoords = [ i * 2 for i in coords ] # scale factor determined by user
        line = " ".join(map(str,newcoords))
	line = line+"\n"
        F.write(line)
