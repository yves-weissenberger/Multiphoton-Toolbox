import sys
import re
import csv
import numpy as np

if __name__== "__main__":

    fPath = sys.argv[1]

    with open(fPath) as f:
        reader = csv.reader(f)

        framelist = []
        for row in reader:

            fNum = int(re.findall(r".*_([0-9]{1,6})",row[0])[0])
            framelist.append(fNum)
    
    framelist = np.array(framelist)
    maxF = np.max(framelist)
    minF = np.min(framelist)
    sys.stdout.write("frames will be between %s and %s\n"%(0,maxF-minF))

    ans = str(raw_input("do you wish to proceed? (y/n): "))

    if ans=='y':
        with open(fPath) as f:

            fixPath = re.findall(r"(.*)\.txt",fPath)[0] + "__frame_fixed.txt"
            print fixPath

            with open(fixPath,'wb') as fix:

                reader = csv.reader(f)
                for idx,row in enumerate(reader):

                    write = re.findall("(.*_.*_)[0-9]{1,6}",row[0])[0]
                    fix.write(write + str(framelist[idx] - minF)+"\n")


    else:
        sys.stdout.write("Nevermind....")


    print "done!"